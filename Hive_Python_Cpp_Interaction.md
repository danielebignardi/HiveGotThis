# Hive GNN — Python ↔ C++ Interaction Spec (libtorch / TorchScript)

**Purpose:** define how the value network crosses from Python (training) to C++ (inference inside the MCTS). Companion to the main encoding spec; it does not repeat the feature tables.

**Chosen runtime:** libtorch with TorchScript.

---

## 1. Mental model: the two languages do not talk at runtime

The network is *born* in Python (PyTorch / PyTorch Geometric, where it's defined and trained) but must *run* in C++ (the MCTS calls the evaluation thousands of times per move). The standard way to bridge this is **not** a live connection between the two. There is no Python interpreter running during gameplay, no socket, no embedded calls.

Instead, the bridge is a **file**. Think of it like compiling:

- Python builds and trains the network, then **exports** it (architecture + learned weights) into a single self-contained model file.
- C++ **loads** that file once at startup and runs it directly, with no Python anywhere.

The interaction is therefore asynchronous and file-mediated: **Python writes a model file, C++ reads it.** Python's job ends the moment the file is written.

---

## 2. The three phases

```
  Phase 1: TRAIN (Python)         Phase 2: EXPORT (Python)        Phase 3: INFER (C++)
  ┌─────────────────────┐         ┌────────────────────┐         ┌─────────────────────┐
  │ encoded tensors      │         │ trained model       │        │ load model file once │
  │ + outcome labels     │  ─────▶ │  → TorchScript .pt  │ ─────▶ │ per leaf:            │
  │ train HiveValueGNN   │         │  (weights+graph)    │  file  │  encode → tensors    │
  │ (PyTorch Geometric)  │         │                     │        │  → forward → value   │
  └─────────────────────┘         └────────────────────┘         └─────────────────────┘
```

The model file (`hive_value_gnn.pt`) is the only artifact that crosses the boundary.

---

## 3. The single-encoder architecture (recommended)

The feature computation (articulation points for break-one-hive, the pinned check, degrees) requires real Hive logic, which lives in C++. The encoder must run in C++ for inference anyway, since self-play happens inside the C++ MCTS. So:

- **There is one encoder, in C++.** It produces the four flat arrays (`x`, `edge_index`, `edge_attr`, `u`).
- **For inference:** the encoder runs live in the MCTS leaf evaluation.
- **For training data:** during self-play, C++ serializes those same arrays plus the game outcome to disk. Python reads them and trains — Python never reconstructs boards or reimplements Hive logic.

**Consequence:** there is no C++↔Python *encoder* parity problem, because there is only one encoder. The validation that matters instead is **export fidelity** — that the TorchScript file reproduces the Python model's outputs (§5). A Python encoder is optional, only as a debugging aid.

### Data augmentation is a pure tensor op

None of the node features are directional (piece type, color, degree, depth, pinned, break-hive, distances, adjacency flags are all rotation-invariant). A `D6` board symmetry therefore changes **only** the planar direction label of each edge. So the 12 symmetries are generated in Python, on the already-serialized tensors, by **permuting the 6 planar slots of `edge_attr`** (slots 6–8, up/down/self, are unchanged; `edge_index`, `x`, and `u` are unchanged). No board reconstruction needed.

---

## 4. Phase 2 — Export (Python)

### Use scripting, not tracing

`torch.jit.trace` records a single execution with **fixed** tensor shapes and control flow — wrong for a GNN, where `N` (nodes) and `E` (edges) vary per board. Use `torch.jit.script`, which compiles the actual code and preserves dynamic shapes.

```python
import torch

model = HiveValueGNN(node_in_dim=18, edge_in_dim=9, u_dim=21)
model.load_state_dict(torch.load("weights.pt"))
model.eval()                       # disables Dropout — MANDATORY before export/inference

scripted = torch.jit.script(model)
scripted.save("hive_value_gnn.pt")
```

### PyG layers and TorchScript

PyTorch Geometric's message-passing layers use scatter/gather operations over variable-size graphs, and need to be made TorchScript-compatible before scripting. Historically PyG exposes a `.jittable()` conversion on message-passing layers for exactly this; the precise API has changed across PyG versions, so **check the JIT section of the docs for your installed PyG version**. This is the single most likely place the export path breaks for a GNN — which is why it must be tested **early, with an untrained model** (§9), not after training.

### What the file contains

A self-contained package: the compiled forward graph **plus** the learned weights. It does not depend on your Python source after export.

---

## 5. Export-fidelity validation (replaces the old parity test)

Immediately after exporting, confirm the scripted model matches the eager model on the same input, in eval mode (so Dropout is off on both, otherwise they differ by design):

```python
model.eval(); scripted.eval()
with torch.no_grad():
    a = model(x, edge_index, edge_attr, u, batch)
    b = scripted(x, edge_index, edge_attr, u, batch)
assert torch.allclose(a, b, atol=1e-5)
```

This is the validation that actually guards correctness in the single-encoder design.

---

## 6. Phase 3 — Inference in C++

### Load once at startup

```cpp
#include <torch/script.h>

torch::jit::script::Module module = torch::jit::load("hive_value_gnn.pt");
module.eval();                       // disable Dropout at inference
```

### One leaf evaluation (single graph)

The `GNNInputs` flat buffers map directly onto tensors via `torch::from_blob`, which **wraps existing memory with no copy** — the same zero-copy reason the encoder uses flat 1-D arrays. You supply the buffer pointer and declare the shape.

```cpp
torch::NoGradGuard no_grad;          // no autograd at inference

GNNInputs g = BoardEncoder::encode(board);
const int64_t N = g.x.size() / 18;
const int64_t E = g.edge_index.size() / 2;

auto fopt = torch::TensorOptions().dtype(torch::kFloat32);
auto iopt = torch::TensorOptions().dtype(torch::kInt64);   // index tensors MUST be int64

torch::Tensor x          = torch::from_blob(g.x.data(),          {N, 18}, fopt);
torch::Tensor edge_index = torch::from_blob(g.edge_index.data(), {2, E},  iopt);
torch::Tensor edge_attr  = torch::from_blob(g.edge_attr.data(),  {E, 9},  fopt);
torch::Tensor u          = torch::from_blob(g.u.data(),          {1, 21}, fopt);  // batch dim of 1
torch::Tensor batch      = torch::zeros({N}, iopt);        // all nodes belong to graph 0

std::vector<torch::jit::IValue> inputs = {x, edge_index, edge_attr, u, batch};
torch::Tensor out = module.forward(inputs).toTensor();     // shape [1, 1]
float value = out.item<float>();                           // in [-1, 1], side-to-move perspective
```

> **Critical — buffer lifetime.** `from_blob` does **not** copy or own the memory; the tensor aliases `g`'s vectors. `g` must stay alive until `forward` returns. If you reuse a scratch `GNNInputs` across calls (recommended to avoid allocation churn), never refill it while a tensor still points into it. If you need to keep a tensor around, `.clone()` it.

The side-to-move perspective and the negamax sign flip (`-v` going up the tree) are handled in the MCTS, not here.

---

## 7. Batching leaf evaluations (the real loop)

The MCTS must not call `forward` one leaf at a time. Collect a batch of `B` boards, encode each, and concatenate into one big graph — exactly what PyTorch Geometric's `Batch` does internally: stack all nodes, **offset each graph's edge IDs by the running node count**, and build a `batch` vector mapping each node to its graph index.

```cpp
std::vector<float>   Xb, EAb, Ub;
std::vector<int64_t> SRC, DST, BATCH;
int64_t node_offset = 0;

for (int b = 0; b < boards.size(); ++b) {
    GNNInputs g = BoardEncoder::encode(boards[b]);
    const int64_t N = g.x.size() / 18;
    const int64_t E = g.edge_index.size() / 2;

    Xb.insert(Xb.end(), g.x.begin(), g.x.end());
    EAb.insert(EAb.end(), g.edge_attr.begin(), g.edge_attr.end());
    Ub.insert(Ub.end(), g.u.begin(), g.u.end());
    BATCH.insert(BATCH.end(), N, (int64_t)b);                 // this graph's nodes → graph b

    for (int64_t k = 0; k < E; ++k) SRC.push_back(g.edge_index[k]     + node_offset);
    for (int64_t k = 0; k < E; ++k) DST.push_back(g.edge_index[E + k] + node_offset);
    node_offset += N;
}

// edge_index [2, sumE] row-major = [all sources..., all destinations...]
std::vector<int64_t> EI;
EI.reserve(SRC.size() + DST.size());
EI.insert(EI.end(), SRC.begin(), SRC.end());
EI.insert(EI.end(), DST.begin(), DST.end());

const int64_t sumN = node_offset;
const int64_t sumE = (int64_t)SRC.size();
const int64_t B    = (int64_t)boards.size();

torch::Tensor x          = torch::from_blob(Xb.data(),  {sumN, 18}, fopt);
torch::Tensor edge_index = torch::from_blob(EI.data(),  {2, sumE},  iopt);
torch::Tensor edge_attr  = torch::from_blob(EAb.data(), {sumE, 9},  fopt);
torch::Tensor u          = torch::from_blob(Ub.data(),  {B, 21},    fopt);
torch::Tensor batch      = torch::from_blob(BATCH.data(),{sumN},    iopt);

auto out = module.forward({x, edge_index, edge_attr, u, batch}).toTensor();  // [B, 1]
// out[b] is the value for boards[b]
```

The model's `global_mean_pool`/`global_max_pool` use the `batch` vector to keep the `B` graphs separate, so you get `B` independent values from one forward call.

---

## 8. Build & linking (libtorch)

1. Download the libtorch C++ distribution from pytorch.org matching your PyTorch version. On Linux with a modern GCC, take the **cxx11 ABI** build — mismatched ABI causes cryptic link errors.
2. CMake:

```cmake
find_package(Torch REQUIRED)
add_executable(hive_engine main.cpp board_encoder.cpp ...)
target_link_libraries(hive_engine "${TORCH_LIBRARIES}")
set_property(TARGET hive_engine PROPERTY CXX_STANDARD 17)
```

3. Configure pointing at the unpacked libtorch: `cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..`

libtorch is a large dependency; that bulk is the main cost of choosing it over ONNX, and it buys you operations that behave identically to training.

---

## 9. Performance & threading notes

- **`module.eval()` + `torch::NoGradGuard`** are both required at inference. `eval()` disables the Dropout in the value head (otherwise inference is stochastic); `NoGradGuard` skips autograd bookkeeping. Note the network uses LayerNorm, not BatchNorm, so there are no batch statistics that differ between train and eval — Dropout is the only train/eval difference.
- **Intra-op threads:** libtorch parallelizes single ops with its own thread pool. For many tiny graphs (≤28 nodes) that overhead can hurt. If you run a multi-threaded MCTS, prefer `torch::set_num_threads(1)` and parallelize across MCTS workers instead — but measure.
- **Always use release builds** for both the C++ engine and any vector indexing benchmarks.

---

## 10. Gotchas checklist

- `from_blob` aliases memory — keep the source buffers alive through `forward`; `.clone()` if you store the tensor.
- `edge_index` and `batch` must be **int64**; `x`, `edge_attr`, `u` must be **float32** (match training dtype).
- `u` always carries a batch dimension: `[1, 21]` for a single graph, `[B, 21]` for a batch.
- Use `torch.jit.script`, never `trace`.
- Call `model.eval()` before exporting and `module.eval()` before inferring.
- Offset edge IDs per graph when batching; forget this and graphs leak messages into each other.
- Test the export path with an **untrained** model on day one of integration.

---

## 11. Where this sits in the build order

This document covers Phases 2–3. The recommended de-risking sequence:

1. Write the C++ `BoardEncoder` (one encoder, per the main spec).
2. **Export an untrained `HiveValueGNN` to TorchScript and run §5 export-fidelity check.** This surfaces any PyG/TorchScript incompatibility immediately, before any training time is spent. If a PyG layer refuses to script, fall back to rewriting the forward pass with export-friendly ops, or — since graphs are tiny — hand-implementing the small forward pass in C++.
3. Load the file in C++, wire it into the MCTS leaf evaluation (single graph first, then batched), confirm it runs and measure per-move latency.
4. Only then: serialize self-play tensors, train, and iterate.
