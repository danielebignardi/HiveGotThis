# Hive GNN — State Representation, Architecture & Integration Spec

**Purpose:** value network as the evaluation function for the Hive MCTS.

This document is the single source of truth for the state→graph encoding, shared by Python (training) and C++ (inference).

---

## 1. Design principles

1. **Hive state is a graph, not a grid.** Tiles are nodes; hex adjacencies are edges. There are no meaningful absolute coordinates — the position "floats" in the plane.
2. **Strict local / global separation.** A *node*'s features describe **only** that piece and its immediate neighborhood. Anything constant across the whole board (hand inventories, turn, queen status) lives **only** in the global vector `u`, injected at the readout. No global information is replicated on nodes — that dilutes the signal and wastes message-passing capacity.
3. **Only on-board pieces are nodes** (including buried pieces under a stack). Pieces in hand are **not** nodes — they would be isolated, geometry-free nodes that pollute the pooling. Hand availability is encoded entirely in `u`.
4. **Canonical perspective.** The state is always seen from the **side to move** (MINE = +1, opponent = −1). The produced value is "how good the position is for the side to move."
5. **One encoder, in C++.** Feature computation needs Hive logic (articulation points, the pinned check, degrees), which lives in C++, and the encoder must run in C++ for inference anyway. The same C++ encoder also serializes tensors for training, so there is no second encoder to keep in sync. The validation that matters is therefore export fidelity (the exported model reproduces the trained model), not C++/Python encoder parity. See §9 and the interaction spec.

---

## 2. Graph construction

Each board produces four tensors (described as flat 1-D arrays in §6):

| Tensor | Logical shape | Contents |
|---|---|---|
| `x` | `[N, F]` | Node features. `N` = pieces **on the board**, `F` = 18 (§3). |
| `edge_index` | `[2, E]` | COO format: row 0 = source IDs, row 1 = destination IDs. |
| `edge_attr` | `[E, 9]` | One-hot edge type/direction. Aligned 1:1 with `edge_index`. |
| `u` | `[U]` | Global vector, `U` = 21 (§4). |

### Indexing & alignment rules

- **Per-edge positional alignment:** row `k` of `edge_attr` corresponds to column `k` of `edge_index`. This is the hard "sacred" alignment, and it holds between `edge_index` and `edge_attr`, both indexed by edge.
- **Node referential consistency:** if row `i` of `x` is piece P, then **every** edge that touches P must use the integer ID `i` inside `edge_index`. This is a referential constraint (the node ID must point to the same row), *not* a positional "row k = column k" one — `x` has `N` rows while `edge_index` has `E` columns.
- **Permutation invariance:** because the readout uses `global_*_pool`, the *labeling* of nodes (which piece is node 0, 1, 2…) is arbitrary. C++ and Python do **not** need to agree on node ordering; each side only needs to be internally consistent within a single graph.

---

## 3. Node features (`x`) — LOCAL information only

`F = 18` features per node. All normalized.

| Idx | Name | Definition | Range |
|---|---|---|---|
| 0 | Queen (Bee) | piece-type one-hot | 0/1 |
| 1 | Ant | one-hot | 0/1 |
| 2 | Spider | one-hot | 0/1 |
| 3 | Grasshopper | one-hot | 0/1 |
| 4 | Beetle | one-hot | 0/1 |
| 5 | Mosquito | one-hot | 0/1 |
| 6 | Ladybug | one-hot | 0/1 |
| 7 | Pillbug | one-hot | 0/1 |
| 8 | Relative color | +1 if MINE (side to move), −1 if opponent | ±1 |
| 9 | Planar degree | number of occupied neighbor cells / 6 | [0,1] |
| 10 | Is on top | 1 if the piece is on top of its stack (free to act this turn), 0 if covered | 0/1 |
| 11 | Depth from top | 0 if on top, 1 if one piece is above it, 2 if two, … (positive, normalized e.g. /3) | [0,1] |
| 12 | Is pinned | 1 if it cannot legally move this turn for a reason **not** already captured by depth or break-hive — i.e. the freedom-to-move / gate rule, or an ability-based lock (Pillbug freeze) | 0/1 |
| 13 | Break One-Hive | 1 if removing it would disconnect the hive (illegal move) | 0/1 |
| 14 | Cubic distance to MY queen | hex cubic distance to own queen, normalized by a max radius | [0,1] |
| 15 | Cubic distance to ENEMY queen | same, to the opponent's queen | [0,1] |
| 16 | Adjacent to MY queen | binary flag | 0/1 |
| 17 | Adjacent to ENEMY queen | binary flag | 0/1 |

**Notes**

- **Why keep both `is_on_top` (10) and `depth_from_top` (11):** the boundary between "on top, can act" and "covered, cannot act at all" is a hard nonlinearity, whereas the difference between depth 2 and depth 3 is nearly irrelevant. A raw normalized depth encodes that threshold poorly, so `is_on_top` provides a clean, linearly-separable flag for the decisive "can it act this turn" boundary, while `depth_from_top` carries the finer "how buried" signal. The redundancy is benign and cheap, and improves sample efficiency.
- **`is_pinned` (12) is distinct from depth and break-hive** and captures cases neither of the others does. It is 1 when an on-top, hive-safe piece still cannot move this turn — either because it cannot slide out (the freedom-to-move / gate rule) or because of a piece-ability lock such as the Pillbug freeze (a piece that moved, or was moved by a Pillbug, on the previous turn cannot move this turn). Derivable from board state, since the engine already tracks the last-moved marker for legality. This is the cheap, local residue of a mobility score; a full mobility score is intentionally left out.
- **Why keep the queen-adjacency flags (16, 17):** the win condition is a *threshold* on exact adjacency (6 neighbors around a queen = loss), not a smooth function of distance — "small distance" is not the same as "touching." Also, `u` only holds the aggregate count of pieces around each queen; the per-node flag says *which specific piece*, of which color and type, is doing the surrounding. Cheap, decisive, keep it.
- **Compute structural features once per board, not per node.** `break_one_hive` = the set of **articulation points** of the planar-adjacency graph, found in a single `O(V+E)` Tarjan pass — not recomputed per piece. Same idea for planar degree.

---

## 4. Global vector (`u`) — GLOBAL information, replicated zero times on nodes

`U = 21` elements, injected directly at the readout (§7); never passes through message passing.

| Idx | Name | Definition | Range |
|---|---|---|---|
| 0 | My queen surrounded | pieces around my queen / 6 | [0,1] |
| 1 | Enemy queen surrounded | pieces around enemy queen / 6 | [0,1] |
| 2 | Normalized turn | current turn / 40 (or chosen constant) | [0,1] |
| 3 | My queen in play | 1 if placed, 0 if in hand | 0/1 |
| 4 | Enemy queen in play | 1 if placed, 0 if in hand | 0/1 |
| 5 | My Bee in hand | 0/1 | [0,1] |
| 6 | My Ants in hand | n / 3 | [0,1] |
| 7 | My Spiders in hand | n / 2 | [0,1] |
| 8 | My Grasshoppers in hand | n / 3 | [0,1] |
| 9 | My Beetles in hand | n / 2 | [0,1] |
| 10 | My Mosquito in hand | 0/1 | [0,1] |
| 11 | My Ladybug in hand | 0/1 | [0,1] |
| 12 | My Pillbug in hand | 0/1 | [0,1] |
| 13–20 | ENEMY inventory in hand | same structure as 5–12, computed for the opponent | [0,1] |

> The two "queen surrounded" values and the inventories appear **only** here, never on the nodes.

---

## 5. Edges (`edge_index` + `edge_attr`)

### 5.1 Edge encoding — 9-dim one-hot

Hive adjacency is discrete: planar neighbors are always exactly one unit apart and stack steps are always ±1, so there is no magnitude to encode. Each edge is therefore exactly **one** of 9 types (`edge_attr` is a 9-dim one-hot, exactly one slot = 1.0):

| Slot | Meaning |
|---|---|
| 0–5 | Planar direction (6 hex directions on the plane) |
| 6 | Up (toward the piece directly above in the same cell) |
| 7 | Down (toward the piece directly below) |
| 8 | Self (self-loop) |

This folds verticality (slots 6–7) and self-loops (slot 8) into one scheme. A self-loop is an explicit type, not a degenerate zero vector, so there is no ambiguity between "edge to the piece I sit on" and "edge to myself." `edge_dim = 9` in the conv layers.

### 5.2 Edge types and how they are drawn

Edges are **directional**; each undirected relation is two columns (forward + reverse).

1. **Planar adjacency** — connect the **top piece of each occupied cell** to the top pieces of its neighboring occupied cells. **Height-independent by design** (we do not require equal stack heights): the planar edge only encodes "these two cells are hex-neighbors in direction `d`." The forward edge uses planar slot `d`; the reverse uses the opposite slot `(d+3) mod 6`. Stack-height information is carried by the node depth feature and the up/down edges, not here. *(This is a modeling choice; revisit if height-aware planar contact proves necessary.)*
2. **Stacking** — for each piece directly above another in the same cell: forward (lower→upper) uses slot 6 (up), reverse (upper→lower) uses slot 7 (down). These are essential: without them, buried pieces are isolated and the net cannot reason about beetles pinning a queen or pieces trapped under a stack. A buried piece reaches the rest of the hive via stack edge → top piece → planar neighbor (≤2 hops).
3. **Self-loop** — for every node `i`, one column `[i, i]` with slot 8 (self). Added **manually in C++** because the conv layers use `add_self_loops=False` (PyG's automatic self-loops would have no valid `edge_attr` and crash / produce NaNs).

---

## 6. C++ output struct

```cpp
#include <vector>
#include <cstdint>

struct GNNInputs {
    // Node features, row-major: N rows of 18 → N*18 floats
    std::vector<float> x;

    // COO connectivity, flattened as [all sources..., all destinations...]: length 2*E
    std::vector<int64_t> edge_index;

    // One-hot edge type/direction per edge, aligned to edge_index: length E*9
    std::vector<float> edge_attr;

    // Global vector: length 21
    std::vector<float> u;
};

class BoardEncoder {
public:
    static GNNInputs encode(const Board& board) {
        GNNInputs t;
        // 1. Nodes (x): on-board pieces only, 18 features each.
        //    Precompute structural features once per board:
        //      - planar degrees
        //      - articulation points (single Tarjan pass) -> break_one_hive
        // 2. Edges:
        //      - planar (top-of-cell to top-of-neighbor-cell), both directions
        //      - stacking (up / down), both directions
        //    edge_attr = one-hot of {6 planar dirs, up, down, self}.
        // 3. Manual self-loops: [i, i] with the "self" slot.
        // 4. u: 21 global values.
        return t;
    }
};
```

> **`edge_index` flattening convention:** all sources first, then all destinations (`[src_0..src_{E-1}, dst_0..dst_{E-1}]`), so the Python side reshapes with `tensor.view(2, E)`. Document and test this.

---

## 7. Network architecture (PyTorch Geometric)

```python
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout, Tanh, LayerNorm
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

class HiveValueGNN(torch.nn.Module):
    def __init__(self, node_in_dim=18, edge_in_dim=9, u_dim=21,
                 hidden_dim=64, heads=4):
        super().__init__()
        # Backbone: 3 GATv2 layers with edge attributes, no automatic self-loops.
        self.conv1 = GATv2Conv(node_in_dim, hidden_dim, edge_dim=edge_in_dim,
                               heads=heads, concat=False, add_self_loops=False)
        self.norm1 = LayerNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim,
                               heads=heads, concat=False, add_self_loops=False)
        self.norm2 = LayerNorm(hidden_dim)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim,
                               heads=heads, concat=False, add_self_loops=False)
        self.norm3 = LayerNorm(hidden_dim)

        # Readout: [mean_pool || max_pool || u] = 2*hidden_dim + u_dim
        self.value_head = Sequential(
            Linear(2 * hidden_dim + u_dim, 32),
            LayerNorm(32),
            ReLU(),
            Dropout(p=0.2),
            Linear(32, 1),
            Tanh(),   # output in [-1, 1]: +1 win / 0 draw / -1 loss for the side to move
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        x = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x = F.elu(self.norm2(self.conv2(x, edge_index, edge_attr)))
        x = F.elu(self.norm3(self.conv3(x, edge_index, edge_attr)))

        h = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch),
                       u], dim=1)
        return self.value_head(h)
```

**Rationale**

- **3 layers** ≈ 2–3 hop receptive field; `u` + global pooling supply the macroscopic context. Very-long-range structure (an ant circling the whole perimeter) stays out of range; add layers with residual connections later if needed.
- **`heads=4, concat=False`**: multi-head attention averaged across heads (output `= hidden_dim`, stable dimensions across layers).
- **`add_self_loops=False`** is mandatory: self-loops carry the explicit "self" one-hot from C++; letting PyG inject its own would leave `edge_attr` undefined for those edges.
- **mean + max pooling**: max preserves "there is a critical node" (e.g. a piece adjacent to the enemy queen); mean preserves overall hive density.
- **Tanh** output in [−1, 1]: symmetric around the draw, and the MCTS backup is a clean negation (see §9). Pair with **MSE** loss.

---

## 8. Symmetries & data augmentation — high training ROI

A position and its dihedral-`D6` transforms (6 rotations of 60° × 2 reflections = **12 transforms**) are the **same** state but produce different `edge_attr`. The network is not invariant by construction and would otherwise have to learn this from data — slow and expensive.

**Best-ROI intervention:** for every self-play position, generate the 12 symmetries. With the one-hot edge encoding this is just a **permutation of the 6 planar slots** (a 60° rotation maps planar slot `d` → `(d+1) mod 6`; reflections apply the corresponding reflection permutation). Slots 6–8 (up/down/self) are invariant. Color is already canonicalized, so this is the remaining symmetry to exploit — the same trick AlphaZero uses on Go.

---

## 9. Value semantics & MCTS integration

- State is canonicalized to the **side to move**, so the value is "goodness for the side to move."
- With **tanh** output in [−1, 1]: target = `+1` win, `0` draw, `−1` loss. The backup toward the parent (opponent) node is **`−v`** (negamax). Loss: MSE.
- **Batch leaf evaluations across self-play games (essential for feasibility):**
  each MCTS keeps one evaluation in flight and remains sequential. Concurrent
  games feed a shared inference queue, which sends their independent leaves to
  the network in batches. This avoids virtual loss and keeps the GPU occupied.

---

## 10. Open decisions

| Topic | Options | Note |
|---|---|---|
| Mobility score | reintroduce later | Expensive in the MCTS hot path; measure the real gain first. |
| Inference runtime | libtorch (TorchScript) / ONNX Runtime | Decide early — it dictates the export path. |

---

## 11. De-risking checklist (recommended order)

1. **Integration before model quality.** Run an **untrained** network inside the C++ MCTS first.
2. **Export-fidelity check (do this with the untrained network):** export the model to TorchScript and confirm the exported model reproduces the eager model's output on the same input (tolerance ~1e-5). With a single C++ encoder there is no dual-encoder parity to test; the real risk shifts to PyG/TorchScript export compatibility, so surface it here with a dummy model — before any training time is spent.
3. **Verify batching** and per-move MCTS latency with the network in the loop.
4. **Only then** tune the architecture and start the self-play loop.
