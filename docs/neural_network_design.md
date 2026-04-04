# Neural Network Design for Hive Board Evaluation

## 1. Problem Statement

Replace (or augment) the handcrafted evaluation oracle with a learned model that:
- **Value head**: evaluates board positions (probability of winning from a state)
- **Policy head** (optional): predicts move probabilities to guide MCTS expansion

The current handcrafted evaluation has ~15 features with manually tuned weights.
A neural network can learn far more complex patterns from self-play data.

---

## 2. Why Hive is Unusual for Neural Networks

Unlike Chess or Go, Hive has properties that make standard approaches (CNN on a grid) suboptimal:

| Property | Chess/Go | Hive |
|----------|----------|------|
| Board shape | Fixed grid (8x8, 19x19) | Dynamic graph, no fixed boundaries |
| Piece count | Fixed (32 chess, ≤361 Go) | Variable (0-28 in play) |
| Topology | Square grid | Hexagonal adjacency, pieces touching |
| Stacking | No (chess), No (Go) | Yes (beetles climb on others) |
| Board growth | Board fills up | Board grows outward from center |

This makes **Graph Neural Networks (GNN)** the most natural fit, though Transformers are also viable.

---

## 3. Architecture Option A: Graph Neural Network (Recommended)

### 3.1 Graph Representation

**Nodes** = all 28 pieces (14 white + 14 black), each always present in the graph.
Pieces in hand are "virtual" nodes with a special flag.

**Node features** (per piece, ~20 dimensions):

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| piece_type | 8 (one-hot) | QueenBee, Spider, Beetle, Grasshopper, Ant, Mosquito, Ladybug, Pillbug |
| color | 2 (one-hot) | White, Black |
| is_in_play | 1 | 1 if on board, 0 if in hand |
| is_on_top | 1 | 1 if top of stack |
| is_pinned | 1 | 1 if can't move without breaking hive |
| is_buried | 1 | 1 if under another piece |
| cannot_be_moved | 1 | 1 if Pillbug-blocked this turn |
| stack_height_at_pos | 1 | Height of stack at this piece's position (0 if in hand) |
| queen_neighbor_count | 1 | Number of occupied neighbors if this is a queen (0 otherwise) |
| relative_position | 2 | (delta_q, delta_r) relative to board center of mass |
| is_current_player | 1 | 1 if this piece belongs to the player to move |

Total: ~20 features per node.

**Edges** (relationships between pieces):

| Edge type | Description |
|-----------|-------------|
| adjacent | Two pieces are hexagonal neighbors on the board |
| stacked_on | Piece A is directly on top of piece B |
| same_color | Both pieces belong to the same player |

Using typed edges allows the GNN to learn different interaction patterns for different relationships.

### 3.2 GNN Architecture

```
Input: Graph G = (V, E) with |V| = 28 nodes, |E| = dynamic edges

Layer 1: Linear(20, 64) + ReLU          -- Project node features to hidden dim
Layer 2: GATv2Conv(64, 64, heads=4)      -- Multi-head attention over neighbors
Layer 3: GATv2Conv(64, 64, heads=4)      -- Second message passing layer
Layer 4: GATv2Conv(64, 64, heads=4)      -- Third message passing layer
Layer 5: GATv2Conv(64, 64, heads=4)      -- Fourth layer (captures 4-hop patterns)

Global pooling: Separate mean pooling for white pieces and black pieces
Concatenate: [white_pool(128) | black_pool(128) | global_features(8)]

Value head:  Linear(264, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 1) -> Sigmoid
Policy head: Linear(264, 128) -> ReLU -> Linear(128, max_moves) -> Softmax
```

**Why GATv2 (Graph Attention Network v2)?**
- Learns which neighbor interactions matter most
- A beetle next to a queen should attend strongly to the queen
- An ant far from action should attend weakly
- Multi-head attention captures different relationship types
- GATv2 fixes the expressiveness limitation of original GAT

**Why 4 layers?**
- Layer 1: Each piece sees its immediate neighbors
- Layer 2: Each piece sees 2-hop neighborhood (pieces near my neighbors)
- Layer 3: 3-hop patterns (strategic positioning)
- Layer 4: Global patterns (full board awareness for 10-15 piece games)

Hive games typically have 10-20 pieces in play within a 4-hop radius, so 4 layers give sufficient receptive field.

### 3.3 Global Features (concatenated before heads)

| Feature | Description |
|---------|-------------|
| turn_number | Normalized current turn (turn / 50) |
| current_player | 0 = White, 1 = Black |
| white_pieces_in_play | Count / 14 |
| black_pieces_in_play | Count / 14 |
| white_queen_in_play | 0 or 1 |
| black_queen_in_play | 0 or 1 |
| white_queen_surround | Count / 6 (if in play) |
| black_queen_surround | Count / 6 (if in play) |

### 3.4 Parameter Count

- Node projection: 20 × 64 = 1,280
- 4 × GATv2(64, 64, 4 heads): ~4 × (64 × 64 × 4 × 3) ≈ 196,608
- Value head: 264×128 + 128×64 + 64×1 ≈ 42,048
- Total: **~240K parameters**

This is very small — inference should be fast even on CPU.

---

## 4. Architecture Option B: Transformer

### 4.1 Token Representation

Each of the 28 pieces becomes a token. Same node features as GNN (20 dims).
Add a special [CLS] token for global classification.

### 4.2 Transformer Architecture

```
Input: 29 tokens × 20 features (28 pieces + 1 CLS token)

Embedding: Linear(20, 128)
Positional encoding: Learned embeddings based on piece identity (not position — position is in features)

Transformer blocks × 4:
  - Multi-head self-attention (4 heads, dim=128)
  - Feed-forward (128 -> 256 -> 128)
  - LayerNorm + residual connections

Output: CLS token → Value head: Linear(128, 1) → Sigmoid
        All tokens → Policy head (optional)
```

### 4.3 Pros and Cons vs GNN

| Aspect | GNN | Transformer |
|--------|-----|-------------|
| Inductive bias | Strong (graph structure) | None (learns from data) |
| Data efficiency | Better (exploits adjacency) | Needs more data |
| Scalability | O(|E|) per layer | O(n²) self-attention |
| Implementation | Needs graph library | Standard PyTorch |
| Board topology | Naturally encoded via edges | Must be learned from features |
| Stacking | Naturally via edge types | Must be learned |
| Piece interactions | Only between connected pieces (efficient) | All pairs (wasteful but complete) |

**Recommendation: Start with GNN**, fall back to Transformer if GNN doesn't converge.

---

## 5. Training Pipeline

### 5.1 Data Generation via Self-Play

```
for each game:
    board = new_game()
    while not game_over:
        move = MCTS.Search(board, time_limit=1s)  // Uses current NN (or handcrafted initially)

        # Record training sample:
        record(board_state, mcts_visit_distribution, current_player)

        board.ApplyMove(move)

    # After game ends, label all positions with the outcome:
    for each recorded position:
        value_label = 1.0 if position's player won, 0.0 if lost, 0.5 if draw
        policy_label = mcts_visit_distribution (normalized to probabilities)
```

### 5.2 Training Loop (AlphaZero-style)

```python
# Phase 1: Bootstrap with handcrafted evaluation
# Generate 10K games using current MCTS + handcrafted eval
# Train initial NN on these games

# Phase 2: Self-play improvement loop
for iteration in range(100):
    # Generate 1000 games using MCTS + current NN
    games = self_play(nn_model, num_games=1000, time_per_move=0.5s)

    # Extract training data
    positions, value_labels, policy_labels = extract_data(games)

    # Train on combined data (recent + some older data)
    train(nn_model, positions, value_labels, policy_labels, epochs=10)

    # Evaluate: new model vs old model (100 games)
    win_rate = evaluate(new_model, old_model, num_games=100)

    # Accept new model only if it wins > 55%
    if win_rate > 0.55:
        accept(new_model)
```

### 5.3 Loss Function

```python
loss = value_loss + policy_loss + regularization

value_loss  = MSE(predicted_value, actual_outcome)              # Or cross-entropy
policy_loss = -sum(mcts_policy * log(predicted_policy))         # KL divergence
regularization = 1e-4 * sum(param ** 2)                         # L2 weight decay
```

### 5.4 Data Representation for Training

Each training sample is a dictionary:
```python
{
    "node_features": Tensor[28, 20],        # 28 pieces × 20 features
    "edge_index": Tensor[2, num_edges],     # COO format adjacency
    "edge_type": Tensor[num_edges],         # 0=adjacent, 1=stacked, 2=same_color
    "global_features": Tensor[8],           # Turn, player, queen info
    "value_target": float,                  # 0.0, 0.5, or 1.0
    "policy_target": Tensor[max_moves],     # MCTS visit distribution (optional)
}
```

---

## 6. C++ Integration

### 6.1 Inference Options

| Option | Pros | Cons |
|--------|------|------|
| **ONNX Runtime** | Fast, cross-platform, well-supported | Extra dependency, graph ops may need custom export |
| **LibTorch** | Native PyTorch, easy export | Large binary (~200MB), may be overkill |
| **Custom C++ forward pass** | No dependencies, fastest, small binary | Must reimplement all layers manually |
| **TensorRT** | Fastest on NVIDIA GPU | GPU-only, complex setup |

**Recommendation: Custom C++ forward pass** for a model this small (240K params).

The model is small enough (~1MB weights) that we can implement the forward pass directly:
- Matrix multiplications: use Eigen or hand-written loops
- GATv2 attention: implement the attention mechanism directly
- ReLU, Sigmoid, Softmax: trivial

This eliminates external dependencies and keeps the engine self-contained.

### 6.2 Board → Graph Feature Extraction (C++)

```cpp
struct GraphFeatures
{
    float nodeFeatures[28][20];     // 28 pieces × 20 features
    int edgeFrom[MAX_EDGES];        // Source node of each edge
    int edgeTo[MAX_EDGES];          // Target node of each edge
    int edgeType[MAX_EDGES];        // 0=adjacent, 1=stacked, 2=same_color
    int numEdges;
    float globalFeatures[8];
};

GraphFeatures ExtractGraphFeatures(const Board& board)
{
    GraphFeatures gf;
    memset(&gf, 0, sizeof(gf));
    gf.numEdges = 0;

    // Compute center of mass for relative positioning
    float centerQ = 0, centerR = 0;
    int inPlayCount = 0;
    for (int p = 0; p < NumPieceNames; p++) {
        if (board.PieceInPlay(static_cast<PieceName>(p))) {
            Index pos = board.GetPosition(static_cast<PieceName>(p));
            centerQ += pos % BoardWidth;
            centerR += pos / BoardWidth;
            inPlayCount++;
        }
    }
    if (inPlayCount > 0) { centerQ /= inPlayCount; centerR /= inPlayCount; }

    // Fill node features
    for (int p = 0; p < NumPieceNames; p++) {
        PieceName piece = static_cast<PieceName>(p);
        BugType type = GetBugType(piece);
        Color color = GetColor(piece);

        // One-hot piece type
        gf.nodeFeatures[p][static_cast<int>(type)] = 1.0f;
        // Color
        gf.nodeFeatures[p][8] = (color == Color::White) ? 1.0f : 0.0f;
        gf.nodeFeatures[p][9] = (color == Color::Black) ? 1.0f : 0.0f;
        // In play
        gf.nodeFeatures[p][10] = board.PieceInPlay(piece) ? 1.0f : 0.0f;

        if (board.PieceInPlay(piece)) {
            Index pos = board.GetPosition(piece);
            gf.nodeFeatures[p][11] = board.PieceIsOnTop(piece) ? 1.0f : 0.0f;
            gf.nodeFeatures[p][12] = (!board.PieceIsOnTop(piece)) ? 1.0f : 0.0f; // buried

            if (board.PieceIsOnTop(piece))
                gf.nodeFeatures[p][13] = board.CanMoveWithoutBreakingHive(piece) ? 0.0f : 1.0f; // pinned

            gf.nodeFeatures[p][14] = board.cannotBeMoved[piece] ? 1.0f : 0.0f;
            gf.nodeFeatures[p][15] = static_cast<float>(board.stackHeight[pos]) / 4.0f;

            // Relative position
            float q = static_cast<float>(pos % BoardWidth) - centerQ;
            float r = static_cast<float>(pos / BoardWidth) - centerR;
            gf.nodeFeatures[p][16] = q / 10.0f;  // Normalize
            gf.nodeFeatures[p][17] = r / 10.0f;

            gf.nodeFeatures[p][18] = (color == board.currentColor) ? 1.0f : 0.0f;

            // Queen surround count
            if (type == BugType::QueenBee) {
                int surround = 0;
                for (int d = 0; d < 6; d++) {
                    Index n = pos + NeighborOffsets[d];
                    if (IsValidIndex(n) && board.HasPieceAt(n)) surround++;
                }
                gf.nodeFeatures[p][19] = static_cast<float>(surround) / 6.0f;
            }
        }
    }

    // Build edges: adjacency
    for (int p1 = 0; p1 < NumPieceNames; p1++) {
        if (!board.PieceInPlay(static_cast<PieceName>(p1))) continue;
        Index pos1 = board.GetPosition(static_cast<PieceName>(p1));

        for (int p2 = p1 + 1; p2 < NumPieceNames; p2++) {
            if (!board.PieceInPlay(static_cast<PieceName>(p2))) continue;
            Index pos2 = board.GetPosition(static_cast<PieceName>(p2));

            // Check adjacency
            for (int d = 0; d < 6; d++) {
                if (pos1 + NeighborOffsets[d] == pos2) {
                    // Bidirectional edge
                    gf.edgeFrom[gf.numEdges] = p1;
                    gf.edgeTo[gf.numEdges] = p2;
                    gf.edgeType[gf.numEdges] = 0; // adjacent
                    gf.numEdges++;
                    gf.edgeFrom[gf.numEdges] = p2;
                    gf.edgeTo[gf.numEdges] = p1;
                    gf.edgeType[gf.numEdges] = 0;
                    gf.numEdges++;
                    break;
                }
            }
        }

        // Stacking edge
        PieceName below = board.GetPieceUnder(static_cast<PieceName>(p1));
        if (below != PieceName::INVALID) {
            gf.edgeFrom[gf.numEdges] = p1;
            gf.edgeTo[gf.numEdges] = static_cast<int>(below);
            gf.edgeType[gf.numEdges] = 1; // stacked_on
            gf.numEdges++;
        }
    }

    // Same-color edges (implicit — can be computed in the network or added here)

    // Global features
    gf.globalFeatures[0] = static_cast<float>(board.GetCurrentTurn()) / 50.0f;
    gf.globalFeatures[1] = (board.currentColor == Color::Black) ? 1.0f : 0.0f;
    // ... (white/black pieces in play, queen info)

    return gf;
}
```

### 6.3 Hybrid Evaluation Strategy

Don't replace the handcrafted evaluation — **combine** them:

```cpp
double HybridEvaluate(const Board& board, Color perspective, int depth)
{
    // Shallow nodes: use fast handcrafted eval
    if (depth < 3)
        return EvaluateBoard(board, perspective);

    // Deep nodes: use NN for more accurate evaluation
    GraphFeatures gf = ExtractGraphFeatures(board);
    double nnValue = NeuralNetwork::Forward(gf);

    // Blend: trust NN more at deeper nodes
    double blend = std::min(1.0, (depth - 3) / 5.0);
    double hcValue = EvaluateBoard(board, perspective);

    return (1.0 - blend) * hcValue + blend * nnValue;
}
```

This keeps the MCTS fast (handcrafted eval for most nodes) while using the NN
for critical deep evaluations where accuracy matters most.

---

## 7. Training Infrastructure (Python)

### 7.1 Required Libraries

```
torch >= 2.0
torch-geometric >= 2.3  (for GNN layers)
numpy
```

### 7.2 Project Structure

```
training/
├── model.py              # GNN model definition (PyTorch)
├── features.py           # Board → graph feature extraction (Python mirror of C++)
├── self_play.py          # Self-play game generation (calls C++ engine via UHP)
├── train.py              # Training loop
├── export.py             # Export weights to binary format for C++
├── evaluate.py           # Model vs model evaluation
└── data/
    ├── games/            # Self-play game records
    └── models/           # Saved model checkpoints
```

### 7.3 Model Definition (PyTorch)

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class HiveGNN(nn.Module):
    def __init__(self, node_dim=20, hidden_dim=64, global_dim=8, num_heads=4, num_layers=4):
        super().__init__()

        self.node_embed = nn.Linear(node_dim, hidden_dim)

        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, batch, global_features):
        # x: [num_nodes, 20], edge_index: [2, num_edges]
        h = torch.relu(self.node_embed(x))

        for gat, norm in zip(self.gat_layers, self.norms):
            h_new = torch.relu(gat(h, edge_index))
            h = norm(h + h_new)  # Residual connection

        # Separate pooling for white and black pieces
        # (assuming first 14 nodes are white, last 14 are black per graph)
        white_mask = ...  # Derive from node features
        black_mask = ...

        white_pool = global_mean_pool(h[white_mask], batch[white_mask])
        black_pool = global_mean_pool(h[black_mask], batch[black_mask])

        combined = torch.cat([white_pool, black_pool, global_features], dim=1)
        value = self.value_head(combined)

        return value
```

### 7.4 Self-Play via UHP Protocol

```python
import subprocess

def play_game(engine_path, time_per_move="00:00:01"):
    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )

    proc.stdin.write("newgame Base\n")
    proc.stdin.flush()
    # Read game string
    game_string = read_response(proc)

    positions = []
    while "InProgress" in game_string:
        proc.stdin.write(f"bestmove time {time_per_move}\n")
        proc.stdin.flush()
        move = read_response(proc)

        # Record position before move
        positions.append(extract_features_from_game_string(game_string))

        proc.stdin.write(f"play {move}\n")
        proc.stdin.flush()
        game_string = read_response(proc)

    # Determine winner and label all positions
    winner = parse_winner(game_string)
    labeled_data = label_positions(positions, winner)

    proc.stdin.write("exit\n")
    proc.stdin.flush()
    proc.wait()

    return labeled_data
```

---

## 8. Weights Export Format (for C++ inference)

Simple binary format:

```
Header:
  4 bytes: magic number (0x48495645 = "HIVE")
  4 bytes: version
  4 bytes: number of weight tensors

For each tensor:
  4 bytes: name length
  N bytes: name string
  4 bytes: number of dimensions
  4 bytes × ndim: shape
  4 bytes × product(shape): float32 weights
```

C++ loader:
```cpp
class NeuralNetwork {
    std::vector<std::vector<float>> weights;

    bool LoadWeights(const std::string& path);
    float Forward(const GraphFeatures& features);
};
```

---

## 9. Implementation Roadmap

### Phase 1: Data Generation (1-2 weeks)
- Implement `ExtractGraphFeatures()` in C++ (added to Evaluation.cpp)
- Add UHP command to dump features: `features` → outputs JSON graph
- Write Python self-play script using current engine
- Generate 50K games for bootstrap training

### Phase 2: Model Training (1-2 weeks)
- Implement GNN model in PyTorch + torch_geometric
- Train on bootstrap data (handcrafted eval labels)
- Validate: NN eval should correlate with handcrafted eval (r² > 0.8)

### Phase 3: C++ Inference (1 week)
- Implement custom forward pass in C++
- Binary weight loader
- Benchmark: target < 0.5ms per evaluation on CPU

### Phase 4: Integration & Self-Play Loop (2-3 weeks)
- Integrate NN into MCTS as hybrid evaluator
- Run self-play improvement loop (AlphaZero-style)
- Target: 10 iterations of 1000 games each
- Evaluate improvement via tournament against handcrafted-only version

### Phase 5: Policy Head (optional, 1-2 weeks)
- Add policy head to guide MCTS expansion (replaces ScoreMove heuristic)
- This dramatically reduces the branching factor explored by MCTS
- Train jointly with value head

---

## 10. Expected Improvements

| Component | Current | With NN |
|-----------|---------|---------|
| Evaluation accuracy | ~15 features, manually tuned | Hundreds of learned features |
| Pattern recognition | None (pure heuristics) | Learns tactical/strategic patterns |
| Position understanding | Local (queen + neighbors) | Global (full board context via message passing) |
| Move ordering | Handcrafted ScoreMove | Policy head (learned from MCTS statistics) |
| Adaptability | Fixed weights | Improves with more training data |

The GNN approach is particularly promising for Hive because:
1. The game IS a graph — pieces touching pieces, with stacking
2. Small model size (240K params) means fast inference
3. 28 fixed nodes means no variable-size padding issues
4. Edge types naturally capture the 3D structure (adjacency + stacking)
5. Message passing learns "if beetle is near queen AND ant is near queen, that's more dangerous than either alone" — interactions the handcrafted eval can't easily capture
