#ifndef BOARDENCODER_H
#define BOARDENCODER_H

#include "Board.h"

#include <cstdint>
#include <vector>

namespace HiveGotThis
{

// ── Tensor dimensions ────────────────────────────────────────────────────────
constexpr int GNNNodeDim   = 18;
constexpr int GNNEdgeDim   = 9;    // slots 0–5 planar dirs, 6 up, 7 down, 8 self
constexpr int GNNGlobalDim = 21;

// ── Edge-attr slots ───────────────────────────────────────────────────────────
// Planar direction d  →  slot d  (Right=0, DownRight=1, …, UpRight=5)
// Opposite of slot d  →  (d+3)%6
constexpr int GNNSlotUp   = 6;
constexpr int GNNSlotDown = 7;
constexpr int GNNSlotSelf = 8;

// ── Normalization constants ───────────────────────────────────────────────────
constexpr float GNNMaxStackDepth   = 4.0f;
constexpr float GNNMaxQueenDist    = 10.0f;
constexpr float GNNTurnNorm        = 40.0f;
constexpr float GNNNormAnt         = 3.0f;
constexpr float GNNNormSpider      = 2.0f;
constexpr float GNNNormGrasshopper = 3.0f;
constexpr float GNNNormBeetle      = 2.0f;

// ── Output struct ─────────────────────────────────────────────────────────────
struct GNNGraph {
    // Row-major node features: N rows × GNNNodeDim cols → N*GNNNodeDim floats
    std::vector<float> x;

    // COO edge list, flattened [src_0…src_{E-1}, dst_0…dst_{E-1}]: length 2*E
    // Python side reshapes with tensor.view(2, E)
    std::vector<int64_t> edge_index;

    // One-hot edge type per edge, GNNEdgeDim floats per edge: length E*GNNEdgeDim
    std::vector<float> edge_attr;

    // Global vector: length GNNGlobalDim
    std::vector<float> u;
};

// ── Encoder ───────────────────────────────────────────────────────────────────
class BoardEncoder
{
public:
    // Encode `board` into four flat tensors ready for PyG / libtorch inference.
    // Only on-board pieces become nodes; pieces in hand appear only in u.
    static GNNGraph encode(const Board& board);
};

} // namespace HiveGotThis

#endif // BOARDENCODER_H
