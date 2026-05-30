// =============================================================================
// TestEncoderShapes.cpp
//
// Strict shape/conformance tests for BoardEncoder::encode() against the GNN
// spec (Hive_GNN_Spec.md §2, §6):
//   - u.size()          == 21
//   - x.size()          == N * 18           (here N = 10 -> 180)
//   - edge_index.size() == 2 * E            (COO flattened [src..., dst...])
//   - edge_attr.size()  == E * 9            (9-dim one-hot per edge)
//
// We build a board with EXACTLY 10 pieces in play and check the four tensors.
//
// BoardEncoder is linked from the hive_core library (see CMakeLists.txt).
// =============================================================================

#include <iostream>
#include <cstddef>

#include "Board.h"
#include "BoardEncoder.h"

using namespace HiveGotThis;
using namespace std;

// --- Tiny home-made test framework (same style as the other test files) ------
int g_testsPassed = 0;
int g_testsFailed = 0;

#define REQUIRE(condition, message) \
    do { \
        if (condition) { \
            g_testsPassed++; \
        } else { \
            cout << "[FAIL] " << message << " (Riga " << __LINE__ << ")" << endl; \
            g_testsFailed++; \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// 10 pieces laid out as a straight, connected line along the Right direction
// (consecutive cells C, C+1, ... C+9 are hex-adjacent), so the hive is valid
// and every piece is on top of its own cell (no stacking).
// -----------------------------------------------------------------------------
void Test_EncoderOutputShapes()
{
    cout << "--- Encoder output shapes (10 pieces) ---" << endl;

    Board board(GameType::Base);
    const Index c = BoardCenter;

    const PieceName pieces[10] = {
        wQ, wS1, wS2, wB1, wB2, wG1, wG2, wG3, wA1, wA2
    };
    for (int i = 0; i < 10; ++i)
        board.PushAt(pieces[i], static_cast<Index>(c + i)); // Right offset = +1

    // Precondition: exactly 10 pieces in play.
    int inPlay = 0;
    for (int i = 0; i < NumPieceNames; ++i)
        if (board.PieceInPlay(static_cast<PieceName>(i))) ++inPlay;
    REQUIRE(inPlay == 10, "La board deve avere esattamente 10 pezzi in gioco");

    GNNGraph g = BoardEncoder::encode(board);

    // --- Global vector and node features -------------------------------------
    REQUIRE(g.u.size() == 21, "u.size() deve essere esattamente 21");
    REQUIRE(g.x.size() == 180, "x.size() deve essere esattamente 180 (10 * 18)");
    REQUIRE(g.x.size() == static_cast<size_t>(10) * GNNNodeDim,
            "x.size() deve essere N * GNNNodeDim");

    // --- Edges: derive E independently from each tensor and require agreement -
    REQUIRE(g.edge_index.size() % 2 == 0,
            "edge_index.size() deve essere pari (formato COO [src..., dst...])");
    REQUIRE(g.edge_attr.size() % GNNEdgeDim == 0,
            "edge_attr.size() deve essere multiplo di 9 (one-hot per arco)");

    const size_t edgesFromIndex = g.edge_index.size() / 2;
    const size_t edgesFromAttr  = g.edge_attr.size() / GNNEdgeDim;
    REQUIRE(edgesFromIndex == edgesFromAttr,
            "edge_index ed edge_attr devono concordare sul numero di archi E");

    const size_t E = edgesFromIndex;
    REQUIRE(g.edge_index.size() == 2 * E, "edge_index.size() deve essere 2 * E");
    REQUIRE(g.edge_attr.size() == E * 9, "edge_attr.size() deve essere E * 9");

    // --- Bonus: exact topology of a straight 10-line --------------------------
    // 9 adjacent pairs * 2 directed planar edges = 18, no stacking, + 10 self
    // loops = 28 edges total.
    REQUIRE(E == 28,
            "Linea di 10 pezzi: 18 archi planari + 10 self-loop = 28 archi");
}

// -----------------------------------------------------------------------------
int main()
{
    Board::InitializeZobristTable();

    cout << "===== TestEncoderShapes: GNN tensor sizes =====" << endl;

    Test_EncoderOutputShapes();

    cout << "\n----------------------------------------" << endl;
    cout << "Passati: " << g_testsPassed << " | Falliti: " << g_testsFailed << endl;
    return g_testsFailed > 0 ? 1 : 0;
}
