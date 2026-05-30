// =============================================================================
// TestIsPinned.cpp
//
// Tests for the Gate Rule and node feature 12 (is_pinned) of the GNN encoder.
// See Hive_GNN_Spec.md §3 (feature 12) and BoardEncoder.cpp::isPinned.
//
// isPinned is a file-local (static) helper and is NOT part of the public API,
// so these tests exercise it through the public BoardEncoder::encode() path by
// reading feature 12 (x[node*18 + 12]) of the relevant node. BoardEncoder is
// linked from the hive_core library (see CMakeLists.txt).
// =============================================================================

#include <iostream>

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

// Bug-type one-hot slots (mirror BoardEncoder::bugTypeToSlot).
constexpr int kSlotAnt      = 1;
constexpr int kSlotMosquito = 5;

// Convenience: place `piece` on top of cell `idx`.
static void place(Board& board, PieceName piece, Index idx)
{
    board.PushAt(piece, idx);
}

// The six hex neighbours of `c`, indexed by Direction (Right, DownRight,
// DownLeft, Left, UpLeft, UpRight) — mirrors NeighborOffsets.
static Index neighbour(Index c, int dir)
{
    return c + NeighborOffsets[dir];
}

// Feature 12 (is_pinned) of the unique on-board node whose bug-type one-hot is
// `bugSlot`. Returns -1.0f if no such node exists.
static float pinnedFeatureOf(const GNNGraph& g, int bugSlot)
{
    const int N = static_cast<int>(g.x.size()) / GNNNodeDim;
    for (int i = 0; i < N; ++i)
        if (g.x[i * GNNNodeDim + bugSlot] == 1.0f)
            return g.x[i * GNNNodeDim + 12];
    return -1.0f;
}

// -----------------------------------------------------------------------------
// Test 1: Mosquito touching ONLY Ants -> standard ground gate check applies.
//
// The Mosquito copies its neighbours' movement. With no Beetle/Grasshopper/
// Ladybug around, it behaves as a ground slider and must pass the gate check.
// We box it in completely with 6 Ants: every direction has both gate cells
// occupied, so it cannot slide out -> feature 12 must be 1.0 (PINNED).
// The 6-Ant ring is connected, so the Mosquito is on top and hive-safe (not a
// cut vertex), which is exactly the case where the gate check runs.
// -----------------------------------------------------------------------------
void Test_Mosquito_OnlyAnt_GateCheck()
{
    cout << "--- Test 1: Mosquito + Ants (gate check) ---" << endl;

    Board board(GameType::BaseM);
    const Index c = BoardCenter;

    place(board, wM, c);                       // Mosquito at the centre
    // Surround it on all 6 sides with Ants (3 white + 3 black ants).
    place(board, wA1, neighbour(c, 0));
    place(board, wA2, neighbour(c, 1));
    place(board, wA3, neighbour(c, 2));
    place(board, bA1, neighbour(c, 3));
    place(board, bA2, neighbour(c, 4));
    place(board, bA3, neighbour(c, 5));

    GNNGraph g = BoardEncoder::encode(board);
    REQUIRE(pinnedFeatureOf(g, kSlotMosquito) == 1.0f,
            "Mosquito circondata da sole Formiche e murata deve risultare PINNED (gate rule)");
}

// -----------------------------------------------------------------------------
// Test 2: Mosquito touching a Beetle -> bypasses the gate check (not pinned).
//
// Identical, fully-gated geometry as Test 1, but one neighbouring Ant is
// swapped for a Beetle. The Mosquito can now copy the Beetle's climb, so the
// gate rule no longer applies and feature 12 must be 0.0 (NOT pinned).
// -----------------------------------------------------------------------------
void Test_Mosquito_Beetle_BypassesGate()
{
    cout << "--- Test 2: Mosquito + Beetle (bypass) ---" << endl;

    Board board(GameType::BaseM);
    const Index c = BoardCenter;

    place(board, wM, c);                       // Mosquito at the centre
    place(board, wB1, neighbour(c, 0));        // Beetle neighbour -> climb ability
    // The remaining 5 neighbours are Ants -> geometry is still fully gated.
    place(board, wA1, neighbour(c, 1));
    place(board, wA2, neighbour(c, 2));
    place(board, wA3, neighbour(c, 3));
    place(board, bA1, neighbour(c, 4));
    place(board, bA2, neighbour(c, 5));

    GNNGraph g = BoardEncoder::encode(board);
    REQUIRE(pinnedFeatureOf(g, kSlotMosquito) == 0.0f,
            "Mosquito adiacente a un Beetle ignora la gate rule -> NON PINNED");
}

// -----------------------------------------------------------------------------
// Test 3: Pillbug freeze -> cannotBeMoved triggers the pin.
//
// cannotBeMoved is the marker the engine sets on a piece that moved (or was
// relocated by a Pillbug) on the previous ply; such a piece cannot move this
// turn. We use a freely-mobile Ant as a control: feature 12 is 0.0 normally,
// but flipping its cannotBeMoved flag must drive feature 12 to 1.0.
// -----------------------------------------------------------------------------
void Test_Pillbug_Freeze()
{
    cout << "--- Test 3: Pillbug freeze (cannotBeMoved) ---" << endl;

    Board board(GameType::BaseP);
    const Index c = BoardCenter;

    place(board, wQ, c);                        // Queen
    place(board, wA1, neighbour(c, 0));         // Ant with a single neighbour

    // Control: the Ant can slide around the Queen, so it is NOT pinned.
    GNNGraph before = BoardEncoder::encode(board);
    REQUIRE(pinnedFeatureOf(before, kSlotAnt) == 0.0f,
            "Formica libera (un solo vicino) NON deve essere pinned");

    // Apply the freeze: the piece was just moved / Pillbug-locked this turn.
    board.cannotBeMoved[wA1] = true;
    GNNGraph after = BoardEncoder::encode(board);
    REQUIRE(pinnedFeatureOf(after, kSlotAnt) == 1.0f,
            "cannotBeMoved (Pillbug freeze) deve rendere il pezzo PINNED");
}

// -----------------------------------------------------------------------------
int main()
{
    Board::InitializeZobristTable();

    cout << "===== TestIsPinned: Gate Rule & feature 12 =====" << endl;

    Test_Mosquito_OnlyAnt_GateCheck();
    Test_Mosquito_Beetle_BypassesGate();
    Test_Pillbug_Freeze();

    cout << "\n----------------------------------------" << endl;
    cout << "Passati: " << g_testsPassed << " | Falliti: " << g_testsFailed << endl;
    return g_testsFailed > 0 ? 1 : 0;
}
