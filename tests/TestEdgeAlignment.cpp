// =============================================================================
// TestEdgeAlignment.cpp
//
// Geometric-consistency tests for the edges produced by BoardEncoder::encode()
// (Hive_GNN_Spec.md §5):
//   - Bidirectional planar alignment: every planar edge A->B with direction slot
//     D (0..5) has a matching reverse edge B->A with slot (D + 3) % 6.
//   - Self-loops: exactly N edges carry GNNSlotSelf, one per node, with src==dst.
//
// Edges are decoded from the flat COO tensors:
//   edge k:  src  = edge_index[k]
//            dst  = edge_index[E + k]
//            slot = argmax(edge_attr[k*9 .. k*9+8])   (exactly one 1.0)
//
// BoardEncoder is linked from the hive_core library (see CMakeLists.txt).
// =============================================================================

#include <iostream>
#include <cstddef>
#include <vector>
#include <set>
#include <tuple>

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

struct DecodedEdge { int64_t src; int64_t dst; int slot; };

// -----------------------------------------------------------------------------
// Build a compact cluster of 4 mutually-adjacent pieces so that planar edges
// appear in several different direction slots:
//
//        UpRight
//   Q(C) -Right- S1(C+1)
//    \DownRight   /DownLeft
//     S2(C+128) -Right- B1 sits at C+127 (DownLeft of Q, Left of S2)
//
// Adjacent pairs: Q-S1, Q-S2, Q-B1, S1-S2, S2-B1  -> planar edges in dirs 0,1,2.
// No stacking -> only planar (0..5) and self (8) slots are produced.
// -----------------------------------------------------------------------------
void Test_EdgeBidirectionalAndSelfLoops()
{
    cout << "--- Edge bidirectional alignment & self-loops ---" << endl;

    Board board(GameType::Base);
    const Index c = BoardCenter;

    board.PushAt(wQ,  c);                                  // centre
    board.PushAt(wS1, static_cast<Index>(c + NeighborOffsets[0])); // Right
    board.PushAt(wS2, static_cast<Index>(c + NeighborOffsets[1])); // DownRight
    board.PushAt(wB1, static_cast<Index>(c + NeighborOffsets[2])); // DownLeft

    GNNGraph g = BoardEncoder::encode(board);

    const int    N = static_cast<int>(g.x.size()) / GNNNodeDim;
    const size_t E = g.edge_attr.size() / GNNEdgeDim;
    REQUIRE(g.edge_index.size() == 2 * E,
            "Precondizione: edge_index coerente con E");

    // --- Decode every edge into (src, dst, slot) -----------------------------
    vector<DecodedEdge> edges;
    edges.reserve(E);
    bool allOneHot = true;
    for (size_t k = 0; k < E; ++k)
    {
        int slot = -1, ones = 0;
        for (int s = 0; s < GNNEdgeDim; ++s)
            if (g.edge_attr[k * GNNEdgeDim + s] == 1.0f) { slot = s; ++ones; }
        if (ones != 1) allOneHot = false;
        edges.push_back({ g.edge_index[k], g.edge_index[E + k], slot });
    }
    REQUIRE(allOneHot, "Ogni edge_attr deve essere un one-hot con esattamente un 1.0");

    // Fast membership lookup of (src, dst, slot).
    set<tuple<int64_t, int64_t, int>> edgeSet;
    for (const auto& e : edges)
        edgeSet.insert(make_tuple(e.src, e.dst, e.slot));

    // --- 1. Bidirectional planar alignment -----------------------------------
    int planarChecked = 0;
    bool allReversesFound = true;
    for (const auto& e : edges)
    {
        if (e.slot < 0 || e.slot > 5) continue; // planar slots only (0..5)
        ++planarChecked;
        const int revSlot = (e.slot + 3) % 6;
        const bool found = edgeSet.count(make_tuple(e.dst, e.src, revSlot)) > 0;
        if (!found) allReversesFound = false;
    }
    REQUIRE(planarChecked > 0, "Devono esistere archi planari da verificare");
    REQUIRE(allReversesFound,
            "Ogni arco planare A->B (dir D) deve avere il reverse B->A (dir (D+3)%6)");

    // --- 2. Exactly N self-loops, one per node, with src == dst --------------
    int selfLoops = 0;
    bool selfWellFormed = true;
    vector<bool> nodeHasSelf(N, false);
    for (const auto& e : edges)
    {
        if (e.slot != GNNSlotSelf) continue;
        ++selfLoops;
        if (e.src != e.dst) selfWellFormed = false;
        if (e.src >= 0 && e.src < N)
        {
            if (nodeHasSelf[e.src]) selfWellFormed = false; // duplicate self-loop
            nodeHasSelf[e.src] = true;
        }
        else selfWellFormed = false; // out-of-range node id
    }
    REQUIRE(selfLoops == N, "Devono esserci esattamente N self-loop (uno per nodo)");
    REQUIRE(selfWellFormed,
            "Ogni self-loop deve avere src==dst, id valido e comparire una sola volta");

    bool everyNodeCovered = true;
    for (int i = 0; i < N; ++i)
        if (!nodeHasSelf[i]) everyNodeCovered = false;
    REQUIRE(everyNodeCovered, "Ogni nodo deve possedere il proprio self-loop");
}

// -----------------------------------------------------------------------------
int main()
{
    Board::InitializeZobristTable();

    cout << "===== TestEdgeAlignment: edge geometry =====" << endl;

    Test_EdgeBidirectionalAndSelfLoops();

    cout << "\n----------------------------------------" << endl;
    cout << "Passati: " << g_testsPassed << " | Falliti: " << g_testsFailed << endl;
    return g_testsFailed > 0 ? 1 : 0;
}