#include "BoardEncoder.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

namespace HiveGotThis
{

// =============================================================================
// Private helpers
// =============================================================================

// Bug type → node feature slot [0, 7]
static int bugTypeToSlot(BugType t)
{
    switch (t)
    {
        case BugType::QueenBee:    return 0;
        case BugType::SoldierAnt:  return 1;
        case BugType::Spider:      return 2;
        case BugType::Grasshopper: return 3;
        case BugType::Beetle:      return 4;
        case BugType::Mosquito:    return 5;
        case BugType::Ladybug:     return 6;
        case BugType::Pillbug:     return 7;
        default:                   return -1;
    }
}

// Number of pieces stacked below `piece` (0 = piece sits on the ground).
static int depthFromBottom(const Board& board, PieceName piece)
{
    int depth = 0;
    PieceName below = board.GetPieceUnder(piece);
    while (below != PieceName::INVALID)
    {
        ++depth;
        below = board.GetPieceUnder(below);
    }
    return depth;
}

// Hex cubic distance between two flat-indexed cells.
static int hexDistance(Index a, Index b)
{
    int aq = a % BoardWidth, ar = a / BoardWidth;
    int bq = b % BoardWidth, br = b / BoardWidth;
    int dq = aq - bq, dr = ar - br;
    return (std::abs(dq) + std::abs(dr) + std::abs(dq + dr)) / 2;
}

// Number of occupied hex neighbors of `pos`.
static int occupiedNeighborCount(const Board& board, Index pos)
{
    int n = 0;
    for (int d = 0; d < 6; ++d)
    {
        Index nb = pos + NeighborOffsets[d];
        if (IsValidIndex(nb) && board.HasPieceAt(nb))
            ++n;
    }
    return n;
}

// True if any of the 6 neighbors of `pos` equals `targetPos`.
static bool isAdjacentTo(Index pos, Index targetPos)
{
    for (int d = 0; d < 6; ++d)
        if (pos + NeighborOffsets[d] == targetPos)
            return true;
    return false;
}

// Cheap "is pinned" test for node feature 12: an on-top, hive-safe piece that
// still cannot move this turn. Deliberately avoids full move generation — it
// uses the gate (freedom-to-move) rule directly plus the engine's last-moved /
// pillbug-freeze marker, as the spec frames feature 12 as the cheap, local
// residue of a mobility score.
//
//   - not on top / buried        -> already encoded by features 10–11 -> not pinned here
//   - removing it breaks hive     -> already encoded by feature 13     -> not pinned here
//   - cannotBeMoved (pillbug/last-moved lock) -> pinned
//   - ground sliders (Queen, Ant, Spider, Pillbug): pinned iff they cannot
//     slide out in ANY of the 6 directions (6 O(1) gate checks).
//   - jumpers/climbers (Grasshopper, Beetle, Ladybug, Mosquito) are not subject
//     to the ground slide-gate and, being in the hive, always have a move
//     available, so they are treated as not gate-pinned.
static bool isPinned(const Board& board, PieceName piece, bool isOnTop, bool breaksHive)
{
    if (!isOnTop || breaksHive)
        return false;
    if (board.cannotBeMoved[piece])
        return true;

    BugType type = GetBugType(piece);

    // Gestione speciale per la Zanzara: si comporta da "strisciante" 
    // se non tocca nessun insetto in grado di saltare o arrampicarsi.
    if (type == BugType::Mosquito)
    {
        bool hasJumperAbility = false;
        Index pos = board.GetPosition(piece);
        for (int d = 0; d < 6; ++d)
        {
            Index nbPos = pos + NeighborOffsets[d];
            if (IsValidIndex(nbPos) && board.HasPieceAt(nbPos))
            {
                PieceName neighbor = board.GetPieceAt(nbPos); // Il pezzo in cima a quella cella
                BugType nType = GetBugType(neighbor);
                if (nType == BugType::Grasshopper || nType == BugType::Beetle || nType == BugType::Ladybug)
                {
                    hasJumperAbility = true;
                    break;
                }
            }
        }
        
        // Se ha l'abilità di saltare/arrampicarsi, non soffre la Gate Rule.
        if (hasJumperAbility) return false; 
        
        // Altrimenti (solo striscianti intorno), prosegui al gate check!
    }
    else if (type == BugType::Grasshopper || type == BugType::Beetle || type == BugType::Ladybug)
    {
        return false; // I saltatori puri non sono mai bloccati dalla Gate Rule
    }

    // A questo punto rimangono solo: QueenBee, SoldierAnt, Spider, Pillbug 
    // (e la Mosquito "strisciante"). Eseguiamo il gate check.
    Index pos = board.GetPosition(piece);
    for (int d = 0; d < 6; ++d)
        if (board.CanSlide(pos, static_cast<Direction>(d), SlideMode::Ground, pos))
            return false;

    return true;
}

// Fill out[8] with normalized in-hand counts for `color`:
//   [0] Queen   (0/1)
//   [1] Ants    / GNNNormAnt
//   [2] Spiders / GNNNormSpider
//   [3] Grass   / GNNNormGrasshopper
//   [4] Beetles / GNNNormBeetle
//   [5] Mosquito (0/1)
//   [6] Ladybug  (0/1)
//   [7] Pillbug  (0/1)
static void fillInventory(const Board& board, Color color, float out[8])
{
    int q = 0, a = 0, s = 0, g = 0, b = 0, m = 0, l = 0, p = 0;
    PieceName first = (color == Color::White) ? PieceName::wQ : PieceName::bQ;
    PieceName last  = (color == Color::White) ? PieceName::wP : PieceName::bP;

    for (uint8_t i = static_cast<uint8_t>(first); i <= static_cast<uint8_t>(last); ++i)
    {
        PieceName piece = static_cast<PieceName>(i);
        if (!PieceNameIsEnabledForGameType(piece, board.gameType)) continue;
        if (!board.PieceInHand(piece)) continue;

        switch (GetBugType(piece))
        {
            case BugType::QueenBee:    ++q; break;
            case BugType::SoldierAnt:  ++a; break;
            case BugType::Spider:      ++s; break;
            case BugType::Grasshopper: ++g; break;
            case BugType::Beetle:      ++b; break;
            case BugType::Mosquito:    ++m; break;
            case BugType::Ladybug:     ++l; break;
            case BugType::Pillbug:     ++p; break;
            default: break;
        }
    }

    out[0] = static_cast<float>(q);
    out[1] = static_cast<float>(a) / GNNNormAnt;
    out[2] = static_cast<float>(s) / GNNNormSpider;
    out[3] = static_cast<float>(g) / GNNNormGrasshopper;
    out[4] = static_cast<float>(b) / GNNNormBeetle;
    out[5] = static_cast<float>(m);
    out[6] = static_cast<float>(l);
    out[7] = static_cast<float>(p);
}

// Append one directed edge to the accumulator vectors.
static void addEdge(
    std::vector<int64_t>& sources,
    std::vector<int64_t>& dests,
    std::vector<float>&   edgeAttr,
    int64_t src, int64_t dst, int slot)
{
    sources.push_back(src);
    dests.push_back(dst);
    // One-hot: GNNEdgeDim floats, all 0 except position `slot`.
    for (int k = 0; k < GNNEdgeDim; ++k)
        edgeAttr.push_back(k == slot ? 1.0f : 0.0f);
}

// =============================================================================
// BoardEncoder::encode
// =============================================================================

GNNGraph BoardEncoder::encode(const Board& board)
{
    // ── Phase A: node index ──────────────────────────────────────────────────
    // Only on-board pieces become nodes; in-hand pieces are skipped.
    int nodeIndex[NumPieceNames];
    std::fill(nodeIndex, nodeIndex + NumPieceNames, -1);
    std::vector<PieceName> nodeList;
    nodeList.reserve(NumPieceNames);

    for (int i = 0; i < NumPieceNames; ++i)
    {
        PieceName p = static_cast<PieceName>(i);
        if (board.PieceInPlay(p))
        {
            nodeIndex[i] = static_cast<int>(nodeList.size());
            nodeList.push_back(p);
        }
    }
    const int N = static_cast<int>(nodeList.size());

    // ── Phase B: board-level precomputation ──────────────────────────────────
    Color myColor    = board.currentColor;
    Color enemyColor = (myColor == Color::White) ? Color::Black : Color::White;
    PieceName myQueen    = (myColor == Color::White) ? PieceName::wQ : PieceName::bQ;
    PieceName enemyQueen = (myColor == Color::White) ? PieceName::bQ : PieceName::wQ;

    // breakHive[i] = true if removing nodeList[i] disconnects the hive
    std::vector<bool> breakHive(N, false);
    // planarDegree[i] = occupied neighbor cells of nodeList[i]'s position
    std::vector<int> planarDegree(N, 0);

    for (int i = 0; i < N; ++i)
    {
        PieceName p = nodeList[i];
        Index pos   = board.GetPosition(p);
        breakHive[i]    = !board.CanMoveWithoutBreakingHive(p);
        planarDegree[i] = occupiedNeighborCount(board, pos);
    }

    Index myQueenPos    = board.PieceInPlay(myQueen)    ? board.GetPosition(myQueen)    : NullIndex;
    Index enemyQueenPos = board.PieceInPlay(enemyQueen) ? board.GetPosition(enemyQueen) : NullIndex;

    float myQueenSurround    = (myQueenPos    != NullIndex) ? static_cast<float>(occupiedNeighborCount(board, myQueenPos))    / 6.0f : 0.0f;
    float enemyQueenSurround = (enemyQueenPos != NullIndex) ? static_cast<float>(occupiedNeighborCount(board, enemyQueenPos)) / 6.0f : 0.0f;

    float myInv[8]    = {};
    float enemyInv[8] = {};
    fillInventory(board, myColor,    myInv);
    fillInventory(board, enemyColor, enemyInv);

    // ── Phase C: node features (x) ───────────────────────────────────────────
    GNNGraph out;
    out.x.resize(N * GNNNodeDim, 0.0f);

    for (int i = 0; i < N; ++i)
    {
        PieceName p  = nodeList[i];
        Index pos    = board.GetPosition(p);
        float* feat  = out.x.data() + i * GNNNodeDim;

        // 0–7: bug-type one-hot
        int slot = bugTypeToSlot(GetBugType(p));
        if (slot >= 0) feat[slot] = 1.0f;

        // 8: relative color
        feat[8] = (GetColor(p) == myColor) ? +1.0f : -1.0f;

        // 9: planar degree / 6
        feat[9] = static_cast<float>(planarDegree[i]) / 6.0f;

        // 10: is on top
        bool isOnTop = board.PieceIsOnTop(p);
        feat[10] = isOnTop ? 1.0f : 0.0f;

        // 11: depth from top / GNNMaxStackDepth
        int dFromBottom = depthFromBottom(board, p);
        int dFromTop    = static_cast<int>(board.stackHeight[pos]) - 1 - dFromBottom;
        feat[11] = std::min(1.0f, static_cast<float>(dFromTop) / GNNMaxStackDepth);

        // 12: is pinned (on-top, hive-safe, but still immobile) — always assigned
        feat[12] = isPinned(board, p, isOnTop, breakHive[i]) ? 1.0f : 0.0f;

        // 13: break one-hive
        feat[13] = breakHive[i] ? 1.0f : 0.0f;

        // 14: normalized distance to my queen
        if (myQueenPos != NullIndex)
            feat[14] = std::min(1.0f, static_cast<float>(hexDistance(pos, myQueenPos)) / GNNMaxQueenDist);
        else
            feat[14] = 1.0f;

        // 15: normalized distance to enemy queen
        if (enemyQueenPos != NullIndex)
            feat[15] = std::min(1.0f, static_cast<float>(hexDistance(pos, enemyQueenPos)) / GNNMaxQueenDist);
        else
            feat[15] = 1.0f;

        // 16: adjacent to my queen
        feat[16] = (myQueenPos != NullIndex && isAdjacentTo(pos, myQueenPos)) ? 1.0f : 0.0f;

        // 17: adjacent to enemy queen
        feat[17] = (enemyQueenPos != NullIndex && isAdjacentTo(pos, enemyQueenPos)) ? 1.0f : 0.0f;
    }

    // ── Phase D: edges ───────────────────────────────────────────────────────
    std::vector<int64_t> sources, dests;
    // Upper bound: planar (≤ 6N directed) + stack (≤ 2*(N-1)) + self (N)
    sources.reserve(8 * N);
    dests.reserve(8 * N);
    out.edge_attr.reserve(8 * N * GNNEdgeDim);

    // Planar edges: connect top-of-cell pieces to their occupied hex neighbors.
    // Each directed edge is emitted once: A→B when processing A in direction d;
    // B→A is emitted when processing B in direction (d+3)%6.
    for (int i = 0; i < N; ++i)
    {
        PieceName p = nodeList[i];
        if (!board.PieceIsOnTop(p)) continue;

        Index pos = board.GetPosition(p);
        for (int d = 0; d < 6; ++d)
        {
            Index nbPos = pos + NeighborOffsets[d];
            if (!IsValidIndex(nbPos) || !board.HasPieceAt(nbPos)) continue;

            PieceName q = board.GetPieceAt(nbPos); // top of neighbor cell
            int j = nodeIndex[static_cast<int>(q)];
            addEdge(sources, dests, out.edge_attr, i, j, d);
        }
    }

    // Stack edges: upper→lower (slot down), lower→upper (slot up).
    // Iterate from each piece; GetPieceUnder() gives the piece directly below.
    for (int i = 0; i < N; ++i)
    {
        PieceName upper = nodeList[i];
        PieceName lower = board.GetPieceUnder(upper);
        if (lower == PieceName::INVALID) continue;

        int j = nodeIndex[static_cast<int>(lower)];
        addEdge(sources, dests, out.edge_attr, i, j, GNNSlotDown); // upper → lower
        addEdge(sources, dests, out.edge_attr, j, i, GNNSlotUp);   // lower → upper
    }

    // Self-loops: one per node with the self slot.
    for (int i = 0; i < N; ++i)
        addEdge(sources, dests, out.edge_attr, i, i, GNNSlotSelf);

    // Flatten edge_index: all sources first, then all destinations.
    out.edge_index.reserve(sources.size() * 2);
    for (int64_t s : sources) out.edge_index.push_back(s);
    for (int64_t d : dests)   out.edge_index.push_back(d);

    // ── Phase E: global vector u ─────────────────────────────────────────────
    out.u.resize(GNNGlobalDim, 0.0f);
    out.u[0]  = myQueenSurround;
    out.u[1]  = enemyQueenSurround;
    out.u[2]  = std::min(1.0f, static_cast<float>(board.GetCurrentTurn()) / GNNTurnNorm);
    out.u[3]  = board.PieceInPlay(myQueen)    ? 1.0f : 0.0f;
    out.u[4]  = board.PieceInPlay(enemyQueen) ? 1.0f : 0.0f;
    for (int k = 0; k < 8; ++k)
    {
        out.u[5  + k] = myInv[k];
        out.u[13 + k] = enemyInv[k];
    }

    return out;
}

// =============================================================================
// JSON serialization
// =============================================================================

std::string GNNGraphToJson(const GNNGraph& graph)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{";

    ss << "\"x\":[";
    for (size_t i = 0; i < graph.x.size(); ++i)
    {
        if (i > 0) ss << ",";
        ss << graph.x[i];
    }
    ss << "],";

    ss << "\"edge_index\":[";
    for (size_t i = 0; i < graph.edge_index.size(); ++i)
    {
        if (i > 0) ss << ",";
        ss << graph.edge_index[i];
    }
    ss << "],";

    ss << "\"edge_attr\":[";
    for (size_t i = 0; i < graph.edge_attr.size(); ++i)
    {
        if (i > 0) ss << ",";
        ss << graph.edge_attr[i];
    }
    ss << "],";

    ss << "\"u\":[";
    for (size_t i = 0; i < graph.u.size(); ++i)
    {
        if (i > 0) ss << ",";
        ss << graph.u[i];
    }
    ss << "]";

    ss << "}";
    return ss.str();
}

} // namespace HiveGotThis
