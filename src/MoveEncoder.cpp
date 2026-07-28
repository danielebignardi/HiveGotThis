#include "MoveEncoder.h"
#include "BoardEncoder.h" // GNNSlotUp: slot direzione "salita" negli edge del grafo

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace HiveGotThis
{

namespace
{
constexpr float MaxQueenDist = 10.0f;
constexpr float MaxStackHeight = 4.0f;
constexpr float TurnNorm = 40.0f;

int bugTypeToPolicySlot(BugType t)
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

int hexDistance(Index a, Index b)
{
    int aq = a % BoardWidth, ar = a / BoardWidth;
    int bq = b % BoardWidth, br = b / BoardWidth;
    int dq = aq - bq, dr = ar - br;
    return (std::abs(dq) + std::abs(dr) + std::abs(dq + dr)) / 2;
}

float normalizedDistance(Index a, Index b)
{
    if (a == NullIndex || b == NullIndex)
        return 1.0f;
    return std::min(1.0f, static_cast<float>(hexDistance(a, b)) / MaxQueenDist);
}

bool isAdjacentTo(Index pos, Index target)
{
    if (pos == NullIndex || target == NullIndex)
        return false;
    for (int d = 0; d < 6; ++d)
        if (pos + NeighborOffsets[d] == target)
            return true;
    return false;
}

int occupiedNeighborCount(const Board& board, Index pos, Index ignoredSource)
{
    if (pos == NullIndex)
        return 0;

    int count = 0;
    for (int d = 0; d < 6; ++d)
    {
        Index nb = pos + NeighborOffsets[d];
        if (!IsValidIndex(nb))
            continue;
        if (nb == ignoredSource && ignoredSource != NullIndex && board.stackHeight[ignoredSource] <= 1)
            continue;
        if (board.HasPieceAt(nb))
            ++count;
    }
    return count;
}

bool isInstantWin(const Board& board, const Move& move, Color perspective)
{
    if (move == PassMove)
        return false;

    PieceName enemyQueen = (perspective == Color::White) ? PieceName::bQ : PieceName::wQ;
    if (!board.PieceInPlay(enemyQueen))
        return false;

    Index queenPos = board.GetPosition(enemyQueen);
    if (move.Destination != queenPos && !isAdjacentTo(move.Destination, queenPos))
        return false;

    int surroundCount = 0;
    for (int d = 0; d < 6; ++d)
    {
        Index nb = queenPos + NeighborOffsets[d];
        if (!IsValidIndex(nb))
            continue;
        if (nb == move.Destination)
        {
            ++surroundCount;
            continue;
        }
        if (move.Source != NullIndex && nb == move.Source && board.stackHeight[move.Source] <= 1)
            continue;
        if (board.HasPieceAt(nb))
            ++surroundCount;
    }
    return surroundCount >= 6;
}
}

std::vector<float> EncodeMoveFeatures(const Board& board, const Move& move)
{
    std::vector<float> f(MoveFeatureDim, 0.0f);

    const Color myColor = board.currentColor;
    const PieceName myQueen = (myColor == Color::White) ? PieceName::wQ : PieceName::bQ;
    const PieceName enemyQueen = (myColor == Color::White) ? PieceName::bQ : PieceName::wQ;
    const Index myQueenPos = board.PieceInPlay(myQueen) ? board.GetPosition(myQueen) : NullIndex;
    const Index enemyQueenPos = board.PieceInPlay(enemyQueen) ? board.GetPosition(enemyQueen) : NullIndex;

    if (move == PassMove)
    {
        f[11] = 1.0f; // is pass
        f[29] = std::min(1.0f, static_cast<float>(board.GetCurrentTurn()) / TurnNorm);
        f[30] = myQueenPos != NullIndex ? 1.0f : 0.0f;
        f[31] = enemyQueenPos != NullIndex ? 1.0f : 0.0f;
        return f;
    }

    int bugSlot = bugTypeToPolicySlot(GetBugType(move.Piece));
    if (bugSlot >= 0)
        f[bugSlot] = 1.0f;

    const bool isPlacement = move.Source == NullIndex;
    const bool isMovement = !isPlacement;
    const bool destOccupiedBefore = IsValidIndex(move.Destination) && board.HasPieceAt(move.Destination);

    f[8] = (GetColor(move.Piece) == myColor) ? 1.0f : -1.0f;
    f[9] = isPlacement ? 1.0f : 0.0f;
    f[10] = isMovement ? 1.0f : 0.0f;
    f[11] = 0.0f;
    f[12] = isMovement ? 1.0f : 0.0f;
    f[13] = destOccupiedBefore ? 1.0f : 0.0f;
    f[14] = isAdjacentTo(move.Destination, myQueenPos) ? 1.0f : 0.0f;
    f[15] = isAdjacentTo(move.Destination, enemyQueenPos) ? 1.0f : 0.0f;

    float destMy = normalizedDistance(move.Destination, myQueenPos);
    float destEnemy = normalizedDistance(move.Destination, enemyQueenPos);
    float srcMy = normalizedDistance(move.Source, myQueenPos);
    float srcEnemy = normalizedDistance(move.Source, enemyQueenPos);

    f[16] = destMy;
    f[17] = destEnemy;
    f[18] = srcMy;
    f[19] = srcEnemy;
    f[20] = isMovement ? std::clamp(srcEnemy - destEnemy, -1.0f, 1.0f) : 0.0f;
    f[21] = isMovement ? std::clamp(srcMy - destMy, -1.0f, 1.0f) : 0.0f;
    f[22] = static_cast<float>(occupiedNeighborCount(board, move.Destination, move.Source)) / 6.0f;
    f[23] = isMovement ? static_cast<float>(occupiedNeighborCount(board, move.Source, NullIndex)) / 6.0f : 0.0f;
    f[24] = destOccupiedBefore ? std::min(1.0f, static_cast<float>(board.stackHeight[move.Destination]) / MaxStackHeight) : 0.0f;
    f[25] = isMovement ? std::min(1.0f, static_cast<float>(board.stackHeight[move.Source]) / MaxStackHeight) : 0.0f;
    f[26] = isInstantWin(board, move, myColor) ? 1.0f : 0.0f;
    f[27] = (isMovement && !isAdjacentTo(move.Source, enemyQueenPos) && isAdjacentTo(move.Destination, enemyQueenPos)) ? 1.0f : 0.0f;
    f[28] = isAdjacentTo(move.Destination, myQueenPos) ? 1.0f : 0.0f;
    f[29] = std::min(1.0f, static_cast<float>(board.GetCurrentTurn()) / TurnNorm);
    f[30] = myQueenPos != NullIndex ? 1.0f : 0.0f;
    f[31] = enemyQueenPos != NullIndex ? 1.0f : 0.0f;

    return f;
}

std::string MoveFeaturesToJson(const std::vector<float>& features)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "[";
    for (size_t i = 0; i < features.size(); ++i)
    {
        if (i > 0) ss << ",";
        ss << features[i];
    }
    ss << "]";
    return ss.str();
}

int GraphNodeIndex(const Board& board, PieceName piece)
{
    if (piece == PieceName::INVALID || !board.PieceInPlay(piece))
        return -1;

    int index = 0;
    for (int i = 0; i < static_cast<int>(piece); ++i)
        if (board.PieceInPlay(static_cast<PieceName>(i)))
            ++index;
    return index;
}

std::string MoveStructuralJson(const Board& board, const Move& move)
{
    std::ostringstream ss;
    ss << "\"src\":" << (move == PassMove ? -1 : GraphNodeIndex(board, move.Piece));
    ss << ",\"dst\":[";

    if (!(move == PassMove))
    {
        Index dest = move.Destination;
        bool first = true;

        PieceName onTop = board.GetPieceAt(dest);
        if (onTop != PieceName::INVALID)
        {
            ss << "[" << GraphNodeIndex(board, onTop) << "," << GNNSlotUp << "]";
            first = false;
        }
        else
        {
            for (int d = 0; d < 6; ++d)
            {
                Index nb = GetNeighborAt(dest, static_cast<Direction>(d));
                if (!IsValidIndex(nb)) continue;

                PieceName np = board.GetPieceAt(nb);
                if (np == move.Piece)
                    np = board.GetPieceUnder(np);
                if (np == PieceName::INVALID) continue;

                if (!first) ss << ",";
                ss << "[" << GraphNodeIndex(board, np) << ","
                   << static_cast<int>(Opposite(static_cast<Direction>(d))) << "]";
                first = false;
            }
        }
    }

    ss << "]";
    return ss.str();
}

} // namespace HiveGotThis
