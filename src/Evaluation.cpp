#include "Evaluation.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace HiveGotThis
{

// =============================================================================
// PESI EURISTICI
// =============================================================================

// Tabella non lineare per il numero di vicini occupati della regina (0-6).
// Andamento esponenziale: pochi vicini sono poco pericolosi, 4+ e' critico.
static constexpr double QueenSurroundTable[7] = {
    0.0,    // 0 vicini
    2.0,    // 1 vicino
    6.0,    // 2 vicini
    14.0,   // 3 vicini
    28.0,   // 4 vicini - situazione seria
    55.0,   // 5 vicini - quasi persa/vinta
    100.0   // 6 vicini - terminale
};

// Bonus per scarabeo sopra la regina avversaria
static constexpr double W_BEETLE_ON_QUEEN      = 50.0;

// Peso per tipo di pezzo adiacente alla regina avversaria
static double PieceTypeQueenThreat(BugType type)
{
    switch (type)
    {
        case BugType::Beetle:      return 5.0;
        case BugType::SoldierAnt:  return 3.5;
        case BugType::Mosquito:    return 3.5;
        case BugType::Ladybug:     return 2.5;
        case BugType::Spider:      return 1.5;
        case BugType::Grasshopper: return 1.5;
        case BugType::Pillbug:     return 1.5;
        case BugType::QueenBee:    return 1.0;
        default:                   return 0.0;
    }
}

// Pesi per le feature
static constexpr double W_QUEEN_CAN_SLIDE     = 2.0;   // Per ogni direzione in cui la regina puo' effettivamente scivolare
static constexpr double W_QUEEN_ESCAPE_EMPTY   = 0.5;   // Bonus base per ogni vicino vuoto (piu' leggero del can-slide)
static constexpr double W_PINNED_OWN           = -0.6;  // Penalita' per ogni nostro pezzo bloccato
static constexpr double W_PINNED_OPP           = 0.5;   // Bonus per ogni pezzo avversario bloccato
static constexpr double W_MOBILE_PIECE         = 0.5;   // Bonus per ogni nostro pezzo mobile
static constexpr double W_BURIED_OWN           = -2.0;  // Penalita' per ogni nostro pezzo sepolto
static constexpr double W_BURIED_OPP           = 2.0;   // Bonus per ogni pezzo avversario sepolto
static constexpr double W_DIST2_THREAT         = 0.7;   // Pezzo mobile a distanza 2 dalla regina avversaria
static constexpr double W_QUEEN_NOT_PLACED     = -4.0;  // Penalita' se la regina non e' piazzata e turno >= 6
static constexpr double W_PLACEMENT_FLEX       = 0.15;  // Per ogni posizione di piazzamento disponibile
static constexpr double W_QUEEN_DEADLINE_NEAR  = -2.0;  // Penalita' aggiuntiva se mancano 1-2 turni alla deadline

// Temperatura per la sigmoide (board evaluation)
static constexpr double SIGMOID_K              = 0.04;
// Temperatura per la sigmoide (move evaluation)
static constexpr double SIGMOID_K_MOVE         = 0.05;

// Valore di sviluppo base per tipo di pezzo
static double PieceDevelopmentValue(BugType type)
{
    switch (type)
    {
        case BugType::SoldierAnt:  return 1.8;
        case BugType::Beetle:      return 1.2;
        case BugType::Mosquito:    return 1.3;
        case BugType::Ladybug:     return 1.2;
        case BugType::Pillbug:     return 1.0;
        case BugType::Spider:      return 0.8;
        case BugType::Grasshopper: return 0.7;
        case BugType::QueenBee:    return 0.3;
        default:                   return 0.0;
    }
}


// =============================================================================
// FUNZIONI HELPER -> SERVONO PER LE FEATURES E INPUT DELLA GNN
// =============================================================================

static int CountOccupiedNeighbors(const Board& board, Index pos)
{
    int count = 0;
    for (int d = 0; d < 6; d++)
    {
        Index neighbor = pos + NeighborOffsets[d];
        if (IsValidIndex(neighbor) && board.HasPieceAt(neighbor))
            count++;
    }
    return count;
}

static void AddGraphEdge(GraphFeatures& gf, int from, int to, GraphEdgeType type)
{
    if (gf.numEdges >= GraphFeatureMaxEdges)
        return;

    gf.edgeFrom[gf.numEdges] = from;
    gf.edgeTo[gf.numEdges] = to;
    gf.edgeType[gf.numEdges] = static_cast<int>(type);
    gf.numEdges++;
}

static int GraphBugFeatureIndex(BugType type)
{
    switch (type)
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

static double MobilityNormalizer(BugType type)
{
    switch (type)
    {
        case BugType::QueenBee:    return 6.0;
        case BugType::SoldierAnt:  return 14.0;
        case BugType::Spider:      return 6.0;
        case BugType::Grasshopper: return 8.0;
        case BugType::Beetle:      return 6.0;
        case BugType::Mosquito:    return 14.0;
        case BugType::Ladybug:     return 8.0;
        case BugType::Pillbug:     return 6.0;
        default:                   return 1.0;
    }
}

static Color OpponentOf(Color color)
{
    return (color == Color::White) ? Color::Black : Color::White;
}

static PieceName QueenOf(Color color)
{
    return (color == Color::White) ? PieceName::wQ : PieceName::bQ;
}
// Calcola la distanza esagonale tra due celle convertendo l'indice 1D in coordinate assiali (q,r), ricavando la terza coordinata cubica implicitamente, e normalizzando il risultato su un raggio massimo.
static int HexDistance(Index a, Index b)
{
    int aq = a % BoardWidth;
    int ar = a / BoardWidth;
    int bq = b % BoardWidth;
    int br = b / BoardWidth;
    int dq = aq - bq;
    int dr = ar - br;
    return (std::abs(dq) + std::abs(dr) + std::abs(dq + dr)) / 2;
}
//Calcola la distanza esagonale reale tra il pezzo e la regina.
static float NormalizedHexDistance(const Board& board, PieceName piece, PieceName queen)
{
    static constexpr float MaxQueenDistance = 10.0f;

    if (!board.PieceInPlay(piece) || !board.PieceInPlay(queen))
        return 1.0f;

    float distance = static_cast<float>(HexDistance(board.GetPosition(piece), board.GetPosition(queen)));
    return std::min(1.0f, distance / MaxQueenDistance);
}

static bool IsAdjacentToPiece(const Board& board, PieceName piece, PieceName target)
{
    if (!board.PieceInPlay(piece) || !board.PieceInPlay(target))
        return false;

    Index pos = board.GetPosition(piece);
    Index targetPos = board.GetPosition(target);
    for (int d = 0; d < 6; d++)
    {
        if (pos + NeighborOffsets[d] == targetPos)
            return true;
    }

    return false;
}

static int CountLegalMovesForPiece(const Board& board, PieceName piece)
{
    if (!board.PieceInPlay(piece))
        return 0;
    if (!board.PieceIsOnTop(piece))
        return 0;
    if (board.cannotBeMoved[piece])
        return 0;

    std::vector<Move> moves;
    switch (GetBugType(piece))
    {
        case BugType::QueenBee:    board.GetQueenBeeMoves(piece, moves);    break;
        case BugType::SoldierAnt:  board.GetSoldierAntMoves(piece, moves);  break;
        case BugType::Spider:      board.GetSpiderMoves(piece, moves);      break;
        case BugType::Grasshopper: board.GetGrasshopperMoves(piece, moves); break;
        case BugType::Beetle:      board.GetBeetleMoves(piece, moves);      break;
        case BugType::Mosquito:    board.GetMosquitoMoves(piece, moves);    break;
        case BugType::Ladybug:     board.GetLadybugMoves(piece, moves);     break;
        case BugType::Pillbug:     board.GetPillbugMoves(piece, moves);     break;
        default: break;
    }

    std::unordered_set<size_t> seen;
    for (const Move& move : moves)
        seen.insert(hash(move));

    return static_cast<int>(seen.size());
}

static float MobilityScore(const Board& board, PieceName piece)
{
    if (!board.PieceInPlay(piece))
        return -1.0f;

    double normalizer = MobilityNormalizer(GetBugType(piece));
    double score = static_cast<double>(CountLegalMovesForPiece(board, piece)) / normalizer;
    return static_cast<float>(std::min(1.0, score));
}

static int CountInHandByType(const Board& board, Color color, BugType type)
{
    int count = 0;
    PieceName start = (color == Color::White) ? PieceName::wQ : PieceName::bQ;
    PieceName end = (color == Color::White) ? PieceName::wP : PieceName::bP;

    for (uint8_t p = start; p <= end; p++)
    {
        PieceName piece = static_cast<PieceName>(p);
        if (!PieceNameIsEnabledForGameType(piece, board.gameType))
            continue;
        if (GetBugType(piece) == type && board.PieceInHand(piece))
            count++;
    }

    return count;
}

static void FillInventoryFeatures(const Board& board, Color color, std::array<float, 8>& out)
{
    out[0] = static_cast<float>(CountInHandByType(board, color, BugType::QueenBee));
    out[1] = static_cast<float>(CountInHandByType(board, color, BugType::SoldierAnt)) / 3.0f;
    out[2] = static_cast<float>(CountInHandByType(board, color, BugType::Spider)) / 2.0f;
    out[3] = static_cast<float>(CountInHandByType(board, color, BugType::Grasshopper)) / 3.0f;
    out[4] = static_cast<float>(CountInHandByType(board, color, BugType::Beetle)) / 2.0f;
    out[5] = static_cast<float>(CountInHandByType(board, color, BugType::Mosquito));
    out[6] = static_cast<float>(CountInHandByType(board, color, BugType::Ladybug));
    out[7] = static_cast<float>(CountInHandByType(board, color, BugType::Pillbug));
}



/*
QUA FINISCONO LE FUNZIONI CHE HO AGGIUNTO PER FEATURES GNN
*/

// Conta le direzioni in cui la regina puo' effettivamente scivolare via (CanSlide ground)
static int CountQueenSlideEscapes(const Board& board, Index queenPos)
{
    int count = 0;
    for (int d = 0; d < 6; d++)
    {
        Index neighbor = queenPos + NeighborOffsets[d];
        if (!IsValidIndex(neighbor) || board.HasPieceAt(neighbor))
            continue;
        // La casella e' vuota: verifica se la regina puo' scivolarci
        if (board.CanSlide(queenPos, static_cast<Direction>(d), SlideMode::Ground))
            count++;
    }
    return count;
}

// Controlla se un pezzo del colore specificato si trova sopra una posizione
static bool HasColorPieceOnTop(const Board& board, Index pos, Color color)
{
    if (!board.HasPieceAt(pos))
        return false;
    PieceName top = board.GetPieceAt(pos);
    return GetColor(top) == color && board.stackHeight[pos] > 1;
}

// Calcola il punteggio di minaccia dei pezzi di un colore adiacenti a una posizione
static double ComputeTypedQueenThreat(const Board& board, Index queenPos, Color attackerColor)
{
    double threat = 0.0;
    for (int d = 0; d < 6; d++)
    {
        Index neighbor = queenPos + NeighborOffsets[d];
        if (IsValidIndex(neighbor) && board.HasPieceAt(neighbor))
        {
            PieceName piece = board.GetPieceAt(neighbor);
            if (GetColor(piece) == attackerColor)
            {
                BugType type = GetBugType(piece);
                double baseThreat = PieceTypeQueenThreat(type);

                // Bonus aggiuntivo se il pezzo e' mobile (puo' essere riposizionato)
                if (board.PieceIsOnTop(piece) && board.CanMoveWithoutBreakingHive(piece))
                    baseThreat *= 1.2;
                // Penalita' se il pezzo e' bloccato (non puo' muoversi, ma occupa spazio)
                // Il pezzo contribuisce comunque al surround, solo la minaccia offensiva e' ridotta

                threat += baseThreat;
            }
        }
    }
    return threat;
}

// Conta i pezzi mobili a distanza 2 dalla regina avversaria (usando dedup)
static int CountDist2Threats(const Board& board, Index queenPos, Color attackerColor)
{
    // Usa un piccolo set per evitare di contare lo stesso pezzo piu' volte
    PieceName seen[28];
    int seenCount = 0;

    for (int d1 = 0; d1 < 6; d1++)
    {
        Index n1 = queenPos + NeighborOffsets[d1];
        if (!IsValidIndex(n1))
            continue;

        for (int d2 = 0; d2 < 6; d2++)
        {
            Index n2 = n1 + NeighborOffsets[d2];
            if (!IsValidIndex(n2) || n2 == queenPos)
                continue;

            if (!board.HasPieceAt(n2))
                continue;

            PieceName piece = board.GetPieceAt(n2);
            if (GetColor(piece) != attackerColor)
                continue;
            if (!board.PieceIsOnTop(piece))
                continue;

            // Verifica che non sia gia' adiacente alla regina
            bool isAdjacent = false;
            for (int da = 0; da < 6; da++)
            {
                if (queenPos + NeighborOffsets[da] == n2)
                {
                    isAdjacent = true;
                    break;
                }
            }
            if (isAdjacent)
                continue;

            // Deduplicazione
            bool alreadySeen = false;
            for (int s = 0; s < seenCount; s++)
            {
                if (seen[s] == piece)
                {
                    alreadySeen = true;
                    break;
                }
            }
            if (!alreadySeen && seenCount < 28)
            {
                seen[seenCount++] = piece;
            }
        }
    }
    return seenCount;
}

// Conta le posizioni di piazzamento valide per un colore (senza allocare vector)
static int CountValidPlacements(const Board& board, Color color)
{
    int turn = board.GetCurrentTurn();
    if (turn == 0) return 1;

    int count = 0;
    // Usa un array booleano per deduplicare le posizioni candidate
    // Questo e' grande ma e' stack-allocated e molto veloce
    static bool candidateChecked[BoardSize];
    std::memset(candidateChecked, 0, sizeof(candidateChecked));

    for (int p = 0; p < NumPieceNames; p++)
    {
        PieceName piece = static_cast<PieceName>(p);
        if (!board.PieceInPlay(piece))
            continue;

        Index pos = board.GetPosition(piece);
        for (int d = 0; d < 6; d++)
        {
            Index neighbor = pos + NeighborOffsets[d];
            if (!IsValidIndex(neighbor) || candidateChecked[neighbor])
                continue;
            candidateChecked[neighbor] = true;

            if (!board.HasPieceAt(neighbor) &&
                board.CanPlaceAt(neighbor, color, turn))
            {
                count++;
            }
        }
    }
    return count;
}


// =============================================================================
// VALUTAZIONE PRINCIPALE
// =============================================================================

static double ComputeColorScore(const Board& board, Color color, Color opponent, int turn)
{
    double score = 0.0;

    PieceName myQueen  = (color == Color::White) ? PieceName::wQ  : PieceName::bQ;
    PieceName oppQueen = (color == Color::White) ? PieceName::bQ  : PieceName::wQ;
    PieceName myStart  = (color == Color::White) ? PieceName::wQ  : PieceName::bQ;
    PieceName myEnd    = (color == Color::White) ? PieceName::wP  : PieceName::bP;

    // =========================================================================
    // FEATURE 1: Accerchiamento della regina avversaria (scoring non lineare)
    // =========================================================================
    if (board.PieceInPlay(oppQueen))
    {
        Index oppQueenPos = board.GetPosition(oppQueen);
        int oppSurrounded = CountOccupiedNeighbors(board, oppQueenPos);
        score += QueenSurroundTable[oppSurrounded];

        // FEATURE 2: Minaccia tipizzata alla regina avversaria
        score += ComputeTypedQueenThreat(board, oppQueenPos, color);

        // FEATURE 3: Scarabeo sopra la regina avversaria
        if (board.stackHeight[oppQueenPos] > 1 &&
            HasColorPieceOnTop(board, oppQueenPos, color))
        {
            score += W_BEETLE_ON_QUEEN;
        }

        // FEATURE 4: Minacce a distanza 2
        int dist2 = CountDist2Threats(board, oppQueenPos, color);
        score += W_DIST2_THREAT * dist2;
    }

    // =========================================================================
    // FEATURE 5: Sicurezza della propria regina
    // =========================================================================
    if (board.PieceInPlay(myQueen))
    {
        Index myQueenPos = board.GetPosition(myQueen);
        int mySurrounded = CountOccupiedNeighbors(board, myQueenPos);
        score -= QueenSurroundTable[mySurrounded];

        // Vie di fuga REALI della propria regina (CanSlide, non solo empty neighbors)
        int slideEscapes = CountQueenSlideEscapes(board, myQueenPos);
        score += W_QUEEN_CAN_SLIDE * slideEscapes;

        // Bonus leggero per vicini vuoti (anche se non ci si puo' scivolare, danno respiro)
        int emptyNeighbors = 6 - mySurrounded;
        score += W_QUEEN_ESCAPE_EMPTY * emptyNeighbors;

        // Penalita' per scarabeo avversario sopra la nostra regina
        if (board.stackHeight[myQueenPos] > 1 &&
            HasColorPieceOnTop(board, myQueenPos, opponent))
        {
            score -= W_BEETLE_ON_QUEEN;
        }
    }
    else
    {
        // FEATURE 6: Penalita' progressiva se la regina non e' piazzata
        // La deadline e' turno 6 per White (turni 0,2,4,6) e turno 7 per Black (turni 1,3,5,7)
        int deadline = (color == Color::White) ? 6 : 7;
        if (turn >= deadline)
            score += W_QUEEN_NOT_PLACED;
        else if (turn >= deadline - 2)
            score += W_QUEEN_DEADLINE_NEAR; // Penalita' leggera: deadline vicina
    }

    // =========================================================================
    // FEATURE 7: Sviluppo, mobilita', pinning, pezzi sepolti
    // =========================================================================
    int mobilePieces = 0;
    int pinnedPieces = 0;
    int buriedPieces = 0;

    for (uint8_t p = myStart; p <= myEnd; p++)
    {
        PieceName piece = static_cast<PieceName>(p);
        if (!PieceNameIsEnabledForGameType(piece, board.gameType))
            continue;

        if (board.PieceInPlay(piece))
        {
            BugType type = GetBugType(piece);
            score += PieceDevelopmentValue(type);

            if (board.PieceIsOnTop(piece))
            {
                if (!board.cannotBeMoved[piece] && board.CanMoveWithoutBreakingHive(piece))
                    mobilePieces++;
                else
                    pinnedPieces++;
            }
            else
            {
                buriedPieces++;
            }
        }
    }

    score += W_MOBILE_PIECE * mobilePieces;
    score += W_PINNED_OWN * pinnedPieces;
    score += W_BURIED_OWN * buriedPieces;

    // =========================================================================
    // FEATURE 8: Pezzi avversari bloccati/sepolti
    // =========================================================================
    PieceName oppStart = (opponent == Color::White) ? PieceName::wQ : PieceName::bQ;
    PieceName oppEnd   = (opponent == Color::White) ? PieceName::wP : PieceName::bP;

    int oppPinnedPieces = 0;
    int oppBuriedPieces = 0;

    for (uint8_t p = oppStart; p <= oppEnd; p++)
    {
        PieceName piece = static_cast<PieceName>(p);
        if (!PieceNameIsEnabledForGameType(piece, board.gameType))
            continue;

        if (board.PieceInPlay(piece))
        {
            if (board.PieceIsOnTop(piece))
            {
                if (!board.cannotBeMoved[piece] && !board.CanMoveWithoutBreakingHive(piece))
                    oppPinnedPieces++;
            }
            else
            {
                oppBuriedPieces++;
            }
        }
    }

    score += W_PINNED_OPP * oppPinnedPieces;
    score += W_BURIED_OPP * oppBuriedPieces;

    // =========================================================================
    // FEATURE 9: Flessibilita' di piazzamento
    // =========================================================================
    // Piu' posizioni disponibili per piazzare = piu' opzioni strategiche
    // Controlliamo se abbiamo ancora pezzi in mano da piazzare
    bool hasPiecesInHand = false;
    for (uint8_t p = myStart; p <= myEnd; p++)
    {
        PieceName piece = static_cast<PieceName>(p);
        if (PieceNameIsEnabledForGameType(piece, board.gameType) && board.PieceInHand(piece))
        {
            hasPiecesInHand = true;
            break;
        }
    }

    if (hasPiecesInHand)
    {
        int placements = CountValidPlacements(board, color);
        score += W_PLACEMENT_FLEX * placements;
    }

    return score;
}


// =============================================================================
// FUNZIONE PUBBLICA DI VALUTAZIONE
// =============================================================================

double EvaluateBoard(const Board& board, Color perspective)
{
    BoardState state = board.GetBoardState();
    if (state == BoardState::WhiteWins)
        return (perspective == Color::White) ? 1.0 : 0.0;
    if (state == BoardState::BlackWins)
        return (perspective == Color::White) ? 0.0 : 1.0;
    if (state == BoardState::Draw)
        return 0.5;

    Color opponent = (perspective == Color::White) ? Color::Black : Color::White;
    int turn = board.GetCurrentTurn();

    double myScore  = ComputeColorScore(board, perspective, opponent, turn);
    double oppScore = ComputeColorScore(board, opponent, perspective, turn);
    double rawDiff  = myScore - oppScore;

    return 1.0 / (1.0 + std::exp(-rawDiff * SIGMOID_K));
}


// =============================================================================
// VALUTAZIONE EURISTICA MOSSE (usato da MCTS)
// =============================================================================

double EvaluateMove(const Board& board, const Move& move, Color perspective)
{
    double score = 0.0;

    if (move.Piece == PieceName::INVALID)
        return 0.0;

    Color opponent = (perspective == Color::White) ? Color::Black : Color::White;
    PieceName oppQueen = (perspective == Color::White) ? PieceName::bQ : PieceName::wQ;
    PieceName myQueen  = (perspective == Color::White) ? PieceName::wQ : PieceName::bQ;

    bool isPlacement = (move.Source == NullIndex);
    Index dest = move.Destination;

    // --- La regina avversaria e' in gioco? ---
    if (board.PieceInPlay(oppQueen))
    {
        Index oppQueenPos = board.GetPosition(oppQueen);

        // Mossa sopra la regina avversaria (scarabeo)
        if (dest == oppQueenPos)
        {
            score += 80.0;
        }
        else
        {
            // Controlla se la mossa piazza/muove un pezzo adiacente alla regina avversaria
            bool movesToQueenAdj = false;
            for (int d = 0; d < 6; d++)
            {
                if (dest == oppQueenPos + NeighborOffsets[d])
                {
                    movesToQueenAdj = true;
                    break;
                }
            }

            if (movesToQueenAdj)
            {
                int currentSurround = CountOccupiedNeighbors(board, oppQueenPos);

                bool comesFromQueenAdj = false;
                if (!isPlacement)
                {
                    for (int d = 0; d < 6; d++)
                    {
                        if (move.Source == oppQueenPos + NeighborOffsets[d])
                        {
                            comesFromQueenAdj = true;
                            break;
                        }
                    }
                }

                int newSurround = currentSurround + (comesFromQueenAdj ? 0 : 1);

                // VITTORIA IMMEDIATA
                if (newSurround >= 6)
                    return 1.0;

                score += 20.0 + 10.0 * newSurround;

                BugType type = GetBugType(move.Piece);
                score += PieceTypeQueenThreat(type);
            }
        }
    }

    // --- Difesa: la nostra regina e' in pericolo ---
    if (board.PieceInPlay(myQueen))
    {
        Index myQueenPos = board.GetPosition(myQueen);
        int myQueenSurround = CountOccupiedNeighbors(board, myQueenPos);

        if (myQueenSurround >= 3 && !isPlacement)
        {
            bool comesFromOwnQueenAdj = false;
            bool movesToOwnQueenAdj = false;
            for (int d = 0; d < 6; d++)
            {
                Index adj = myQueenPos + NeighborOffsets[d];
                if (move.Source == adj) comesFromOwnQueenAdj = true;
                if (dest == adj) movesToOwnQueenAdj = true;
            }

            // Muovere un pezzo lontano dalla propria regina quando e' in pericolo
            if (comesFromOwnQueenAdj && !movesToOwnQueenAdj)
                score += 5.0 + 3.0 * (myQueenSurround - 3);

            // Muovere la regina stessa fuori pericolo
            if (move.Piece == myQueen)
                score += 8.0 + 4.0 * (myQueenSurround - 3);
        }
    }

    // --- Piazzamento della regina quando la deadline e' vicina ---
    if (isPlacement)
    {
        BugType type = GetBugType(move.Piece);

        if (type == BugType::QueenBee)
        {
            int turn = board.GetCurrentTurn();
            int deadline = (perspective == Color::White) ? 6 : 7;
            if (turn >= deadline - 2)
                score += 15.0; // Priorita' alta: piazza la regina prima della deadline
        }
        else
        {
            score += PieceDevelopmentValue(type) * 0.5;
        }
    }

    return 1.0 / (1.0 + std::exp(-score * SIGMOID_K_MOVE));
}



/*
FUNZIONI PER ESTRAZIONI FEATURE GNN E CREAZIONE JSON
*/

GraphFeatures ExtractGraphFeatures(const Board& board)
{
    GraphFeatures gf;
    //Raccolta informazioni generali per capire di chi è il turno e di chi è la regina, per calcolare le feature in modo coerente
    Color myColor = board.currentColor;
    Color enemyColor = OpponentOf(myColor);
    PieceName myQueen = QueenOf(myColor);
    PieceName enemyQueen = QueenOf(enemyColor);

    // Calcola se la regina è in gioco, da quanti pezzi è circondata per entrambe le parti, normalizzato a [0,1]
    float myQueenSurround = board.PieceInPlay(myQueen)
        ? static_cast<float>(CountOccupiedNeighbors(board, board.GetPosition(myQueen))) / 6.0f
        : 0.0f;
    float enemyQueenSurround = board.PieceInPlay(enemyQueen)
        ? static_cast<float>(CountOccupiedNeighbors(board, board.GetPosition(enemyQueen))) / 6.0f
        : 0.0f;

    // Inventario dei pezzi in mano per entrambe le parti   
    std::array<float, 8> myInventory{};
    std::array<float, 8> enemyInventory{};
    FillInventoryFeatures(board, myColor, myInventory);
    FillInventoryFeatures(board, enemyColor, enemyInventory);

    // Vettore globale separato u: contesto macro passato al readout finale.
    gf.globalFeatures[0] = myQueenSurround;
    gf.globalFeatures[1] = enemyQueenSurround;
    gf.globalFeatures[2] = static_cast<float>(board.GetCurrentTurn()) / 40.0f;
    gf.globalFeatures[3] = board.PieceInPlay(myQueen) ? 1.0f : 0.0f;
    gf.globalFeatures[4] = board.PieceInPlay(enemyQueen) ? 1.0f : 0.0f;

    // Estrazione delle feature per ogni pezzo
    for (int p = 0; p < NumPieceNames; p++)
    {
        PieceName piece = static_cast<PieceName>(p);
        BugType type = GetBugType(piece);
        Color color = GetColor(piece);
        int typeFeature = GraphBugFeatureIndex(type); //funzione di mapping: prende il tipo di insetto (BugType) e restituisce quale indice tra 0 e 7 deve essere impostato a 1.0 (definita sopra riga 124)

        if (typeFeature >= 0)
            gf.nodeFeatures[p][typeFeature] = 1.0f;

        gf.nodeFeatures[p][8] = (color == myColor) ? 1.0f : -1.0f;
        //se il pezzo è ancora in mano, restituisce -1.0f;
        //se il pezzo è sul tavolo, conta quante mosse legali può fare;   //CountLegalMovesForPiece(board, piece)
        //divide quel numero per il massimo teorico del tipo di insetto; //MobilityNormalizer(GetBugType(piece))
        //taglia il valore massimo a 1.0
        gf.nodeFeatures[p][12] = MobilityScore(board, piece); //(riga 219)


        gf.nodeFeatures[p][19] = board.PieceInPlay(piece) ? 1.0f : 0.0f; //Piece in play funzione definita in board se il pezzo è in gioco è pari a 1 altrimenti è pari a 0
        gf.nodeFeatures[p][20] = myQueenSurround; //riga 733 -> num pezzi attorno alla mia regina
        gf.nodeFeatures[p][21] = enemyQueenSurround; //riga 736 -> num pezzi attorno alla regina avversaria

        //Inventario dei pezzi in mano per entrambe le parti 
        //myInventory[22] = Ape/Regina ancora in mano: 1.0 oppure 0.0
        //myInventory[23] = Formiche in mano / 3.0
        //myInventory[2] = Ragni in mano / 2.0
        for (int i = 0; i < 8; i++)
        {
            gf.nodeFeatures[p][22 + i] = myInventory[i];
            gf.nodeFeatures[p][30 + i] = enemyInventory[i];
        }

        //Se il pezzo non è sul tavolo, salta il resto del ciclo e passa al prossimo pezzo
        if (!board.PieceInPlay(piece))
            continue;
        //Features che dipendono dalla posizione reale del pezzo
        Index pos = board.GetPosition(piece); // posizione
        bool isOnTop = board.PieceIsOnTop(piece); //pezzo on top
        bool breaksHive = !board.CanMoveWithoutBreakingHive(piece); //muovere il pezzo rompe la regola one-hive?

        gf.nodeFeatures[p][9] = static_cast<float>(CountOccupiedNeighbors(board, pos)) / 6.0f; //numero di pezzi adiacenti/6
        gf.nodeFeatures[p][10] = static_cast<float>(std::max(0, static_cast<int>(board.stackHeight[pos]) - 1)) / 3.0f; //altezza dello stack
        //stackHeight = 1 -> (1 - 1) / 3 = 0.0
        // stackHeight = 2 -> (2 - 1) / 3 = 0.333
        //stackHeight = 3 -> (3 - 1) / 3 = 0.666
        //stackHeight = 4 -> (4 - 1) / 3 = 1.0
        gf.nodeFeatures[p][11] = isOnTop ? 1.0f : 0.0f; //è in cima allo stack?
        //Feature [13]: Is Pinned.
        //Vale 1.0 se almeno una di queste condizioni è vera:
        //il pezzo non è in cima alla pila;
        // il pezzo è bloccato da cannotBeMoved;
        //il pezzo non ha nessuna mossa legale.
        gf.nodeFeatures[p][13] = (!isOnTop || board.cannotBeMoved[piece] || CountLegalMovesForPiece(board, piece) == 0) ? 1.0f : 0.0f;
        // Feature [14]: Breaks Hive. Vale 1.0 se muovere il pezzo romperebbe l'hive, altrimenti 0.0. 
        gf.nodeFeatures[p][14] = breaksHive ? 1.0f : 0.0f;
        //Feature [15]: distanza dalla mia regina, normalizzata.
        gf.nodeFeatures[p][15] = NormalizedHexDistance(board, piece, myQueen); //funzione riga 165
        //Feature [16]: distanza dalla regina avversaria, normalizzata.
        gf.nodeFeatures[p][16] = NormalizedHexDistance(board, piece, enemyQueen);
        gf.nodeFeatures[p][17] = IsAdjacentToPiece(board, piece, myQueen) ? 1.0f : 0.0f; //pezzo tocca la mia regina?
        gf.nodeFeatures[p][18] = IsAdjacentToPiece(board, piece, enemyQueen) ? 1.0f : 0.0f; //pezzo tocca la regina avversaria?
    }

    for (int p1 = 0; p1 < NumPieceNames; p1++)
    {
        PieceName piece1 = static_cast<PieceName>(p1);
        //se il pezzo non è in gioco, salta al prossimo pezzo perchè non puo' avere archi
        if (!board.PieceInPlay(piece1))
            continue;

        Index pos1 = board.GetPosition(piece1);
        //Confronta piece1 con tutti i pezzi successivi:
        for (int p2 = p1 + 1; p2 < NumPieceNames; p2++)
        {
            PieceName piece2 = static_cast<PieceName>(p2);
            if (!board.PieceInPlay(piece2))
                continue;
            //Controlla le 6 direzioni esagonali attorno a piece1
            Index pos2 = board.GetPosition(piece2);
            for (int d = 0; d < 6; d++)
            {
                //Se piece2 si trova in una delle 6 celle vicine a piece1, allora i due pezzi sono adiacenti.
                if (pos1 + NeighborOffsets[d] == pos2)
                {
                    //aggiungo due archi perchè relazione bidirezionale
                    AddGraphEdge(gf, p1, p2, GraphEdgeType::Adjacent);
                    AddGraphEdge(gf, p2, p1, GraphEdgeType::Adjacent);
                    break;
                }
            }
        }
        //Controlla se piece1 è sopra un altro pezzo (stacked) e aggiunge un arco StackedOn
        PieceName below = board.GetPieceUnder(piece1);
        if (below != PieceName::INVALID)
            AddGraphEdge(gf, p1, static_cast<int>(below), GraphEdgeType::StackedOn);
    }
    /*
    for (int p1 = 0; p1 < NumPieceNames; p1++)
    {
        PieceName piece1 = static_cast<PieceName>(p1);
        for (int p2 = p1 + 1; p2 < NumPieceNames; p2++)
        {
            PieceName piece2 = static_cast<PieceName>(p2);
            if (GetColor(piece1) != GetColor(piece2))
                continue;

            AddGraphEdge(gf, p1, p2, GraphEdgeType::SameColor);
            AddGraphEdge(gf, p2, p1, GraphEdgeType::SameColor);
        }
    }
    */

    return gf;
}

std::string GraphFeaturesToJson(const GraphFeatures& features)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{";

    ss << "\"node_features\":[";
    for (int n = 0; n < GraphFeatureNodeCount; n++)
    {
        if (n > 0) ss << ",";
        ss << "[";
        for (int f = 0; f < GraphFeatureNodeDim; f++)
        {
            if (f > 0) ss << ",";
            ss << features.nodeFeatures[n][f];
        }
        ss << "]";
    }
    ss << "],";

    ss << "\"edge_index\":[[";
    for (int e = 0; e < features.numEdges; e++)
    {
        if (e > 0) ss << ",";
        ss << features.edgeFrom[e];
    }
    ss << "],[";
    for (int e = 0; e < features.numEdges; e++)
    {
        if (e > 0) ss << ",";
        ss << features.edgeTo[e];
    }
    ss << "]],";

    ss << "\"edge_type\":[";
    for (int e = 0; e < features.numEdges; e++)
    {
        if (e > 0) ss << ",";
        ss << features.edgeType[e];
    }
    ss << "],";

    ss << "\"global_features\":[";
    for (int f = 0; f < GraphFeatureGlobalDim; f++)
    {
        if (f > 0) ss << ",";
        ss << features.globalFeatures[f];
    }
    ss << "],";

    ss << "\"num_edges\":" << features.numEdges;
    ss << "}";
    return ss.str();
}

} // namespace HiveGotThis
