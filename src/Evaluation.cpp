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

// Conta i vicini occupati (su 6) di una posizione.
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
// Mobilita' e pin sono ora pesati per tipo (MobilityValueT / PinWeightT, vedi sotto):
// una formica libera/pinnata pesa molto piu' di un ragno o una cavalletta.
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

// Tabelle per tipo (BugType: Queen=0,Spider=1,Beetle=2,Grasshopper=3,Ant=4,Mosquito=5,Ladybug=6,Pillbug=7)
// La formica e' il pezzo che domina la mobilita': libera vale molto, pinnata e' un disastro
// ("a pinned Ant is a worthless Ant", Ingersoll cap. 8.1).
static constexpr double MobilityValueT[8] = { 0.3, 0.4, 0.7, 0.5, 1.3, 0.9, 0.7, 0.5 };
static constexpr double PinWeightT[8]     = { 0.5, 0.4, 0.6, 0.5, 1.5, 0.9, 0.6, 0.6 };
static inline double MobilityValueOf(BugType t){ return MobilityValueT[static_cast<uint8_t>(t)]; }
static inline double PinWeightOf(BugType t){ return PinWeightT[static_cast<uint8_t>(t)]; }


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
                    score += MobilityValueOf(type);          // type-weighted: formica libera vale di piu'
                else
                    score -= PinWeightOf(type);              // type-weighted: formica pinnata e' molto grave
            }
            else
            {
                buriedPieces++;
            }
        }
    }

    score += W_BURIED_OWN * buriedPieces;

    // =========================================================================
    // FEATURE 8: Pezzi avversari bloccati/sepolti
    // =========================================================================
    PieceName oppStart = (opponent == Color::White) ? PieceName::wQ : PieceName::bQ;
    PieceName oppEnd   = (opponent == Color::White) ? PieceName::wP : PieceName::bP;

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
                    score += PinWeightOf(GetBugType(piece));  // pinnare una formica avversaria vale molto
            }
            else
            {
                oppBuriedPieces++;
            }
        }
    }

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


} // namespace HiveGotThis
