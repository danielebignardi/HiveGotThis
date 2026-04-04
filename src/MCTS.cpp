#include "MCTS.h"
#include "Evaluation.h"

#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cstring>

namespace HiveGotThis
{

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

TranspositionTable::TranspositionTable(size_t sizePower2)
{
    size_t size = static_cast<size_t>(1) << sizePower2;
    m_mask = size - 1;
    m_table = new TTEntry[size];
    Clear();
}

TranspositionTable::~TranspositionTable()
{
    delete[] m_table;
}

void TranspositionTable::Clear()
{
    size_t size = m_mask + 1;
    std::memset(m_table, 0, size * sizeof(TTEntry));
}

bool TranspositionTable::Probe(uint64_t hash, TTEntry& entry) const
{
    size_t idx = hash & m_mask;
    const TTEntry& stored = m_table[idx];
    if (stored.hash == hash && stored.depth > 0)
    {
        entry = stored;
        return true;
    }
    return false;
}

void TranspositionTable::Store(uint64_t hash, double value, int16_t depth, bool isExact)
{
    size_t idx = hash & m_mask;
    TTEntry& stored = m_table[idx];

    if (depth >= stored.depth)
    {
        stored.hash = hash;
        stored.value = value;
        stored.depth = depth;
        stored.isExact = isExact;
    }
}


// =============================================================================
// MCTSNode
// =============================================================================

MCTSNode::MCTSNode(Move move, MCTSNode* parent, bool isMaxNode, double heuristicScore)
    : move(move)
    , parent(parent)
    , visitCount(0)
    , totalValue(0.0)
    , isExpanded(false)
    , isTerminal(false)
    , isMaxNode(isMaxNode)
    , provenResult(0)
    , heuristicScore(heuristicScore)
{
}

MCTSNode::~MCTSNode()
{
    for (MCTSNode* child : children)
        delete child;
}

double MCTSNode::UCB1(double explorationC, double logParentVisits, bool maximizing) const
{
    // Nodo con risultato provato: restituisce +/- infinito a seconda di chi sceglie.
    // MAX (rootColor) preferisce provenResult=1, MIN (avversario) preferisce provenResult=-1.
    if (provenResult != 0)
    {
        if (maximizing)
        {
            return (provenResult == 1) ? std::numeric_limits<double>::max() : -std::numeric_limits<double>::max();
        }
        else
        {
            return (provenResult == -1) ? std::numeric_limits<double>::max() : -std::numeric_limits<double>::max();
        }
    }

    // Win rate dal punto di vista di rootColor
    double winRate = totalValue / static_cast<double>(visitCount);

    // Nodo MAX: massimizza winRate. Nodo MIN: massimizza (1 - winRate).
    double exploitation = maximizing ? winRate : (1.0 - winRate);

    double exploration = explorationC * std::sqrt(logParentVisits / static_cast<double>(visitCount));
    double bias = PROGRESSIVE_BIAS_WEIGHT * heuristicScore / static_cast<double>(visitCount + 1);

    return exploitation + exploration + bias;
}

double MCTSNode::WinRate() const
{
    if (visitCount == 0) return 0.5;
    return totalValue / static_cast<double>(visitCount);
}


// =============================================================================
// MCTS - Utility
// =============================================================================

// Verifica se una mossa circonda completamente la regina avversaria (vittoria immediata).
// Tiene conto del fatto che un pezzo spostato libera la casella di partenza.
bool MCTS::IsInstantWin(const Board& board, const Move& move, Color perspective)
{
    if (move.Piece == PieceName::INVALID)
        return false;

    PieceName oppQueen = (perspective == Color::White) ? PieceName::bQ : PieceName::wQ;

    if (!board.PieceInPlay(oppQueen))
        return false;

    Index oppQueenPos = board.GetPosition(oppQueen);
    Index dest = move.Destination;

    // Scarta subito le mosse che non finiscono adiacenti o sopra alla regina avversaria
    bool isAdjacentToOppQueen = false;
    for (int d = 0; d < 6; d++)
    {
        if (dest == oppQueenPos + NeighborOffsets[d])
        {
            isAdjacentToOppQueen = true;
            break;
        }
    }
    if (!isAdjacentToOppQueen && dest != oppQueenPos)
        return false;

    // Conta i vicini occupati della regina dopo la mossa
    int surroundCount = 0;
    for (int d = 0; d < 6; d++)
    {
        Index neighbor = oppQueenPos + NeighborOffsets[d];
        if (!IsValidIndex(neighbor))
            continue;

        if (neighbor == dest)
        {
            surroundCount++;
        }
        else if (board.HasPieceAt(neighbor))
        {
            // Se il pezzo viene spostato via da qui (e non e' impilato), non conta piu'
            if (move.Source != NullIndex && neighbor == move.Source && board.stackHeight[move.Source] <= 1)
                continue;
            surroundCount++;
        }
    }

    return surroundCount >= 6;
}

// Ordina le mosse per valutazione euristica crescente (le migliori in coda,
// cosi' vengono estratte per prime con pop_back).
void MCTS::OrderMoves(std::vector<Move>& moves, const Board& board, Color perspective)
{
    std::sort(moves.begin(), moves.end(),
        [&board, perspective](const Move& a, const Move& b)
        {
            return EvaluateMove(board, a, perspective) < EvaluateMove(board, b, perspective);
        }
    );
}


// =============================================================================
// MCTS - Solver
// =============================================================================

// Prova a determinare se il risultato di `node` e' certo, risalendo verso la radice.
// Logica minimax: MAX vince se almeno un figlio e' vittoria; perde se tutti sono sconfitta.
//                 MIN vince (per se') se almeno un figlio e' sconfitta di rootColor; perde se tutti sono vittoria.
void MCTS::TrySolve(MCTSNode* node)
{
    if (!node->isExpanded || node->children.empty())
        return;
    if (!node->unexpandedMoves.empty())
        return; // Ci sono ancora figli da generare: non possiamo concludere nulla

    bool allChildrenProven = true;

    if (node->isMaxNode)
    {
        for (MCTSNode* child : node->children)
        {
            if (child->provenResult == 0)
            {
                allChildrenProven = false;
            }
            else if (child->provenResult == 1)
            {
                // Basta un figlio vincente: rootColor sceglie quello
                node->provenResult = 1;
                if (node->parent != nullptr)
                    TrySolve(node->parent);
                return;
            }
        }
        // Tutti figli provati e nessuna vittoria: tutte le mosse perdono
        if (allChildrenProven)
        {
            node->provenResult = -1;
            if (node->parent != nullptr)
                TrySolve(node->parent);
        }
    }
    else
    {
        for (MCTSNode* child : node->children)
        {
            if (child->provenResult == 0)
            {
                allChildrenProven = false;
            }
            else if (child->provenResult == -1)
            {
                // Basta un figlio perdente: l'avversario sceglie quello
                node->provenResult = -1;
                if (node->parent != nullptr)
                    TrySolve(node->parent);
                return;
            }
        }
        // Tutti figli provati e nessuna sconfitta: rootColor vince comunque
        if (allChildrenProven)
        {
            node->provenResult = 1;
            if (node->parent != nullptr)
                TrySolve(node->parent);
        }
    }
}


// =============================================================================
// MCTS - Algoritmo principale
// =============================================================================

// Esegue una singola iterazione MCTS: selezione -> espansione -> valutazione -> backpropagation.
// Tutti i valori sono dal punto di vista di rootColor (1.0 = vittoria, 0.0 = sconfitta).
void MCTS::RunIteration(MCTSNode* root, const Board& rootBoard, Color rootColor, TranspositionTable& tt)
{
    // === SELEZIONE ===
    // Scende nell'albero seguendo UCB1 fino a un nodo da espandere, terminale, o provato.
    MCTSNode* node = root;
    Board board = rootBoard;
    int depth = 0;

    while (node->isExpanded && !node->isTerminal && node->provenResult == 0)
    {
        double logParent = std::log(static_cast<double>(node->visitCount + 1));

        MCTSNode* best = nullptr;
        double bestUCB = -std::numeric_limits<double>::max();

        for (MCTSNode* child : node->children)
        {
            double ucb = child->UCB1(EXPLORATION_C, logParent, node->isMaxNode);
            if (ucb > bestUCB)
            {
                bestUCB = ucb;
                best = child;
            }
        }

        board.ApplyMove(best->move);
        node = best;
        depth++;
    }

    // Nodo gia' provato: backpropaga direttamente senza espandere
    if (node->provenResult != 0)
    {
        double value = (node->provenResult == 1) ? 1.0 : 0.0;
        MCTSNode* current = node;
        while (current != nullptr)
        {
            current->visitCount++;
            current->totalValue += value;
            current = current->parent;
        }
        return;
    }

    // === ESPANSIONE ===
    if (!node->isTerminal)
    {
        // Primo accesso: genera le mosse valide e controllase la partita e' finita
        if (node->unexpandedMoves.empty() && node->children.empty())
        {
            BoardState state = board.GetBoardState();
            if (GameIsOver(state))
            {
                node->isTerminal = true;
                node->isExpanded = true;

                if (state == BoardState::WhiteWins)
                    node->provenResult = (rootColor == Color::White) ? 1 : -1;
                else if (state == BoardState::BlackWins)
                    node->provenResult = (rootColor == Color::White) ? -1 : 1;
            }
            else
            {
                board.GetValidMoves(node->unexpandedMoves);
                OrderMoves(node->unexpandedMoves, board, board.currentColor);
            }
        }

        // Estrae la mossa migliore (in coda) e crea il nodo figlio
        if (!node->unexpandedMoves.empty())
        {
            Move move = node->unexpandedMoves.back();
            node->unexpandedMoves.pop_back();

            if (node->unexpandedMoves.empty())
                node->isExpanded = true;

            double hScore = EvaluateMove(board, move, board.currentColor);

            // Il figlio e' MAX se dopo la mossa tocca a rootColor.
            // Ora muove board.currentColor, dopo ApplyMove tocchera' all'altro.
            bool childIsMax = (board.currentColor != rootColor);

            MCTSNode* child = new MCTSNode(move, node, childIsMax, hScore);
            node->children.push_back(child);

            board.ApplyMove(move);
            node = child;
            depth++;

            // Controlla se il nuovo stato e' terminale
            BoardState childState = board.GetBoardState();
            if (GameIsOver(childState))
            {
                node->isTerminal = true;
                node->isExpanded = true;

                if (childState == BoardState::WhiteWins)
                    node->provenResult = (rootColor == Color::White) ? 1 : -1;
                else if (childState == BoardState::BlackWins)
                    node->provenResult = (rootColor == Color::White) ? -1 : 1;
            }
        }
    }

    // === VALUTAZIONE ===
    // Determina il valore della foglia: risultato provato, cache TT, o euristica.
    double value;

    if (node->provenResult != 0)
    {
        value = (node->provenResult == 1) ? 1.0 : 0.0;
    }
    else
    {
        uint64_t hash = board.GetHash();
        TTEntry ttEntry;

        if (tt.Probe(hash, ttEntry))
        {
            value = ttEntry.value;
        }
        else
        {
            value = EvaluateBoard(board, rootColor);
            tt.Store(hash, value, static_cast<int16_t>(depth), node->isTerminal);
        }
    }

    // === BACKPROPAGATION ===
    // Risale fino alla radice, aggiornando visite e valore cumulativo di ogni nodo.
    MCTSNode* current = node;
    while (current != nullptr)
    {
        current->visitCount++;
        current->totalValue += value;
        current = current->parent;
    }

    // === SOLVER ===
    // Se la foglia ha un risultato certo, prova a propagarlo verso la radice
    if (node->provenResult != 0 && node->parent != nullptr)
    {
        TrySolve(node->parent);
    }
}

// Sceglie la mossa finale dalla radice.
// Priorita':   1) vittoria provata
//              2) mossa piu' visitata (escluse sconfitte provate),
//              3) se tutto e' perso, mossa piu' visitata (ritarda la sconfitta).
Move MCTS::SelectBestMove(MCTSNode* root)
{
    MCTSNode* bestChild = nullptr;
    int bestVisits = -1;

    // 1) Cerca una vittoria provata
    for (MCTSNode* child : root->children)
    {
        if (child->provenResult == 1)
        {
            if (child->visitCount > bestVisits)
            {
                bestVisits = child->visitCount;
                bestChild = child;
            }
        }
    }
    if (bestChild)
        return bestChild->move;

    // 2) Mossa piu' visitata, escludendo le sconfitte provate
    bestVisits = -1;
    bestChild = nullptr;
    for (MCTSNode* child : root->children)
    {
        if (child->provenResult == -1)
            continue;

        if (child->visitCount > bestVisits)
        {
            bestVisits = child->visitCount;
            bestChild = child;
        }
    }

    // 3) Tutte le mosse sono sconfitte provate: scegli quella piu' visitata
    if (!bestChild)
    {
        bestVisits = -1;
        for (MCTSNode* child : root->children)
        {
            if (child->visitCount > bestVisits)
            {
                bestVisits = child->visitCount;
                bestChild = child;
            }
        }
    }

    return bestChild->move;
}


// =============================================================================
// MCTS - Entry point
// =============================================================================

// Cerca la mossa migliore entro un limite di tempo (millisecondi).
// Prima controlla casi banali (nessuna mossa, mossa unica, vittoria immediata),
// poi esegue iterazioni MCTS fino a scadenza o soluzione provata.
Move MCTS::Search(const Board& rootBoard, int timeLimitMs)
{
    Color rootColor = rootBoard.currentColor;

    std::vector<Move> moves;
    rootBoard.GetValidMoves(moves);

    if (moves.empty())
        return PassMove;
    if (moves.size() == 1)
        return moves[0];

    for (const Move& m : moves)
    {
        if (IsInstantWin(rootBoard, m, rootColor))
            return m;
    }

    MCTSNode root(PassMove, nullptr, /*isMaxNode=*/true);
    TranspositionTable tt(18);

    auto startTime = std::chrono::steady_clock::now();
    auto endTime   = startTime + std::chrono::milliseconds(timeLimitMs);

    int iterations = 0;
    while (true)
    {
        RunIteration(&root, rootBoard, rootColor, tt);
        iterations++;

        if ((iterations & (TIME_CHECK_INTERVAL - 1)) == 0)
        {
            if (std::chrono::steady_clock::now() >= endTime)
                break;
            if (root.provenResult != 0)
                break;
        }
    }

    return SelectBestMove(&root);
}

// Come Search, ma con un limite sul numero di iterazioni anziche' sul tempo.
Move MCTS::SearchIterations(const Board& rootBoard, int maxIterations)
{
    Color rootColor = rootBoard.currentColor;

    std::vector<Move> moves;
    rootBoard.GetValidMoves(moves);

    if (moves.empty())
        return PassMove;
    if (moves.size() == 1)
        return moves[0];

    for (const Move& m : moves)
    {
        if (IsInstantWin(rootBoard, m, rootColor))
            return m;
    }

    MCTSNode root(PassMove, nullptr, /*isMaxNode=*/true);
    TranspositionTable tt(18);

    for (int i = 0; i < maxIterations; i++)
    {
        RunIteration(&root, rootBoard, rootColor, tt);
        if (root.provenResult != 0)
            break;
    }

    return SelectBestMove(&root);
}

} // namespace HiveGotThis
