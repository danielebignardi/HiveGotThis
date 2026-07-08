#include "MCTS.h"
#include "Evaluation.h"
#include "NeuralEvaluator.h"

#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <memory>

namespace HiveGotThis
{

namespace
{
    // Non posseduta: chi chiama SetValueNetwork resta proprietario dell'oggetto.
    TorchScriptValueEvaluator* g_valueNetwork = nullptr;

    // Profiling: tempo cumulativo e numero di chiamate reali alla rete
    // (solo cache miss). Non usate nel normale funzionamento del motore.
    double g_networkTimeMs = 0.0;
    long   g_networkCallCount = 0;

    // Profiling: numero totale di iterazioni MCTS eseguite (cache hit + miss).
    long g_iterationCount = 0;

    // Transposition table persistente per tutta la durata del processo: le
    // entry sono valutazioni pure della value network per una board, valide
    // finche' i pesi della rete non cambiano (caricati una volta sola in
    // main.cpp) - quindi e' sicuro riusarle tra chiamate di Search/SearchIterations
    // diverse (turni diversi, anche partite diverse nello stesso processo).
    std::unique_ptr<TranspositionTable> g_tt;

    // Con la tabella persistente l'allocazione avviene una sola volta per
    // processo (non piu' ad ogni Search), quindi possiamo permetterci una
    // tabella piu' grande della vecchia 2^18 per-chiamata.
    constexpr size_t PERSISTENT_TT_SIZE_POWER2 = 21; // ~2.1M entries, ~48 MB
}

void MCTS::SetValueNetwork(TorchScriptValueEvaluator* evaluator)
{
    g_valueNetwork = evaluator;
}

void MCTS::ResetNetworkStats()
{
    g_networkTimeMs = 0.0;
    g_networkCallCount = 0;
    g_iterationCount = 0;
}

double MCTS::GetNetworkTimeMs()
{
    return g_networkTimeMs;
}

long MCTS::GetNetworkCallCount()
{
    return g_networkCallCount;
}

long MCTS::GetIterationCount()
{
    return g_iterationCount;
}

TranspositionTable& MCTS::GetPersistentTranspositionTable()
{
    if (!g_tt)
        g_tt = std::make_unique<TranspositionTable>(PERSISTENT_TT_SIZE_POWER2);
    return *g_tt;
}

void MCTS::ClearTranspositionTable()
{
    if (g_tt)
        g_tt->Clear();
}

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

MCTSNode::MCTSNode(Move move, MCTSNode* parent, double heuristicScore)
    : move(move)
    , parent(parent)
    , visitCount(0)
    , totalValue(0.0)
    , isExpanded(false)
    , isTerminal(false)
    , provenResult(0)
    , heuristicScore(heuristicScore)
{
}

MCTSNode::~MCTSNode()
{
    for (MCTSNode* child : children)
        delete child;
}

double MCTSNode::UCB1(double explorationC, double logParentVisits) const
{
    // Negamax: il valore di questo figlio e' dal punto di vista del SUO giocatore di turno
    // (l'avversario di chi sta scegliendo). Il genitore preferisce il figlio il cui
    // giocatore di turno PERDE (provenResult == -1).
    if (provenResult != 0)
    {
        return (provenResult == -1) ? std::numeric_limits<double>::max()
                                    : -std::numeric_limits<double>::max();
    }

    // Valore del figlio in [-1,1] dal punto di vista del suo giocatore di turno.
    // L'utilita' per il genitore e' (1 - childValue) / 2, rimappata in [0,1] cosi'
    // che EXPLORATION_C e PROGRESSIVE_BIAS_WEIGHT mantengano la scala originale.
    double childValue = totalValue / static_cast<double>(visitCount);
    double exploitation = (1.0 - childValue) / 2.0;

    double exploration = explorationC * std::sqrt(logParentVisits / static_cast<double>(visitCount));

    double bias = PROGRESSIVE_BIAS_WEIGHT * heuristicScore / static_cast<double>(visitCount + 1);

    return exploitation + exploration + bias;
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
// Negamax: provenResult e' dal punto di vista del giocatore di turno in `node`; i figli
// portano il provenResult dal punto di vista dell'avversario. Quindi:
//   - vinco se ALMENO un figlio e' una sconfitta per l'avversario (provenResult == -1);
//   - perdo se TUTTI i figli sono vittorie per l'avversario (provenResult == +1).
void MCTS::TrySolve(MCTSNode* node)
{
    // Il lucchetto globale isExpanded è stato rimosso.
    // Ci fermiamo solo se non abbiamo ancora generato alcun figlio da valutare.
    if (node->children.empty())
        return;

    bool allChildrenProven = true;

    for (MCTSNode* child : node->children)
    {
        if (child->provenResult == -1)
        {
            // EARLY WIN: esiste una mossa in cui l'avversario perde.
            // Propaghiamo subito la vittoria senza aspettare l'espansione totale.
            node->provenResult = 1;
            if (node->parent != nullptr)
                TrySolve(node->parent);
            return;
        }
        else if (child->provenResult == 0)
        {
            allChildrenProven = false;
        }
    }

    // TRAPPOLA INEVITABILE (Sconfitta):
    // Per arrenderci, dobbiamo accertarci che TUTTI i figli siano stati provati (allChildrenProven)
    // come vittorie per l'avversario E che non ci siano più mosse da esplorare (node->isExpanded).
    if (allChildrenProven && node->isExpanded)
    {
        node->provenResult = -1;
        if (node->parent != nullptr)
            TrySolve(node->parent);
    }
}


// =============================================================================
// MCTS - Algoritmo principale
// =============================================================================

// Esegue una singola iterazione MCTS: selezione -> espansione -> valutazione -> backpropagation.
// Convenzione negamax: ogni valore e' in [-1,1] dal punto di vista del giocatore di turno nel
// nodo (+1 = chi muove vince, -1 = perde, 0 = patta); il segno si inverte ad ogni livello in backup.
void MCTS::RunIteration(MCTSNode* root, const Board& rootBoard, TranspositionTable& tt)
{
    ++g_iterationCount;

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
            double ucb = child->UCB1(EXPLORATION_C, logParent);
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

    // Nodo gia' provato: backpropaga direttamente senza espandere.
    // value e' dal punto di vista del giocatore di turno nel nodo; negamax inverte il segno ad ogni livello.
    if (node->provenResult != 0)
    {
        double value = (node->provenResult == 1) ? 1.0 : -1.0;
        MCTSNode* current = node;
        while (current != nullptr)
        {
            current->visitCount++;
            current->totalValue += value;
            value = -value;
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

                // provenResult dal punto di vista del giocatore di turno in questo nodo (board.currentColor)
                if (state == BoardState::WhiteWins)
                    node->provenResult = (board.currentColor == Color::White) ? 1 : -1;
                else if (state == BoardState::BlackWins)
                    node->provenResult = (board.currentColor == Color::White) ? -1 : 1;
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

            // heuristicScore dal punto di vista di chi muove ora (board.currentColor),
            // ovvero il genitore: si allinea con l'utilita' del genitore nel termine di bias di UCB1.
            double hScore = EvaluateMove(board, move, board.currentColor);

            MCTSNode* child = new MCTSNode(move, node, hScore);
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

                // provenResult dal punto di vista del giocatore di turno nel figlio (board.currentColor dopo ApplyMove)
                if (childState == BoardState::WhiteWins)
                    node->provenResult = (board.currentColor == Color::White) ? 1 : -1;
                else if (childState == BoardState::BlackWins)
                    node->provenResult = (board.currentColor == Color::White) ? -1 : 1;
            }
        }
    }

    // === VALUTAZIONE ===
    // Determina il valore della foglia: risultato provato, cache TT, o euristica.
    // value e' in [-1,1] dal punto di vista del giocatore di turno nella foglia (board.currentColor).
    double value;

    if (node->provenResult != 0)
    {
        value = (node->provenResult == 1) ? 1.0 : -1.0;
    }
    else
    {
        uint64_t hash = board.GetHash();
        TTEntry ttEntry;

        if (tt.Probe(hash, ttEntry))
        {
            // L'hash Zobrist include il turno, quindi il valore side-to-move memorizzato e' coerente.
            value = ttEntry.value;
        }
        else
        {
            if (g_valueNetwork == nullptr)
                throw std::runtime_error("MCTS: value network non impostata (chiamare SetValueNetwork prima di Search)");

            // La value network restituisce gia' [-1,1] dal punto di vista di chi muove.
            auto netStart = std::chrono::steady_clock::now();
            value = static_cast<double>(g_valueNetwork->EvaluateBoard(board));
            auto netEnd = std::chrono::steady_clock::now();
            g_networkTimeMs += std::chrono::duration<double, std::milli>(netEnd - netStart).count();
            ++g_networkCallCount;

            tt.Store(hash, value, static_cast<int16_t>(depth), node->isTerminal);
        }
    }

    // === BACKPROPAGATION ===
    // Risale fino alla radice (negamax): visite +1, valore += value, poi inverte il segno ad ogni livello.
    MCTSNode* current = node;
    while (current != nullptr)
    {
        current->visitCount++;
        current->totalValue += value;
        value = -value;
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
// NB: i figli della radice hanno provenResult dal punto di vista dell'avversario (chi muove dopo).
// Una vittoria per noi e' un figlio dove l'avversario perde (== -1); una sconfitta nostra e' (== +1).
Move MCTS::SelectBestMove(MCTSNode* root)
{
    MCTSNode* bestChild = nullptr;
    int bestVisits = -1;

    // 1) Cerca una vittoria provata (figlio in cui l'avversario perde)
    for (MCTSNode* child : root->children)
    {
        if (child->provenResult == -1)
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

    // 2) Mossa piu' visitata, escludendo le sconfitte provate (figli in cui l'avversario vince)
    bestVisits = -1;
    bestChild = nullptr;
    for (MCTSNode* child : root->children)
    {
        if (child->provenResult == 1)
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

    MCTSNode root(PassMove, nullptr);
    root.visitCount = 1;
    TranspositionTable& tt = GetPersistentTranspositionTable();

    auto startTime = std::chrono::steady_clock::now();
    auto endTime   = startTime + std::chrono::milliseconds(timeLimitMs);

    int iterations = 0;
    while (true)
    {
        RunIteration(&root, rootBoard, tt);
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

    MCTSNode root(PassMove, nullptr);
    root.visitCount = 1;
    TranspositionTable& tt = GetPersistentTranspositionTable();

    for (int i = 0; i < maxIterations; i++)
    {
        RunIteration(&root, rootBoard, tt);
        if (root.provenResult != 0)
            break;
    }

    return SelectBestMove(&root);
}

} // namespace HiveGotThis
