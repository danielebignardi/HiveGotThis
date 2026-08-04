#ifndef MCTS_H
#define MCTS_H

#include "Board.h"
#include "Move.h"

#include <vector>
#include <cstdint>

namespace HiveGotThis
{

class TorchScriptValueEvaluator; // definita in NeuralEvaluator.h

struct PolicyTarget
{
    Move move;
    int visitCount;
    double pi;

    // Risultato provato dal solver per il nodo figlio raggiunto dalla mossa,
    // dal punto di vista di chi muove DOPO di essa: -1 = l'avversario perde
    // (la mossa vince per chi la gioca), +1 = l'avversario vince (mossa
    // perdente), 0 = non provato. Il self-play lo usa per non campionare
    // mai via una vittoria provata.
    int8_t provenResult = 0;
};

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

struct TTEntry
{
    uint64_t hash;      // Hash Zobrist della posizione
    double value;       // Valutazione [-1,1] dal punto di vista del giocatore di turno (side-to-move)
    int16_t depth;      // Profondita' nell'albero a cui e' stata calcolata
    bool isExact;       // true se il valore e' terminale (non stimato)
};

class TranspositionTable
{
public:
    TranspositionTable(size_t sizePower2 = 20); // 2^20 entries (~24 MB)
    ~TranspositionTable();

    // Cerca nella tabella. Restituisce true se la posizione e' presente.
    bool Probe(uint64_t hash, TTEntry& entry) const;

    // Salva una valutazione. Sovrascrive solo se depth >= quella gia' memorizzata.
    void Store(uint64_t hash, double value, int16_t depth, bool isExact);

    void Clear();

private:
    TTEntry* m_table;
    size_t m_mask; // (dimensione - 1), usata come maschera per indicizzare con AND
};


// =============================================================================
// MCTS NODE
// =============================================================================

// Peso del progressive bias (scala l'influenza dell'euristica sui nodi poco visitati)
static constexpr double PROGRESSIVE_BIAS_WEIGHT = 5.0;

struct MCTSNode
{
    Move move;                          // Mossa che ha generato questo nodo
    MCTSNode* parent;
    std::vector<MCTSNode*> children;

    int visitCount;
    double totalValue;                  // Somma [-1,1] delle valutazioni negamax dal punto di vista del giocatore di turno in questo nodo

    bool isExpanded;                    // true quando tutte le mosse figlie sono state generate
    bool isTerminal;

    // Risultato provato dal solver, dal punto di vista del giocatore di turno in questo nodo:
    //   0 = non provato, 1 = il giocatore di turno vince, -1 = il giocatore di turno perde
    int8_t provenResult;

    // Valutazione euristica della mossa [0,1], usata dal progressive bias
    double heuristicScore;

    // Prior della policy head [0,1], usato da PUCT quando disponibile.
    double policyPrior;

    // Mosse ancora da espandere (ordinate: peggiori in testa, migliori in coda)
    std::vector<Move> unexpandedMoves;

    // Prior allineate a unexpandedMoves, vuote quando la policy non e' disponibile.
    std::vector<double> unexpandedPriors;

    MCTSNode(Move move, MCTSNode* parent, double heuristicScore = 0.5);
    ~MCTSNode();

    // Calcola il punteggio di selezione dal punto di vista del genitore (negamax).
    double SelectionScore(double explorationC, double logParentVisits, double sqrtParentVisits) const;
};


// =============================================================================
// MCTS
// =============================================================================

class MCTS
{
public:
    static Move Search(const Board& rootBoard, int timeLimitMs);
    static Move SearchIterations(const Board& rootBoard, int maxIterations);

    // Esegue una ricerca a iterazioni fisse e restituisce, per ogni mossa legale
    // alla radice, la distribuzione policy target derivata dalle visite MCTS.
    static std::vector<PolicyTarget> SearchPolicyTargets(const Board& rootBoard, int maxIterations);

    // Deve essere impostata prima di chiamare Search/SearchIterations: l'MCTS
    // valuta le foglie sempre con la value network, non esiste piu' un fallback
    // euristico. Il chiamante mantiene la proprieta' dell'evaluator (nessun
    // ownership qui).
    static void SetValueNetwork(TorchScriptValueEvaluator* evaluator);

    // Statistiche di profiling sulle chiamate reali alla value network durante
    // l'ultima ricerca del thread corrente.
    static void ResetNetworkStats();
    static double GetNetworkTimeMs();
    static long GetNetworkCallCount();

    // Numero totale di iterazioni MCTS eseguite nell'ultima ricerca (cache hit
    // + miss). Confrontato con GetNetworkCallCount() da' il cache-hit ratio;
    // diviso per il tempo reale da' le iterazioni/secondo di una ricerca a
    // tempo. Azzerato da ResetNetworkStats().
    static long GetIterationCount();

    // Svuota esplicitamente la transposition table persistente. Non necessaria
    // nel normale funzionamento - utile per benchmark o test che vogliono una
    // misurazione "a freddo".
    static void ClearTranspositionTable();

private:
    // Iperparametri
    static constexpr double EXPLORATION_C = 1.0;        // Peso dell'esplorazione in UCB/PUCT
    static constexpr int TIME_CHECK_INTERVAL = 128;     // Controlla il tempo ogni N iterazioni

    // Restituisce la transposition table persistente del thread corrente,
    // allocandola al primo uso.
    static TranspositionTable& GetPersistentTranspositionTable();

    // Algoritmo
    static void RunIteration(MCTSNode* root, const Board& rootBoard, TranspositionTable& tt);
    static Move SelectBestMove(MCTSNode* root);
    static void OrderMoves(std::vector<Move>& moves, const Board& board, Color perspective);
    static bool IsInstantWin(const Board& board, const Move& move, Color perspective);

    // Propaga risultati provati (vittoria/sconfitta certa) verso la radice
    static void TrySolve(MCTSNode* node);
};

} // namespace HiveGotThis

#endif // MCTS_H
