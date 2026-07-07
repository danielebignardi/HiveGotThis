#ifndef MCTS_H
#define MCTS_H

#include "Board.h"
#include "Move.h"

#include <vector>
#include <cstdint>

namespace HiveGotThis
{

class TorchScriptValueEvaluator; // definita in NeuralEvaluator.h

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

    // Mosse ancora da espandere (ordinate: peggiori in testa, migliori in coda)
    std::vector<Move> unexpandedMoves;

    MCTSNode(Move move, MCTSNode* parent, double heuristicScore = 0.5);
    ~MCTSNode();

    // Calcola il punteggio UCB1 di questo nodo dal punto di vista del genitore (negamax).
    double UCB1(double explorationC, double logParentVisits) const;
};


// =============================================================================
// MCTS
// =============================================================================

class MCTS
{
public:
    static Move Search(const Board& rootBoard, int timeLimitMs);
    static Move SearchIterations(const Board& rootBoard, int maxIterations);

    // Deve essere impostata prima di chiamare Search/SearchIterations: l'MCTS
    // valuta le foglie sempre con la value network, non esiste piu' un fallback
    // euristico. Il chiamante mantiene la proprieta' dell'evaluator (nessun
    // ownership qui).
    static void SetValueNetwork(TorchScriptValueEvaluator* evaluator);

    // Statistiche di profiling sulle chiamate reali alla value network durante
    // l'ultima ricerca (esclude le foglie servite dalla transposition table).
    // Usate per benchmark (vedi tests/TestInferenceBenchmark.cpp), non nel
    // normale funzionamento del motore.
    static void ResetNetworkStats();
    static double GetNetworkTimeMs();
    static long GetNetworkCallCount();

private:
    // Iperparametri
    static constexpr double EXPLORATION_C = 1.0;        // Peso dell'esplorazione in UCB1
    static constexpr int TIME_CHECK_INTERVAL = 128;     // Controlla il tempo ogni N iterazioni

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
