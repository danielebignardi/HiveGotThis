// Benchmark: quanto tempo impiega la value network rispetto al resto di una
// iterazione MCTS. Costruisce una posizione di meta' partita (stesso approccio
// di TestPerftMidgame), poi misura separatamente:
//   1) il tempo di una singola chiamata a EvaluateBoard (solo forward della rete)
//   2) il tempo medio di una iterazione MCTS completa durante una ricerca reale
//
// NB: la transposition table fa si' che non tutte le iterazioni chiamino
// davvero la rete (alcune posizioni sono cache hit) -> il tempo medio per
// iterazione tende a SOTTOSTIMARE il peso reale della rete quando e' chiamata.
//
// Uso:
//   TestInferenceBenchmark <model.pt> [targetPezzi] [pliesSetup] [seed] [chiamateRete] [iterazioniMCTS]
//   default: target 22 pezzi, 60 plies, seed 12345, 200 chiamate, 2000 iterazioni

#include "Board.h"
#include "MCTS.h"
#include "NeuralEvaluator.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace HiveGotThis;

static int PieceCount(const Board& b)
{
    int n = 0;
    for (int p = 0; p < NumPieceNames; ++p)
        if (b.piecesPositions[p] != NullIndex) ++n;
    return n;
}

static void PlayToMidgame(Board& board, int targetPieces, int maxPlies, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::vector<Move> moves;
    std::vector<const Move*> placements;

    for (int ply = 0; ply < maxPlies; ++ply)
    {
        if (GameIsOver(board.GetBoardState())) break;

        board.GetValidMoves(moves);
        if (moves.empty()) break;
        if (moves.size() == 1 && moves[0] == PassMove) { board.ApplyMove(PassMove); continue; }

        const Move* chosen = nullptr;
        if (PieceCount(board) < targetPieces)
        {
            placements.clear();
            for (const Move& m : moves)
                if (m.Source == NullIndex) placements.push_back(&m);
            if (!placements.empty())
                chosen = placements[rng() % placements.size()];
        }
        if (!chosen) chosen = &moves[rng() % moves.size()];

        board.ApplyMove(*chosen);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Uso: TestInferenceBenchmark <model.pt> [targetPezzi] [pliesSetup] [seed] [chiamateRete] [iterazioniMCTS]\n";
        return 1;
    }

    Board::InitializeZobristTable();

    std::string modelPath   = argv[1];
    int      targetPieces   = (argc >= 3) ? std::atoi(argv[2]) : 22;
    int      maxPlies       = (argc >= 4) ? std::atoi(argv[3]) : 60;
    uint32_t seed           = (argc >= 5) ? static_cast<uint32_t>(std::atoll(argv[4])) : 12345u;
    int      networkCalls   = (argc >= 6) ? std::atoi(argv[5]) : 200;
    int      mctsIterations = (argc >= 7) ? std::atoi(argv[6]) : 2000;

    TorchScriptValueEvaluator evaluator(modelPath);

    Board board(GameType::BaseMLP);
    board.StartGame();
    PlayToMidgame(board, targetPieces, maxPlies, seed);

    std::cout << "Posizione: pezzi sulla board = " << PieceCount(board)
              << "  turno = " << board.GetCurrentTurn()
              << "  (seed " << seed << ")\n\n";

    // --- 1) Chiamata isolata alla rete (encode + forward + estrazione) ---
    // Proviamo sia con il numero di thread di default di libtorch, sia forzato
    // a 1: per grafi piccoli come questi il thread pool interno puo' costare
    // piu' di quanto aiuti (overhead di sincronizzazione >> lavoro reale).
    volatile float sink = 0.0f; // impedisce che il compilatore ottimizzi via il ciclo

    evaluator.EvaluateBoard(board); // warm-up: la prima chiamata scalda buffer/cache interne

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < networkCalls; ++i)
        sink = evaluator.EvaluateBoard(board);
    auto t1 = std::chrono::steady_clock::now();

    double networkTotalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double networkAvgUs   = (networkTotalMs * 1000.0) / networkCalls;

    std::cout << "Rete neurale (thread default): " << networkCalls << " chiamate in "
              << networkTotalMs << " ms  ->  media " << networkAvgUs << " us/chiamata\n";

    TorchScriptValueEvaluator::SetTorchThreads(1);
    evaluator.EvaluateBoard(board); // warm-up dopo il cambio di thread pool

    auto t0b = std::chrono::steady_clock::now();
    for (int i = 0; i < networkCalls; ++i)
        sink = evaluator.EvaluateBoard(board);
    auto t1b = std::chrono::steady_clock::now();

    double networkTotalMs1T = std::chrono::duration<double, std::milli>(t1b - t0b).count();
    double networkAvgUs1T   = (networkTotalMs1T * 1000.0) / networkCalls;

    std::cout << "Rete neurale (1 thread):       " << networkCalls << " chiamate in "
              << networkTotalMs1T << " ms  ->  media " << networkAvgUs1T << " us/chiamata\n";

    // --- 2) Scomposizione reale del tempo durante una ricerca vera ---
    // ResetNetworkStats/GetNetworkTimeMs/GetNetworkCallCount misurano, dentro
    // RunIteration, solo le chiamate che superano la transposition table (cache
    // miss) -> questo ci dice esattamente quanto della ricerca e' inferenza
    // pura e quanto e' "il resto" (selezione, espansione, TT, backprop, ecc.).
    MCTS::SetValueNetwork(&evaluator);
    MCTS::ResetNetworkStats();

    auto t2 = std::chrono::steady_clock::now();
    MCTS::SearchIterations(board, mctsIterations);
    auto t3 = std::chrono::steady_clock::now();

    double mctsTotalMs   = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double mctsAvgUs     = (mctsTotalMs * 1000.0) / mctsIterations;
    double netTimeMs     = MCTS::GetNetworkTimeMs();
    long   netCalls      = MCTS::GetNetworkCallCount();
    double restTimeMs    = mctsTotalMs - netTimeMs;
    double netSharePct   = (mctsTotalMs > 0.0) ? (netTimeMs / mctsTotalMs) * 100.0 : 0.0;
    double cacheHitRatio = (mctsIterations > 0)
        ? 100.0 * static_cast<double>(mctsIterations - netCalls) / mctsIterations
        : 0.0;

    std::cout << "Ricerca reale: " << mctsIterations << " iterazioni in "
              << mctsTotalMs << " ms totali  ->  media " << mctsAvgUs << " us/iterazione\n"
              << "  di cui chiamate reali alla rete (cache miss): " << netCalls
              << " (" << (100.0 - cacheHitRatio) << "% delle iterazioni)\n"
              << "  tempo cumulativo in inferenza: " << netTimeMs << " ms ("
              << netSharePct << "% del totale)\n"
              << "  tempo cumulativo nel resto (selezione/espansione/TT/backprop): "
              << restTimeMs << " ms (" << (100.0 - netSharePct) << "% del totale)\n"
              << "  cache hit sulla transposition table: " << cacheHitRatio << "% delle iterazioni\n";

    (void)sink;
    return 0;
}
