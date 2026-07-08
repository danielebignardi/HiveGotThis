// Benchmark "da torneo": simula una partita reale in cui SOLO il nostro
// colore (White) e' giocato dal motore con una vera ricerca MCTS a tempo
// (esattamente come "bestmove time" in torneo). L'avversario (Black) sceglie
// la mossa migliore secondo l'euristica gia' esistente (EvaluateMove, la
// stessa usata da MCTS::OrderMoves) - nessuna ricerca, nessuna rete, nessuna
// transposition table condivisa: simula un motore avversario indipendente,
// che nella realta' non condivide la sua cache interna con noi.
//
// Questo evita l'errore del design precedente (entrambi i colori tramite
// MCTS::Search, che condividono la transposition table persistente): in un
// torneo vero facciamo ricerca solo sulle nostre mosse, e la cache si popola
// solo con cio' che valutiamo noi stessi.
//
// Per ogni nostra mossa misura, usando i contatori di profiling di MCTS
// (GetIterationCount/GetNetworkTimeMs/GetNetworkCallCount):
//   - quante iterazioni MCTS sono state completate nel tempo disponibile
//   - quante di quelle hanno richiesto una vera chiamata alla rete (cache miss)
//   - quanto del tempo e' stato speso in inferenza vs nel resto dell'algoritmo
//   - la latenza media per chiamata alla rete (il numero che ottimizzazioni
//     future come GPU o batching delle valutazioni cercherebbero di ridurre)
//   - di quanto si sfora il budget di tempo richiesto (rilevante per la
//     sicurezza in torneo, e per capire se una modifica futura introduce
//     latenze meno prevedibili)
//
// Il riepilogo finale riporta sia le medie per mossa sia i TOTALI di tutta la
// partita, cosi' l'intera run si riduce a una manciata di numeri da
// confrontare rapidamente prima/dopo una modifica (es. inferenza su GPU,
// parallelizzazione di MCTS, batching delle valutazioni).
//
// La transposition table e' persistente per tutta la durata del processo
// (vedi MCTS::GetPersistentTranspositionTable in src/MCTS.cpp): non viene mai
// svuotata tra una nostra mossa e la successiva, quindi ci aspettiamo che il
// cache-hit ratio salga progressivamente nel corso della partita.
//
// Uso:
//   TestTournamentBenchmark <model.pt> [timeLimitMs] [maxPlies]
//   default: 5000 ms/mossa (solo per le nostre mosse), 40 ply totali

#include "Board.h"
#include "Evaluation.h"
#include "MCTS.h"
#include "NeuralEvaluator.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace HiveGotThis;

static constexpr Color OUR_COLOR = Color::White;

// Sceglie la mossa migliore secondo l'euristica a singolo livello (nessuna
// ricerca in profondita'), per simulare un avversario indipendente che non
// tocca la nostra transposition table.
static Move ChooseOpponentMove(const Board& board, const std::vector<Move>& moves)
{
    Move best = moves[0];
    double bestScore = -std::numeric_limits<double>::max();
    for (const Move& m : moves)
    {
        double score = EvaluateMove(board, m, board.currentColor);
        if (score > bestScore)
        {
            bestScore = score;
            best = m;
        }
    }
    return best;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Uso: TestTournamentBenchmark <model.pt> [timeLimitMs] [maxPlies]\n";
        return 1;
    }

    Board::InitializeZobristTable();

    std::string modelPath  = argv[1];
    int timeLimitMs = (argc >= 3) ? std::atoi(argv[2]) : 5000;
    int maxPlies    = (argc >= 4) ? std::atoi(argv[3]) : 40;

    TorchScriptValueEvaluator evaluator(modelPath);
    TorchScriptValueEvaluator::SetTorchThreads(1);
    MCTS::SetValueNetwork(&evaluator);

    Board board(GameType::BaseMLP);
    board.StartGame();

    std::vector<double> cacheHitRatios;
    std::vector<double> netSharePcts;
    std::vector<double> iterationsPerSec;
    std::vector<double> netLatencyUs;
    std::vector<double> overshootMs;
    std::vector<double> overshootPcts;

    double totalRealMs = 0.0;
    double totalNetMs  = 0.0;
    long   totalIters    = 0;
    long   totalNetCalls = 0;

    int ourMovesPlayed = 0;
    for (int ply = 0; ply < maxPlies; ++ply)
    {
        if (GameIsOver(board.GetBoardState()))
            break;

        std::vector<Move> moves;
        board.GetValidMoves(moves);
        if (moves.empty())
            break;

        Color mover = board.currentColor;

        if (mover == OUR_COLOR)
        {
            MCTS::ResetNetworkStats();

            auto t0 = std::chrono::steady_clock::now();
            Move chosen = MCTS::Search(board, timeLimitMs);
            auto t1 = std::chrono::steady_clock::now();

            double realMs   = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double netMs    = MCTS::GetNetworkTimeMs();
            long   netCalls = MCTS::GetNetworkCallCount();
            long   iters    = MCTS::GetIterationCount();

            double cacheHitRatio = (iters > 0)
                ? 100.0 * static_cast<double>(iters - netCalls) / static_cast<double>(iters)
                : 0.0;
            double netSharePct = (realMs > 0.0) ? (netMs / realMs) * 100.0 : 0.0;
            double itersPerSec = (realMs > 0.0) ? (static_cast<double>(iters) / realMs) * 1000.0 : 0.0;
            double avgNetLatencyUs = (netCalls > 0) ? (netMs * 1000.0) / static_cast<double>(netCalls) : 0.0;
            double overshoot = realMs - static_cast<double>(timeLimitMs);
            double overshootPct = (timeLimitMs > 0) ? (overshoot / static_cast<double>(timeLimitMs)) * 100.0 : 0.0;

            cacheHitRatios.push_back(cacheHitRatio);
            netSharePcts.push_back(netSharePct);
            iterationsPerSec.push_back(itersPerSec);
            netLatencyUs.push_back(avgNetLatencyUs);
            overshootMs.push_back(overshoot);
            overshootPcts.push_back(overshootPct);

            totalRealMs   += realMs;
            totalNetMs    += netMs;
            totalIters    += iters;
            totalNetCalls += netCalls;

            std::cout << "Ply " << (ply + 1) << " (" << (mover == Color::White ? "White" : "Black")
                      << ", nostra mossa): " << realMs << " ms reali (budget " << timeLimitMs
                      << " ms, overshoot " << overshoot << " ms / " << overshootPct << "%), "
                      << iters << " iterazioni (" << itersPerSec << " it/s), " << netCalls
                      << " chiamate rete (" << avgNetLatencyUs << " us/chiamata), "
                      << "cache hit " << cacheHitRatio << "%, "
                      << "tempo in inferenza " << netSharePct << "%\n";

            board.ApplyMove(chosen);
            ++ourMovesPlayed;
        }
        else
        {
            Move chosen = ChooseOpponentMove(board, moves);
            board.ApplyMove(chosen);
        }
    }

    std::cout << "\nPartita terminata dopo " << ourMovesPlayed << " nostre mosse. Stato finale: ";
    switch (board.GetBoardState())
    {
        case BoardState::WhiteWins: std::cout << "White wins\n"; break;
        case BoardState::BlackWins: std::cout << "Black wins\n"; break;
        case BoardState::Draw:      std::cout << "Draw\n"; break;
        default:                    std::cout << "In progress (limite mosse raggiunto)\n"; break;
    }

    if (!cacheHitRatios.empty())
    {
        auto average = [](const std::vector<double>& v, size_t begin, size_t end)
        {
            double sum = 0.0;
            for (size_t i = begin; i < end; ++i) sum += v[i];
            return sum / static_cast<double>(end - begin);
        };
        auto maxOf = [](const std::vector<double>& v)
        {
            double m = v[0];
            for (double x : v) if (x > m) m = x;
            return m;
        };

        size_t n    = cacheHitRatios.size();
        size_t half = n / 2;

        double overallCacheHitRatio = (totalIters > 0)
            ? 100.0 * static_cast<double>(totalIters - totalNetCalls) / static_cast<double>(totalIters)
            : 0.0;
        double overallNetSharePct = (totalRealMs > 0.0) ? (totalNetMs / totalRealMs) * 100.0 : 0.0;
        double overallNetLatencyUs = (totalNetCalls > 0) ? (totalNetMs * 1000.0) / static_cast<double>(totalNetCalls) : 0.0;
        double overallItersPerSec = (totalRealMs > 0.0) ? (static_cast<double>(totalIters) / totalRealMs) * 1000.0 : 0.0;

        std::cout << "\nRiepilogo per mossa (medie):\n"
                  << "  cache hit ratio medio, prima meta': " << average(cacheHitRatios, 0, half > 0 ? half : n) << "%\n"
                  << "  cache hit ratio medio, seconda meta': " << average(cacheHitRatios, half, n) << "%\n"
                  << "  quota media di tempo in inferenza: " << average(netSharePcts, 0, n) << "%\n"
                  << "  iterazioni/secondo medie: " << average(iterationsPerSec, 0, n) << "\n"
                  << "  latenza media per chiamata rete: " << average(netLatencyUs, 0, n) << " us\n"
                  << "  overshoot medio sul budget: " << average(overshootMs, 0, n) << " ms ("
                  << average(overshootPcts, 0, n) << "%)\n"
                  << "  overshoot massimo sul budget: " << maxOf(overshootMs) << " ms ("
                  << maxOf(overshootPcts) << "%)\n";

        std::cout << "\nRiepilogo TOTALI partita (fingerprint da confrontare tra run diverse):\n"
                  << "  tempo totale di ricerca: " << totalRealMs << " ms\n"
                  << "  iterazioni totali: " << totalIters << " (" << overallItersPerSec << " it/s medie)\n"
                  << "  chiamate rete totali: " << totalNetCalls << "\n"
                  << "  cache hit ratio complessivo: " << overallCacheHitRatio << "%\n"
                  << "  quota di tempo in inferenza complessiva: " << overallNetSharePct << "%\n"
                  << "  latenza media per chiamata rete (complessiva): " << overallNetLatencyUs << " us\n";
    }

    return 0;
}
