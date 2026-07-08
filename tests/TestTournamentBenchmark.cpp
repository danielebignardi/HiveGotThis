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

            cacheHitRatios.push_back(cacheHitRatio);
            netSharePcts.push_back(netSharePct);
            iterationsPerSec.push_back(itersPerSec);

            std::cout << "Ply " << (ply + 1) << " (" << (mover == Color::White ? "White" : "Black")
                      << ", nostra mossa): " << realMs << " ms reali, " << iters << " iterazioni ("
                      << itersPerSec << " it/s), " << netCalls << " chiamate rete, "
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

        size_t n    = cacheHitRatios.size();
        size_t half = n / 2;

        std::cout << "\nRiepilogo (solo nostre mosse):\n"
                  << "  cache hit ratio medio, prima meta': " << average(cacheHitRatios, 0, half > 0 ? half : n) << "%\n"
                  << "  cache hit ratio medio, seconda meta': " << average(cacheHitRatios, half, n) << "%\n"
                  << "  quota media di tempo in inferenza: " << average(netSharePcts, 0, n) << "%\n"
                  << "  iterazioni/secondo medie: " << average(iterationsPerSec, 0, n) << "\n";
    }

    return 0;
}
