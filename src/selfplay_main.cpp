// Generatore di dati self-play parallelo.
//
// Gioca N partite del motore contro se stesso. Tutti i worker usano la stessa
// value network; ogni worker possiede una transposition table thread-local e
// la riusa tra i turni e le partite che gli vengono assegnate. Il coordinatore
// dell'evaluator aggrega sulla GPU richieste provenienti da partite diverse.
// scrive una riga JSON per ogni posizione incontrata: feature della board da
// BoardEncoder, label z, e i target di policy (le mosse legali con la
// distribuzione pi delle visite MCTS alla radice, piu' move_features e
// descrizione strutturale src/dst da MoveEncoder).
//
// Giocare piu' partite nello stesso processo evita di ricaricare il modello e
// permette a CPU e GPU di lavorare contemporaneamente.
//
// Formato output (JSONL, una riga per posizione, append al file esistente):
//   {"game_id":0,"ply":12,"side_to_move":"White","z":1,
//    "moves":[{"visits":37,"pi":0.0925,"features":[...],"src":3,"dst":[[0,2]]},...],
//    "x":[...],"edge_index":[...],"edge_attr":[...],"u":[...]}
//
// z e' l'esito finale della partita dal punto di vista di CHI MUOVE nella
// posizione (stessa convenzione side-to-move/negamax dell'output della rete):
//   +1 = chi muove ha poi vinto, -1 = ha perso, 0 = patta o partita troncata
//   al tetto di mosse.
//
// La varieta' tra le partite viene dalla temperatura: nei primi `tempPlies`
// ply la mossa giocata si campiona proporzionalmente alle visite MCTS
// (pi ∝ N, tau=1) invece di prendere sempre la piu' visitata - mosse sensate
// ma non sempre uguali, quindi etichette piu' pulite delle vecchie aperture
// uniformi e partite comunque tutte diverse. Dal ply tempPlies in poi la
// scelta e' deterministica (la piu' visitata). In entrambi i casi una
// vittoria provata dal solver si gioca subito e una sconfitta provata non
// viene mai campionata (se esiste un'alternativa).
//
// Uso:
//   SelfPlay <model.pt> <output.jsonl> [iterations] [maxPlies] [seed] [tempPlies] [game_id] [numGames] [device] [workers] [inferenceBatch] [batchWaitUs]
//   default: 400 iterazioni/mossa, 200 ply massimi, seed casuale, 10 ply di
//   temperatura, game_id 0, 1 partita, device auto, un worker per core.
//
// La partita i-esima usa seed+i e game_id+i. Per riproducibilita' bit-a-bit
// usare workers=1 e inferenceBatch=1: batch GPU diversi possono introdurre
// minime differenze floating point e quindi cambiare una scelta al limite.

#include "Board.h"
#include "BoardEncoder.h"
#include "MCTS.h"
#include "MoveEncoder.h"
#include "NeuralEvaluator.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace HiveGotThis;

// Una posizione registrata durante la partita, in attesa del label z finale.
struct PositionRecord
{
    std::string graphJson;   // output di GNNGraphToJson: {"x":[...],...}
    std::string movesJson;   // target policy: "moves":[{...},...]
    int ply;
    Color sideToMove;
    // Valore Q della radice MCTS in [-1, 1] side-to-move: la stima della
    // ricerca per questa posizione. Target "q" di distillazione (blend con z
    // via --q-blend del trainer). hasQ=false quando la ricerca non ha
    // prodotto statistiche (mossa singola obbligata).
    double q;
    bool hasQ;
};

struct GameResult
{
    std::string jsonl;
    std::string summary;
};

// Frammento JSON '"moves":[...]' con i target di policy della posizione:
// per ogni mossa legale visite, pi, move_features e descrizione src/dst.
static std::string PolicyMovesJson(const Board& board, const std::vector<PolicyTarget>& targets)
{
    std::ostringstream ss;
    ss << "\"moves\":[";
    for (size_t i = 0; i < targets.size(); ++i)
    {
        const PolicyTarget& t = targets[i];
        if (i > 0)
            ss << ",";
        ss << "{\"visits\":" << t.visitCount
           << ",\"pi\":" << t.pi
           << ",\"features\":" << MoveFeaturesToJson(EncodeMoveFeatures(board, t.move))
           << "," << MoveStructuralJson(board, t.move)
           << "}";
    }
    ss << "]";
    return ss.str();
}

// Indice della mossa piu' visitata, con la stessa etica del solver di
// SelectBestMove: una vittoria provata si gioca subito, una sconfitta
// provata si evita finche' esiste un'alternativa.
// provenResult e' dal punto di vista di chi muove DOPO la mossa:
// -1 = l'avversario perde (mossa vincente), +1 = l'avversario vince.
static size_t BestTargetIndex(const std::vector<PolicyTarget>& targets)
{
    for (size_t i = 0; i < targets.size(); ++i)
        if (targets[i].provenResult == -1)
            return i;

    size_t best = targets.size();
    for (size_t i = 0; i < targets.size(); ++i)
    {
        if (targets[i].provenResult == 1)
            continue;
        if (best == targets.size() || targets[i].visitCount > targets[best].visitCount)
            best = i;
    }
    if (best != targets.size())
        return best;

    // Tutte le mosse sono sconfitte provate: la piu' visitata.
    best = 0;
    for (size_t i = 1; i < targets.size(); ++i)
        if (targets[i].visitCount > targets[best].visitCount)
            best = i;
    return best;
}

// Campionamento con temperatura tau=1: mossa scelta proporzionalmente alle
// visite MCTS, escludendo le sconfitte provate. Vittorie provate e casi
// degeneri ricadono sulla scelta deterministica.
static size_t SampleTargetIndex(const std::vector<PolicyTarget>& targets, std::mt19937& rng)
{
    for (size_t i = 0; i < targets.size(); ++i)
        if (targets[i].provenResult == -1)
            return i;

    long total = 0;
    for (const PolicyTarget& t : targets)
        if (t.provenResult != 1)
            total += t.visitCount;

    if (total <= 0)
        return BestTargetIndex(targets);

    long pick = static_cast<long>(rng() % static_cast<unsigned long>(total));
    for (size_t i = 0; i < targets.size(); ++i)
    {
        if (targets[i].provenResult == 1)
            continue;
        pick -= targets[i].visitCount;
        if (pick < 0)
            return i;
    }
    return BestTargetIndex(targets); // non raggiungibile, per sicurezza
}

// Gioca una partita completa. Ogni chiamata possiede board, albero MCTS e RNG;
// il risultato viene restituito al writer principale, che mantiene l'ordine.
static GameResult PlayOneGame(long gameId, uint32_t seed, int iterations,
                              int maxPlies, int tempPlies)
{
    std::mt19937 rng(seed);

    Board board(GameType::BaseMLP);
    board.StartGame();

    std::vector<PositionRecord> records;

    int ply = 0;
    for (; ply < maxPlies; ++ply)
    {
        if (GameIsOver(board.GetBoardState()))
            break;

        // La ricerca passa sempre da SearchPolicyTargets: stesso costo di
        // SearchIterations, ma espone la distribuzione delle visite alla
        // radice - il target della policy head e la base del campionamento -
        // e il valore Q della radice, che diventa il target "q" di
        // distillazione (la stima della ricerca e' un'etichetta piu' densa
        // dell'esito finale z).
        double rootQ = std::numeric_limits<double>::quiet_NaN();
        std::vector<PolicyTarget> targets = MCTS::SearchPolicyTargets(board, iterations, &rootQ);
        if (targets.empty())
            break;
        bool hasRootQ = !std::isnan(rootQ);

        // Registra la posizione che il giocatore di turno si trova davanti,
        // PRIMA di scegliere la mossa. La board vuota di ply 0 viene saltata:
        // zero nodi non danno nulla da imparare (e il max-pooling su un grafo
        // vuoto e' mal definito).
        GNNGraph graph = BoardEncoder::encode(board);
        if (!graph.x.empty())
            records.push_back({GNNGraphToJson(graph), PolicyMovesJson(board, targets),
                               ply, board.currentColor, rootQ, hasRootQ});

        size_t chosen = (ply < tempPlies) ? SampleTargetIndex(targets, rng)
                                          : BestTargetIndex(targets);
        board.ApplyMove(targets[chosen].move);
    }

    // Esito finale -> label z side-to-move per ogni posizione registrata.
    BoardState finalState = board.GetBoardState();
    auto zFor = [finalState](Color sideToMove) -> int
    {
        if (finalState == BoardState::WhiteWins)
            return (sideToMove == Color::White) ? 1 : -1;
        if (finalState == BoardState::BlackWins)
            return (sideToMove == Color::Black) ? 1 : -1;
        return 0; // patta o partita troncata al tetto di mosse
    };

    std::ostringstream gameJson;
    for (const PositionRecord& rec : records)
    {
        // Inserisce metadati e target policy in testa all'oggetto JSON
        // prodotto da GNNGraphToJson ({"x":...}), dopo la graffa di apertura.
        gameJson << "{\"game_id\":" << gameId
                 << ",\"ply\":" << rec.ply
                 << ",\"side_to_move\":\""
                 << (rec.sideToMove == Color::White ? "White" : "Black") << "\""
                 << ",\"z\":" << zFor(rec.sideToMove);
        if (rec.hasQ)
            gameJson << ",\"q\":" << rec.q;
        gameJson << "," << rec.movesJson
                 << "," << rec.graphJson.substr(1)
                 << "\n";
    }

    const char* resultStr = "PlyCapReached";
    if (finalState == BoardState::WhiteWins)      resultStr = "WhiteWins";
    else if (finalState == BoardState::BlackWins) resultStr = "BlackWins";
    else if (finalState == BoardState::Draw)      resultStr = "Draw";

    std::ostringstream summary;
    summary << "Partita " << gameId << " terminata: " << resultStr
            << " in " << ply << " ply (seed " << seed
            << ", " << iterations << " iterazioni/mossa, "
            << records.size() << " posizioni scritte)";

    return {gameJson.str(), summary.str()};
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Uso: SelfPlay <model.pt> <output.jsonl> [iterations] [maxPlies] [seed] [tempPlies] [game_id] [numGames] [device] [workers] [inferenceBatch] [batchWaitUs]\n";
        return 1;
    }

    Board::InitializeZobristTable();

    std::string modelPath  = argv[1];
    std::string outputPath = argv[2];
    int      iterations   = (argc >= 4) ? std::atoi(argv[3]) : 400;
    int      maxPlies     = (argc >= 5) ? std::atoi(argv[4]) : 200;
    uint32_t seed         = (argc >= 6) ? static_cast<uint32_t>(std::atoll(argv[5]))
                                        : std::random_device{}();
    int      tempPlies    = (argc >= 7) ? std::atoi(argv[6]) : 10;
    long     gameId       = (argc >= 8) ? std::atol(argv[7]) : 0;
    int      numGames     = (argc >= 9) ? std::atoi(argv[8]) : 1;
    std::string device    = (argc >= 10) ? argv[9] : "auto";
    unsigned int hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    int      workers      = (argc >= 11) ? std::atoi(argv[10])
                                         : std::min(numGames, static_cast<int>(hardwareThreads));
    int      inferenceBatch = (argc >= 12) ? std::atoi(argv[11]) : workers;
    int      batchWaitUs  = (argc >= 13) ? std::atoi(argv[12]) : 2000;

    if (iterations < 1 || maxPlies < 1 || numGames < 1 || workers < 1
        || inferenceBatch < 1 || batchWaitUs < 1)
    {
        std::cerr << "Errore: iterations, maxPlies, numGames, workers, "
                     "inferenceBatch e batchWaitUs devono essere >= 1\n";
        return 1;
    }
    workers = std::min(workers, numGames);
    inferenceBatch = std::min(inferenceBatch, workers);

    std::unique_ptr<TorchScriptValueEvaluator> evaluator;
    try
    {
        evaluator = std::make_unique<TorchScriptValueEvaluator>(modelPath, device);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Errore caricando la value network (" << modelPath << "): " << e.what() << "\n";
        return 1;
    }
    TorchScriptValueEvaluator::SetTorchThreads(1);
    evaluator->EnableCrossGameBatching(inferenceBatch, batchWaitUs);
    MCTS::SetValueNetwork(evaluator.get());

    std::cerr << "Inferenza su " << evaluator->GetDevice()
              << ", " << workers << " partite concorrenti"
              << ", batch condiviso fino a " << inferenceBatch
              << " richieste (" << batchWaitUs << " us max)\n";

    std::ofstream out(outputPath, std::ios::app);
    if (!out)
    {
        std::cerr << "Errore: impossibile aprire " << outputPath << " in scrittura\n";
        return 1;
    }

    std::atomic<int> nextGame{0};
    std::mutex resultMutex;
    std::condition_variable resultReady;
    std::vector<std::unique_ptr<GameResult>> results(static_cast<size_t>(numGames));
    std::vector<std::exception_ptr> errors(static_cast<size_t>(numGames));
    std::vector<bool> ready(static_cast<size_t>(numGames), false);
    std::vector<std::thread> gameThreads;
    gameThreads.reserve(static_cast<size_t>(workers));

    for (int worker = 0; worker < workers; ++worker)
    {
        gameThreads.emplace_back([&]
        {
            while (true)
            {
                int index = nextGame.fetch_add(1);
                if (index >= numGames)
                    return;

                std::unique_ptr<GameResult> result;
                std::exception_ptr error;
                try
                {
                    result = std::make_unique<GameResult>(
                        PlayOneGame(gameId + index,
                                    seed + static_cast<uint32_t>(index),
                                    iterations, maxPlies, tempPlies));
                }
                catch (...)
                {
                    error = std::current_exception();
                }

                {
                    std::lock_guard<std::mutex> lock(resultMutex);
                    results[static_cast<size_t>(index)] = std::move(result);
                    errors[static_cast<size_t>(index)] = error;
                    ready[static_cast<size_t>(index)] = true;
                }
                resultReady.notify_all();
            }
        });
    }

    for (int i = 0; i < numGames; ++i)
    {
        std::unique_ptr<GameResult> result;
        std::exception_ptr error;
        {
            std::unique_lock<std::mutex> lock(resultMutex);
            resultReady.wait(lock, [&ready, i]
            {
                return ready[static_cast<size_t>(i)];
            });
            result = std::move(results[static_cast<size_t>(i)]);
            error = errors[static_cast<size_t>(i)];
        }

        if (result)
        {
            out << result->jsonl;
            out.flush();
            std::cerr << result->summary << "\n";
        }
        else if (error)
        {
            try
            {
                std::rethrow_exception(error);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Errore nella partita " << gameId + i
                          << ": " << e.what() << "\n";
            }
        }
    }

    for (std::thread& thread : gameThreads)
        thread.join();

    for (const std::exception_ptr& error : errors)
        if (error)
            return 1;

    std::cerr << numGames << " partite completate, output in " << outputPath << "\n";
    return 0;
}
