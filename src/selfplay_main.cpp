// Generatore di dati self-play (versione minima, single-process).
//
// Gioca N partite del motore contro se stesso (entrambi i colori usano
// MCTS::SearchIterations con la stessa value network e la stessa
// transposition table persistente - in self-play condividerla e' corretto:
// stessa rete, stesse valutazioni) e scrive una riga JSON per ogni posizione
// incontrata, con le feature gia' encodate da BoardEncoder e il label z.
//
// Giocare piu' partite nello stesso processo evita di ricaricare il modello
// ad ogni partita e mantiene calda la transposition table: partite successive
// condividono molte posizioni (soprattutto in apertura) e i valori in cache
// restano validi, essendo funzione solo di (board, pesi della rete).
//
// Formato output (JSONL, una riga per posizione, append al file esistente):
//   {"game_id":0,"ply":12,"side_to_move":"White","z":1,"x":[...],
//    "edge_index":[...],"edge_attr":[...],"u":[...]}
//
// z e' l'esito finale della partita dal punto di vista di CHI MUOVE nella
// posizione (stessa convenzione side-to-move/negamax dell'output della rete):
//   +1 = chi muove ha poi vinto, -1 = ha perso, 0 = patta o partita troncata
//   al tetto di mosse (col modello non ancora addestrato succede spesso: e'
//   una situazione transitoria, il bootstrap verra' da partite umane).
//
// Le prime `openingPlies` mosse sono casuali uniformi tra le legali: senza
// questa fonte di rumore ogni partita giocata con lo stesso modello sarebbe
// identica alla precedente e il dataset conterrebbe una sola partita ripetuta.
//
// Uso:
//   SelfPlay <model.pt> <output.jsonl> [iterations] [maxPlies] [seed] [openingPlies] [game_id] [numGames]
//   default: 400 iterazioni/mossa, 200 ply massimi, seed casuale, 6 ply di
//   apertura casuale, game_id 0, 1 partita
//
// La partita i-esima del lotto usa seed+i e game_id+i: cosi' ogni singola
// partita resta riproducibile da sola rilanciando con numGames=1 e i valori
// corrispondenti.

#include "Board.h"
#include "BoardEncoder.h"
#include "MCTS.h"
#include "NeuralEvaluator.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace HiveGotThis;

// Una posizione registrata durante la partita, in attesa del label z finale.
struct PositionRecord
{
    std::string graphJson;   // output di GNNGraphToJson: {"x":[...],...}
    int ply;
    Color sideToMove;
};

// Gioca una partita di self-play e scrive le sue posizioni su `out`.
static void PlayOneGame(std::ofstream& out, long gameId, uint32_t seed,
                        int iterations, int maxPlies, int openingPlies)
{
    std::mt19937 rng(seed);

    Board board(GameType::BaseMLP);
    board.StartGame();

    std::vector<PositionRecord> records;
    std::vector<Move> moves;

    int ply = 0;
    for (; ply < maxPlies; ++ply)
    {
        if (GameIsOver(board.GetBoardState()))
            break;

        board.GetValidMoves(moves);
        if (moves.empty())
            break;

        // Registra la posizione che il giocatore di turno si trova davanti,
        // PRIMA di scegliere la mossa. La board vuota di ply 0 viene saltata:
        // zero nodi non danno nulla da imparare (e il max-pooling su un grafo
        // vuoto e' mal definito).
        GNNGraph graph = BoardEncoder::encode(board);
        if (!graph.x.empty())
            records.push_back({GNNGraphToJson(graph), ply, board.currentColor});

        Move chosen;
        if (ply < openingPlies)
            chosen = moves[rng() % moves.size()];
        else
            chosen = MCTS::SearchIterations(board, iterations);

        board.ApplyMove(chosen);
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

    for (const PositionRecord& rec : records)
    {
        // Inserisce i metadati in testa all'oggetto JSON prodotto da
        // GNNGraphToJson ({"x":...}), subito dopo la graffa di apertura.
        out << "{\"game_id\":" << gameId
            << ",\"ply\":" << rec.ply
            << ",\"side_to_move\":\"" << (rec.sideToMove == Color::White ? "White" : "Black") << "\""
            << ",\"z\":" << zFor(rec.sideToMove)
            << "," << rec.graphJson.substr(1)
            << "\n";
    }
    out.flush(); // ogni partita e' subito intera su disco

    const char* resultStr = "PlyCapReached";
    if (finalState == BoardState::WhiteWins)      resultStr = "WhiteWins";
    else if (finalState == BoardState::BlackWins) resultStr = "BlackWins";
    else if (finalState == BoardState::Draw)      resultStr = "Draw";

    std::cerr << "Partita " << gameId << " terminata: " << resultStr
              << " in " << ply << " ply (seed " << seed
              << ", " << iterations << " iterazioni/mossa, "
              << records.size() << " posizioni scritte)\n";
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Uso: SelfPlay <model.pt> <output.jsonl> [iterations] [maxPlies] [seed] [openingPlies] [game_id] [numGames]\n";
        return 1;
    }

    Board::InitializeZobristTable();

    std::string modelPath  = argv[1];
    std::string outputPath = argv[2];
    int      iterations   = (argc >= 4) ? std::atoi(argv[3]) : 400;
    int      maxPlies     = (argc >= 5) ? std::atoi(argv[4]) : 200;
    uint32_t seed         = (argc >= 6) ? static_cast<uint32_t>(std::atoll(argv[5]))
                                        : std::random_device{}();
    int      openingPlies = (argc >= 7) ? std::atoi(argv[6]) : 6;
    long     gameId       = (argc >= 8) ? std::atol(argv[7]) : 0;
    int      numGames     = (argc >= 9) ? std::atoi(argv[8]) : 1;

    std::unique_ptr<TorchScriptValueEvaluator> evaluator;
    try
    {
        evaluator = std::make_unique<TorchScriptValueEvaluator>(modelPath);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Errore caricando la value network (" << modelPath << "): " << e.what() << "\n";
        return 1;
    }
    TorchScriptValueEvaluator::SetTorchThreads(1);
    MCTS::SetValueNetwork(evaluator.get());

    std::ofstream out(outputPath, std::ios::app);
    if (!out)
    {
        std::cerr << "Errore: impossibile aprire " << outputPath << " in scrittura\n";
        return 1;
    }

    for (int i = 0; i < numGames; ++i)
        PlayOneGame(out, gameId + i, seed + static_cast<uint32_t>(i),
                    iterations, maxPlies, openingPlies);

    std::cerr << numGames << " partite completate, output in " << outputPath << "\n";
    return 0;
}
