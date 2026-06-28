// Benchmark della generazione mosse su posizioni di META' PARTITA (non da board vuota).
// Il perft normale parte da 0 pezzi: le posizioni esplorate sono quasi tutte piccole,
// e quindi sottostimano il costo del controllo "one hive" (che cresce ~O(k^2) col numero
// di pezzi k). Qui prima portiamo la board a ~20-25 pezzi giocando una partita pseudo-casuale
// con seed fisso, poi misuriamo il perft a partire da quella posizione densa.
//
// Uso:
//   TestPerftMidgame [perftDepth] [GameType] [targetPezzi] [pliesTotali] [seed]
//   default: depth 3, Base+MLP, target 22 pezzi, 60 plies di setup, seed 12345
//
// Stesso seed + stessa logica di GetValidMoves => stessa posizione di partenza, quindi
// il confronto fra build diverse (con/senza punti di articolazione) è apples-to-apples:
// i conteggi dei nodi DEVONO coincidere, cambia solo il tempo.

#include "Board.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace HiveGotThis;

static int PieceCount(const Board& b)
{
    int n = 0;
    for (int p = 0; p < NumPieceNames; ++p)
        if (b.piecesPositions[p] != NullIndex) ++n;
    return n;
}

static uint64_t Perft(Board& board, int depth)
{
    if (depth == 0) return 1;

    static std::vector<std::vector<Move>> pool;
    if (static_cast<int>(pool.size()) <= depth) pool.resize(depth + 1);
    std::vector<Move>& moves = pool[depth];
    moves.clear();

    board.GetValidMoves(moves);
    if (depth == 1) return moves.size();

    uint64_t nodes = 0;
    MoveUndo undo;
    for (const Move& m : moves)
    {
        board.ApplyMoveSavingUndo(m, undo);
        nodes += Perft(board, depth - 1);
        board.UndoMove(m, undo);
    }
    return nodes;
}

// Porta la board a una posizione densa di meta' partita giocando 'maxPlies' mosse.
// Finche' siamo sotto 'targetPieces' preferiamo i piazzamenti (Source == NullIndex)
// per riempire la board; poi giochiamo mosse a caso per dare realismo (pezzi mossi).
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
    Board::InitializeZobristTable();

    int      depth        = (argc >= 2) ? std::atoi(argv[1]) : 3;
    GameType gameType     = (argc >= 3) ? GetGameTypeValue(argv[2]) : GameType::BaseMLP;
    int      targetPieces = (argc >= 4) ? std::atoi(argv[3]) : 22;
    int      maxPlies     = (argc >= 5) ? std::atoi(argv[4]) : 60;
    uint32_t seed         = (argc >= 6) ? static_cast<uint32_t>(std::atoll(argv[5])) : 12345u;
    if (gameType == GameType::INVALID) gameType = GameType::BaseMLP;

    Board board(gameType);
    board.StartGame();
    PlayToMidgame(board, targetPieces, maxPlies, seed);

    std::cout << "Posizione di partenza: GameType=" << GetEnumString(gameType)
              << "  pezzi sulla board = " << PieceCount(board)
              << "  turno = " << board.GetCurrentTurn()
              << "  (seed " << seed << ")\n";

    // Misuriamo il perft due volte (i buffer statici si scaldano alla prima chiamata).
    for (int run = 1; run <= 2; ++run)
    {
        uint64_t hashBefore = board.GetHash();
        int      turnBefore = board.GetCurrentTurn();

        auto t0 = std::chrono::steady_clock::now();
        uint64_t nodes = Perft(board, depth);
        auto t1 = std::chrono::steady_clock::now();

        double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double nps = (ms > 0.0) ? (nodes / (ms / 1000.0)) : 0.0;

        std::cout << "run " << run
                  << "  depth " << depth
                  << "  nodes " << nodes
                  << "  time " << ms << " ms"
                  << "  (" << static_cast<uint64_t>(nps) << " nodes/s)\n";

        if (board.GetHash() != hashBefore || board.GetCurrentTurn() != turnBefore)
        {
            std::cerr << "ERRORE: UndoMove non ha ripristinato lo stato\n";
            return 1;
        }
    }

    return 0;
}
