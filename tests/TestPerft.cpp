// Perft (PERformance Test): conta le foglie dell'albero delle mosse fino a una certa profondità.
// Serve sia come benchmark per la generazione delle mosse, sia come oracolo di correttezza
// (i conteggi devono restare stabili fra refactor di GetValidMoves).
//
// Uso:
//   TestPerft                 -> perft da profondità 1 a 4 su GameType::Base
//   TestPerft <maxDepth>      -> perft fino a maxDepth
//   TestPerft <maxDepth> <gt> -> usa GameType <gt> (Base, BaseM, BaseL, BaseP, BaseML, BaseMP, BaseLP, BaseMLP)

#include "Board.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace HiveGotThis;

static uint64_t Perft(Board& board, int depth)
{
    if (depth == 0) return 1;

    std::vector<Move> moves;
    board.GetValidMoves(moves);

    // Ottimizzazione tipica del perft: a profondità 1 il numero di foglie
    // è esattamente il numero di mosse legali, niente ricorsione.
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

// Variante diagnostica: per ogni mossa alla radice stampa il sotto-conteggio.
// Indispensabile per fare diff contro un altro motore quando i totali divergono.
static uint64_t PerftDivide(Board& board, int depth)
{
    std::vector<Move> moves;
    board.GetValidMoves(moves);

    uint64_t total = 0;
    MoveUndo undo;
    for (const Move& m : moves)
    {
        board.ApplyMoveSavingUndo(m, undo);
        uint64_t sub = (depth <= 1) ? 1 : Perft(board, depth - 1);
        board.UndoMove(m, undo);
        std::cout << "  " << m << " : " << sub << "\n";
        total += sub;
    }
    return total;
}

int main(int argc, char** argv)
{
    Board::InitializeZobristTable();

    int      maxDepth = (argc >= 2) ? std::atoi(argv[1]) : 4;
    GameType gameType = (argc >= 3) ? GetGameTypeValue(argv[2]) : GameType::Base;
    if (gameType == GameType::INVALID) gameType = GameType::Base;

    Board board(gameType);
    board.StartGame();

    std::cout << "Perft su GameType=" << GetEnumString(gameType)
              << " fino a profondita' " << maxDepth << "\n";

    for (int d = 1; d <= maxDepth; ++d)
    {
        // Controllo di consistenza: hash e turno devono tornare identici dopo perft.
        uint64_t hashBefore = board.GetHash();
        int      turnBefore = board.GetCurrentTurn();

        auto t0 = std::chrono::steady_clock::now();
        uint64_t nodes = Perft(board, d);
        auto t1 = std::chrono::steady_clock::now();

        double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double nps = (ms > 0.0) ? (nodes / (ms / 1000.0)) : 0.0;

        std::cout << "depth " << d
                  << "  nodes " << nodes
                  << "  time " << ms << " ms"
                  << "  (" << static_cast<uint64_t>(nps) << " nodes/s)\n";

        if (board.GetHash() != hashBefore || board.GetCurrentTurn() != turnBefore)
        {
            std::cerr << "ERRORE: UndoMove non ha ripristinato lo stato (depth=" << d << ")\n";
            return 1;
        }
    }

    return 0;
}
