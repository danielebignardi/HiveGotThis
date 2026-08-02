#include "Board.h"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

using namespace HiveGotThis;

static uint64_t Perft(Board& board, int depth)
{
    if (depth == 0)
        return 1;

    std::vector<Move> moves;
    board.GetValidMoves(moves);
    if (depth == 1)
        return moves.size();

    uint64_t nodes = 0;
    MoveUndo undo;
    for (const Move& move : moves)
    {
        board.ApplyMoveSavingUndo(move, undo);
        nodes += Perft(board, depth - 1);
        board.UndoMove(move, undo);
    }
    return nodes;
}

int main()
{
    Board::InitializeZobristTable();

    constexpr int WorkerCount = 8;
    constexpr int Repetitions = 20;
    constexpr uint64_t ExpectedNodes = 151686;

    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    workers.reserve(WorkerCount);

    for (int i = 0; i < WorkerCount; ++i)
    {
        workers.emplace_back([&failed]
        {
            for (int repeat = 0; repeat < Repetitions; ++repeat)
            {
                Board board(GameType::BaseMLP);
                board.StartGame();
                if (Perft(board, 4) != ExpectedNodes)
                    failed = true;
            }
        });
    }

    for (std::thread& worker : workers)
        worker.join();

    if (failed)
    {
        std::cerr << "Parallel board generation produced inconsistent results\n";
        return 1;
    }

    std::cout << "Parallel board generation: OK\n";
    return 0;
}
