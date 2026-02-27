#include "Board.h"
#include <iostream>

using namespace HiveGotThis;

void CheckTest(const std::string& testName, bool passed)
{
    std::cout << (passed ? "[OK] " : "[FAIL] ") << testName << std::endl;
}

// Test 1: al turno 0 White ha esattamente 14 mosse valide (BaseMLP)
void TestInitialMoves()
{
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;

    std::vector<Move> moves;
    board.GetValidMoves(moves);

    CheckTest("Turno 0: 14 mosse disponibili", moves.size() == 14);
}

// Test 2: al turno 0 tutte le mosse hanno come destinazione BoardCenter
void TestInitialMovesDestination()
{
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;

    std::vector<Move> moves;
    board.GetValidMoves(moves);

    bool allCenter = true;
    for (Move& m : moves)
    {
        if (m.Destination != BoardCenter) { allCenter = false; break; }
    }

    CheckTest("Turno 0: tutte le mosse puntano a BoardCenter", allCenter);
}

// Test 3: al turno 0 tutte le mosse hanno Source == NullIndex (piazzamento)
void TestInitialMovesSources()
{
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;

    std::vector<Move> moves;
    board.GetValidMoves(moves);

    bool allNull = true;
    for (Move& m : moves)
    {
        if (m.Source != NullIndex) { allNull = false; break; }
    }

    CheckTest("Turno 0: tutte le mosse hanno Source == NullIndex", allNull);
}

// Test 4: al turno 1 Black ha 14 mosse, tutte adiacenti a BoardCenter
void TestBlackFirstTurn()
{
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;

    board.MovePiece(PieceName::wQ, BoardCenter);
    board.currentTurn = 1;
    board.currentColor = Color::Black;

    std::vector<Move> moves;
    board.GetValidMoves(moves);

    // 14 pezzi × 6 celle adiacenti = 84 mosse
    CheckTest("Turno 1: Black ha 84 mosse (14 pezzi × 6 celle)", moves.size() == 84);

    bool allAdjacent = true;
    for (Move& m : moves)
    {
        bool adjacent = false;
        for (int i = 0; i < 6; i++)
        {
            if (BoardCenter + NeighborOffsets[i] == m.Destination)
            {
                adjacent = true;
                break;
            }
        }
        if (!adjacent) { allAdjacent = false; break; }
    }

    CheckTest("Turno 1: tutte le mosse Black sono adiacenti a BoardCenter", allAdjacent);
}

// Test 5: al turno 2 White non può piazzare adiacente a pezzi Black
void TestWhiteCannotPlaceAdjacentToBlack()
{
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;

    board.MovePiece(PieceName::wQ, BoardCenter);
    board.MovePiece(PieceName::bQ, BoardCenter + NeighborOffsets[0]);
    board.currentTurn = 2;
    board.currentColor = Color::White;

    std::vector<Move> moves;
    board.GetValidMoves(moves);

    Index bQPos = BoardCenter + NeighborOffsets[0];
    bool noneAdjacentToBlack = true;
    for (Move& m : moves)
    {
        for (int i = 0; i < 6; i++)
        {
            if (bQPos + NeighborOffsets[i] == m.Destination)
            {
                std::cout << "  Dest=" << m.Destination << " è adiacente a bQ" << std::endl;
                noneAdjacentToBlack = false;
                break;
            }
        }
    }

    CheckTest("Turno 2: White non può piazzare adiacente a Black", noneAdjacentToBlack);
}

// Test 6: regola della Regina — al turno 6 White deve piazzare wQ se non l'ha fatto
void TestQueenRule()
{
    Board board(GameType::Base);
    board.boardState = BoardState::InProgress;

    // Piazza 3 pezzi White e 3 pezzi Black senza usare le Regine
    board.MovePiece(PieceName::wS1, BoardCenter);
    board.MovePiece(PieceName::bS1, BoardCenter + NeighborOffsets[0]);
    board.MovePiece(PieceName::wS2, BoardCenter + NeighborOffsets[3]);
    board.MovePiece(PieceName::bS2, BoardCenter + NeighborOffsets[0] + NeighborOffsets[0]);
    board.MovePiece(PieceName::wG1, BoardCenter + NeighborOffsets[3] + NeighborOffsets[3]);
    board.MovePiece(PieceName::bG1, BoardCenter + NeighborOffsets[0] + NeighborOffsets[0] + NeighborOffsets[0]);
    board.currentTurn = 6;
    board.currentColor = Color::White;

    std::vector<Move> moves;
    board.GetValidMoves(moves);

    // Tutte le mosse devono essere piazzamenti di wQ
    bool allQueenPlacements = true;
    for (Move& m : moves)
    {
        if (m.Piece != PieceName::wQ) { allQueenPlacements = false; break; }
    }

    std::cout << "  Destinazioni White turno 2:" << std::endl;
    for (Move& m : moves)
    {
        // stampa solo le destinazioni uniche
        std::cout << "  Dest=" << m.Destination << std::endl;
    }

    std::cout << "  Destinazioni White turno 2:" << std::endl;
    for (Move& m : moves)
    {
        // stampa solo le destinazioni uniche
        std::cout << "  Dest=" << m.Destination << std::endl;
    }

    CheckTest("Turno 6: White deve piazzare wQ", allQueenPlacements);
}

int main()
{
    Board::InitializeZobristTable();

    TestInitialMoves();
    TestInitialMovesDestination();
    TestInitialMovesSources();
    TestBlackFirstTurn();
    TestWhiteCannotPlaceAdjacentToBlack();
    TestQueenRule();

    return 0;
}