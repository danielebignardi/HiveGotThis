#include "Board.h"
#include <iostream>

using namespace HiveGotThis;

void CheckTest(const std::string& testName, bool passed)
{
    std::cout << (passed ? "[OK] " : "[FAIL] ") << testName << std::endl;
}

// Restituisce true se una delle mosse ha quella destinazione
static bool HasDest(const std::vector<Move>& moves, Index dest)
{
    for (const Move& m : moves) if (m.Destination == dest) return true;
    return false;
}

// Restituisce true se `to` è adiacente a `from`
static bool IsAdjacentTo(Index from, Index to)
{
    for (int i = 0; i < 6; i++)
        if (from + NeighborOffsets[i] == to) return true;
    return false;
}



// - - - - - - - - - - MOVEMENT TESTS - - - - - - - - - -

// QueenBee scivola di 1 passo lungo l'hive
// Setup: wQ al centro, bQ a destra -> wQ può andare solo a DownRight e UpRight
void TestQueenBeeMoves()
{
    Board board(GameType::Base);
    board.boardState = BoardState::InProgress;
    board.MovePiece(PieceName::wQ, BoardCenter);
    board.MovePiece(PieceName::bQ, BoardCenter + NeighborOffsets[0]); // Right

    std::vector<Move> moves;
    board.GetQueenBeeMoves(PieceName::wQ, moves);

    CheckTest("QueenBee: 2 mosse con bQ a destra",           moves.size() == 2);
    CheckTest("QueenBee: può andare a DownRight",             HasDest(moves, BoardCenter + NeighborOffsets[1]));
    CheckTest("QueenBee: può andare a UpRight",               HasDest(moves, BoardCenter + NeighborOffsets[5]));
    CheckTest("QueenBee: non può andare a Right (occupato)",  !HasDest(moves, BoardCenter + NeighborOffsets[0]));
}

// Beetle può salire su pezzi adiacenti (climbing)
// Setup: wB1 al centro, bQ a destra -> Beetle può salire su bQ, QueenBee non può
void TestBeetleMoves()
{
    Board board(GameType::Base);
    board.boardState = BoardState::InProgress;
    board.MovePiece(PieceName::wB1, BoardCenter);
    board.MovePiece(PieceName::bQ, BoardCenter + NeighborOffsets[0]); // Right

    std::vector<Move> beetleMoves;
    board.GetBeetleMoves(PieceName::wB1, beetleMoves);
    CheckTest("Beetle: può salire su bQ (cella occupata)", HasDest(beetleMoves, BoardCenter + NeighborOffsets[0]));

    std::vector<Move> queenMoves;
    board.GetQueenBeeMoves(PieceName::wB1, queenMoves);
    CheckTest("Beetle vs QueenBee: la Queen non può salire su celle occupate",
              !HasDest(queenMoves, BoardCenter + NeighborOffsets[0]));
}

// Grasshopper salta in linea retta oltre i pezzi
// Setup: wG1 al centro, bQ a destra -> salta oltre bQ e atterra a Right+Right
void TestGrasshopperMoves()
{
    Board board(GameType::Base);
    board.boardState = BoardState::InProgress;
    board.MovePiece(PieceName::wG1, BoardCenter);
    board.MovePiece(PieceName::bQ, BoardCenter + NeighborOffsets[0]); // Right

    std::vector<Move> moves;
    board.GetGrasshopperMoves(PieceName::wG1, moves);

    Index landingSpot = BoardCenter + 2 * NeighborOffsets[0]; // Right+Right
    CheckTest("Grasshopper: 1 mossa (solo direzione destra)",        moves.size() == 1);
    CheckTest("Grasshopper: atterra oltre bQ a Right+Right",         HasDest(moves, landingSpot));
    CheckTest("Grasshopper: destinazione non adiacente all'origine", !IsAdjacentTo(BoardCenter, landingSpot));
}

// Spider si muove esattamente 3 passi (non 1 o 2)
// Setup: linea wS1 - bQ - wQ - bS1
void TestSpiderMoves()
{
    Board board(GameType::Base);
    board.boardState = BoardState::InProgress;
    Index pos0 = BoardCenter;
    Index pos1 = pos0 + NeighborOffsets[0]; // +1
    Index pos2 = pos1 + NeighborOffsets[0]; // +2
    Index pos3 = pos2 + NeighborOffsets[0]; // +3

    board.MovePiece(PieceName::wS1, pos0);
    board.MovePiece(PieceName::bQ,  pos1);
    board.MovePiece(PieceName::wQ,  pos2);
    board.MovePiece(PieceName::bS1, pos3);

    std::vector<Move> moves;
    board.GetSpiderMoves(PieceName::wS1, moves);

    CheckTest("Spider: ha mosse valide",                   moves.size() > 0);
    CheckTest("Spider: non atterra a 1 passo (pos1=bQ)",   !HasDest(moves, pos1));
    CheckTest("Spider: non atterra a 2 passi (pos2=wQ)",   !HasDest(moves, pos2));
}

// SoldierAnt raggiunge posizioni non adiacenti alla propria origine
// Setup: wA1 al centro, bQ a destra
void TestSoldierAntMoves()
{
    Board board(GameType::Base);
    board.boardState = BoardState::InProgress;
    board.MovePiece(PieceName::wA1, BoardCenter);
    board.MovePiece(PieceName::bQ, BoardCenter + NeighborOffsets[0]); // Right

    std::vector<Move> moves;
    board.GetSoldierAntMoves(PieceName::wA1, moves);

    // Right+Right non è adiacente a center ma è raggiungibile dall'Ant
    Index nonAdjacentDest = BoardCenter + 2 * NeighborOffsets[0];
    CheckTest("SoldierAnt: raggiunge Right+Right (non adiacente all'origine)", HasDest(moves, nonAdjacentDest));
    CheckTest("SoldierAnt: più mosse della QueenBee (>2)",                     moves.size() > 2);
}

// Ladybug raggiunge posizioni passando sopra l'hive (2 passi su + 1 giù)
// Setup: triangolo wL - bQ - wQ
void TestLadybugMoves()
{
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;
    board.MovePiece(PieceName::wL, BoardCenter);
    board.MovePiece(PieceName::bQ, BoardCenter + NeighborOffsets[0]); // Right
    board.MovePiece(PieceName::wQ, BoardCenter + NeighborOffsets[1]); // DownRight

    std::vector<Move> moves;
    board.GetLadybugMoves(PieceName::wL, moves);

    CheckTest("Ladybug: genera mosse valide", moves.size() > 0);

    bool hasNonAdjacentDest = false;
    for (const Move& m : moves)
        if (!IsAdjacentTo(BoardCenter, m.Destination)) { hasNonAdjacentDest = true; break; }
    CheckTest("Ladybug: può raggiungere posizioni non adiacenti all'origine", hasNonAdjacentDest);
}

// Pillbug sposta pezzi adiacenti (oltre al proprio movimento di 1 passo)
// Setup: wP al centro, bQ a destra, wQ a DownRight
void TestPillbugMoves()
{
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;
    board.MovePiece(PieceName::wP, BoardCenter);
    board.MovePiece(PieceName::bQ, BoardCenter + NeighborOffsets[0]); // Right
    board.MovePiece(PieceName::wQ, BoardCenter + NeighborOffsets[1]); // DownRight

    std::vector<Move> moves;
    board.GetPillbugMoves(PieceName::wP, moves);

    bool movesOtherPieces = false;
    for (const Move& m : moves)
        if (m.Piece != PieceName::wP) { movesOtherPieces = true; break; }
    CheckTest("Pillbug: genera mosse per pezzi adiacenti (non solo se stesso)", movesOtherPieces);
}

// Mosquito copia il tipo di movimento del vicino
// Setup: wM al centro, bQ a destra -> Mosquito si comporta come QueenBee
void TestMosquitoMoves()
{
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;
    board.MovePiece(PieceName::wM, BoardCenter);
    board.MovePiece(PieceName::bQ, BoardCenter + NeighborOffsets[0]); // Right

    std::vector<Move> queenLikeMoves;
    board.GetQueenBeeMoves(PieceName::wM, queenLikeMoves);

    std::vector<Move> mosquitoMoves;
    board.GetMosquitoMoves(PieceName::wM, mosquitoMoves);

    CheckTest("Mosquito: stessa quantità di mosse di QueenBee (adiacente a bQ)",
              mosquitoMoves.size() == queenLikeMoves.size());
}



int main()
{
    Board::InitializeZobristTable();

    TestQueenBeeMoves();
    TestBeetleMoves();
    TestGrasshopperMoves();
    TestSpiderMoves();
    TestSoldierAntMoves();
    TestLadybugMoves();
    TestPillbugMoves();
    TestMosquitoMoves();

    return 0;
}
