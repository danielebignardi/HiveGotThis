#include "Board.h"
#include <iostream>
#include <iomanip>
#include <sstream>
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



// - - - - - - - - - - TEST SINGOLI INSETTI - - - - - - - - - -

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

// - - - - - - - - - - TEST CONFIGURAZIONE - - - - - - - - - -

// Genera tutte le rappresentazioni UHP valide per una mossa.
// La stessa mossa può essere descritta in più modi in UHP (un riferimento diverso per ogni vicino),
// quindi generiamo tutte le varianti per confrontarle con quelle di Mzinga.
std::vector<std::string> GetAllUHPRepresentations(const Board& board, const Move& move) {std::vector<std::string> result;

    // USA QUELLI STANDARD CHE ORA NON SERVE PIU AVERLI IN UNA DATA POSIZIONE (E COSI PUOI USARE IL PDF DELLA GRIGLIA)
    Index CustomNeighborOffsets[6] = {
        -BoardWidth,       // Direction::UpLeft
        -BoardWidth + 1,   // Direction::UpRight
        1,                 // Direction::Right
        BoardWidth,        // Direction::DownRight
        BoardWidth - 1,    // Direction::DownLeft
        -1                 // Direction::Left
    };

    char Indicators[] = "\\/-";

    for (int i = 0; i < 6; i++) {
        Index neighborPos = move.Destination + CustomNeighborOffsets[i];
        if (!IsValidIndex(neighborPos)) continue;
        if (!board.HasPieceAt(neighborPos)) continue;
        if (board.GetPieceAt(neighborPos) == move.Piece) continue; // skip: il vicino è il pezzo stesso (movimenti)

        std::string str = GetEnumString(move.Piece);
        str.append(" ");
        if (i == 1 || i == 2 || i == 3) {
            str.append(GetEnumString(board.GetPieceAt(neighborPos)));
            str.append(1, Indicators[i % 3]);
        } else {
            str.append(1, Indicators[i % 3]);
            str.append(GetEnumString(board.GetPieceAt(neighborPos)));
        }
        result.push_back(str);
    }

    if (result.empty()) {
        if (board.currentTurn == 0) // prima mossa: board vuota, nessun vicino atteso
            result.push_back(GetEnumString(move.Piece));
        else
            result.push_back("!NO!"); // nessun vicino inatteso: mossa invalida, non deve matchare
    }

    return result;
}

void CheckValidMoves(const Board& board, const std::string& message, const std::string& mzingaMoves) {

    std::cout << message << '\n';
    //std::cout << "mzinga moves: " << mzingaMoves << '\n';

    std::vector<Move> moves;
    board.GetValidMoves(moves);
    
    // conteggio delle mosse
    int ourCount = moves.size();
    int mzingaCount = std::count(mzingaMoves.begin(), mzingaMoves.end(), ';');
    std::cout << ourCount << " our valid moves" << '\n';
    std::cout << mzingaCount << " mzinga valid moves" << '\n';    

    // le nostre mosse devono essere tra quelle di mzinga
    for (Move move : moves)
    {
        auto reps = GetAllUHPRepresentations(board, move);
        bool found = false;
        for (const std::string& rep : reps)
            if (mzingaMoves.find(rep + ";") != std::string::npos) { found = true; break; }
        
        std::stringstream ss;
        ss << move;

        std::cout << std::left          
          << std::setw(25) << ss.str() 
          << std::setw(25) << reps[0] 
          << found << '\n';

        assert(found);
    }
    std::cout << '\n';
    moves.clear();

    assert(ourCount == mzingaCount);
}

void playMove(Board& board, PieceName piece, Index dest){
    board.MovePiece(piece, dest);
    board.currentColor = (Color)(((int)board.currentColor+1)%2);
    board.currentTurn++;
}

// Le mosse disponibili all'inizio della Board
void testBoard1()
{
    // starting from empty board
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;

    CheckValidMoves(board, "\nEMPTY BOARD", "wS1;wB1;wG1;wA1;wM;wL;wP;");

    playMove(board, wS1, 8256);
    CheckValidMoves(board, "\nEXECUTED wS1", "bS1 \\wS1;bS1 wS1/;bS1 wS1-;bS1 wS1\\;bS1 /wS1;bS1 -wS1;bB1 \\wS1;bB1 wS1/;bB1 wS1-;bB1 wS1\\;bB1 /wS1;bB1 -wS1;bG1 \\wS1;bG1 wS1/;bG1 wS1-;bG1 wS1\\;bG1 /wS1;bG1 -wS1;bA1 \\wS1;bA1 wS1/;bA1 wS1-;bA1 wS1\\;bA1 /wS1;bA1 -wS1;bM \\wS1;bM wS1/;bM wS1-;bM wS1\\;bM /wS1;bM -wS1;bL \\wS1;bL wS1/;bL wS1-;bL wS1\\;bL /wS1;bL -wS1;bP \\wS1;bP wS1/;bP wS1-;bP wS1\\;bP /wS1;bP -wS1;");

    playMove(board, bB1, 8129);
    CheckValidMoves(board, "\nEXECUTED bB1 /wS1", "wQ \\wS1;wQ wS1/;wQ wS1-;wS2 \\wS1;wS2 wS1/;wS2 wS1-;wB1 \\wS1;wB1 wS1/;wB1 wS1-;wG1 \\wS1;wG1 wS1/;wG1 wS1-;wA1 \\wS1;wA1 wS1/;wA1 wS1-;wM \\wS1;wM wS1/;wM wS1-;wL \\wS1;wL wS1/;wL wS1-;wP \\wS1;wP wS1/;wP wS1-;");

    playMove(board, wG1, 8383);
    CheckValidMoves(board, "\nEXECUTED wG1 wS1/", "bQ bB1\\;bQ /bB1;bQ -bB1;bS1 bB1\\;bS1 /bB1;bS1 -bB1;bB2 bB1\\;bB2 /bB1;bB2 -bB1;bG1 bB1\\;bG1 /bB1;bG1 -bB1;bA1 bB1\\;bA1 /bB1;bA1 -bB1;bM bB1\\;bM /bB1;bM -bB1;bL bB1\\;bL /bB1;bL -bB1;bP bB1\\;bP /bB1;bP -bB1;");

    playMove(board, bA1, 8002);
    CheckValidMoves(board, "EXECTUED bA1 /bB1", "wQ -wG1;wQ wG1\\;wQ \\wG1;wQ wG1/;wQ wG1-;wS2 -wG1;wS2 wG1\\;wS2 \\wG1;wS2 wG1/;wS2 wG1-;wB1 -wG1;wB1 wG1\\;wB1 \\wG1;wB1 wG1/;wB1 wG1-;wG2 -wG1;wG2 wG1\\;wG2 \\wG1;wG2 wG1/;wG2 wG1-;wA1 -wG1;wA1 wG1\\;wA1 \\wG1;wA1 wG1/;wA1 wG1-;wM -wG1;wM wG1\\;wM \\wG1;wM wG1/;wM wG1-;wL -wG1;wL wG1\\;wL \\wG1;wL wG1/;wL wG1-;wP -wG1;wP wG1\\;wP \\wG1;wP wG1/;wP wG1-;");

    playMove(board, wQ, 8382);
    CheckValidMoves(board, "EXECUTED wQ wG1-", "bQ bB1\\;bQ -bB1;bQ bA1\\;bQ /bA1;bQ -bA1;bS1 bB1\\;bS1 -bB1;bS1 bA1\\;bS1 /bA1;bS1 -bA1;bB2 bB1\\;bB2 -bB1;bB2 bA1\\;bB2 /bA1;bB2 -bA1;bG1 bB1\\;bG1 -bB1;bG1 bA1\\;bG1 /bA1;bG1 -bA1;bA2 bB1\\;bA2 -bB1;bA2 bA1\\;bA2 /bA1;bA2 -bA1;bM bB1\\;bM -bB1;bM bA1\\;bM /bA1;bM -bA1;bL bB1\\;bL -bB1;bL bA1\\;bL /bA1;bL -bA1;bP bB1\\;bP -bB1;bP bA1\\;bP /bA1;bP -bA1;");

    playMove(board, bM, 7875);
    CheckValidMoves(board, "EXECUTED bM /bA1", "wQ wG1/;wQ wG1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 -wG1;wS2 \\wG1;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 -wG1;wB1 \\wG1;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 -wG1;wG2 \\wG1;wA1 \\wQ;wA1 wQ/;wA1 wQ-;wA1 wQ\\;wA1 wG1\\;wA1 -wG1;wA1 \\wG1;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM -wG1;wM \\wG1;wL \\wQ;wL wQ/;wL wQ-;wL wQ\\;wL wG1\\;wL -wG1;wL \\wG1;wP \\wQ;wP wQ/;wP wQ-;wP wQ\\;wP wG1\\;wP -wG1;wP \\wG1;");

    playMove(board, wP, 8384);
    CheckValidMoves(board, "EXECUTED wP \\wS1", "bQ bB1\\;bQ -bB1;bQ bA1\\;bQ -bA1;bQ bM\\;bQ /bM;bQ -bM;");

    playMove(board, bQ, 8130);
    CheckValidMoves(board, "EXECUTED bQ \bA1", "wQ wG1/;wQ wG1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 \\wG1;wS2 \\wP;wS2 -wP;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 \\wG1;wB1 \\wP;wB1 -wP;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 \\wG1;wG2 \\wP;wG2 -wP;wA1 \\wQ;wA1 wQ/;wA1 wQ-;wA1 wQ\\;wA1 wG1\\;wA1 \\wG1;wA1 \\wP;wA1 -wP;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM \\wG1;wM \\wP;wM -wP;wL \\wQ;wL wQ/;wL wQ-;wL wQ\\;wL wG1\\;wL \\wG1;wL \\wP;wL -wP;wP \\wG1;wP -wS1;");

    playMove(board, wA1, 8385);
    CheckValidMoves(board, "EXECUTED wA1 -wP", "bQ wA1\\;bQ -bA1;bS1 /bQ;bS1 -bQ;bS1 bB1\\;bS1 bA1\\;bS1 bM\\;bS1 /bM;bS1 -bM;bB2 /bQ;bB2 -bQ;bB2 bB1\\;bB2 bA1\\;bB2 bM\\;bB2 /bM;bB2 -bM;bG1 /bQ;bG1 -bQ;bG1 bB1\\;bG1 bA1\\;bG1 bM\\;bG1 /bM;bG1 -bM;bA2 /bQ;bA2 -bQ;bA2 bB1\\;bA2 bA1\\;bA2 bM\\;bA2 /bM;bA2 -bM;bM /bQ;bM -bQ;bM /wA1;bM -wA1;bM \\wA1;bM \\wP;bM \\wG1;bM \\wQ;bM wQ/;bM wQ-;bM wQ\\;bM wG1\\;bM wS1\\;bM bB1\\;bM bA1\\;bL /bQ;bL -bQ;bL bB1\\;bL bA1\\;bL bM\\;bL /bM;bL -bM;bP /bQ;bP -bQ;bP bB1\\;bP bA1\\;bP bM\\;bP /bM;bP -bM;");
}

int main()
{
    Board::InitializeZobristTable();

    //TestQueenBeeMoves();
    //TestBeetleMoves();
    //TestGrasshopperMoves();
    //TestSpiderMoves();
    //TestSoldierAntMoves();
    //TestLadybugMoves();
    //TestPillbugMoves();
    //TestMosquitoMoves();

    testBoard1();



    return 0;
}
