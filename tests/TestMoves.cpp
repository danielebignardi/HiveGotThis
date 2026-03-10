#include "Board.h"
#include <iostream>
#include <iomanip>
#include <sstream>

// questi include servono solo per interagire con MzingaEngine
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <unordered_map>

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

/*
{
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
}
*/

// - - - - - - - - - - TEST CONFIGURAZIONE - - - - - - - - - -

// Genera tutte le rappresentazioni UHP valide per una mossa.
// La stessa mossa può essere descritta in più modi in UHP (un riferimento diverso per ogni vicino),
// quindi generiamo tutte le varianti per confrontarle con quelle di Mzinga.
std::vector<std::string> GetAllUHPRepresentations(const Board& board, const Move& move) {
    std::vector<std::string> result;

    // NeighborOffsets: Right(0), DownRight(1), DownLeft(2), Left(3), UpLeft(4), UpRight(5)
    static const char IndicatorChar[]  = { '-', '\\', '/', '-', '\\', '/' };
    static const bool IndicatorAfter[] = { true, true, false, false, false, true };

    // Caso Beetle: destinazione occupata → l'unica notazione valida è "pezzo pezzoDest" (senza indicatore)
    if (board.HasPieceAt(move.Destination)) {
        PieceName destPiece = board.GetPieceAt(move.Destination);
        if (destPiece != move.Piece) {
            std::string str = GetEnumString(move.Piece);
            str.append(" ");
            str.append(GetEnumString(destPiece));
            result.push_back(str);
        }
        return result;
    }

    // Destinazione vuota: genera tutte le notazioni valide basate sui vicini occupati
    for (int i = 0; i < 6; i++) {
        Index neighborPos = move.Destination + NeighborOffsets[i];
        if (!IsValidIndex(neighborPos)) continue;
        if (!board.HasPieceAt(neighborPos)) continue;
        // Se il vicino è il pezzo stesso (Beetle che scende dallo stack), usa il pezzo sotto come riferimento
        PieceName topAtNeighbor = board.GetPieceAt(neighborPos);
        PieceName ref;
        if (topAtNeighbor == move.Piece) {
            ref = board.GetPieceUnder(move.Piece); // il pezzo che rimarrà visibile dopo che il Beetle si sposta
            if (ref == PieceName::INVALID) continue;
        } else {
            ref = topAtNeighbor;
        }

        std::string str = GetEnumString(move.Piece);
        str.append(" ");
        std::string refPiece = GetEnumString(ref);
        if (IndicatorAfter[i]) {
            str.append(refPiece);
            str.append(1, IndicatorChar[i]);
        } else {
            str.append(1, IndicatorChar[i]);
            str.append(refPiece);
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

void CheckValidMoves(const Board& board, const std::string& message, const std::string& mzingaMoves, bool verbose = true) {

    std::vector<Move> moves;
    board.GetValidMoves(moves);

    int ourCount = moves.size();
    int mzingaCount = std::count(mzingaMoves.begin(), mzingaMoves.end(), ';');

    if (verbose) {
        std::cout << message << '\n';
        std::cout << ourCount << " our / " << mzingaCount << " mzinga\n";
        std::cout << "Mzinga moves: " << mzingaMoves << '\n';
    }

    for (Move move : moves)
    {
        auto reps = GetAllUHPRepresentations(board, move);
        bool found = false;
        for (const std::string& rep : reps)
            if (mzingaMoves.find(rep + ";") != std::string::npos) { found = true; break; }

        if (verbose) {
            std::stringstream ss;
            ss << move;
            std::cout << std::left
              << std::setw(25) << ss.str()
              << std::setw(25) << reps[0]
              << found << '\n';
        } else if (!found) {
            // In modalità silenziosa stampa contesto solo in caso di failure
            std::stringstream ss;
            ss << move;
            std::cout << "\nFAIL: " << message << '\n';
            std::cout << ourCount << " our / " << mzingaCount << " mzinga\n";
            std::cout << ss.str() << " -> " << reps[0] << " not found\n" << std::flush;
        }

        assert(found);
    }

    if (verbose) std::cout << '\n';

    if (!verbose && ourCount != mzingaCount) {
        std::cout << "\nFAIL: " << message << '\n';
        std::cout << ourCount << " our / " << mzingaCount << " mzinga\n";
    }

    if (ourCount != mzingaCount) {
        std::istringstream mss(mzingaMoves);
        std::string tok;
        while (std::getline(mss, tok, ';')) {
            if (tok.empty()) continue;
            bool found = false;
            for (Move mv : moves) {
                auto reps = GetAllUHPRepresentations(board, mv);
                for (const std::string& rep : reps)
                    if (rep == tok) { found = true; break; }
                if (found) break;
            }
            if (!found) std::cout << "  MISSING: " << tok << '\n';
        }
        std::cout << std::flush;
    }
    assert(ourCount == mzingaCount);
}

void playMove(Board& board, PieceName piece, Index dest){
    board.MovePiece(piece, dest);
    board.ApplyTurnEffects(piece);
}

// Le mosse disponibili all'inizio della Board
void testGame48Moves()
{
    // starting from empty board
    Board board(GameType::BaseMLP);
    board.boardState = BoardState::InProgress;

    CheckValidMoves(board, "\nEMPTY BOARD", "wS1;wB1;wG1;wA1;wM;wL;wP;");

    // wS1:-1->8256             wS1                          1
    playMove(board, wS1, 8256);
    CheckValidMoves(board, "\nEXECUTED wS1", "bS1 \\wS1;bS1 wS1/;bS1 wS1-;bS1 wS1\\;bS1 /wS1;bS1 -wS1;bB1 \\wS1;bB1 wS1/;bB1 wS1-;bB1 wS1\\;bB1 /wS1;bB1 -wS1;bG1 \\wS1;bG1 wS1/;bG1 wS1-;bG1 wS1\\;bG1 /wS1;bG1 -wS1;bA1 \\wS1;bA1 wS1/;bA1 wS1-;bA1 wS1\\;bA1 /wS1;bA1 -wS1;bM \\wS1;bM wS1/;bM wS1-;bM wS1\\;bM /wS1;bM -wS1;bL \\wS1;bL wS1/;bL wS1-;bL wS1\\;bL /wS1;bL -wS1;bP \\wS1;bP wS1/;bP wS1-;bP wS1\\;bP /wS1;bP -wS1;");

    // bB1:-1->8129             bB1 /wS1                     1
    playMove(board, bB1, 8129);
    CheckValidMoves(board, "\nEXECUTED bB1 /wS1", "wQ \\wS1;wQ wS1/;wQ wS1-;wS2 \\wS1;wS2 wS1/;wS2 wS1-;wB1 \\wS1;wB1 wS1/;wB1 wS1-;wG1 \\wS1;wG1 wS1/;wG1 wS1-;wA1 \\wS1;wA1 wS1/;wA1 wS1-;wM \\wS1;wM wS1/;wM wS1-;wL \\wS1;wL wS1/;wL wS1-;wP \\wS1;wP wS1/;wP wS1-;");

    // wG1:-1->8383             wG1 wS1/                     1
    playMove(board, wG1, 8383);
    CheckValidMoves(board, "\nEXECUTED wG1 wS1/", "bQ bB1\\;bQ /bB1;bQ -bB1;bS1 bB1\\;bS1 /bB1;bS1 -bB1;bB2 bB1\\;bB2 /bB1;bB2 -bB1;bG1 bB1\\;bG1 /bB1;bG1 -bB1;bA1 bB1\\;bA1 /bB1;bA1 -bB1;bM bB1\\;bM /bB1;bM -bB1;bL bB1\\;bL /bB1;bL -bB1;bP bB1\\;bP /bB1;bP -bB1;");

    // bA1:-1->8002             bA1 /bB1                     1
    playMove(board, bA1, 8002);
    CheckValidMoves(board, "EXECTUED bA1 /bB1", "wQ -wG1;wQ wG1\\;wQ \\wG1;wQ wG1/;wQ wG1-;wS2 -wG1;wS2 wG1\\;wS2 \\wG1;wS2 wG1/;wS2 wG1-;wB1 -wG1;wB1 wG1\\;wB1 \\wG1;wB1 wG1/;wB1 wG1-;wG2 -wG1;wG2 wG1\\;wG2 \\wG1;wG2 wG1/;wG2 wG1-;wA1 -wG1;wA1 wG1\\;wA1 \\wG1;wA1 wG1/;wA1 wG1-;wM -wG1;wM wG1\\;wM \\wG1;wM wG1/;wM wG1-;wL -wG1;wL wG1\\;wL \\wG1;wL wG1/;wL wG1-;wP -wG1;wP wG1\\;wP \\wG1;wP wG1/;wP wG1-;");

    // wQ:-1->8382              wQ wG1-                      1
    playMove(board, wQ, 8382);
    CheckValidMoves(board, "EXECUTED wQ wG1-", "bQ bB1\\;bQ -bB1;bQ bA1\\;bQ /bA1;bQ -bA1;bS1 bB1\\;bS1 -bB1;bS1 bA1\\;bS1 /bA1;bS1 -bA1;bB2 bB1\\;bB2 -bB1;bB2 bA1\\;bB2 /bA1;bB2 -bA1;bG1 bB1\\;bG1 -bB1;bG1 bA1\\;bG1 /bA1;bG1 -bA1;bA2 bB1\\;bA2 -bB1;bA2 bA1\\;bA2 /bA1;bA2 -bA1;bM bB1\\;bM -bB1;bM bA1\\;bM /bA1;bM -bA1;bL bB1\\;bL -bB1;bL bA1\\;bL /bA1;bL -bA1;bP bB1\\;bP -bB1;bP bA1\\;bP /bA1;bP -bA1;");

    // bM:-1->7875              bM /bA1                      1
    playMove(board, bM, 7875);
    CheckValidMoves(board, "EXECUTED bM /bA1", "wQ wG1/;wQ wG1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 -wG1;wS2 \\wG1;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 -wG1;wB1 \\wG1;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 -wG1;wG2 \\wG1;wA1 \\wQ;wA1 wQ/;wA1 wQ-;wA1 wQ\\;wA1 wG1\\;wA1 -wG1;wA1 \\wG1;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM -wG1;wM \\wG1;wL \\wQ;wL wQ/;wL wQ-;wL wQ\\;wL wG1\\;wL -wG1;wL \\wG1;wP \\wQ;wP wQ/;wP wQ-;wP wQ\\;wP wG1\\;wP -wG1;wP \\wG1;");

    // wP:-1->8384              wP \wS1                      1
    playMove(board, wP, 8384);
    CheckValidMoves(board, "EXECUTED wP \\wS1", "bQ bB1\\;bQ -bB1;bQ bA1\\;bQ -bA1;bQ bM\\;bQ /bM;bQ -bM;");

    // bQ:-1->8130              bQ \bA1                      1
    playMove(board, bQ, 8130);
    CheckValidMoves(board, "EXECUTED bQ \\bA1", "wQ wG1/;wQ wG1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 \\wG1;wS2 \\wP;wS2 -wP;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 \\wG1;wB1 \\wP;wB1 -wP;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 \\wG1;wG2 \\wP;wG2 -wP;wA1 \\wQ;wA1 wQ/;wA1 wQ-;wA1 wQ\\;wA1 wG1\\;wA1 \\wG1;wA1 \\wP;wA1 -wP;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM \\wG1;wM \\wP;wM -wP;wL \\wQ;wL wQ/;wL wQ-;wL wQ\\;wL wG1\\;wL \\wG1;wL \\wP;wL -wP;wP \\wG1;wP -wS1;");

    // wA1:-1->8385             wA1 -wP                      1
    playMove(board, wA1, 8385);
    CheckValidMoves(board, "EXECUTED wA1 -wP", "bQ wA1\\;bQ -bA1;bS1 /bQ;bS1 -bQ;bS1 bB1\\;bS1 bA1\\;bS1 bM\\;bS1 /bM;bS1 -bM;bB2 /bQ;bB2 -bQ;bB2 bB1\\;bB2 bA1\\;bB2 bM\\;bB2 /bM;bB2 -bM;bG1 /bQ;bG1 -bQ;bG1 bB1\\;bG1 bA1\\;bG1 bM\\;bG1 /bM;bG1 -bM;bA2 /bQ;bA2 -bQ;bA2 bB1\\;bA2 bA1\\;bA2 bM\\;bA2 /bM;bA2 -bM;bM /bQ;bM -bQ;bM /wA1;bM -wA1;bM \\wA1;bM \\wP;bM \\wG1;bM \\wQ;bM wQ/;bM wQ-;bM wQ\\;bM wG1\\;bM wS1\\;bM bB1\\;bM bA1\\;bL /bQ;bL -bQ;bL bB1\\;bL bA1\\;bL bM\\;bL /bM;bL -bM;bP /bQ;bP -bQ;bP bB1\\;bP bA1\\;bP bM\\;bP /bM;bP -bM;");

    // bL:-1->7748              bL /bM                       1
    playMove(board, bL, 7748);
    CheckValidMoves(board, "EXECUTED bL /bM", "wQ wG1/;wQ wG1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 \\wG1;wS2 \\wA1;wS2 \\wP;wS2 -wA1;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 \\wG1;wB1 \\wA1;wB1 \\wP;wB1 -wA1;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 \\wG1;wG2 \\wA1;wG2 \\wP;wG2 -wA1;wA1 \\wP;wA1 \\wG1;wA1 \\wQ;wA1 wQ/;wA1 wQ-;wA1 wQ\\;wA1 wG1\\;wA1 wS1\\;wA1 bB1\\;wA1 bA1\\;wA1 bM\\;wA1 bL\\;wA1 /bL;wA1 -bL;wA1 -bM;wA1 /bQ;wA1 -bQ;wA1 \\bQ;wA1 /wP;wA2 \\wQ;wA2 wQ/;wA2 wQ-;wA2 wQ\\;wA2 wG1\\;wA2 \\wG1;wA2 \\wA1;wA2 \\wP;wA2 -wA1;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM \\wG1;wM \\wA1;wM \\wP;wM -wA1;wL \\wQ;wL wQ/;wL wQ-;wL wQ\\;wL wG1\\;wL \\wG1;wL \\wA1;wL \\wP;wL -wA1;");

    // wA1:8385->8258           wA1 \bQ                      1
    playMove(board, wA1, 8258);
    CheckValidMoves(board, "EXECTUDED wA1 \\bQ", "bS1 /bQ;bS1 bB1\\;bS1 bA1\\;bS1 bM\\;bS1 -bM;bS1 bL\\;bS1 /bL;bS1 -bL;bB2 /bQ;bB2 bB1\\;bB2 bA1\\;bB2 bM\\;bB2 -bM;bB2 bL\\;bB2 /bL;bB2 -bL;bG1 /bQ;bG1 bB1\\;bG1 bA1\\;bG1 bM\\;bG1 -bM;bG1 bL\\;bG1 /bL;bG1 -bL;bA2 /bQ;bA2 bB1\\;bA2 bA1\\;bA2 bM\\;bA2 -bM;bA2 bL\\;bA2 /bL;bA2 -bL;bL bB1\\;bL bA1\\;bL /bQ;bP /bQ;bP bB1\\;bP bA1\\;bP bM\\;bP -bM;bP bL\\;bP /bL;bP -bL;");

    // bA2:-1->7749             bA2 -bL                      1
    playMove(board, bA2, 7749);
    CheckValidMoves(board, "EXECUTED bA2 -bL", "wQ wG1/;wQ wG1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 \\wG1;wS2 \\wA1;wS2 -wP;wS2 -wA1;wS2 \\wP;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 \\wG1;wB1 \\wA1;wB1 -wP;wB1 -wA1;wB1 \\wP;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 \\wG1;wG2 \\wA1;wG2 -wP;wG2 -wA1;wG2 \\wP;wA1 /wP;wA1 -wP;wA1 \\wP;wA1 \\wG1;wA1 \\wQ;wA1 wQ/;wA1 wQ-;wA1 wQ\\;wA1 wG1\\;wA1 wS1\\;wA1 bB1\\;wA1 bA1\\;wA1 bM\\;wA1 bL\\;wA1 bA2\\;wA1 /bA2;wA1 -bA2;wA1 \\bA2;wA1 -bM;wA1 /bQ;wA1 -bQ;wA2 \\wQ;wA2 wQ/;wA2 wQ-;wA2 wQ\\;wA2 wG1\\;wA2 \\wG1;wA2 \\wA1;wA2 -wP;wA2 -wA1;wA2 \\wP;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM \\wG1;wM \\wA1;wM -wP;wM -wA1;wM \\wP;wL \\wQ;wL wQ/;wL wQ-;wL wQ\\;wL wG1\\;wL \\wG1;wL \\wA1;wL -wP;wL -wA1;wL \\wP;wP \\wG1;wP -wS1;");

    // wA2:-1->8259             wA2 -wA1                     1
    playMove(board, wA2, 8259);
    CheckValidMoves(board, "EXECUTED wA2 -wA1", "bS1 /bQ;bS1 bB1\\;bS1 bA1\\;bS1 \\bA2;bS1 -bM;bS1 bA2\\;bS1 /bA2;bS1 -bA2;bS1 bM\\;bS1 bL\\;bB2 /bQ;bB2 bB1\\;bB2 bA1\\;bB2 \\bA2;bB2 -bM;bB2 bA2\\;bB2 /bA2;bB2 -bA2;bB2 bM\\;bB2 bL\\;bG1 /bQ;bG1 bB1\\;bG1 bA1\\;bG1 \\bA2;bG1 -bM;bG1 bA2\\;bG1 /bA2;bG1 -bA2;bG1 bM\\;bG1 bL\\;bA2 -bM;bA2 /bQ;bA2 wA2\\;bA2 /wA2;bA2 -wA2;bA2 \\wA2;bA2 \\wA1;bA2 -wP;bA2 \\wP;bA2 \\wG1;bA2 \\wQ;bA2 wQ/;bA2 wQ-;bA2 wQ\\;bA2 wG1\\;bA2 wS1\\;bA2 bB1\\;bA2 bA1\\;bA2 bM\\;bA2 bL\\;bA2 /bL;bA3 /bQ;bA3 bB1\\;bA3 bA1\\;bA3 \\bA2;bA3 -bM;bA3 bA2\\;bA3 /bA2;bA3 -bA2;bA3 bM\\;bA3 bL\\;bP /bQ;bP bB1\\;bP bA1\\;bP \\bA2;bP -bM;bP bA2\\;bP /bA2;bP -bA2;bP bM\\;bP bL\\;");

    // bG1:-1->7620             bG1 bL\                  1
    playMove(board, bG1, 7620);
    CheckValidMoves(board, "EXECUTED bG1 bL\\", "wQ wG1/;wQ wG1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 \\wG1;wS2 \\wA1;wS2 -wP;wS2 \\wA2;wS2 /wA2;wS2 -wA2;wS2 \\wP;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 \\wG1;wB1 \\wA1;wB1 -wP;wB1 \\wA2;wB1 /wA2;wB1 -wA2;wB1 \\wP;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 \\wG1;wG2 \\wA1;wG2 -wP;wG2 \\wA2;wG2 /wA2;wG2 -wA2;wG2 \\wP;wA2 \\wA1;wA2 -wP;wA2 \\wP;wA2 \\wG1;wA2 \\wQ;wA2 wQ/;wA2 wQ-;wA2 wQ\\;wA2 wG1\\;wA2 wS1\\;wA2 bB1\\;wA2 bA1\\;wA2 bM\\;wA2 bG1-;wA2 bG1\\;wA2 /bG1;wA2 bA2\\;wA2 /bA2;wA2 -bA2;wA2 \\bA2;wA2 -bM;wA2 /bQ;wA2 /wA1;wA3 \\wQ;wA3 wQ/;wA3 wQ-;wA3 wQ\\;wA3 wG1\\;wA3 \\wG1;wA3 \\wA1;wA3 -wP;wA3 \\wA2;wA3 /wA2;wA3 -wA2;wA3 \\wP;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM \\wG1;wM \\wA1;wM -wP;wM \\wA2;wM /wA2;wM -wA2;wM \\wP;wL \\wQ;wL wQ/;wL wQ-;wL wQ\\;wL wG1\\;wL \\wG1;wL \\wA1;wL -wP;wL \\wA2;wL /wA2;wL -wA2;wL \\wP;wP \\wG1;wP -wS1;");

    // wM:-1->8385              wM -wP                   1
    playMove(board, wM, 8385);
    CheckValidMoves(board, "EXECUTED wM -wP", "bQ -bA1;bQ wA2\\;bS1 /bQ;bS1 bB1\\;bS1 bM\\;bS1 bG1-;bS1 bG1\\;bS1 /bG1;bS1 bA2\\;bS1 bA1\\;bS1 \\bA2;bS1 -bM;bS1 /bA2;bS1 -bA2;bB1 wS1;bB1 wS1\\;bB1 bA1-;bB1 bA1;bB1 bQ;bB2 /bQ;bB2 bB1\\;bB2 bM\\;bB2 bG1-;bB2 bG1\\;bB2 /bG1;bB2 bA2\\;bB2 bA1\\;bB2 \\bA2;bB2 -bM;bB2 /bA2;bB2 -bA2;bG1 -bM;bG2 /bQ;bG2 bB1\\;bG2 bM\\;bG2 bG1-;bG2 bG1\\;bG2 /bG1;bG2 bA2\\;bG2 bA1\\;bG2 \\bA2;bG2 -bM;bG2 /bA2;bG2 -bA2;bA2 -bM;bA2 /bQ;bA2 wA2\\;bA2 /wA2;bA2 -wA2;bA2 \\wA2;bA2 -wM;bA2 \\wM;bA2 \\wP;bA2 \\wG1;bA2 \\wQ;bA2 wQ/;bA2 wQ-;bA2 wQ\\;bA2 wG1\\;bA2 wS1\\;bA2 bB1\\;bA2 bA1\\;bA2 bM\\;bA2 bG1-;bA2 bG1\\;bA2 /bG1;bA2 /bL;bA3 /bQ;bA3 bB1\\;bA3 bM\\;bA3 bG1-;bA3 bG1\\;bA3 /bG1;bA3 bA2\\;bA3 bA1\\;bA3 \\bA2;bA3 -bM;bA3 /bA2;bA3 -bA2;bP /bQ;bP bB1\\;bP bM\\;bP bG1-;bP bG1\\;bP /bG1;bP bA2\\;bP bA1\\;bP \\bA2;bP -bM;bP /bA2;bP -bA2;");

    // bS1:-1->7876             bS1 -bM                  1
    playMove(board, bS1, 7876);
    CheckValidMoves(board, "EXECUTED bS1 -bM", "wQ wG1/;wQ wG1\\;wS1 wQ-;wS1 bA1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 \\wG1;wS2 -wM;wS2 \\wA2;wS2 /wA2;wS2 -wA2;wS2 \\wM;wS2 \\wP;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 \\wG1;wB1 -wM;wB1 \\wA2;wB1 /wA2;wB1 -wA2;wB1 \\wM;wB1 \\wP;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 \\wG1;wG2 -wM;wG2 \\wA2;wG2 /wA2;wG2 -wA2;wG2 \\wM;wG2 \\wP;wA2 -wM;wA2 \\wM;wA2 \\wP;wA2 \\wG1;wA2 \\wQ;wA2 wQ/;wA2 wQ-;wA2 wQ\\;wA2 wG1\\;wA2 wS1\\;wA2 bB1\\;wA2 bA1\\;wA2 bM\\;wA2 bG1-;wA2 bG1\\;wA2 /bG1;wA2 bA2\\;wA2 /bA2;wA2 -bA2;wA2 -bS1;wA2 \\bS1;wA2 /bQ;wA2 /wA1;wA3 \\wQ;wA3 wQ/;wA3 wQ-;wA3 wQ\\;wA3 wG1\\;wA3 \\wG1;wA3 -wM;wA3 \\wA2;wA3 /wA2;wA3 -wA2;wA3 \\wM;wA3 \\wP;wM \\wP;wM \\wA1;wP \\wM;wP wM/;wP wM\\;wP -wM;wM \\wG1;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM wS1\\;wM bB1\\;wM bA1\\;wM bM\\;wM bG1-;wM bG1\\;wM /bG1;wM bA2\\;wM /bA2;wM -bA2;wM -bS1;wM \\bS1;wM /bQ;wM wA2\\;wM /wA2;wM -wA2;wM \\wA2;wL \\wQ;wL wQ/;wL wQ-;wL wQ\\;wL wG1\\;wL \\wG1;wL -wM;wL \\wA2;wL /wA2;wL -wA2;wL \\wM;wL \\wP;wP \\wG1;wS1 \\wP;wS1 \\wG1;wS1 wM\\;wM /wP;");

    // wL:-1->8132              wL /wA2                  1
    playMove(board, wL, 8132);
    CheckValidMoves(board, "EXECUTED wL /wA2", "bQ -bA1;bQ wA2\\;bS1 wL\\;bS1 /bA2;bS2 /bQ;bS2 -bS1;bS2 bB1\\;bS2 bM\\;bS2 bG1-;bS2 bG1\\;bS2 /bG1;bS2 bA2\\;bS2 bA1\\;bS2 /bA2;bS2 -bA2;bB1 wS1;bB1 wS1\\;bB1 bA1-;bB1 bA1;bB1 bQ;bB2 /bQ;bB2 -bS1;bB2 bB1\\;bB2 bM\\;bB2 bG1-;bB2 bG1\\;bB2 /bG1;bB2 bA2\\;bB2 bA1\\;bB2 /bA2;bB2 -bA2;bG1 wL\\;bG2 /bQ;bG2 -bS1;bG2 bB1\\;bG2 bM\\;bG2 bG1-;bG2 bG1\\;bG2 /bG1;bG2 bA2\\;bG2 bA1\\;bG2 /bA2;bG2 -bA2;bA2 -bS1;bA2 wL\\;bA2 wA2\\;bA2 /bQ;bA2 /wL;bA2 -wL;bA2 -wA2;bA2 \\wA2;bA2 -wM;bA2 \\wM;bA2 \\wP;bA2 \\wG1;bA2 \\wQ;bA2 wQ/;bA2 wQ-;bA2 wQ\\;bA2 wG1\\;bA2 wS1\\;bA2 bB1\\;bA2 bA1\\;bA2 bM\\;bA2 bG1-;bA2 bG1\\;bA2 /bG1;bA2 /bL;bA3 /bQ;bA3 -bS1;bA3 bB1\\;bA3 bM\\;bA3 bG1-;bA3 bG1\\;bA3 /bG1;bA3 bA2\\;bA3 bA1\\;bA3 /bA2;bA3 -bA2;bP /bQ;bP -bS1;bP bB1\\;bP bM\\;bP bG1-;bP bG1\\;bP /bG1;bP bA2\\;bP bA1\\;bP /bA2;bP -bA2;");

    // bP:-1->7750              bP -bA2                  1
    playMove(board, bP, 7750);
    CheckValidMoves(board, "EXECUTED bP -bA2", "wQ wG1/;wQ wG1\\;wS1 wQ-;wS1 bA1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 \\wG1;wS2 -wM;wS2 \\wA2;wS2 -wA2;wS2 \\wM;wS2 \\wP;wS2 /wL;wS2 -wL;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 \\wG1;wB1 -wM;wB1 \\wA2;wB1 -wA2;wB1 \\wM;wB1 \\wP;wB1 /wL;wB1 -wL;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 \\wG1;wG2 -wM;wG2 \\wA2;wG2 -wA2;wG2 \\wM;wG2 \\wP;wG2 /wL;wG2 -wL;wA3 \\wQ;wA3 wQ/;wA3 wQ-;wA3 wQ\\;wA3 wG1\\;wA3 \\wG1;wA3 -wM;wA3 \\wA2;wA3 -wA2;wA3 \\wM;wA3 \\wP;wA3 /wL;wA3 -wL;wM \\wP;wM \\wA1;wP \\wM;wP wM/;wP wM\\;wP -wM;wM \\wG1;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM wS1\\;wM bB1\\;wM bA1\\;wM bM\\;wM bG1-;wM bG1\\;wM /bG1;wM bA2\\;wM bP\\;wM /bP;wM -bP;wM \\bP;wM -bS1;wM wL\\;wM wA2\\;wM /bQ;wM /wL;wM -wL;wM -wA2;wM \\wA2;wL -wM;wL wM\\;wL wA2\\;wP \\wG1;wS1 \\wP;wS1 \\wG1;wS1 wM\\;wM /wP;");

    // wM:8385->8260            wM -wA2                  1
    playMove(board, wM, 8260);
    CheckValidMoves(board, "EXECUTED wM -wA2", "bS1 wL\\;bS1 -bP;bS2 /bQ;bS2 -bS1;bS2 bB1\\;bS2 bM\\;bS2 bG1-;bS2 bG1\\;bS2 /bG1;bS2 bA2\\;bS2 bA1\\;bS2 bP\\;bS2 \\bP;bS2 /bP;bS2 -bP;bB2 /bQ;bB2 -bS1;bB2 bB1\\;bB2 bM\\;bB2 bG1-;bB2 bG1\\;bB2 /bG1;bB2 bA2\\;bB2 bA1\\;bB2 bP\\;bB2 \\bP;bB2 /bP;bB2 -bP;bG1 wL\\;bG2 /bQ;bG2 -bS1;bG2 bB1\\;bG2 bM\\;bG2 bG1-;bG2 bG1\\;bG2 /bG1;bG2 bA2\\;bG2 bA1\\;bG2 bP\\;bG2 \\bP;bG2 /bP;bG2 -bP;bA3 /bQ;bA3 -bS1;bA3 bB1\\;bA3 bM\\;bA3 bG1-;bA3 bG1\\;bA3 /bG1;bA3 bA2\\;bA3 bA1\\;bA3 bP\\;bA3 \\bP;bA3 /bP;bA3 -bP;bP -bS1;bP /bA2;");

    // bG2:-1->7878             bG2 \bP                  1
    playMove(board, bG2, 7878);
    CheckValidMoves(board, "EXECUTED bG2 \bP", "wQ wG1/;wQ wG1\\;wS2 \\wQ;wS2 wQ/;wS2 wQ-;wS2 wQ\\;wS2 wG1\\;wS2 \\wG1;wS2 \\wA1;wS2 -wP;wS2 \\wA2;wS2 \\wM;wS2 /wM;wS2 -wM;wS2 \\wP;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 \\wG1;wB1 \\wA1;wB1 -wP;wB1 \\wA2;wB1 \\wM;wB1 /wM;wB1 -wM;wB1 \\wP;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 \\wG1;wG2 \\wA1;wG2 -wP;wG2 \\wA2;wG2 \\wM;wG2 /wM;wG2 -wM;wG2 \\wP;wA3 \\wQ;wA3 wQ/;wA3 wQ-;wA3 wQ\\;wA3 wG1\\;wA3 \\wG1;wA3 \\wA1;wA3 -wP;wA3 \\wA2;wA3 \\wM;wA3 /wM;wA3 -wM;wA3 \\wP;wM \\wA2;wM \\wA1;wM -wP;wM \\wP;wM \\wG1;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM wS1\\;wM bB1\\;wM bA1\\;wM bM\\;wM bG1-;wM bG1\\;wM /bG1;wM bA2\\;wM bP\\;wM /bP;wM /bG2;wM -bG2;wM \\bG2;wM /wL;wM -wL;wM wL\\;wM wA2\\;wM /bQ;wM -bS1;wM /wP;wL \\wA2;wL \\wA1;wL wA2\\;wL -wP;wL /wP;wL \\wM;wL /wM;wL -wM;wP \\wG1;wP -wS1;");

    // wS2:-1->8386             wS2 \wA1                 1
    playMove(board, wS2, 8386);
    CheckValidMoves(board, "EXECUTED wS2 \\wA1", "bS1 wL\\;bS1 /wM;bS1 \\bG2;bS2 /bQ;bS2 -bS1;bS2 bB1\\;bS2 bM\\;bS2 bG1-;bS2 bG1\\;bS2 /bG1;bS2 bA2\\;bS2 \\bG2;bS2 /bG2;bS2 -bG2;bS2 bA1\\;bS2 bP\\;bS2 /bP;bB2 /bQ;bB2 -bS1;bB2 bB1\\;bB2 bM\\;bB2 bG1-;bB2 bG1\\;bB2 /bG1;bB2 bA2\\;bB2 \\bG2;bB2 /bG2;bB2 -bG2;bB2 bA1\\;bB2 bP\\;bB2 /bP;bG1 wL\\;bG2 bP\\;bG3 /bQ;bG3 -bS1;bG3 bB1\\;bG3 bM\\;bG3 bG1-;bG3 bG1\\;bG3 /bG1;bG3 bA2\\;bG3 \\bG2;bG3 /bG2;bG3 -bG2;bG3 bA1\\;bG3 bP\\;bG3 /bP;bA3 /bQ;bA3 -bS1;bA3 bB1\\;bA3 bM\\;bA3 bG1-;bA3 bG1\\;bA3 /bG1;bA3 bA2\\;bA3 \\bG2;bA3 /bG2;bA3 -bG2;bA3 bA1\\;bA3 bP\\;bA3 /bP;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bB2:-1->7621             bB2 bA2\                 1
    playMove(board, bB2, 7621);
    CheckValidMoves(board, "EXECUTED bB2 bA2\\", "wQ wG1/;wQ wG1\\;wS2 \\wG1;wS2 -wM;wB1 \\wQ;wB1 wQ/;wB1 wQ-;wB1 wQ\\;wB1 wG1\\;wB1 \\wS2;wB1 wS2/;wB1 -wP;wB1 -wS2;wB1 \\wG1;wB1 \\wM;wB1 /wM;wB1 -wM;wB1 \\wP;wG2 \\wQ;wG2 wQ/;wG2 wQ-;wG2 wQ\\;wG2 wG1\\;wG2 \\wS2;wG2 wS2/;wG2 -wP;wG2 -wS2;wG2 \\wG1;wG2 \\wM;wG2 /wM;wG2 -wM;wG2 \\wP;wA3 \\wQ;wA3 wQ/;wA3 wQ-;wA3 wQ\\;wA3 wG1\\;wA3 \\wS2;wA3 wS2/;wA3 -wP;wA3 -wS2;wA3 \\wG1;wA3 \\wM;wA3 /wM;wA3 -wM;wA3 \\wP;wM -wS2;wM \\wS2;wM wS2/;wM -wP;wM \\wP;wM \\wG1;wM \\wQ;wM wQ/;wM wQ-;wM wQ\\;wM wG1\\;wM wS1\\;wM bB1\\;wM bA1\\;wM bM\\;wM bG1-;wM bG1\\;wM bB2\\;wM /bB2;wM bP\\;wM /bP;wM /bG2;wM -bG2;wM \\bG2;wM /wL;wM -wL;wM wL\\;wM wA2\\;wM /bQ;wM -bS1;wM /wP;wL -wS2;wL wA2\\;wL \\wS2;wL wS2/;wL -wP;wL /wP;wL \\wM;wL /wM;wL -wM;wP \\wG1;wP -wS1;");

    // wG2:-1->8509             wG2 wQ/                  1
    playMove(board, wG2, 8509);
    CheckValidMoves(board, "EXECUTED wG2 wQ/", "bS1 wL\\;bS1 /wM;bS1 \\bG2;bS2 /bQ;bS2 -bS1;bS2 bB1\\;bS2 bB2\\;bS2 /bB2;bS2 bP\\;bS2 bM\\;bS2 bG1-;bS2 bG1\\;bS2 \\bG2;bS2 /bG2;bS2 -bG2;bS2 bA1\\;bS2 /bP;bB2 bA2;bB2 bL;bB2 bG1;bB2 /bG1;bB2 bP\\;bG1 wL\\;bG1 bP\\;bG2 bP\\;bG3 /bQ;bG3 -bS1;bG3 bB1\\;bG3 bB2\\;bG3 /bB2;bG3 bP\\;bG3 bM\\;bG3 bG1-;bG3 bG1\\;bG3 \\bG2;bG3 /bG2;bG3 -bG2;bG3 bA1\\;bG3 /bP;bA3 /bQ;bA3 -bS1;bA3 bB1\\;bA3 bB2\\;bA3 /bB2;bA3 bP\\;bA3 bM\\;bA3 bG1-;bA3 bG1\\;bA3 \\bG2;bA3 /bG2;bA3 -bG2;bA3 bA1\\;bA3 /bP;bL /bQ;bL bA1\\;bL bM\\;bL -bS1;bL bP\\;bL bB1\\;bL wL\\;bL bB2\\;bL /bB2;bL bG1-;bL bG1\\;bL /bP;bL /bG2;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bS2:-1->8003             bS2 /bQ                  1
    playMove(board, bS2, 8003);
    CheckValidMoves(board, "EXECUTED bS2 /bQ", "wS2 \\wG1;wS2 -wM;wB1 -wG2;wB1 wG2\\;wB1 wQ\\;wB1 wG1\\;wB1 \\wS2;wB1 wS2/;wB1 -wP;wB1 -wS2;wB1 \\wG1;wB1 \\wG2;wB1 wG2/;wB1 wG2-;wB1 \\wM;wB1 /wM;wB1 -wM;wB1 \\wP;wG2 wG1\\;wG3 -wG2;wG3 wG2\\;wG3 wQ\\;wG3 wG1\\;wG3 \\wS2;wG3 wS2/;wG3 -wP;wG3 -wS2;wG3 \\wG1;wG3 \\wG2;wG3 wG2/;wG3 wG2-;wG3 \\wM;wG3 /wM;wG3 -wM;wG3 \\wP;wA3 -wG2;wA3 wG2\\;wA3 wQ\\;wA3 wG1\\;wA3 \\wS2;wA3 wS2/;wA3 -wP;wA3 -wS2;wA3 \\wG1;wA3 \\wG2;wA3 wG2/;wA3 wG2-;wA3 \\wM;wA3 /wM;wA3 -wM;wA3 \\wP;wM -wS2;wM \\wS2;wM wS2/;wM -wP;wM \\wP;wM \\wG1;wM -wG2;wM \\wG2;wM wG2/;wM wG2-;wM wG2\\;wM wQ\\;wM wG1\\;wM wS1\\;wM bB1\\;wM bA1\\;wM bM\\;wM bG1-;wM bG1\\;wM bB2\\;wM /bB2;wM bP\\;wM /bP;wM /bG2;wM -bG2;wM \\bG2;wM /wL;wM -wL;wM wL\\;wM -bS1;wM /wP;wM wA2\\;wL -wS2;wL wA2\\;wL \\wS2;wL wS2/;wL -wP;wL /wP;wL \\wM;wL /wM;wL -wM;wP \\wG1;wP -wS1;");

    // wB1:-1->8254             wB1 wQ\                  1
    playMove(board, wB1, 8254);
    CheckValidMoves(board, "EXECUTED wB1 wQ\\", "bS1 /wM;bS1 \\bA2;bS1 \\bG2;bS1 wL\\;bS2 -bS1;bS2 /wL;bS2 /wM;bS2 \\bG2;bB2 bA2;bB2 bL;bB2 bG1;bB2 /bG1;bB2 bP\\;bG1 wL\\;bG1 bP\\;bG2 bP\\;bG3 -bS1;bG3 bB1\\;bG3 bB2\\;bG3 /bB2;bG3 bP\\;bG3 bM\\;bG3 bG1-;bG3 bG1\\;bG3 \\bG2;bG3 /bG2;bG3 -bG2;bG3 bA1\\;bG3 /bP;bA1 bB1\\;bA1 wS1\\;bA1 wG1\\;bA1 /wB1;bA1 wB1\\;bA1 wB1-;bA1 wG2\\;bA1 wG2-;bA1 wG2/;bA1 \\wG2;bA1 -wG2;bA1 \\wG1;bA1 \\wP;bA1 -wP;bA1 wS2/;bA1 \\wS2;bA1 -wS2;bA1 \\wM;bA1 -wM;bA1 /wM;bA1 /wL;bA1 wL\\;bA1 -bS1;bA1 \\bG2;bA1 -bG2;bA1 /bG2;bA1 /bP;bA1 bP\\;bA1 /bB2;bA1 bB2\\;bA1 bG1\\;bA1 bG1-;bA1 bM\\;bA1 bM-;bA3 -bS1;bA3 bB1\\;bA3 bB2\\;bA3 /bB2;bA3 bP\\;bA3 bM\\;bA3 bG1-;bA3 bG1\\;bA3 \\bG2;bA3 /bG2;bA3 -bG2;bA3 bA1\\;bA3 /bP;bM wS1\\;bM bG1\\;bM bA1\\;bM bB1\\;bM wG1\\;bM /wB1;bM wB1\\;bM wB1-;bM wG2\\;bM wG2-;bM wG2/;bM \\wG2;bM -wG2;bM \\wG1;bM \\wP;bM -wP;bM wS2/;bM \\wS2;bM -wS2;bM \\wM;bM -wM;bM /wM;bM /wL;bM wL\\;bM -bS1;bM \\bG2;bM -bG2;bM /bG2;bM /bP;bM bP\\;bM /bB2;bM bB2\\;bM bG1-;bM bG1/;bM /wP;bM wA2\\;bL wA2\\;bL wL\\;bL bA1\\;bL bM\\;bL -bS1;bL bP\\;bL bB1\\;bL bB2\\;bL /bB2;bL bG1-;bL bG1\\;bL /bP;bL /bG2;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bA3:-1->7492             bA3 bG1\                 1
    playMove(board, bA3, 7492);
    CheckValidMoves(board, "EXECUTED bA3 bG1\\", "wS2 \\wG1;wS2 -wM;wB1 wQ;wB1 wG2\\;wB1 wG1\\;wB2 -wG2;wB2 wG2\\;wB2 wG1\\;wB2 \\wS2;wB2 wS2/;wB2 -wP;wB2 -wS2;wB2 wB1-;wB2 wB1\\;wB2 /wB1;wB2 \\wG1;wB2 \\wG2;wB2 wG2/;wB2 wG2-;wB2 \\wM;wB2 /wM;wB2 -wM;wB2 \\wP;wG2 wG1\\;wG3 -wG2;wG3 wG2\\;wG3 wG1\\;wG3 \\wS2;wG3 wS2/;wG3 -wP;wG3 -wS2;wG3 wB1-;wG3 wB1\\;wG3 /wB1;wG3 \\wG1;wG3 \\wG2;wG3 wG2/;wG3 wG2-;wG3 \\wM;wG3 /wM;wG3 -wM;wG3 \\wP;wA3 -wG2;wA3 wG2\\;wA3 wG1\\;wA3 \\wS2;wA3 wS2/;wA3 -wP;wA3 -wS2;wA3 wB1-;wA3 wB1\\;wA3 /wB1;wA3 \\wG1;wA3 \\wG2;wA3 wG2/;wA3 wG2-;wA3 \\wM;wA3 /wM;wA3 -wM;wA3 \\wP;wM -wS2;wM \\wS2;wM wS2/;wM -wP;wM \\wP;wM \\wG1;wM -wG2;wM \\wG2;wM wG2/;wM wG2-;wM wG2\\;wM wB1-;wM wB1\\;wM /wB1;wM wG1\\;wM wS1\\;wM bB1\\;wM bA1\\;wM bM\\;wM bA3/;wM bA3-;wM bA3\\;wM /bA3;wM bB2\\;wM /bB2;wM bP\\;wM /bP;wM /bG2;wM -bG2;wM \\bG2;wM /wL;wM -wL;wM wL\\;wM -bS1;wM /wP;wM wA2\\;wL -wS2;wL wA2\\;wL \\wS2;wL wS2/;wL -wP;wL /wP;wL \\wM;wL /wM;wL -wM;wP \\wG1;wP -wS1;");

    // wB2:-1->8133             wB2 /wM                  1
    playMove(board, wB2, 8133);
    CheckValidMoves(board, "EXECUTED wB2 /wM", "bS1 \\bA2;bS1 wL\\;bS2 -bS1;bS2 wB2\\;bB2 bA2;bB2 bL;bB2 bG1;bB2 /bG1;bB2 bP\\;bG2 bP\\;bG3 -bS1;bG3 bB1\\;bG3 bB2\\;bG3 /bB2;bG3 bP\\;bG3 bM\\;bG3 bA3/;bG3 /bG2;bG3 -bG2;bG3 bA1\\;bG3 bA3-;bG3 bA3\\;bG3 /bA3;bG3 /bP;bA1 bB1\\;bA1 wS1\\;bA1 wG1\\;bA1 /wB1;bA1 wB1\\;bA1 wB1-;bA1 wG2\\;bA1 wG2-;bA1 wG2/;bA1 \\wG2;bA1 -wG2;bA1 \\wG1;bA1 \\wP;bA1 -wP;bA1 wS2/;bA1 \\wS2;bA1 -wS2;bA1 \\wM;bA1 -wM;bA1 -wB2;bA1 /wB2;bA1 -bG2;bA1 /bG2;bA1 /bP;bA1 bP\\;bA1 /bB2;bA1 bB2\\;bA1 /bA3;bA1 bA3\\;bA1 bA3-;bA1 bA3/;bA1 bM\\;bA1 bM-;bA3 bG1-;bA3 bM\\;bA3 bA1\\;bA3 bB1\\;bA3 wS1\\;bA3 wG1\\;bA3 /wB1;bA3 wB1\\;bA3 wB1-;bA3 wG2\\;bA3 wG2-;bA3 wG2/;bA3 \\wG2;bA3 -wG2;bA3 \\wG1;bA3 \\wP;bA3 -wP;bA3 wS2/;bA3 \\wS2;bA3 -wS2;bA3 \\wM;bA3 -wM;bA3 -wB2;bA3 /wB2;bA3 -bG2;bA3 /bG2;bA3 /bP;bA3 bP\\;bA3 /bB2;bA3 bB2\\;bM wS1\\;bM bA3-;bM bA1\\;bM bB1\\;bM wG1\\;bM /wB1;bM wB1\\;bM wB1-;bM wG2\\;bM wG2-;bM wG2/;bM \\wG2;bM -wG2;bM \\wG1;bM \\wP;bM -wP;bM wS2/;bM \\wS2;bM -wS2;bM \\wM;bM -wM;bM -wB2;bM /wB2;bM -bG2;bM /bG2;bM /bP;bM bP\\;bM /bB2;bM bB2\\;bM /bA3;bM bA3\\;bM bA3/;bM bG1/;bM /wP;bM wA2\\;bM wL\\;bM -bS1;bL wA2\\;bL wL\\;bL bA1\\;bL bM\\;bL -bS1;bL bP\\;bL bB1\\;bL bA3/;bL bA3-;bL bA3\\;bL /bA3;bL bB2\\;bL /bB2;bL /bP;bL /bG2;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bG3:-1->7619             bG3 bG1-                 1
    playMove(board, bG3, 7619);
    CheckValidMoves(board, "EXECUTED bG3 bG1-", "wS2 \\wG1;wS2 -wM;wB1 wQ;wB1 wG2\\;wB1 wG1\\;wB2 -wM;wB2 wM;wB2 wL;wB2 /wL;wG2 wG1\\;wG3 -wG2;wG3 wG2\\;wG3 wG1\\;wG3 \\wS2;wG3 wS2/;wG3 -wP;wG3 -wS2;wG3 wB1-;wG3 wB1\\;wG3 /wB1;wG3 -wM;wG3 -wB2;wG3 \\wG1;wG3 \\wG2;wG3 wG2/;wG3 wG2-;wG3 \\wM;wG3 \\wP;wA3 -wG2;wA3 wG2\\;wA3 wG1\\;wA3 \\wS2;wA3 wS2/;wA3 -wP;wA3 -wS2;wA3 wB1-;wA3 wB1\\;wA3 /wB1;wA3 -wM;wA3 -wB2;wA3 \\wG1;wA3 \\wG2;wA3 wG2/;wA3 wG2-;wA3 \\wM;wA3 \\wP;wM -wS2;wM \\wS2;wM wS2/;wM -wP;wM \\wP;wM \\wG1;wM -wG2;wM \\wG2;wM wG2/;wM wG2-;wM wG2\\;wM wB1-;wM wB1\\;wM /wB1;wM wG1\\;wM wS1\\;wM bB1\\;wM bA1\\;wM bM\\;wM bG3/;wM bG3-;wM bG3\\;wM bA3\\;wM /bA3;wM bB2\\;wM /bB2;wM bP\\;wM /bP;wM /bG2;wM -bG2;wM /wB2;wM -wB2;wM \\wB2;wM /wP;wM wA2\\;wM wL\\;wM wB2\\;wM wA2;wM wL;wM wB2;wL -wS2;wL wA2\\;wL -wM;wL wB2\\;wL /wB2;wL -wB2;wL \\wS2;wL wS2/;wL -wP;wL /wP;wL \\wM;wP \\wG1;wP -wS1;");

    // wA3:-1->8510             wA3 -wG2                 1
    playMove(board, wA3, 8510);
    CheckValidMoves(board, "EXECUTED wA3 -wG2", "bS1 \\bA2;bS1 wL\\;bS2 -bS1;bS2 wB2\\;bB2 bA2;bB2 bL;bB2 bG1;bB2 /bG1;bB2 bP\\;bG2 bP\\;bG3 /bA3;bG3 bP\\;bA1 bB1\\;bA1 wS1\\;bA1 wG1\\;bA1 /wB1;bA1 wB1\\;bA1 wB1-;bA1 wG2\\;bA1 wG2-;bA1 wG2/;bA1 \\wG2;bA1 \\wA3;bA1 -wA3;bA1 \\wP;bA1 -wP;bA1 wS2/;bA1 \\wS2;bA1 -wS2;bA1 \\wM;bA1 -wM;bA1 -wB2;bA1 /wB2;bA1 -bG2;bA1 /bG2;bA1 /bP;bA1 bP\\;bA1 /bB2;bA1 bB2\\;bA1 /bA3;bA1 bA3\\;bA1 bG3\\;bA1 bG3-;bA1 bG3/;bA1 bM\\;bA1 bM-;bA3 bG3\\;bA3 bG3-;bA3 bG3/;bA3 bM\\;bA3 bA1\\;bA3 bB1\\;bA3 wS1\\;bA3 wG1\\;bA3 /wB1;bA3 wB1\\;bA3 wB1-;bA3 wG2\\;bA3 wG2-;bA3 wG2/;bA3 \\wG2;bA3 \\wA3;bA3 -wA3;bA3 \\wP;bA3 -wP;bA3 wS2/;bA3 \\wS2;bA3 -wS2;bA3 \\wM;bA3 -wM;bA3 -wB2;bA3 /wB2;bA3 -bG2;bA3 /bG2;bA3 /bP;bA3 bP\\;bA3 /bB2;bA3 bB2\\;bM wS1\\;bM bG3-;bM bA1\\;bM bB1\\;bM wG1\\;bM /wB1;bM wB1\\;bM wB1-;bM wG2\\;bM wG2-;bM wG2/;bM \\wG2;bM \\wA3;bM -wA3;bM \\wP;bM -wP;bM wS2/;bM \\wS2;bM -wS2;bM \\wM;bM -wM;bM -wB2;bM /wB2;bM -bG2;bM /bG2;bM /bP;bM bP\\;bM /bB2;bM bB2\\;bM /bA3;bM bA3\\;bM bG3\\;bM bG3/;bM \\bG3;bM /wP;bM wA2\\;bM wL\\;bM -bS1;bL wA2\\;bL wL\\;bL bA1\\;bL bM\\;bL -bS1;bL bP\\;bL bB1\\;bL bG3/;bL bG3-;bL bG3\\;bL bA3\\;bL /bA3;bL bB2\\;bL /bB2;bL /bP;bL /bG2;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bM:7875->8511            bM -wA3                  1
    playMove(board, bM, 8511);
    CheckValidMoves(board, "EXECUTED bM -wA3", "wS2 \\bM;wS2 -wM;wB1 wQ;wB1 wG2\\;wB1 wG1\\;wB2 -wM;wB2 wM;wB2 wL;wB2 /wL;wG1 \\bM;wG1 \\wG2;wG1 wG2\\;wG1 bS2\\;wG1 -wP;wG2 wG1\\;wG2 -bM;wG3 wG2\\;wG3 wG1\\;wG3 \\wS2;wG3 wS2/;wG3 -wP;wG3 -wS2;wG3 wB1-;wG3 wB1\\;wG3 /wB1;wG3 -wM;wG3 -wB2;wG3 \\wG2;wG3 wG2/;wG3 wG2-;wG3 \\wM;wA3 bM/;wA3 \\bM;wA3 -bM;wA3 -wP;wA3 wS2/;wA3 \\wS2;wA3 -wS2;wA3 \\wM;wA3 -wM;wA3 -wB2;wA3 /wB2;wA3 -bG2;wA3 /bG2;wA3 /bP;wA3 bP\\;wA3 /bB2;wA3 bB2\\;wA3 /bA3;wA3 bA3\\;wA3 bG3\\;wA3 bG3-;wA3 bG3/;wA3 \\bG3;wA3 bS2\\;wA3 bA1\\;wA3 bB1\\;wA3 wS1\\;wA3 wG1\\;wA3 /wB1;wA3 wB1\\;wA3 wB1-;wA3 wG2\\;wA3 wG2-;wA3 wG2/;wA3 \\wG2;wM -wS2;wM \\wS2;wM wS2/;wM -wP;wM -bM;wM \\bM;wM \\wA3;wM \\wG2;wM wG2/;wM wG2-;wM wG2\\;wM wB1-;wM wB1\\;wM /wB1;wM wG1\\;wM wS1\\;wM bB1\\;wM bA1\\;wM bS2\\;wM \\bG3;wM bG3/;wM bG3-;wM bG3\\;wM bA3\\;wM /bA3;wM bB2\\;wM /bB2;wM bP\\;wM /bP;wM /bG2;wM -bG2;wM /wB2;wM -wB2;wM \\wB2;wM /wP;wM wA2\\;wM wL\\;wM wB2\\;wM wA2;wM wL;wM wB2;wL -wS2;wL wA2\\;wL -wM;wL wB2\\;wL /wB2;wL -wB2;wL \\wS2;wL wS2/;wL -wP;wL /wP;wL \\wM;wP -bM;wP -wS1;wG1 -bM;wG1 /wP;");

    // wG3:-1->8513             wG3 wS2/                 1
    playMove(board, wG3, 8513);
    CheckValidMoves(board, "EXECUTED wG3 wS2/", "bB2 bA2;bB2 bL;bB2 bG1;bB2 /bG1;bB2 bP\\;bG2 bP\\;bG3 /bA3;bG3 bP\\;bA1 bB1\\;bA1 wS1\\;bA1 wG1\\;bA1 /wB1;bA1 wB1\\;bA1 wB1-;bA1 wG2\\;bA1 wG2-;bA1 wG2/;bA1 \\wG2;bA1 \\wA3;bA1 \\bM;bA1 -bM;bA1 wG3/;bA1 \\wG3;bA1 -wG3;bA1 -wS2;bA1 \\wM;bA1 -wM;bA1 -wB2;bA1 /wB2;bA1 -bG2;bA1 /bG2;bA1 /bP;bA1 bP\\;bA1 /bB2;bA1 bB2\\;bA1 /bA3;bA1 bA3\\;bA1 bG3\\;bA1 bG3-;bA1 bG3/;bA1 \\bG3;bA1 bS2\\;bA3 bG3\\;bA3 bG3-;bA3 bG3/;bA3 \\bG3;bA3 bS2\\;bA3 bA1\\;bA3 bB1\\;bA3 wS1\\;bA3 wG1\\;bA3 /wB1;bA3 wB1\\;bA3 wB1-;bA3 wG2\\;bA3 wG2-;bA3 wG2/;bA3 \\wG2;bA3 \\wA3;bA3 \\bM;bA3 -bM;bA3 wG3/;bA3 \\wG3;bA3 -wG3;bA3 -wS2;bA3 \\wM;bA3 -wM;bA3 -wB2;bA3 /wB2;bA3 -bG2;bA3 /bG2;bA3 /bP;bA3 bP\\;bA3 /bB2;bA3 bB2\\;bM \\wA3;bM \\wG2;bM wG2/;bM wG2-;bM wG2\\;bM wB1-;bM wB1\\;bM /wB1;bM wG1\\;bM wS1\\;bM bB1\\;bM bA1\\;bM bS2\\;bM \\bG3;bM bG3/;bM bG3-;bM bG3\\;bM bA3\\;bM /bA3;bM bB2\\;bM /bB2;bM bP\\;bM /bP;bM /bG2;bM -bG2;bM /wB2;bM -wB2;bM -wM;bM \\wM;bM -wS2;bM -wG3;bM \\wG3;bM wG3/;bM \\wP;bM /wP;wA3 \\bM;wA3 bM/;wA3 -bM;wG1 \\bM;wG1 \\wA3;wG1 -bM;wP \\bM;wP \\wA3;wP -bM;bL wA2\\;bL bS2\\;bL wL\\;bL -bS1;bL bP\\;bL \\bG3;bL bG3/;bL bG3-;bL bG3\\;bL bA3\\;bL /bA3;bL bB2\\;bL /bB2;bL /bP;bL /bG2;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bB2:7621->7749           bB2 bA2                  1
    playMove(board, bB2, 7749);
    CheckValidMoves(board, "EXECUTED bB2 bA2", "wB1 wQ;wB1 wG2\\;wB1 wG1\\;wB2 -wM;wB2 wM;wB2 wL;wB2 /wL;wG1 \\bM;wG1 \\wG2;wG1 wG2\\;wG1 bS2\\;wG1 wG3\\;wG2 wG1\\;wG2 -bM;wG3 wB2\\;wA3 bM/;wA3 \\bM;wA3 -bM;wA3 wG3/;wA3 \\wG3;wA3 -wG3;wA3 -wS2;wA3 \\wM;wA3 -wM;wA3 -wB2;wA3 /wB2;wA3 -bG2;wA3 /bG2;wA3 /bP;wA3 bP\\;wA3 bB2\\;wA3 /bG1;wA3 /bA3;wA3 bA3\\;wA3 bG3\\;wA3 bG3-;wA3 bG3/;wA3 \\bG3;wA3 bS2\\;wA3 bA1\\;wA3 bB1\\;wA3 wS1\\;wA3 wG1\\;wA3 /wB1;wA3 wB1\\;wA3 wB1-;wA3 wG2\\;wA3 wG2-;wA3 wG2/;wA3 \\wG2;wM -wS2;wM -wG3;wM \\wG3;wM wG3/;wM -bM;wM \\bM;wM \\wA3;wM \\wG2;wM wG2/;wM wG2-;wM wG2\\;wM wB1-;wM wB1\\;wM /wB1;wM wG1\\;wM wS1\\;wM bB1\\;wM bA1\\;wM bS2\\;wM \\bG3;wM bG3/;wM bG3-;wM bG3\\;wM bA3\\;wM /bA3;wM /bG1;wM bB2\\;wM bP\\;wM /bP;wM /bG2;wM -bG2;wM /wB2;wM -wB2;wM \\wB2;wM wG3\\;wM /wP;wM wA2\\;wM wL\\;wM wB2\\;wM wA2;wM wL;wM wB2;wL -wS2;wL wA2\\;wL -wM;wL wB2\\;wL /wB2;wL -wB2;wL -wG3;wL wG3\\;wL /wP;wL \\wM;wP -bM;wP -wS1;bM \\wP;bM /wP;bM wG3\\;wG1 -bM;wG1 /wP;");

    // wL:8132->8385            wL wS2-                  1
    playMove(board, wL, 8385);
    CheckValidMoves(board, "EXECUTED wL wS2-", "bB1 wS1;bB1 wS1\\;bB1 bA1-;bB1 bA1;bB1 bQ;bB2 -bS1;bB2 bS1;bB2 bL;bB2 bA2\\;bB2 bP\\;bB2 bP;bG2 bP\\;bG3 /bA3;bG3 bB2\\;bA1 bB1\\;bA1 wS1\\;bA1 wG1\\;bA1 /wB1;bA1 wB1\\;bA1 wB1-;bA1 wG2\\;bA1 wG2-;bA1 wG2/;bA1 \\wG2;bA1 \\wA3;bA1 \\bM;bA1 -bM;bA1 wG3/;bA1 \\wG3;bA1 -wG3;bA1 -wS2;bA1 \\wM;bA1 -wM;bA1 -wB2;bA1 /wB2;bA1 -bG2;bA1 /bG2;bA1 /bP;bA1 bP\\;bA1 bB2\\;bA1 /bG1;bA1 /bA3;bA1 bA3\\;bA1 bG3\\;bA1 bG3-;bA1 bG3/;bA1 \\bG3;bA1 bS2\\;bA3 bG3\\;bA3 bG3-;bA3 bG3/;bA3 \\bG3;bA3 bS2\\;bA3 bA1\\;bA3 bB1\\;bA3 wS1\\;bA3 wG1\\;bA3 /wB1;bA3 wB1\\;bA3 wB1-;bA3 wG2\\;bA3 wG2-;bA3 wG2/;bA3 \\wG2;bA3 \\wA3;bA3 \\bM;bA3 -bM;bA3 wG3/;bA3 \\wG3;bA3 -wG3;bA3 -wS2;bA3 \\wM;bA3 -wM;bA3 -wB2;bA3 /wB2;bA3 -bG2;bA3 /bG2;bA3 /bP;bA3 bP\\;bA3 bB2\\;bA3 /bG1;bM \\wA3;bM \\wG2;bM wG2/;bM wG2-;bM wG2\\;bM wB1-;bM wB1\\;bM /wB1;bM wG1\\;bM wS1\\;bM bB1\\;bM bA1\\;bM bS2\\;bM \\bG3;bM bG3/;bM bG3-;bM bG3\\;bM bA3\\;bM /bA3;bM /bG1;bM bB2\\;bM bP\\;bM /bP;bM /bG2;bM -bG2;bM /wB2;bM -wB2;bM -wM;bM \\wM;bM -wS2;bM -wG3;bM \\wG3;bM wG3/;bM \\wP;bM wL\\;wA3 \\bM;wA3 bM/;wA3 -bM;wG1 \\bM;wG1 \\wA3;wG1 -bM;wP \\bM;wP \\wA3;wP -bM;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bA3:7492->8128           bA3 bB1-                 1
    playMove(board, bA3, 8128);
    CheckValidMoves(board, "EXECUTED bA3 bB1-", "wS2 wG3/;wS2 -wM;wB1 wQ;wB1 wG2\\;wB1 wG1\\;wB2 -wM;wB2 wM;wB2 wM\\;wG1 \\bM;wG1 \\wG2;wG1 wG2\\;wG1 bS2\\;wG1 -wS2;wG2 wG1\\;wG2 -bM;wG3 wL\\;wG3 wM\\;wA3 bM/;wA3 \\bM;wA3 -bM;wA3 wG3/;wA3 \\wG3;wA3 -wG3;wA3 -wS2;wA3 \\wM;wA3 -wM;wA3 -wB2;wA3 /wB2;wA3 -bG2;wA3 /bG2;wA3 /bP;wA3 bP\\;wA3 bB2\\;wA3 /bG1;wA3 bG1\\;wA3 bG3\\;wA3 bG3-;wA3 bG3/;wA3 \\bG3;wA3 bS2\\;wA3 bA1\\;wA3 bB1\\;wA3 bA3\\;wA3 /wB1;wA3 wB1\\;wA3 wB1-;wA3 wG2\\;wA3 wG2-;wA3 wG2/;wA3 \\wG2;wL -wG3;wL -wS2;wL \\bM;wL \\wA3;wL -bM;wL wG1\\;wL /wP;wL wA2\\;wL wM\\;wL \\wG3;wL wG3/;bM \\wP;bM wL\\;wG1 -bM;wG1 wL\\;wS1 -bM;wS1 wL\\;");

    // wB1:8254->8382           wB1 wQ                   1
    playMove(board, wB1, 8382);
    CheckValidMoves(board, "EXECUTED wB1 wQ", "bB1 wS1;bB1 bA3;bB1 bA1;bB1 bQ;bB2 -bS1;bB2 bS1;bB2 bL;bB2 bA2\\;bB2 bP\\;bB2 bP;bG2 bP\\;bG3 bB2\\;bA1 bB1\\;bA1 bA3\\;bA1 bA3-;bA1 wG1\\;bA1 wB1\\;bA1 wG2\\;bA1 wG2-;bA1 wG2/;bA1 \\wG2;bA1 \\wA3;bA1 \\bM;bA1 -bM;bA1 wG3/;bA1 \\wG3;bA1 -wG3;bA1 -wS2;bA1 \\wM;bA1 -wM;bA1 -wB2;bA1 /wB2;bA1 -bG2;bA1 /bG2;bA1 /bP;bA1 bP\\;bA1 bB2\\;bA1 /bG1;bA1 bG1\\;bA1 bG3\\;bA1 bG3-;bA1 bG3/;bA1 \\bG3;bA1 bS2\\;bA3 wG1\\;bA3 wB1\\;bA3 wG2\\;bA3 wG2-;bA3 wG2/;bA3 \\wG2;bA3 \\wA3;bA3 \\bM;bA3 -bM;bA3 wG3/;bA3 \\wG3;bA3 -wG3;bA3 -wS2;bA3 \\wM;bA3 -wM;bA3 -wB2;bA3 /wB2;bA3 -bG2;bA3 /bG2;bA3 /bP;bA3 bP\\;bA3 bB2\\;bA3 /bG1;bA3 bG1\\;bA3 bG3\\;bA3 bG3-;bA3 bG3/;bA3 \\bG3;bA3 bS2\\;bA3 bA1\\;bA3 bB1\\;bM \\wA3;bM \\wG2;bM wG2/;bM wG2-;bM wG2\\;bM wB1\\;bM wG1\\;bM bA3-;bM bA3\\;bM bB1\\;bM bA1\\;bM bS2\\;bM \\bG3;bM bG3/;bM bG3-;bM bG3\\;bM bG1\\;bM /bG1;bM bB2\\;bM bP\\;bM /bP;bM /bG2;bM -bG2;bM /wB2;bM -wB2;bM -wM;bM \\wM;bM -wS2;bM -wG3;bM \\wG3;bM wG3/;bM \\wP;bM wL\\;wA3 \\bM;wA3 bM/;wA3 -bM;wG1 \\bM;wG1 \\wA3;wG1 -bM;wP \\bM;wP \\wA3;wP -bM;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bB1:8129->8256           bB1 wS1                  1
    playMove(board, bB1, 8256);
    CheckValidMoves(board, "EXECUTED bB1 wS1", "wS2 wG3/;wS2 -wM;wB1 wA3;wB1 wG2;wB1 wG2\\;wB1 wQ\\;wB1 wG1\\;wB1 wG1;wB2 -wM;wB2 wM;wB2 wM\\;wG1 \\bM;wG1 \\wG2;wG1 wG2\\;wG1 /bB1;wG1 -wS2;wG2 wG1\\;wG2 -bM;wG3 wL\\;wG3 wM\\;wA3 bM/;wA3 \\bM;wA3 -bM;wA3 wG3/;wA3 \\wG3;wA3 -wG3;wA3 -wS2;wA3 \\wM;wA3 -wM;wA3 -wB2;wA3 /wB2;wA3 -bG2;wA3 /bG2;wA3 /bP;wA3 bP\\;wA3 bB2\\;wA3 /bG1;wA3 bG1\\;wA3 bG3\\;wA3 bG3-;wA3 bG3/;wA3 \\bG3;wA3 bS2\\;wA3 bA1\\;wA3 /bA3;wA3 bA3\\;wA3 bA3-;wA3 wG1\\;wA3 wB1\\;wA3 wG2\\;wA3 wG2-;wA3 wG2/;wA3 \\wG2;bM \\wP;bM wL\\;wG1 -bM;wG1 wL\\;");

    // wB1:8382->8383           wB1 wG1                  1
    playMove(board, wB1, 8383);
    CheckValidMoves(board, "EXECUTED wB1 wG1", "bB1 wP;bB1 wB1;bB1 wB1\\;bB1 bA3;bB1 /wS1;bB1 wL\\;bB2 -bS1;bB2 bS1;bB2 bL;bB2 bA2\\;bB2 bP\\;bB2 bP;bG2 bP\\;bG3 bB2\\;bA1 /bB1;bA1 /bA3;bA1 bA3\\;bA1 bA3-;bA1 wB1\\;bA1 wQ\\;bA1 wG2\\;bA1 wG2-;bA1 wG2/;bA1 \\wG2;bA1 \\wA3;bA1 \\bM;bA1 -bM;bA1 wG3/;bA1 \\wG3;bA1 -wG3;bA1 -wS2;bA1 \\wM;bA1 -wM;bA1 -wB2;bA1 /wB2;bA1 -bG2;bA1 /bG2;bA1 /bP;bA1 bP\\;bA1 bB2\\;bA1 /bG1;bA1 bG1\\;bA1 bG3\\;bA1 bG3-;bA1 bG3/;bA1 \\bG3;bA1 bS2\\;bA3 wB1\\;bA3 wQ\\;bA3 wG2\\;bA3 wG2-;bA3 wG2/;bA3 \\wG2;bA3 \\wA3;bA3 \\bM;bA3 -bM;bA3 wG3/;bA3 \\wG3;bA3 -wG3;bA3 -wS2;bA3 \\wM;bA3 -wM;bA3 -wB2;bA3 /wB2;bA3 -bG2;bA3 /bG2;bA3 /bP;bA3 bP\\;bA3 bB2\\;bA3 /bG1;bA3 bG1\\;bA3 bG3\\;bA3 bG3-;bA3 bG3/;bA3 \\bG3;bA3 bS2\\;bA3 bA1\\;bA3 bA1-;bA3 /bB1;bM \\wA3;bM \\wG2;bM wG2/;bM wG2-;bM wG2\\;bM wQ\\;bM wB1\\;bM bA3-;bM bA3\\;bM /bA3;bM bA1\\;bM bS2\\;bM \\bG3;bM bG3/;bM bG3-;bM bG3\\;bM bG1\\;bM /bG1;bM bB2\\;bM bP\\;bM /bP;bM /bG2;bM -bG2;bM /wB2;bM -wB2;bM -wM;bM \\wM;bM -wS2;bM -wG3;bM \\wG3;bM wG3/;bM \\wP;bM wA3;bM wB1;bM wP;wA3 \\bM;wA3 bM/;wA3 -bM;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bM:8511->8383            bM wB1                   1
    playMove(board, bM, 8383);
    CheckValidMoves(board, "EXECUTED bM wB1", "wQ wG2\\;wQ bM\\;wS2 wG3/;wS2 -wM;wB2 -wM;wB2 wM;wB2 wM\\;wG2 bM\\;wG2 -wA3;wG3 wL\\;wG3 wM\\;wA3 \\wG2;wA3 wG2/;wA3 wG2-;wA3 wG2\\;wA3 wQ\\;wA3 bM\\;wA3 bA3-;wA3 bA3\\;wA3 /bA3;wA3 bA1\\;wA3 bS2\\;wA3 \\bG3;wA3 bG3/;wA3 bG3-;wA3 bG3\\;wA3 bG1\\;wA3 /bG1;wA3 bB2\\;wA3 bP\\;wA3 /bP;wA3 /bG2;wA3 -bG2;wA3 /wB2;wA3 -wB2;wA3 -wM;wA3 \\wM;wA3 -wS2;wA3 -wG3;wA3 \\wG3;wA3 wG3/;wA3 \\wP;wA3 \\bM;");

    // wG3:8513->8257           wG3 wA1-                 1
    playMove(board, wG3, 8257);
    CheckValidMoves(board, "EXECUTED wG3 wA1-", "bB1 wP;bB1 bM;bB1 bM\\;bB1 bA3;bB1 wG3\\;bB1 wG3;bB2 -bS1;bB2 bS1;bB2 bL;bB2 bA2\\;bB2 bP\\;bB2 bP;bG2 bP\\;bG3 bB2\\;bA1 wG3\\;bA1 /bA3;bA1 bA3\\;bA1 bA3-;bA1 bM\\;bA1 wQ\\;bA1 wG2\\;bA1 wG2-;bA1 wG2/;bA1 \\wG2;bA1 \\wA3;bA1 -wA3;bA1 \\wP;bA1 \\wL;bA1 \\wS2;bA1 -wS2;bA1 \\wM;bA1 -wM;bA1 -wB2;bA1 /wB2;bA1 -bG2;bA1 /bG2;bA1 /bP;bA1 bP\\;bA1 bB2\\;bA1 /bG1;bA1 bG1\\;bA1 bG3\\;bA1 bG3-;bA1 bG3/;bA1 \\bG3;bA1 bS2\\;bA3 bM\\;bA3 wQ\\;bA3 wG2\\;bA3 wG2-;bA3 wG2/;bA3 \\wG2;bA3 \\wA3;bA3 -wA3;bA3 \\wP;bA3 \\wL;bA3 \\wS2;bA3 -wS2;bA3 \\wM;bA3 -wM;bA3 -wB2;bA3 /wB2;bA3 -bG2;bA3 /bG2;bA3 /bP;bA3 bP\\;bA3 bB2\\;bA3 /bG1;bA3 bG1\\;bA3 bG3\\;bA3 bG3-;bA3 bG3/;bA3 \\bG3;bA3 bS2\\;bA3 bA1\\;bA3 bA1-;bA3 wG3\\;bM -wA3;bM wA3;bM wQ;bM wB1\\;bM bB1;bM wP;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bA1:8002->8006           bA1 /wB2                 1
    playMove(board, bA1, 8006);
    CheckValidMoves(board, "EXECUTED bA1 /wB2", "wQ wG2\\;wQ bM\\;wS2 -wA3;wS2 -wM;wB2 -wM;wB2 wM;wB2 wM\\;wB2 bG2/;wB2 bA1;wB2 \\bA1;wG2 bM\\;wG2 -wA3;wG3 \\wL;wG3 -wA3;wG3 bM\\;wG3 bP\\;wG3 -wM;wA2 /wA1;wA2 -bS2;wA2 -bS1;wA2 wB2\\;wA2 wM\\;wA3 \\wG2;wA3 wG2/;wA3 wG2-;wA3 wG2\\;wA3 wQ\\;wA3 bM\\;wA3 bA3-;wA3 bA3\\;wA3 /bA3;wA3 wG3\\;wA3 bQ\\;wA3 bS2\\;wA3 \\bG3;wA3 bG3/;wA3 bG3-;wA3 bG3\\;wA3 bG1\\;wA3 /bG1;wA3 bB2\\;wA3 bP\\;wA3 /bP;wA3 /bG2;wA3 /bA1;wA3 -bA1;wA3 -wB2;wA3 -wM;wA3 \\wM;wA3 -wS2;wA3 \\wS2;wA3 \\wL;wA3 \\wP;wA3 \\bM;wM -wS2;wM \\wS2;wM \\wL;wM \\wP;wM -wA3;wM \\wA3;wM \\wG2;wM wG2/;wM wG2-;wM wG2\\;wM wQ\\;wM bM\\;wM bA3-;wM bA3\\;wM /bA3;wM wG3\\;wM bQ\\;wM bS2\\;wM \\bG3;wM bG3/;wM bG3-;wM bG3\\;wM bG1\\;wM /bG1;wM bB2\\;wM bP\\;wM /bP;wM /bG2;wM /bA1;wM -bA1;wM -wB2;wM \\wB2;wM wA2;wM wB2;wL -wA3;wL bM\\;wL wG3\\;wL \\wP;wL bQ\\;wL wA2\\;wL \\wS2;wL wS2/;wL -wS2;wL wM\\;wP wL/;wP -wA3;wG3 \\wP;");

    // wP:8384->8511            wP -wA3                  1
    playMove(board, wP, 8511);
    CheckValidMoves(board, "EXECUTED wP -wA3", "bQ wG3\\;bQ bS2-;bS1 wM\\;bS2 wB2\\;bS2 /bA3;bS2 bG3/;bB1 /wP;bB1 bM;bB1 bM\\;bB1 bA3;bB1 wG3\\;bB1 wG3;bB2 -bS1;bB2 bS1;bB2 bL;bB2 bA2\\;bB2 bP\\;bB2 bP;bG2 -wB2;bG2 bP\\;bG3 bB2\\;bA1 -wB2;bA1 -wM;bA1 \\wM;bA1 -wS2;bA1 \\wS2;bA1 \\wL;bA1 -wP;bA1 \\wP;bA1 \\wA3;bA1 \\wG2;bA1 wG2/;bA1 wG2-;bA1 wG2\\;bA1 wQ\\;bA1 bM\\;bA1 bA3-;bA1 bA3\\;bA1 /bA3;bA1 wG3\\;bA1 bQ\\;bA1 bS2\\;bA1 \\bG3;bA1 bG3/;bA1 bG3-;bA1 bG3\\;bA1 bG1\\;bA1 /bG1;bA1 bB2\\;bA1 bP\\;bA1 /bP;bA1 /bG2;bA1 -bG2;bA3 bM\\;bA3 wQ\\;bA3 wG2\\;bA3 wG2-;bA3 wG2/;bA3 \\wG2;bA3 \\wA3;bA3 \\wP;bA3 -wP;bA3 \\wL;bA3 \\wS2;bA3 -wS2;bA3 \\wM;bA3 -wM;bA3 -wB2;bA3 -bA1;bA3 /bA1;bA3 /bG2;bA3 /bP;bA3 bP\\;bA3 bB2\\;bA3 /bG1;bA3 bG1\\;bA3 bG3\\;bA3 bG3-;bA3 bG3/;bA3 \\bG3;bA3 bS2\\;bA3 bQ\\;bA3 wG3\\;bM wP;bM wA3;bM wQ;bM wB1\\;bM bB1;bM /wP;bP /bB2;bP /bG2;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bS2:8003->8001           bS2 /bA3                 1
    playMove(board, bS2, 8001);
    CheckValidMoves(board, "EXECUTED bS2 /bA3", "wQ wG2\\;wQ bM\\;wS2 \\wP;wS2 -wM;wG2 bM\\;wG2 -wP;wA3 wP/;wA3 \\wP;wA3 -wP;wA3 \\wL;wA3 \\wS2;wA3 -wS2;wA3 \\wM;wA3 -wM;wA3 -wB2;wA3 -bA1;wA3 /bA1;wA3 /bG2;wA3 /bP;wA3 bP\\;wA3 bB2\\;wA3 /bG1;wA3 bG1\\;wA3 bG3\\;wA3 bG3-;wA3 bG3/;wA3 \\bG3;wA3 bL/;wA3 /bQ;wA3 wA2\\;wA3 wM\\;wA3 wB2\\;wA3 -bS1;wA3 \\bS1;wA3 bQ\\;wA3 /bS2;wA3 bS2\\;wA3 bA3\\;wA3 bA3-;wA3 bM\\;wA3 wQ\\;wA3 wG2\\;wA3 wG2-;wA3 wG2/;wA3 \\wG2;wL /wP;wL bM\\;wL wG3\\;wL bQ\\;wL /bQ;wL wA2\\;wL \\wS2;wL wS2/;wL -wS2;wL wM\\;wP \\wA3;wP -bM;wA3 /wP;");

    // wL:8385->8003            wL /bQ                   1
    playMove(board, wL, 8003);
    CheckValidMoves(board, "EXECUTED wL /bQ", "bQ wG3\\;bQ -bS2;bS1 wM\\;bS2 wL\\;bS2 bM\\;bB1 /wP;bB1 bM;bB1 bM\\;bB1 bA3;bB1 wG3\\;bB1 wG3;bB2 -bS1;bB2 bS1;bB2 bL;bB2 bA2\\;bB2 bP\\;bB2 bP;bG2 -wB2;bG2 bP\\;bG3 bB2\\;bA1 -wB2;bA1 -wM;bA1 \\wM;bA1 -wS2;bA1 \\wS2;bA1 wS2/;bA1 \\wG3;bA1 /wP;bA1 -wP;bA1 \\wP;bA1 \\wA3;bA1 \\wG2;bA1 wG2/;bA1 wG2-;bA1 wG2\\;bA1 wQ\\;bA1 bM\\;bA1 bA3-;bA1 bA3\\;bA1 bS2\\;bA1 /bS2;bA1 bQ\\;bA1 wL\\;bA1 \\bG3;bA1 bG3/;bA1 bG3-;bA1 bG3\\;bA1 bG1\\;bA1 /bG1;bA1 bB2\\;bA1 bP\\;bA1 /bP;bA1 /bG2;bA1 -bG2;bM wP;bM wA3;bM wQ;bM wB1\\;bM bB1;bM /wP;bP /bB2;bP /bG2;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bG3:7619->7621           bG3 bB2\                 1
    playMove(board, bG3, 7621);
    CheckValidMoves(board, "EXECUTED bG3 bB2\\", "wQ wG2\\;wQ bM\\;wS2 -wP;wS2 -wM;wB2 -wM;wB2 wM;wB2 wM\\;wB2 bG2/;wB2 bA1;wB2 \\bA1;wG2 bM\\;wG2 -wP;wA2 /wA1;wA2 -wL;wA2 -bS1;wA2 wB2\\;wA2 wM\\;wA3 wP/;wA3 \\wP;wA3 -wP;wA3 /wP;wA3 \\wG3;wA3 wS2/;wA3 \\wS2;wA3 -wS2;wA3 \\wM;wA3 -wM;wA3 -wB2;wA3 -bA1;wA3 /bA1;wA3 /bG2;wA3 /bP;wA3 bP\\;wA3 /bG3;wA3 bG3\\;wA3 bG1\\;wA3 bG1-;wA3 bG1/;wA3 wL\\;wA3 bQ\\;wA3 /bS2;wA3 bS2\\;wA3 bA3\\;wA3 bA3-;wA3 bM\\;wA3 wQ\\;wA3 wG2\\;wA3 wG2-;wA3 wG2/;wA3 \\wG2;wM -wS2;wM \\wS2;wM wS2/;wM \\wG3;wM /wP;wM -wP;wM \\wP;wM \\wA3;wM \\wG2;wM wG2/;wM wG2-;wM wG2\\;wM wQ\\;wM bM\\;wM bA3-;wM bA3\\;wM bS2\\;wM /bS2;wM bQ\\;wM wL\\;wM bG1/;wM bG1-;wM bG1\\;wM bG3\\;wM /bG3;wM bP\\;wM /bP;wM /bG2;wM /bA1;wM -bA1;wM -wB2;wM \\wB2;wM wA2;wM wB2;wL \\wG3;wL wA2\\;wL /wP;wL wG3\\;wL bL/;wL bG1/;wL -bS1;wL bP\\;wP \\wA3;wP -bM;");

    // wA3:8510->8002           wA3 wL-                  1
    playMove(board, wA3, 8002);
    CheckValidMoves(board, "EXECUTED wA3 wL-", "bS1 wM\\;bS2 bM\\;bS2 bG1/;bB1 /wP;bB1 bM;bB1 bM\\;bB1 bA3;bB1 wG3\\;bB1 wG3;bB2 -bS1;bB2 bS1;bB2 bL;bB2 bG3;bB2 bP\\;bB2 bP;bG1 -wL;bG1 bP\\;bG2 -wB2;bG2 bP\\;bG3 -bS1;bG3 wL\\;bG3 bG1-;bA1 -wB2;bA1 -wM;bA1 \\wM;bA1 -wS2;bA1 \\wS2;bA1 wS2/;bA1 \\wG3;bA1 /wP;bA1 -wP;bA1 \\wP;bA1 wP/;bA1 -wG2;bA1 \\wG2;bA1 wG2/;bA1 wG2-;bA1 wG2\\;bA1 wQ\\;bA1 bM\\;bA1 bA3-;bA1 bA3\\;bA1 bS2\\;bA1 wA3\\;bA1 wL\\;bA1 bG1/;bA1 bG1-;bA1 bG1\\;bA1 bG3\\;bA1 /bG3;bA1 bP\\;bA1 /bP;bA1 /bG2;bA1 -bG2;bA3 bM\\;bA3 wQ\\;bA3 wG2\\;bA3 wG2-;bA3 wG2/;bA3 \\wG2;bA3 -wG2;bA3 wP/;bA3 \\wP;bA3 -wP;bA3 /wP;bA3 \\wG3;bA3 wS2/;bA3 \\wS2;bA3 -wS2;bA3 \\wM;bA3 -wM;bA3 -wB2;bA3 -bA1;bA3 /bA1;bA3 /bG2;bA3 /bP;bA3 bP\\;bA3 /bG3;bA3 bG3\\;bA3 bG1\\;bA3 bG1-;bA3 bG1/;bA3 wL\\;bA3 wA3\\;bA3 bS2\\;bA3 bS2-;bM wP;bM -wG2;bM wQ;bM wB1\\;bM bB1;bM /wP;bL wA2\\;bL wL\\;bL -wL;bL -bS1;bL bP\\;bL bG3\\;bL /bG3;bL bG1/;bL bG1-;bL bG1\\;bL /bP;bL /bG2;bP /bB2;bP /bG2;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bL:7748->7747            bL bG1/                  1
    playMove(board, bL, 7747);
    CheckValidMoves(board, "EXECUTED bL bG1/", "wS2 -wP;wS2 -wM;wB2 -wM;wB2 wM;wB2 wM\\;wB2 bG2/;wB2 bA1;wB2 \\bA1;wG2 bM\\;wG3 bM\\;wG3 bP\\;wG3 -wM;wA2 /wA1;wA2 -wL;wA2 -bS1;wA2 wB2\\;wA2 wM\\;wA3 /bS2;wA3 bS2\\;wA3 bA3\\;wA3 bA3-;wA3 bM\\;wA3 wQ\\;wA3 wG2\\;wA3 wG2-;wA3 wG2/;wA3 \\wG2;wA3 -wG2;wA3 wP/;wA3 \\wP;wA3 -wP;wA3 /wP;wA3 \\wG3;wA3 wS2/;wA3 \\wS2;wA3 -wS2;wA3 \\wM;wA3 -wM;wA3 -wB2;wA3 -bA1;wA3 /bA1;wA3 /bG2;wA3 /bP;wA3 bP\\;wA3 /bG3;wA3 bG3\\;wA3 bG1\\;wA3 bL\\;wA3 bL-;wA3 wL\\;wM -wS2;wM \\wS2;wM wS2/;wM \\wG3;wM /wP;wM -wP;wM \\wP;wM wP/;wM -wG2;wM \\wG2;wM wG2/;wM wG2-;wM wG2\\;wM wQ\\;wM bM\\;wM bA3-;wM bA3\\;wM bS2\\;wM wA3\\;wM bL-;wM bL\\;wM bG1\\;wM bG3\\;wM /bG3;wM bP\\;wM /bP;wM /bG2;wM /bA1;wM -bA1;wM -wB2;wM \\wB2;wM wA2;wM wB2;wL \\wG3;wL wA2\\;wL /wP;wL wG3\\;wL wA3\\;wL /wA3;wL bA3\\;wL bS2\\;wL -bS1;wL bS1\\;wL bP\\;wP -wG2;wP -bM;");

    // wG3:8257->8261           wG3 -wM                  1
    playMove(board, wG3, 8261);
    CheckValidMoves(board, "EXECUTED wG3 -wM", "bQ -bB1;bQ /bB1;bS1 wM\\;bB1 /wP;bB1 bM;bB1 bM\\;bB1 bA3;bB1 /wS1;bB1 -wS1;bB2 -bS1;bB2 bS1;bB2 bS1\\;bB2 bG3;bB2 bP\\;bB2 bP;bG2 /wG3;bG2 bP\\;bA1 /wG3;bA1 -wG3;bA1 \\wG3;bA1 \\wM;bA1 -wS2;bA1 \\wS2;bA1 wS2/;bA1 wA1/;bA1 -bB1;bA1 /wP;bA1 -wP;bA1 \\wP;bA1 wP/;bA1 -wG2;bA1 \\wG2;bA1 wG2/;bA1 wG2-;bA1 wG2\\;bA1 wQ\\;bA1 bM\\;bA1 bA3-;bA1 bA3\\;bA1 bS2\\;bA1 wA3\\;bA1 bL-;bA1 bL\\;bA1 bG1\\;bA1 bG3\\;bA1 /bG3;bA1 bP\\;bA1 /bP;bA1 /bG2;bA1 -bG2;bM wP;bM -wG2;bM wQ;bM wB1\\;bM bB1;bM /wP;bL bS1\\;bL bG3\\;bL /bG3;bL bP\\;bP /bB2;bP /bG2;bG2 -bS1;bG2 /bP;bG2 -bP;");

    // bG2:7878->7622           bG2 bP\                  1
    playMove(board, bG2, 7622);
    CheckValidMoves(board, "EXECUTED bG2 bP\\", "wS2 /wP;wS2 \\wG3;wG2 bM\\;wG3 -bB1;wG3 wB2\\;wP -wG2;wP -bM;");

    std::cout << "48 test eseguiti con successo" << '\n';
}

void testMzingaConnection()
{
    const char* enginePath = "/Users/umbertocrema/Desktop/Lavoro/Scuola_Ortogonale/Game/Mzinga/MzingaEngine";

    int toChild[2], fromChild[2];
    pipe(toChild);
    pipe(fromChild);

    pid_t pid = fork();
    if (pid == 0) {
        close(toChild[1]);
        close(fromChild[0]);
        dup2(toChild[0], STDIN_FILENO);
        dup2(fromChild[1], STDOUT_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        close(toChild[0]);
        close(fromChild[1]);
        execl(enginePath, enginePath, nullptr);
        _exit(1);
    }

    close(toChild[0]);
    close(fromChild[1]);
    FILE* toEngine   = fdopen(toChild[1],  "w");
    FILE* fromEngine = fdopen(fromChild[0], "r");

    // Helper: legge finché non arriva "ok" e restituisce la risposta
    auto readUntilOk = [&]() -> std::string {
        std::string result;
        char buf[8192];
        while (fgets(buf, sizeof(buf), fromEngine)) {
            if (strcmp(buf, "ok\n") == 0) break;
            result += buf;
        }
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
            result.pop_back();
        return result;
    };

    // Helper: invia un comando e restituisce la risposta
    auto send = [&](const std::string& cmd) -> std::string {
        fprintf(toEngine, "%s\n", cmd.c_str());
        fflush(toEngine);
        return readUntilOk();
    };

    srand(static_cast<unsigned>(time(nullptr)));

    // Mappa stringa -> PieceName (costruita una volta sola)
    std::unordered_map<std::string, PieceName> nameMap;
    for (int i = 0; i < static_cast<int>(NumPieceNames); i++) {
        PieceName p = static_cast<PieceName>(i);
        nameMap[GetEnumString(p)] = p;
    }

    // Parser UHP: converte una mossa UHP in (PieceName, Index destinazione)
    auto parseUHP = [&](const std::string& uhp, const Board& b) -> std::pair<PieceName, Index> {
        size_t sp = uhp.find(' ');
        PieceName piece = nameMap[uhp.substr(0, sp)];

        if (sp == std::string::npos)
            return {piece, BoardCenter};  // prima mossa

        std::string rest = uhp.substr(sp + 1);
        char first = rest.front();
        char last  = rest.back();

        if (first == '/' || first == '\\' || first == '-') {
            // indicatore PRIMA: dest è nella direzione indicata A PARTIRE DA ref
            // '-' → Right (+1), '\\' → DownRight (+128), '/' → UpRight (-127)
            Index refPos = b.GetPosition(nameMap[rest.substr(1)]);
            Index offset = (first == '/') ? NeighborOffsets[5]   // UpRight  (-127)
                         : (first == '\\') ? NeighborOffsets[1]  // DownRight (+128)
                         : NeighborOffsets[0];                    // Right     (+1)
            return {piece, refPos + offset};
        }
        if (last == '/' || last == '\\' || last == '-') {
            // indicatore DOPO: dest è nella direzione opposta rispetto a sopra
            // '/' → DownLeft (+127), '\\' → UpLeft (-128), '-' → Left (-1)
            Index refPos = b.GetPosition(nameMap[rest.substr(0, rest.size() - 1)]);
            Index offset = (last == '/') ? NeighborOffsets[2]   // DownLeft (+127)
                         : (last == '\\') ? NeighborOffsets[4]  // UpLeft   (-128)
                         : NeighborOffsets[3];                   // Left     (-1)
            return {piece, refPos + offset};
        }
        // Nessun indicatore: beetle stack
        return {piece, b.GetPosition(nameMap[rest])};
    };

    readUntilOk(); // messaggio di avvio

    unsigned masterSeed; //static_cast<unsigned>(time(nullptr));

    for (int game = 0; game < 1; game++)
    {
        unsigned seed = masterSeed + static_cast<unsigned>(game);
        srand(seed);
        std::cout << "Game " << (game + 1) << "/100  seed=" << seed << '\n';

        send("newgame Base+MLP");
        Board board(GameType::BaseMLP);
        board.boardState = BoardState::InProgress;

        for (int turn = 0; turn < 10000; turn++)
        {
            std::string movesStr = send("validmoves");

            // "pass" = nessuna mossa disponibile, la partita è finita
            if (movesStr == "pass") break;

            std::string label = "game=" + std::to_string(game + 1) +
                                " turn=" + std::to_string(turn) +
                                " seed=" + std::to_string(seed);

            if ((board.IsQueenSurrounded(Color::White) || board.IsQueenSurrounded(Color::Black)) && movesStr == "err The game is over. Try 'newgame' to start a new game.") {
                std::cout << "GAME ENDED FOR BOTH" << '\n';
                break;
            }

            CheckValidMoves(board, label, movesStr + ";", true);

            // Sceglie una mossa a caso
            std::vector<std::string> moves;
            std::istringstream ss(movesStr);
            std::string token;
            while (std::getline(ss, token, ';'))
                if (!token.empty()) moves.push_back(token);

            std::string chosen = moves[rand() % moves.size()];
            auto [piece, dest] = parseUHP(chosen, board);
            std::cout << "  play " << chosen << '\n';
            send("play " + chosen);
            board.MovePiece(piece, dest);
            board.ApplyTurnEffects(piece);
        }

        std::cout << " OK\n";
    }

    std::cout << "Tutti i test passati.\n";

    fclose(toEngine);
    fclose(fromEngine);
    waitpid(pid, nullptr, 0);
}

int main()
{
    Board::InitializeZobristTable();

    /*
    TestQueenBeeMoves();
    TestBeetleMoves();
    TestGrasshopperMoves();
    TestSpiderMoves();
    TestSoldierAntMoves();
    TestLadybugMoves();
    TestPillbugMoves();
    TestMosquitoMoves();
    */

    //testGame48Moves();
    testMzingaConnection();

    return 0;
}
