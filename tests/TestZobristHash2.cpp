#include <iostream>
#include <cassert>
#include "Board.h"

using namespace HiveGotThis;

void TestMovePieceBase(Board& b) {
    std::cout << "\n--- TEST 1: MovePiece (Hand -> Board -> Board) ---" << std::endl;

    // 1. Dalla Mano alla Board (Piazzamento)
    Index posA = 1000;
    Index posB = 1001;
    
    // Mossa: Regina da Mano a posA
    b.MovePiece(PieceName::wQ, posA);
    uint64_t hashA = b.GetHash();
    
    assert(hashA != 0);
    assert(b.GetPieceAt(posA) == PieceName::wQ);
    std::cout << "Piazzamento (Hand->PosA): OK. Hash: " << hashA << std::endl;

    // 2. Dalla Board alla Board (Spostamento)
    // Mossa: Regina da posA a posB
    b.MovePiece(PieceName::wQ, posB);
    uint64_t hashB = b.GetHash();

    assert(hashB != hashA); // Deve essere cambiato
    assert(b.GetPieceAt(posA) == PieceName::INVALID); // La vecchia deve essere vuota
    assert(b.GetPieceAt(posB) == PieceName::wQ);      // La nuova deve avere la Regina
    std::cout << "Spostamento (PosA->PosB): OK. Hash: " << hashB << std::endl;

    // 3. Ritorno indietro (Reversibilità)
    b.MovePiece(PieceName::wQ, posA);
    assert(b.GetHash() == hashA);
    std::cout << "Ritorno (PosB->PosA): OK. Hash tornato identico." << std::endl;
}

void TestTransposition() {
    std::cout << "\n--- TEST 2: Transposition (Il Triangolo) ---" << std::endl;
    // Verifica che il percorso non conti, conta solo la destinazione.
    
    Board b1(GameType::Base), b2(GameType::Base);
    Index p1 = 100, p2 = 101, p3 = 102;

    // BOARD 1: Fa il giro lungo (p1 -> p2 -> p3)
    b1.MovePiece(PieceName::wG1, p1); // Piazza
    b1.MovePiece(PieceName::wG1, p2); // Sposta
    b1.MovePiece(PieceName::wG1, p3); // Sposta ancora

    // BOARD 2: Va diretta (p1 -> p3)
    // Nota: simuliamo che sia stata piazzata su p1 e poi mossa subito a p3
    // (Oppure piazzata direttamente su p3, il risultato finale deve essere identico
    // se consideriamo solo la posizione finale, ma MovePiece gestisce la logica "da dove vieni")
    
    // Per essere corretti con la logica di gioco:
    b2.MovePiece(PieceName::wG1, p1); // Piazza start
    b2.MovePiece(PieceName::wG1, p3); // Salta direttamente alla fine

    std::cout << "Board 1 (Giro Lungo) Hash: " << b1.GetHash() << std::endl;
    std::cout << "Board 2 (Giro Corto) Hash: " << b2.GetHash() << std::endl;

    assert(b1.GetHash() == b2.GetHash());
    std::cout << "SUCCESS: Arrivare alla stessa casella con mosse diverse produce lo stesso Hash." << std::endl;
}

void TestBeetleMoveOverStack() {
    std::cout << "\n--- TEST 3: MovePiece su Stack (Beetle) ---" << std::endl;
    
    Board b(GameType::Base);
    Index pos1 = 2000;
    Index pos2 = 2001;

    // Setup: Regina su pos2
    b.MovePiece(PieceName::wQ, pos2);
    
    // 1. Piazza Scarabeo su pos1
    b.MovePiece(PieceName::bS1, pos1);
    uint64_t hashSeparate = b.GetHash();

    // 2. Muovi Scarabeo SOPRA la Regina (pos1 -> pos2)
    b.MovePiece(PieceName::bS1, pos2);
    uint64_t hashStacked = b.GetHash();

    assert(hashStacked != hashSeparate);
    
    // Verifica fisica
    assert(b.GetPieceAt(pos2) == PieceName::bS1); // Sopra c'è lo scarabeo
    
    // Verifica che l'hash sia coerente: 
    // Torniamo indietro
    b.MovePiece(PieceName::bS1, pos1);
    assert(b.GetHash() == hashSeparate);

    std::cout << "SUCCESS: Lo scarabeo si muove sopra e sotto stack aggiornando l'hash correttamente." << std::endl;
}

int main() {
    std::cout << "=== HIVE ENGINE: MOVE PIECE TEST ===" << std::endl;
    
    Board::InitializeZobristTable();

    Board b(GameType::Base);
    TestMovePieceBase(b);
    TestTransposition();
    TestBeetleMoveOverStack();

    std::cout << "\n=== TUTTI I TEST PASSATI ===" << std::endl;
    return 0;
}