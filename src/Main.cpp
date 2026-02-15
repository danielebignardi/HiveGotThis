#include <iostream>
#include <cassert>
#include "Board.h"

using namespace HiveGotThis;

void TestInizializzazione(const Board& b) {
    std::cout << "Test Inizializzazione... ";
    
    // Verifica che la board sia vuota al centro
    assert(b.GetPieceAt(BoardSize / 2) == PieceName::INVALID);
    assert(b.HasPieceAt(BoardSize / 2) == false);
    
    // Verifica che i pezzi siano "in mano"
    assert(b.PieceInHand(PieceName::wQ) == true);
    assert(b.PieceInPlay(PieceName::wQ) == false);
    
    std::cout << "OK!" << std::endl;
}

void TestPiazzamentoEMovimento(Board& b) {
    std::cout << "Test Piazzamento e Stack... ";
    
    Index pos1 = 1000; // Un indice a caso nel mezzo
    Index pos2 = 1001;

    // 1. Piazziamo la Regina Bianca (wQ)
    b.MovePiece(PieceName::wQ, pos1);
    assert(b.GetPieceAt(pos1) == PieceName::wQ);
    assert(b.PieceInPlay(PieceName::wQ) == true);
    assert(b.PieceIsOnTop(PieceName::wQ) == true);
    assert(b.PieceInHand(PieceName::wQ) == false);

    // 2. Test Stack: Piazziamo uno Scarabeo Nero (bS1) sopra la Regina
    // Nota: In Hive solo lo scarabeo sale sopra, ma la tua board lo permette a tutti
    b.MovePiece(PieceName::bS1, pos1);
    assert(b.GetPieceAt(pos1) == PieceName::bS1); // Ora in cima c'è lo scarabeo
    assert(b.PieceIsOnTop(PieceName::bS1) == true);
    assert(b.PieceIsOnTop(PieceName::wQ) == false); // La regina è sotto
    assert(b.PieceInPlay(PieceName::bS1) == true);
    assert(b.PieceInHand(PieceName::bS1) == false);

    // 3. Muoviamo lo scarabeo via (Spostamento)
    b.MovePiece(PieceName::bS1, pos2);
    assert(b.GetPieceAt(pos2) == PieceName::bS1);
    assert(b.GetPieceAt(pos1) == PieceName::wQ); // La regina deve essere tornata visibile!
    assert(b.PieceIsOnTop(PieceName::wQ) == true);

    std::cout << "OK!" << std::endl;
}

int main() {
    std::cout << "--- HIVE ENGINE: BOARD UNIT TEST ---" << std::endl;

    try {
        Board b;

        TestInizializzazione(b);
        TestPiazzamentoEMovimento(b);

        std::cout << "--- TUTTI I TEST PASSATI CON SUCCESSO ---" << std::endl;
    }
    catch (...) {
        std::cerr << "ERRORE CRITICO DURANTE I TEST" << std::endl;
        return 1;
    }

    return 0;
}