#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include "Board.h"

using namespace HiveGotThis;

// Helper per stampare messaggi colorati
void Log(const std::string& msg) {
    std::cout << "[TEST] " << msg << std::endl;
}

void TestBaseConnection() {
    Log("--- Test 1: Connessione Base (Lineare) ---");
    Board b(GameType::Base);
    
    assert(b.IsOneHive(NullIndex));

    // Piazziamo 3 pezzi in fila: A - B - C
    Index p1 = 2000;
    Index p2 = 2001; // Est di p1
    Index p3 = 2002; // Est di p2

    b.MovePiece(PieceName::wA1, p1);
    b.MovePiece(PieceName::bA1, p2);
    b.MovePiece(PieceName::wA2, p3);

    // 1. Muovere A (estremità sinistra) -> OK
    assert(b.CanMoveWithoutBreakingHive(PieceName::wA1) == true);
    
    // 2. Muovere C (estremità destra) -> OK
    assert(b.CanMoveWithoutBreakingHive(PieceName::wA2) == true);

    // 3. Muovere B (centro) -> NO (spezzerebbe A da C)
    assert(b.CanMoveWithoutBreakingHive(PieceName::bA1) == false);

    Log("PASS: Rilevazione corretta dei punti di articolazione lineari.");
}

void TestRingShape() {
    Log("--- Test 2: Forma ad Anello (Ring) ---");
    Board b(GameType::Base);
    // Creiamo un cerchio di 6 pezzi. Tutti dovrebbero potersi muovere
    // perché togliendone uno il cerchio diventa una "C", ma resta connesso.
    
    Index center = 5000;
    // Coordinate vicini: N, NE, E, S, SO, O
    Index n  = center - 128;
    Index ne = center - 127;
    Index se = center + 1;
    Index s  = center + 128;
    Index sw = center + 127;
    Index nw = center - 1;

    b.MovePiece(PieceName::wG1, n);
    b.MovePiece(PieceName::bG1, ne);
    b.MovePiece(PieceName::wG2, se);
    b.MovePiece(PieceName::bG2, s);
    b.MovePiece(PieceName::wA1, sw);
    b.MovePiece(PieceName::bA1, nw);

    // Verifica che TUTTI possano muoversi
    assert(b.CanMoveWithoutBreakingHive(PieceName::wG1) == true);
    assert(b.CanMoveWithoutBreakingHive(PieceName::bG2) == true);
    assert(b.CanMoveWithoutBreakingHive(PieceName::wA1) == true);

    Log("PASS: In un anello chiuso tutti i pezzi sono liberi.");
}

void TestStackLogic() {
    Log("--- Test 3: Logica delle Pile (Beetle Logic) ---");
    Board b(GameType::Base);
    Index pos = 3000;
    Index neighbor = 3001;

    // Setup: Regina Bianca a terra, Scarabeo Nero sopra
    b.MovePiece(PieceName::wQ, pos);
    b.MovePiece(PieceName::bS1, pos); // Sale sopra wQ
    
    // Mettiamo un vicino per avere un "hive" valido
    b.MovePiece(PieceName::bQ, neighbor);

    // 1. Lo Scarabeo (in cima) può muoversi?
    // SÌ, perché sotto c'è la Regina che tiene la connessione
    bool beetleCanMove = b.CanMoveWithoutBreakingHive(PieceName::bS1);
    assert(beetleCanMove == true);

    // 2. La Regina (sotto) può muoversi?
    // NO, fisicamente bloccata (il tuo assert scatterebbe qui se non gestito prima)
    // CanMoveWithoutBreakingHive assumerebbe "false" per logica fisica o assert
    // Nota: nel tuo codice l'assert(false) scatta se chiami questo su un pezzo coperto.
    // Quindi non possiamo testarlo direttamente senza crashare il test, 
    // ma sappiamo che GetPieceAt(pos) != wQ.

    // 3. Caso speciale: Pila come ponte
    //    A -- [Q sotto, S sopra] -- B
    // Se S si muove, Q tiene uniti A e B.
    Index pLeft = 2999;
    b.MovePiece(PieceName::wA1, pLeft); // Sinistra della pila
    
    // Ora abbiamo wA1 <-> [wQ, bS1] <-> bQ
    // Se bS1 si muove, wQ rimane lì e connette wA1 con bQ.
    assert(b.CanMoveWithoutBreakingHive(PieceName::bS1) == true);

    Log("PASS: Lo Scarabeo su una pila non rompe l'alveare muovendosi.");
}

void TestStarConfiguration() {
    Log("--- Test 4: Configurazione a Stella (Hub & Spoke) ---");
    Board b(GameType::Base);
    Index center = 4000;
    
    // Regina al centro
    b.MovePiece(PieceName::wQ, center);
    
    // 3 pezzi attorno (raggi)
    b.MovePiece(PieceName::bA1, center - 128); // Nord
    b.MovePiece(PieceName::bA2, center + 128); // Sud
    b.MovePiece(PieceName::bA3, center + 1);   // Est

    // 1. I raggi possono muoversi? SÌ (sono foglie)
    assert(b.CanMoveWithoutBreakingHive(PieceName::bA1) == true);
    assert(b.CanMoveWithoutBreakingHive(PieceName::bA2) == true);

    // 2. Il centro può muoversi? NO (rompe tutto)
    assert(b.CanMoveWithoutBreakingHive(PieceName::wQ) == false);

    Log("PASS: Il centro di una stella è bloccato correttamente.");
}

void TestWeirdGraph() {
    Log("--- Test 5: Grafo Complesso (The Snake) ---");
    Board b(GameType::Base);
    // A - B - C
    //     |
    //     D
    Index pA = 1000;
    Index pB = 1001;
    Index pC = 1002;
    Index pD = 1001 + 128; // Sud di B

    b.MovePiece(PieceName::wA1, pA); // A
    b.MovePiece(PieceName::bA1, pB); // B (Snodo)
    b.MovePiece(PieceName::wA2, pC); // C
    b.MovePiece(PieceName::bA2, pD); // D

    // B è l'unico punto di articolazione critico
    assert(b.CanMoveWithoutBreakingHive(PieceName::bA1) == false); // B bloccato
    
    assert(b.CanMoveWithoutBreakingHive(PieceName::wA1) == true);  // A libero
    assert(b.CanMoveWithoutBreakingHive(PieceName::wA2) == true);  // C libero
    assert(b.CanMoveWithoutBreakingHive(PieceName::bA2) == true);  // D libero

    Log("PASS: Rilevazione corretta in grafo a T.");
}

int main() {
    std::cout << "=== ESECUZIONE TEST HIVE LOGIC (BFS & PILE) ===" << std::endl;
    
    Board::InitializeZobristTable(); // Se serve per MovePiece

    TestBaseConnection();
    TestRingShape();
    TestStackLogic();
    TestStarConfiguration();
    TestWeirdGraph();

    std::cout << "\n=== TUTTI I TEST LOGICI SUPERATI ===" << std::endl;
    return 0;
}