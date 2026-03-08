#include "Board.h"
#include <iostream>
#include <cassert>
#include <iomanip> // Per stampare in esadecimale


using namespace HiveGotThis;

// Funzione helper per stampare gli hash in modo leggibile
void PrintHash(const std::string& label, uint64_t hash) {
    std::cout << label << ": 0x" << std::hex << std::uppercase << hash << std::dec << std::endl;
}

void TestBaseReversibilita(Board& b) {
    std::cout << "\n--- TEST 1: Reversibilita (Push & Pop) ---" << std::endl;

    uint64_t initialHash = b.GetHash();
    PrintHash("Hash Iniziale", initialHash);
    assert(initialHash == 0); // La board vuota dovrebbe avere hash 0

    // 1. Mossa: Piazza Regina Bianca
    Index pos = 1000;
    b.PushAt(PieceName::wQ, pos);
    uint64_t hashAfterPush = b.GetHash();
    PrintHash("Dopo Push wQ", hashAfterPush);
    
    assert(hashAfterPush != initialHash); // Deve essere cambiato

    // 2. Annulla Mossa: Togli Regina Bianca
    b.PopAt(pos);
    uint64_t hashAfterPop = b.GetHash();
    PrintHash("Dopo Pop wQ ", hashAfterPop);

    // VERIFICA: L'hash deve essere tornato ESATTAMENTE quello iniziale
    assert(hashAfterPop == initialHash);
    std::cout << "SUCCESS: Push e Pop sono perfettamente reversibili." << std::endl;
}

void TestCoerenzaDueBoard() {
    std::cout << "\n--- TEST 2: Coerenza tra Board diverse ---" << std::endl;

    Board b1(GameType::Base);
    Board b2(GameType::Base);

    Index pos = 555;

    // Faccio le stesse identiche cose su due istanze diverse
    b1.PushAt(PieceName::bA1, pos);
    b2.PushAt(PieceName::bA1, pos);

    PrintHash("Board 1 Hash", b1.GetHash());
    PrintHash("Board 2 Hash", b2.GetHash());

    assert(b1.GetHash() == b2.GetHash());
    std::cout << "SUCCESS: Due board con lo stesso stato hanno lo stesso Hash." << std::endl;
}

void TestStackOrder(Board& b1, Board& b2) {
    std::cout << "\n--- TEST 3: Stack Order (Il test CRITICO) ---" << std::endl;
    // Verifica che [A sotto, B sopra] sia diverso da [B sotto, A sopra]
    
    Index pos = 2000;

    // SCENARIO 1: Regina sotto, Scarabeo sopra
    // wQ (livello 0) -> bS1 (livello 1)
    b1.PushAt(PieceName::wQ, pos);
    b1.PushAt(PieceName::bS1, pos);
    b1.PushAt(PieceName::bA1, pos);
    
    // SCENARIO 2: Scarabeo sotto, Regina sopra
    // bS1 (livello 0) -> wQ (livello 1)
    b2.PushAt(PieceName::bS1, pos);
    b2.PushAt(PieceName::wQ, pos);
    b1.PushAt(PieceName::bA1, pos);

    PrintHash("Scenario 1 (Q sotto, S sopra)", b1.GetHash());
    PrintHash("Scenario 2 (S sotto, Q sopra)", b2.GetHash());

    // VERIFICA: Se hai implementato ZobristLevel correttamente, questi devono essere DIVERSI
    assert(b1.GetHash() != b2.GetHash());
    
    if (b1.GetHash() != b2.GetHash()) {
        std::cout << "SUCCESS: L'Hash distingue l'ordine verticale dei pezzi!" << std::endl;
    } else {
        std::cerr << "FAIL: Collisione! L'hash non distingue chi sta sopra e chi sta sotto." << std::endl;
    }
}

void TestTurno() {
    std::cout << "\n--- TEST 4: Cambio Turno (Toggle) ---" << std::endl;
    Board b(GameType::Base);
    uint64_t start = b.GetHash();

    // Simula cambio turno a Nero
    b.ToggleTurnHash();
    uint64_t blackTurn = b.GetHash();
    PrintHash("Turno Nero", blackTurn);
    assert(start != blackTurn);

    // Simula ritorno a Bianco
    b.ToggleTurnHash();
    uint64_t backToWhite = b.GetHash();
    PrintHash("Ritorno a Bianco", backToWhite);
    
    assert(backToWhite == start);
    std::cout << "SUCCESS: Il cambio turno funziona correttamente." << std::endl;
}

int main() {
    std::cout << "=== HIVE ENGINE: ZOBRIST UNIT TEST ===" << std::endl;

    // 1. IMPORTANTE: Inizializza la tabella statica UNA VOLTA SOLA
    Board::InitializeZobristTable();

    Board b1(GameType::Base);
    TestBaseReversibilita(b1);
    
    TestCoerenzaDueBoard();

    Board b2(GameType::Base), b3(GameType::Base);
    TestStackOrder(b2, b3);

    TestTurno();

    std::cout << "\n=== TUTTI I TEST PASSATI ===" << std::endl;
    return 0;
}