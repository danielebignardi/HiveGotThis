/*
#include "Move.h"
#include <iostream>

using namespace HiveGotThis;

int main()
{
    //testare l'hash di move
    Move m1{PieceName::wQ, 0, 128*128-1}; // Queen bianca che si muove dalla posizione 0 alla posizione 16383 (angolo opposto)

    std::cout << "Hash of m1: " << hash(m1) << std::endl;
    std::cout << "Hash of mPass: " << hash(PassMove) << std::endl;



    return 0;
}
    */

#include <iostream>
#include <string>
#include <vector>

// Includiamo i nostri moduli
#include "Constants.h"
#include "Enums.h"
#include "Position.h"
#include "Move.h"

using namespace HiveGotThis;
using namespace std;

// --- PICCOLO FRAMEWORK DI TEST FATTO IN CASA ---
int g_testsPassed = 0;
int g_testsFailed = 0;

#define REQUIRE(condition, message) \
    do { \
        if (condition) { \
            /* cout << "[OK] " << message << endl; */ \
            g_testsPassed++; \
        } else { \
            cout << "[FAIL] " << message << " (Riga " << __LINE__ << ")" << endl; \
            g_testsFailed++; \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// 1. TEST ENUMS & MATEMATICA
// -----------------------------------------------------------------------------
void Test_Enums()
{
    cout << "--- Testing Enums & Math ---" << endl;

    // Test Colori
    REQUIRE(GetColor(PieceName::wQ) == Color::White, "Regina Bianca deve essere White");
    REQUIRE(GetColor(PieceName::bQ) == Color::Black, "Regina Nera deve essere Black");
    
    // Test BugType
    REQUIRE(GetBugType(PieceName::wS1) == BugType::Spider, "wS1 deve essere un Ragno");
    REQUIRE(GetBugType(PieceName::bS1) == BugType::Spider, "bS1 deve essere un Ragno");
}

// -----------------------------------------------------------------------------
// 2. TEST POSIZIONI (Il più critico!)
// -----------------------------------------------------------------------------
void Test_Positions()
{
    cout << "--- Testing Positions ---" << endl;

    // 1. Validità
    REQUIRE(IsValidIndex(0) == true, "Indice 0 deve essere valido");
    REQUIRE(IsValidIndex(static_cast<Index>(BoardSize - 1)) == true, "Ultimo indice deve essere valido");
    REQUIRE(IsValidIndex(-1) == false, "-1 non deve essere valido (tranne come NullIndex)");
    REQUIRE(IsValidIndex(static_cast<Index>(BoardSize)) == false, "Fuori dai bordi non valido");

    // 2. Movimento (Vicinato)
    // Testiamo muovendoci dal Centro
    Index center = BoardCenter;
    
    // Muovi a Destra (+1)
    Index right = GetNeighborAt(center, Direction::Right);
    REQUIRE(right == center + 1, "Vicino Destro corretto");

    // Muovi a Sinistra (-1)
    Index left = GetNeighborAt(center, Direction::Left);
    REQUIRE(left == center - 1, "Vicino Sinistro corretto");

    // Muovi DownRight (+BoardWidth)
    Index downRight = GetNeighborAt(center, Direction::DownRight);
    REQUIRE(downRight == center + BoardWidth, "Vicino DownRight corretto");

    // Test ciclico: Vai a destra, poi a sinistra -> devi tornare al centro
    REQUIRE(GetNeighborAt(right, Direction::Left) == center, "Andata e Ritorno orizzontale");
}

// -----------------------------------------------------------------------------
// 3. TEST MOSSE & HASH
// -----------------------------------------------------------------------------
void Test_Moves()
{
    cout << "--- Testing Moves ---" << endl;

    // 2. Uguaglianza
    Move m1{PieceName::wQ, 100, 200};
    Move m2{PieceName::wQ, 100, 200};
    REQUIRE(m1 == m2, "Mosse identiche devono essere uguali");

    // 3. Hash (Collisioni semplici)
    size_t h1 = HiveGotThis::hash(m1);
    Move m3{PieceName::bQ, 100, 200};
    size_t h2 = HiveGotThis::hash(m3);
    REQUIRE(h1 != h2, "Hash di mosse diverse deve essere diverso");
    
    // Verifica che l'hash sia stabile
    REQUIRE(h1 == HiveGotThis::hash(m1), "Hash deterministico");
}

// -----------------------------------------------------------------------------
// 4. TEST GESTIONE STRINGHE (UHP PARSING)
// -----------------------------------------------------------------------------
void Test_MoveStrings()
{
    cout << "--- Testing Move Strings (UHP) ---" << endl;

    // Variabili di appoggio per catturare i risultati del parsing
    bool isPass = false;
    PieceName startPiece = PieceName::INVALID;
    PieceName endPiece = PieceName::INVALID;
    char beforeSep = '\0';
    char afterSep = '\0';
    std::string resultString;

    // ----------------------------------------------------------
    // TEST 1: BuildMoveString (Assemblaggio)
    // ----------------------------------------------------------
    // Caso A: Mossa di apertura (Solo un pezzo, es. "wS1")
    isPass = false;
    startPiece = PieceName::wS1;
    beforeSep = '\0';
    endPiece = PieceName::INVALID;
    afterSep = '\0';
    
    std::string move1 = BuildMoveString(isPass, startPiece, beforeSep, endPiece, afterSep);
    REQUIRE(move1 == "wS1", "Build: Apertura semplice (wS1)");

    // Caso B: Mossa Relativa ("wQ \ wS1" -> Regina a sx del Ragno)
    startPiece = PieceName::wQ;
    beforeSep = '\\'; // Backslash (Left Of)
    endPiece = PieceName::wS1;
    afterSep = '\0';

    std::string move2 = BuildMoveString(isPass, startPiece, beforeSep, endPiece, afterSep);
    // Nota: Mzinga potrebbe generare "wQ \ wS1" o "wQ \wS1" a seconda degli spazi.
    // Adattiamo il test per accettare entrambi se sensato, ma lo standard UHP vuole spazi.
    REQUIRE(move2 == "wQ \\wS1", "Build: Mossa relativa (wQ \\ wS1)");

    // Caso C: Pass
    isPass = true;
    std::string movePass = BuildMoveString(isPass, startPiece, beforeSep, endPiece, afterSep);
    REQUIRE(movePass == "pass", "Build: Mossa Pass");


    // ----------------------------------------------------------
    // TEST 2: TryNormalizeMoveString (Pulizia Stringa)
    // ----------------------------------------------------------
    // Caso: Input sporco (spazi, minuscole)
    std::string messyInput = "  wq  /  ws1  "; 
    bool success = TryNormalizeMoveString(messyInput, resultString);
    
    REQUIRE(success == true, "Normalize: Input sporco accettato");
    // Ci aspettiamo che diventi maiuscolo e con spazi singoli
    REQUIRE(resultString == "wQ /wS1", "Normalize: Stringa pulita correttamente");

    // Caso: Input spazzatura
    std::string garbageInput = "mossa non valida";
    success = TryNormalizeMoveString(garbageInput, resultString);
    REQUIRE(success == false, "Normalize: Rifiuta garbage");


    // ----------------------------------------------------------
    // TEST 3: TryNormalizeMoveString (Parser Completo)
    // ----------------------------------------------------------
    // Caso: "wB1 - wQ" (Scarabeo si muove accanto alla Regina)
    std::string validMove = "wB1 - wQ";
    
    // Reset variabili
    isPass = false; startPiece = PieceName::INVALID; endPiece = PieceName::INVALID;
    beforeSep = '\0'; afterSep = '\0';

    success = TryNormalizeMoveString(validMove, isPass, startPiece, beforeSep, endPiece, afterSep);

    REQUIRE(success == true, "Parse: Stringa valida");
    REQUIRE(isPass == false, "Parse: Non è un pass");
    REQUIRE(startPiece == PieceName::wB1, "Parse: StartPiece è wB1");
    REQUIRE(endPiece == PieceName::wQ, "Parse: EndPiece è wQ");
    REQUIRE(beforeSep == '-', "Parse: Separatore '-' rilevato");

    // Caso: "pass"
    success = TryNormalizeMoveString("pass", isPass, startPiece, beforeSep, endPiece, afterSep);
    REQUIRE(success == true, "Parse: Pass valido");
    REQUIRE(isPass == true, "Parse: Flag IsPass attivo");
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main()
{
    cout << "AVVIO TEST AUTOMATIZZATI..." << endl << endl;

    Test_Enums();
    Test_Positions();
    Test_Moves();
    Test_MoveStrings();

    cout << endl << "------------------------------------------------" << endl;
    cout << "TEST COMPLETATI." << endl;
    cout << "PASSED: " << g_testsPassed << endl;
    cout << "FAILED: " << g_testsFailed << endl;

    if (g_testsFailed == 0) {
        cout << "TUTTO VERDE! Procedi pure." << endl;
        return 0;
    } else {
        cout << "ATTENZIONE: Correggi gli errori prima di continuare." << endl;
        return 1;
    }
}