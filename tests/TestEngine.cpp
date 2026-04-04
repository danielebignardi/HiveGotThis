// =============================================================================
// Compilazione (dalla cartella tests, con gli header in include):
//   g++ -std=c++17 -Wall -g -I../include \
//       TestEngine.cpp ../src/Board.cpp ../src/Enums.cpp \
//       ../src/Move.cpp ../src/Engine.cpp -o test_engine
// =============================================================================

#include "Board.h"
#include "Engine.h"
#include "Enums.h"
#include "Move.h"
#include "Position.h"
#include "Constants.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace HiveGotThis;

// =============================================================================
// MINI FRAMEWORK DI TEST
// =============================================================================

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(desc, condition)                                              \
    do {                                                                    \
        if (condition) {                                                    \
            std::cout << "  [PASS] " << (desc) << "\n";                    \
            ++g_passed;                                                     \
        } else {                                                            \
            std::cout << "  [FAIL] " << (desc) << "\n";                    \
            ++g_failed;                                                     \
        }                                                                   \
    } while(0)

static void PrintSection(const std::string& name)
{
    std::cout << "\n=== " << name << " ===\n";
}

// =============================================================================
// HELPER: esegue l'engine con i comandi dati e restituisce l'output completo
// =============================================================================
static std::string RunEngine(const std::string& commands)
{
    // Aggiungiamo sempre "exit" alla fine per terminare il loop
    std::string input = commands;
    if (input.find("exit") == std::string::npos)
        input += "\nexit\n";

    std::istringstream fakeInput(input);
    std::ostringstream fakeOutput;

    std::streambuf* origCin  = std::cin.rdbuf(fakeInput.rdbuf());
    std::streambuf* origCout = std::cout.rdbuf(fakeOutput.rdbuf());

    Engine engine;
    engine.Run();

    std::cin.rdbuf(origCin);
    std::cout.rdbuf(origCout);

    return fakeOutput.str();
}

// Conta quante volte una sottostringa appare nell'output
static int CountOccurrences(const std::string& text, const std::string& pattern)
{
    int count = 0;
    size_t pos = 0;
    while ((pos = text.find(pattern, pos)) != std::string::npos)
    {
        ++count;
        pos += pattern.size();
    }
    return count;
}

// =============================================================================
// SEZIONE 1: COMANDO INFO
// =============================================================================
void TestInfo()
{
    PrintSection("Comando: info");

    std::string out = RunEngine("info\n");

    CHECK("Contiene 'id HiveGotThis'",
          out.find("id HiveGotThis") != std::string::npos);

    CHECK("Contiene le capabilities (Mosquito;Ladybug;Pillbug)",
          out.find("Mosquito;Ladybug;Pillbug") != std::string::npos);

    // L'info viene stampata 2 volte: all'avvio automaticamente + al comando esplicito
    CHECK("'ok' presente almeno 2 volte (avvio + comando info)",
          CountOccurrences(out, "ok\n") >= 2);
}

// =============================================================================
// SEZIONE 2: COMANDO NEWGAME
// =============================================================================
void TestNewGame()
{
    PrintSection("Comando: newgame");

    // --- newgame senza parametri ---
    {
        std::string out = RunEngine("newgame\n");
        CHECK("newgame: contiene 'Base'",
              out.find("Base") != std::string::npos);
        CHECK("newgame: contiene 'InProgress'",
              out.find("InProgress") != std::string::npos);
        CHECK("newgame: contiene 'White[1]'",
              out.find("White[1]") != std::string::npos);
        CHECK("newgame: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- newgame Base ---
    {
        std::string out = RunEngine("newgame Base\n");
        CHECK("newgame Base: contiene 'Base;InProgress;White[1]'",
              out.find("Base;InProgress;White[1]") != std::string::npos);
    }

    // --- newgame Base+MLP ---
    {
        std::string out = RunEngine("newgame Base+MLP\n");
        CHECK("newgame Base+MLP: contiene 'Base+MLP'",
              out.find("Base+MLP") != std::string::npos);
        CHECK("newgame Base+MLP: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- newgame con GameTypeString invalida ---
    {
        std::string out = RunEngine("newgame CosaACaso\n");
        CHECK("newgame invalido: contiene 'err '",
              out.find("err ") != std::string::npos);
    }

    // --- newgame da GameString con 1 mossa ---
    {
        std::string out = RunEngine("newgame Base;InProgress;Black[1];wQ\n");
        CHECK("newgame da GameString: contiene 'Black[1]'",
              out.find("Black[1]") != std::string::npos);
        CHECK("newgame da GameString: contiene 'wQ' nello storico",
              out.find("wQ") != std::string::npos);
        CHECK("newgame da GameString: nessun errore",
              out.find("err ") == std::string::npos);
    }
}

// =============================================================================
// SEZIONE 3: COMANDO VALIDMOVES
// =============================================================================
void TestValidMoves()
{
    PrintSection("Comando: validmoves");

    // --- validmoves senza partita in corso ---
    {
        std::string out = RunEngine("validmoves\n");
        CHECK("validmoves senza partita: contiene 'err '",
              out.find("err ") != std::string::npos);
    }

    // --- validmoves al primo turno (board vuota) ---
    {
        std::string out = RunEngine("newgame\nvalidmoves\n");

        CHECK("validmoves turno 1: contiene 'wQ'",
              out.find("wQ") != std::string::npos);
        CHECK("validmoves turno 1: contiene 'wS1'",
              out.find("wS1") != std::string::npos);
        CHECK("validmoves turno 1: contiene 'wG1'",
              out.find("wG1") != std::string::npos);
        CHECK("validmoves turno 1: NON contiene 'pass'",
              out.find("pass") == std::string::npos);
        CHECK("validmoves turno 1: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- validmoves al secondo turno (dopo prima mossa bianca) ---
    {
        std::string out = RunEngine("newgame\nplay wQ\nvalidmoves\n");

        CHECK("validmoves turno 2: contiene pezzi neri",
              out.find("bQ") != std::string::npos || out.find("bS1") != std::string::npos);
        CHECK("validmoves turno 2: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- validmoves termina sempre con ok ---
    {
        std::string out = RunEngine("newgame\nvalidmoves\n");
        CHECK("validmoves: termina sempre con 'ok'",
              out.find("ok\n") != std::string::npos);
    }
}

// =============================================================================
// SEZIONE 4: COMANDO PLAY
// =============================================================================
void TestPlay()
{
    PrintSection("Comando: play");

    // --- play senza partita in corso ---
    {
        std::string out = RunEngine("play wQ\n");
        CHECK("play senza partita: contiene 'err '",
              out.find("err ") != std::string::npos);
    }

    // --- prima mossa valida ---
    {
        std::string out = RunEngine("newgame\nplay wQ\n");
        CHECK("play wQ: contiene 'InProgress'",
              out.find("InProgress") != std::string::npos);
        CHECK("play wQ: turno passa al nero 'Black[1]'",
              out.find("Black[1]") != std::string::npos);
        CHECK("play wQ: storico contiene 'wQ'",
              out.find("wQ") != std::string::npos);
        CHECK("play wQ: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- mossa invalida (pezzo avversario al turno del bianco) ---
    {
        std::string out = RunEngine("newgame\nplay bQ\n");
        CHECK("play bQ al turno del bianco: contiene 'invalidmove'",
              out.find("invalidmove") != std::string::npos);
    }

    // --- due mosse consecutive ---
    {
        std::string out = RunEngine("newgame\nplay wQ\nplay bQ wQ-\n");
        CHECK("due mosse: contiene 'White[2]'",
              out.find("White[2]") != std::string::npos);
        CHECK("due mosse: storico contiene entrambe le mosse",
              out.find("wQ") != std::string::npos &&
              out.find("bQ") != std::string::npos);
        CHECK("due mosse: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- play pass con mosse disponibili deve dare invalidmove ---
    {
        std::string out = RunEngine("newgame\nplay pass\n");
        CHECK("play pass con mosse disponibili: contiene 'invalidmove'",
              out.find("invalidmove") != std::string::npos);
    }

    // --- sequenza di 4 mosse ---
    {
        std::string out = RunEngine(
            "newgame\n"
            "play wQ\n"
            "play bQ wQ-\n"
            "play wS1 -wQ\n"
            "play bS1 bQ-\n"
        );
        CHECK("4 mosse: turno White[3]",
              out.find("White[3]") != std::string::npos);
        CHECK("4 mosse: nessun errore",
              out.find("err ") == std::string::npos);
    }
}

// =============================================================================
// SEZIONE 5: COMANDO BESTMOVE
// =============================================================================
void TestBestMove()
{
    PrintSection("Comando: bestmove");

    // --- bestmove senza partita ---
    {
        std::string out = RunEngine("bestmove depth 1\n");
        CHECK("bestmove senza partita: contiene 'err '",
              out.find("err ") != std::string::npos);
    }

    // --- bestmove depth al primo turno ---
    {
        std::string out = RunEngine("newgame\nbestmove depth 1\n");

        // Controlliamo che ci sia qualsiasi pezzo bianco (w seguito da una lettera)
        // Il bot potrebbe restituire qualsiasi pezzo: wQ, wS1, wB1, wG1, wA1...
        bool hasWhiteMove = out.find("wQ")  != std::string::npos ||
                            out.find("wS1") != std::string::npos ||
                            out.find("wS2") != std::string::npos ||
                            out.find("wB1") != std::string::npos ||
                            out.find("wB2") != std::string::npos ||
                            out.find("wG1") != std::string::npos ||
                            out.find("wG2") != std::string::npos ||
                            out.find("wG3") != std::string::npos ||
                            out.find("wA1") != std::string::npos ||
                            out.find("wA2") != std::string::npos ||
                            out.find("wA3") != std::string::npos;
        CHECK("bestmove depth 1: restituisce una mossa bianca valida", hasWhiteMove);
        CHECK("bestmove depth 1: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- bestmove time ---
    {
        std::string out = RunEngine("newgame\nbestmove time 00:00:01\n");

        // Cerchiamo qualsiasi pezzo bianco nell'output (prefisso 'w')
        // Usiamo find("w") ma verifichiamo che non sia solo nella parola "White"
        bool hasMove = out.find("wQ")  != std::string::npos ||
                       out.find("wS1") != std::string::npos ||
                       out.find("wS2") != std::string::npos ||
                       out.find("wB1") != std::string::npos ||
                       out.find("wB2") != std::string::npos ||
                       out.find("wG1") != std::string::npos ||
                       out.find("wG2") != std::string::npos ||
                       out.find("wG3") != std::string::npos ||
                       out.find("wA1") != std::string::npos ||
                       out.find("wA2") != std::string::npos ||
                       out.find("wA3") != std::string::npos;
        CHECK("bestmove time: restituisce una mossa bianca", hasMove);
        CHECK("bestmove time: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- bestmove dopo alcune mosse ---
    {
        std::string out = RunEngine(
            "newgame\n"
            "play wQ\n"
            "play bQ wQ-\n"
            "bestmove depth 1\n"
        );
        CHECK("bestmove dopo 2 mosse: nessun errore",
              out.find("err ") == std::string::npos);
        CHECK("bestmove dopo 2 mosse: contiene 'ok'",
              out.find("ok\n") != std::string::npos);
    }

    // --- bestmove termina sempre con ok ---
    {
        std::string out = RunEngine("newgame\nbestmove depth 1\n");
        CHECK("bestmove: termina con 'ok'",
              out.rfind("ok\n") != std::string::npos);
    }
}

// =============================================================================
// SEZIONE 6: COMANDO UNDO
// =============================================================================
void TestUndo()
{
    PrintSection("Comando: undo");

    // --- undo senza partita ---
    {
        std::string out = RunEngine("undo\n");
        CHECK("undo senza partita: contiene 'err '",
              out.find("err ") != std::string::npos);
    }

    // --- undo senza mosse da annullare ---
    {
        std::string out = RunEngine("newgame\nundo\n");
        CHECK("undo senza mosse: contiene 'err '",
              out.find("err ") != std::string::npos);
    }

    // --- undo dopo 1 mossa ---
    {
        std::string out = RunEngine("newgame\nplay wQ\nundo\n");
        CHECK("undo dopo 1 mossa: torna a 'White[1]'",
              CountOccurrences(out, "White[1]") >= 2); // newgame + dopo undo
        CHECK("undo dopo 1 mossa: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- undo 2 dopo 2 mosse ---
    {
        std::string out = RunEngine(
            "newgame\n"
            "play wQ\n"
            "play bQ wQ-\n"
            "undo 2\n"
        );
        CHECK("undo 2: torna a 'White[1]'",
              out.find("White[1]") != std::string::npos);
        CHECK("undo 2: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- undo 1 dopo 2 mosse: torna al turno nero ---
    {
        std::string out = RunEngine(
            "newgame\n"
            "play wQ\n"
            "play bQ wQ-\n"
            "undo 1\n"
        );
        CHECK("undo 1 dopo 2 mosse: torna a 'Black[1]'",
              out.find("Black[1]") != std::string::npos);
        CHECK("undo 1 dopo 2 mosse: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // --- undo troppo grande ---
    {
        std::string out = RunEngine("newgame\nplay wQ\nundo 99\n");
        CHECK("undo 99 con 1 mossa: contiene 'err '",
              out.find("err ") != std::string::npos);
    }

    // --- undo con argomento non numerico ---
    {
        std::string out = RunEngine("newgame\nplay wQ\nundo abc\n");
        CHECK("undo con argomento non numerico: contiene 'err '",
              out.find("err ") != std::string::npos);
    }

    // --- undo e poi rigioca una mossa diversa ---
    {
        std::string out = RunEngine(
            "newgame\n"
            "play wQ\n"
            "undo\n"
            "play wS1\n"
        );
        CHECK("undo e rigioca: contiene 'Black[1]'",
              out.find("Black[1]") != std::string::npos);
        CHECK("undo e rigioca: nessun errore",
              out.find("err ") == std::string::npos);
    }
}

// =============================================================================
// SEZIONE 7: COMANDO OPTIONS
// =============================================================================
void TestOptions()
{
    PrintSection("Comando: options");

    std::string out = RunEngine("options\n");
    CHECK("options: termina con 'ok'",
          out.find("ok\n") != std::string::npos);
    CHECK("options: nessun errore imprevisto",
          out.find("err ") == std::string::npos);
}

// =============================================================================
// SEZIONE 8: COMANDI SCONOSCIUTI
// =============================================================================
void TestUnknownCommand()
{
    PrintSection("Comando sconosciuto");

    {
        std::string out = RunEngine("ciaociao\n");
        CHECK("comando sconosciuto: contiene 'err '",
              out.find("err ") != std::string::npos);
    }

    {
        std::string out = RunEngine("NEWGAME\n"); // UHP è case sensitive
        CHECK("comando maiuscolo: contiene 'err '",
              out.find("err ") != std::string::npos);
    }
}

// =============================================================================
// SEZIONE 9: SESSIONE COMPLETA (partita simulata)
// =============================================================================
void TestFullSession()
{
    PrintSection("Sessione completa");

    std::string out = RunEngine(
        "newgame\n"
        "play wQ\n"
        "play bQ wQ-\n"
        "play wS1 -wQ\n"
        "play bS1 bQ-\n"
        "validmoves\n"
        "bestmove depth 1\n"
        "undo 2\n"
        "validmoves\n"
    );

    CHECK("sessione completa: nessun 'err '",
          out.find("err ") == std::string::npos);
    CHECK("sessione completa: nessun 'invalidmove'",
          out.find("invalidmove") == std::string::npos);
    CHECK("sessione completa: raggiunge White[3]",
          out.find("White[3]") != std::string::npos);
    CHECK("sessione completa: dopo undo 2 torna a White[2]",
          out.find("White[2]") != std::string::npos);

    // info(auto) + newgame + 4*play + validmoves + bestmove + undo + validmoves = 10
    CHECK("sessione completa: almeno 10 'ok'",
          CountOccurrences(out, "ok\n") >= 10);
}

// =============================================================================
// SEZIONE 10: NEWGAME DA GAMESTRING (ricarica partita)
// =============================================================================
void TestNewGameFromGameString()
{
    PrintSection("newgame da GameString");

    // Ricarica con 1 mossa
    {
        std::string out = RunEngine(
            "newgame Base;InProgress;Black[1];wQ\n"
            "validmoves\n"
        );
        CHECK("reload 1 mossa: turno è Black[1]",
              out.find("Black[1]") != std::string::npos);
        CHECK("reload 1 mossa: validmoves genera pezzi neri",
              out.find("bQ") != std::string::npos ||
              out.find("bS1") != std::string::npos);
        CHECK("reload 1 mossa: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // Ricarica con 2 mosse
    {
        std::string out = RunEngine(
            "newgame Base;InProgress;White[2];wQ;bQ wQ-\n"
            "validmoves\n"
        );
        CHECK("reload 2 mosse: turno è White[2]",
              out.find("White[2]") != std::string::npos);
        CHECK("reload 2 mosse: nessun errore",
              out.find("err ") == std::string::npos);
    }

    // GameString con tipo non valido
    {
        std::string out = RunEngine("newgame Invalido;InProgress;White[1]\n");
        CHECK("reload GameString invalida: contiene 'err '",
              out.find("err ") != std::string::npos);
    }
}

// =============================================================================
// SEZIONE 11: ROBUSTEZZA (input malformati)
// =============================================================================
void TestRobustness()
{
    PrintSection("Robustezza input");

    // Righe vuote: non deve crashare
    {
        std::string out = RunEngine("\n\n\nnewgame\n\n");
        CHECK("righe vuote: non crasha",
              out.find("InProgress") != std::string::npos);
    }

    // Play senza argomento
    {
        std::string out = RunEngine("newgame\nplay\n");
        CHECK("play senza argomento: contiene 'invalidmove'",
              out.find("invalidmove") != std::string::npos);
    }

    // Doppio newgame: il secondo deve resettare il primo
    {
        std::string out = RunEngine(
            "newgame\n"
            "play wQ\n"
            "newgame\n"
            "validmoves\n"
        );
        CHECK("doppio newgame: validmoves dopo reset contiene 'wQ'",
              out.find("wQ") != std::string::npos);
        CHECK("doppio newgame: nessun errore",
              out.find("err ") == std::string::npos);
    }
}

// =============================================================================
// MAIN
// =============================================================================
int main()
{
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   HiveGotThis - Test Suite Engine        ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    Board::InitializeZobristTable();

    TestInfo();
    TestNewGame();
    TestValidMoves();
    TestPlay();
    TestBestMove();
    TestUndo();
    TestOptions();
    TestUnknownCommand();
    TestFullSession();
    TestNewGameFromGameString();
    TestRobustness();

    int total = g_passed + g_failed;
    std::cout << "\n╔══════════════════════════════════════════╗\n";
    std::cout << "║  RISULTATI FINALI                        ║\n";
    std::cout << "║  Totale: " << total
              << std::string(32 - std::to_string(total).size(), ' ') << "║\n";
    std::cout << "║  PASS:   " << g_passed
              << std::string(32 - std::to_string(g_passed).size(), ' ') << "║\n";
    std::cout << "║  FAIL:   " << g_failed
              << std::string(32 - std::to_string(g_failed).size(), ' ') << "║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    return (g_failed == 0) ? 0 : 1;
}