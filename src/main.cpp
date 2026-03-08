#include "Engine.h"
#include "Board.h"

int main()
{
    // Inizializza la tabella Zobrist una volta sola all'avvio
    HiveGotThis::Board::InitializeZobristTable();

    // Avvia l'engine: legge comandi da stdin e scrive risposte su stdout.
    // L'engine stampa automaticamente "info" all'avvio, poi aspetta comandi.
    // Termina quando riceve il comando "exit".
    HiveGotThis::Engine engine;
    engine.Run();

    return 0;
}