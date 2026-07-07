#include "Engine.h"
#include "Board.h"
#include "MCTS.h"
#include "NeuralEvaluator.h"

#include <cstdlib>
#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
    // Inizializza la tabella Zobrist una volta sola all'avvio
    HiveGotThis::Board::InitializeZobristTable();

    // Il motore richiede sempre una value network: nessun fallback euristico.
    // Path del modello da riga di comando, altrimenti da HIVE_VALUE_NETWORK.
    const char* modelPath = argc > 1 ? argv[1] : std::getenv("HIVE_VALUE_NETWORK");
    if (modelPath == nullptr)
    {
        std::cerr << "Errore: serve il path della value network "
                   << "(argomento da riga di comando oppure variabile HIVE_VALUE_NETWORK)." << std::endl;
        return 1;
    }

    std::unique_ptr<HiveGotThis::TorchScriptValueEvaluator> valueNetwork;
    try
    {
        valueNetwork = std::make_unique<HiveGotThis::TorchScriptValueEvaluator>(modelPath);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Errore caricando la value network (" << modelPath << "): " << e.what() << std::endl;
        return 1;
    }

    HiveGotThis::MCTS::SetValueNetwork(valueNetwork.get());
    std::cerr << "Value network caricata da: " << modelPath << std::endl;

    // Avvia l'engine: legge comandi da stdin e scrive risposte su stdout.
    // L'engine stampa automaticamente "info" all'avvio, poi aspetta comandi.
    // Termina quando riceve il comando "exit".
    HiveGotThis::Engine engine;
    engine.Run();

    return 0;
}