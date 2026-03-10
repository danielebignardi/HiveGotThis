#ifndef ENGINE_H
#define ENGINE_H

#include "Board.h"
#include "Move.h"
#include "Enums.h"
#include "Constants.h"

#include <string>
#include <vector>
#include <sstream>

namespace HiveGotThis
{

// =============================================================================
// ENGINE
// =============================================================================
// La classe Engine implementa il protocollo UHP (Universal Hive Protocol).
// Si occupa di:
//   1. Parsare i comandi testuali ricevuti da stdin
//   2. Mantenere lo stato della partita corrente (via Board)
//   3. Produrre le risposte testuali da mandare a stdout
//   4. Fornire una mossa "migliore" (per ora scelta casualmente)
// =============================================================================
class Engine
{
public:
    // -------------------------------------------------------------------------
    // COSTRUTTORE
    // -------------------------------------------------------------------------
    Engine();
    ~Engine() = default;

    // -------------------------------------------------------------------------
    // LOOP PRINCIPALE
    // -------------------------------------------------------------------------
    // Avvia il loop di lettura comandi da stdin.
    // All'avvio stampa automaticamente l'output del comando "info" (spec UHP).
    // Termina quando riceve il comando "exit".
    void Run();

private:
    // -------------------------------------------------------------------------
    // STATO INTERNO
    // -------------------------------------------------------------------------

    // Il board corrente (nullptr se nessuna partita è in corso)
    Board* m_board;

    // Storico delle mosse giocate nella partita corrente, in ordine cronologico.
    // Ogni elemento è la stringa UHP della mossa (es. "wS1", "bG1 wS1-", "pass")
    std::vector<std::string> m_moveHistory;

    // -------------------------------------------------------------------------
    // GESTIONE COMANDI UHP
    // -------------------------------------------------------------------------

    // "info" - Restituisce l'identificatore del motore e le sue capacità
    void CommandInfo();

    // "newgame [GameTypeString|GameString]" - Avvia una nuova partita
    void CommandNewGame(const std::string& param);

    // "validmoves" - Elenca tutte le mosse valide nel formato UHP
    void CommandValidMoves();

    // "bestmove time hh:mm:ss" oppure "bestmove depth N"
    // Per ora restituisce una mossa casuale tra quelle valide
    void CommandBestMove(const std::string& param);

    // "play MoveString" - Gioca la mossa specificata
    void CommandPlay(const std::string& moveString);

    // "pass" - Gioca una mossa di passaggio (equivalente a "play pass")
    void CommandPass();

    // "undo [N]" - Annulla le ultime N mosse (default 1)
    void CommandUndo(const std::string& param);

    // "options [get/set ...]" - Gestione opzioni (per ora vuota)
    void CommandOptions(const std::string& param);

    // -------------------------------------------------------------------------
    // LOGICA DI GIOCO INTERNA
    // -------------------------------------------------------------------------

    // Applica una mossa alla board aggiornando turno, hash, cannotBeMoved e storico.
    // Usata sia da CommandPlay che da CommandNewGame (riproduzione storico).
    // Non stampa nulla. Restituisce false se la mossa non è parsabile.
    bool ApplyMove(const std::string& moveString);

    // -------------------------------------------------------------------------
    // CONVERSIONE POSIZIONE <-> MOVESTRING UHP
    // -------------------------------------------------------------------------

    // Converte una mossa interna (struct Move) in una stringa UHP.
    // Esempio: Move{wS1, NullIndex, BoardCenter} -> "wS1"
    // Esempio: Move{bG1, NullIndex, pos}         -> "bG1 wS1-"
    std::string MoveToMoveString(const Move& move) const;

    // Dato un indice di destinazione, restituisce la stringa di posizione relativa UHP
    // rispetto ai pezzi già sulla board. Es: "wS1-", "/bQ", ecc.
    std::string PositionToRelativeString(Index pos) const;

    // Converte una MoveString UHP in una mossa interna.
    // Restituisce un Move con Piece==INVALID se la conversione fallisce.
    Move MoveStringToMove(const std::string& moveString) const;

    // -------------------------------------------------------------------------
    // COSTRUZIONE GAMESTRING UHP
    // -------------------------------------------------------------------------

    // Costruisce la GameString completa per lo stato corrente della board.
    // Formato: "GameTypeString;GameStateString;TurnString[;Move1;Move2;...]"
    std::string BuildGameString() const;

    // Costruisce la TurnString (es. "White[3]")
    // Nota: currentTurn parte da 1 e aumenta ad ogni mossa (non solo al turno bianco).
    // Il numero nel TurnString UHP è (currentTurn + 1) / 2, ovvero il "round".
    std::string BuildTurnString() const;

    // Costruisce la GameTypeString (es. "Base", "Base+MLP")
    std::string BuildGameTypeString() const;

    // -------------------------------------------------------------------------
    // UTILITY
    // -------------------------------------------------------------------------

    // Stampa un messaggio di errore UHP ("err <msg>\nok\n")
    void WriteError(const std::string& message) const;

    // Stampa un messaggio di mossa invalida UHP ("invalidmove <msg>\nok\n")
    void WriteInvalidMove(const std::string& message) const;

    // Stampa "ok\n" con flush (fondamentale per il viewer)
    void WriteOk() const;

    // Rimuove spazi iniziali/finali da una stringa
    static std::string Trim(const std::string& s);

    // Splitta una stringa in token separati da spazi
    static std::vector<std::string> Split(const std::string& s);

    // Azzera lo stato della partita (dealloca board, svuota storico)
    void ResetGame();
};

} // namespace HiveGotThis

#endif // ENGINE_H