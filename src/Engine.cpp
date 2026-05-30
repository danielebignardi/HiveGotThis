#include "Engine.h"
#include "Evaluation.h"
#include "BoardEncoder.h"
#include "MCTS.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>
#include <cstdio>

namespace HiveGotThis
{

// =============================================================================
// COSTRUTTORE
// =============================================================================

Engine::Engine()
    : m_board(nullptr)
{
}

// =============================================================================
// LOOP PRINCIPALE
// =============================================================================

void Engine::Run()
{
    // UHP richiede che al momento dell'avvio il motore stampi automaticamente
    // l'output del comando "info" senza aspettare che il viewer lo chieda.
    CommandInfo();

    std::string line;
    while (std::getline(std::cin, line))
    {
        line = Trim(line);
        if (line.empty()) continue;

        // Splitta: il primo token è il comando, il resto è il parametro
        std::vector<std::string> tokens = Split(line);
        const std::string& command = tokens[0];

        // Ricostruisce i parametri unendo i token rimanenti con spazio
        std::string param;
        for (size_t i = 1; i < tokens.size(); ++i)
        {
            if (i > 1) param += " ";
            param += tokens[i];
        }

        // --- Dispatch dei comandi ---
        if (command == CommandString_Info)
        {
            CommandInfo();
        }
        else if (command == CommandString_NewGame)
        {
            CommandNewGame(param);
        }
        else if (command == CommandString_ValidMoves)
        {
            if (m_board == nullptr)
                WriteError(ErrorMessage_NoGameInProgress);
            else if (GameIsOver(m_board->GetBoardState()))
                WriteError(ErrorMessage_GameIsOver);
            else
                CommandValidMoves();
        }
        else if (command == CommandString_BestMove)
        {
            if (m_board == nullptr)
                WriteError(ErrorMessage_NoGameInProgress);
            else if (GameIsOver(m_board->GetBoardState()))
                WriteError(ErrorMessage_GameIsOver);
            else
                CommandBestMove(param);
        }
        else if (command == CommandString_Play)
        {
            if (m_board == nullptr)
                WriteError(ErrorMessage_NoGameInProgress);
            else if (GameIsOver(m_board->GetBoardState()))
                WriteError(ErrorMessage_GameIsOver);
            else
                CommandPlay(param);
        }
        else if (command == CommandString_Pass)
        {
            if (m_board == nullptr)
                WriteError(ErrorMessage_NoGameInProgress);
            else if (GameIsOver(m_board->GetBoardState()))
                WriteError(ErrorMessage_GameIsOver);
            else
                CommandPass();
        }
        else if (command == CommandString_Undo)
        {
            if (m_board == nullptr)
                WriteError(ErrorMessage_NoGameInProgress);
            else
                CommandUndo(param);
        }
        else if (command == CommandString_Options)
        {
            CommandOptions(param);
        }
        else if (command == CommandString_Features)
        {
            if (m_board == nullptr)
                WriteError(ErrorMessage_NoGameInProgress);
            else
                CommandFeatures();
        }
        else if (command == CommandString_Exit)
        {
            break;
        }
        else
        {
            WriteError(ErrorMessage_InvalidCommand);
        }
    }
}

// =============================================================================
// GESTIONE COMANDI UHP
// =============================================================================

void Engine::CommandInfo()
{
    // Formato UHP:
    //   id <NomeMotore>
    //   <Capacità separate da ;>
    //   ok
    std::cout << IdString << "\n"; //Ricorda: IdString e CapabilitiesString sono definiti in Constants.h
    std::cout << CapabilitiesString << "\n";
    WriteOk();
}

void Engine::CommandNewGame(const std::string& param)
{
    ResetGame();

    GameType gt = GameType::Base; // Default: partita base senza espansioni

    if (!param.empty())
    {
        // Il parametro può essere:
        //   1. Una GameTypeString senza ';' (es. "Base+MLP")
        //   2. Una GameString completa con ';' (es. "Base;InProgress;White[3];wS1;bG1 -wS1")
        if (param.find(';') != std::string::npos) //se c'è ';', è sicuramente il caso 2, altrimenti caso 1
        {
            // --- CASO 2: GameString completa ---
            // Splittiamo per ';' per ottenere le parti
            std::vector<std::string> parts;
            std::istringstream iss(param);//trasforma la stringa in uno stream per poter usare getline con ';' come delimitatore
            std::string token;
            while (std::getline(iss, token, ';'))
                parts.push_back(Trim(token));//la funzione Trim rimuove spazi iniziali/finali, utile per evitare problemi di parsing se ci sono spazi extra intorno ai ';'

            if (parts.size() < 3) //nel caso 2 ci aspettiamo almeno 3 parti: GameTypeString, GameStateString, TurnString. Le mosse sono opzionali (possono essere 0 o più)
            {
                WriteError("GameString non valida.");
                return;
            }

            // parts[0] = GameTypeString
            gt = GetGameTypeValue(parts[0].c_str());//c_str() converte la stringa in un array di char*, necessario per la funzione GetGameTypeValue che si aspetta un const char* definita in Enum.cpp
            if (gt == GameType::INVALID)
            {
                WriteError("GameTypeString non riconosciuta: " + parts[0]);
                return;
            }

            m_board = new Board(gt);
            // boardState parte da NotStarted (costruttore).
            // Il primo ApplyMove chiamerà ApplyTurnEffects() che porta a InProgress.

            // parts[1] = GameStateString  -> ignorato, lo ricalcoliamo
            // parts[2] = TurnString       -> ignorato, lo ricaviamo dalle mosse

            // parts[3..N] = mosse da riprodurre in ordine
            for (size_t i = 3; i < parts.size(); ++i)
            {
                if (parts[i].empty()) continue;

                if (!ApplyMove(parts[i]))
                {
                    WriteError("Mossa non valida nello storico: " + parts[i]);
                    ResetGame();
                    return;
                }
            }

        }
        else
        {
            // --- CASO 1: solo GameTypeString ---
            gt = GetGameTypeValue(param.c_str());
            if (gt == GameType::INVALID)
            {
                WriteError("GameTypeString non riconosciuta: " + param);
                return;
            }
            m_board = new Board(gt);
            // m_board->StartGame(); // BUG UHP: setterebbe InProgress su una board vuota (0 mosse); deve restare NotStarted
        }
    }
    else
    {
        // Nessun parametro: partita base standard
        m_board = new Board(gt);
        // m_board->StartGame(); // BUG UHP: setterebbe InProgress su una board vuota (0 mosse); deve restare NotStarted
    }

    std::cout << BuildGameString() << "\n";
    WriteOk();
}

void Engine::CommandValidMoves()
{
    // Chiediamo direttamente alla Board le mosse valide.
    // GetValidMoves gestisce già: piazzamenti, movimenti, obbligo regina,
    // cannotBeMoved e aggiunge PassMove se non ci sono altre mosse.
    std::vector<Move> moves;
    m_board->GetValidMoves(moves);

    // Stampa le mosse separate da ';' su una sola riga
    for (size_t i = 0; i < moves.size(); ++i)
    {
        if (i > 0) std::cout << ";";
        std::cout << MoveToMoveString(moves[i]);
    }
    std::cout << "\n";

    WriteOk();
}

void Engine::CommandBestMove(const std::string& param)
{
    // Genera tutte le mosse valide
    std::vector<Move> moves;
    m_board->GetValidMoves(moves);

    if (moves.empty())
    {
        // Non dovrebbe mai succedere: GetValidMoves aggiunge sempre PassMove
        std::cout << PassMoveString << "\n";
        WriteOk();
        return;
    }

    /*
    // --- SELEZIONE CASUALE (VECCHIA IMPLEMENTAZIONE) ---
    // Per ora il motore sceglie una mossa a caso tra quelle disponibili.
    // In futuro qui andrà l'algoritmo di ricerca (Minimax, MCTS, ecc.)
    // Il parametro "time hh:mm:ss" o "depth N" viene ignorato per ora.
    static std::mt19937 rng(
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count())
    );
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);

    std::cout << MoveToMoveString(moves[dist(rng)]) << "\n";
    WriteOk();
    */

    // ===================== MCTS =====================

    // Se c'e' una sola mossa legale, giocala subito senza cercare
    if (moves.size() == 1)
    {
        std::cout << MoveToMoveString(moves[0]) << "\n";
        WriteOk();
        return;
    }

    // Parsa il parametro "time HH:MM:SS" o "depth N"
    std::vector<std::string> tokens = Split(param);
    int timeLimitMs = 5000; // 5 secondi

    if (tokens.size() >= 2)
    {
        if (tokens[0] == "time")
        {
            int h = 0, m = 0, s = 0;
            std::sscanf(tokens[1].c_str(), "%d:%d:%d", &h, &m, &s);
            timeLimitMs = (h * 3600 + m * 60 + s) * 1000;
            // Usa il 95% del tempo per lasciare margine
            timeLimitMs = static_cast<int>(timeLimitMs * 0.95);
            if (timeLimitMs < 100) timeLimitMs = 100;
        }
        else if (tokens[0] == "depth")
        {
            int depth = std::stoi(tokens[1]);
            Move best = MCTS::SearchIterations(*m_board, depth * 1000);
            std::cout << MoveToMoveString(best) << "\n";
            WriteOk();
            return;
        }
    }

    Move best = MCTS::Search(*m_board, timeLimitMs);
    std::cout << MoveToMoveString(best) << "\n";
    WriteOk();

    // ================================================
}
/*
void Engine::CommandPlay(const std::string& moveString)
{
    if (moveString.empty())
    {
        WriteInvalidMove(InvalidMoveMessage_Generic);
        return;
    }

    // Gestione "play pass" come alias di "pass"
    if (moveString == PassMoveString)
    {
        CommandPass();
        return;
    }

    // Controlla che la mossa sia tra quelle valide
    Move mv = MoveStringToMove(moveString);
    if (mv.Piece == PieceName::INVALID)
    {
        WriteInvalidMove("Impossibile interpretare la mossa: " + moveString);
        return;
    }

    std::vector<Move> validMoves;
    m_board->GetValidMoves(validMoves);

    bool found = false;
    for (const Move& vm : validMoves)
    {
        if (vm.Piece == mv.Piece && vm.Destination == mv.Destination)
        {
            found = true;
            break;
        }
    }

    if (!found)
    {
        WriteInvalidMove(InvalidMoveMessage_Generic);
        return;
    }

    // Applica la mossa
    ApplyMove(moveString);

    std::cout << BuildGameString() << "\n";
    WriteOk();
}*/

void Engine::CommandPlay(const std::string& moveString)
{
    if (moveString.empty()) { WriteInvalidMove(InvalidMoveMessage_Generic); return; }
    if (moveString == PassMoveString) { CommandPass(); return; }

    Move mv = MoveStringToMove(moveString);
    if (mv.Piece == PieceName::INVALID)
    {
        WriteInvalidMove("Impossibile interpretare la mossa: " + moveString);
        return;
    }

    std::vector<Move> validMoves;
    m_board->GetValidMoves(validMoves);

    bool found = false;
    for (const Move& vm : validMoves)
    {
        if (vm.Piece == mv.Piece && vm.Destination == mv.Destination)
        {
            found = true;
            mv = vm; // usa la Move interna (Source corretto)
            break;
        }
    }

    if (!found) { WriteInvalidMove(InvalidMoveMessage_Generic); return; }

    // Normalizza la stringa PRIMA di applicarla
    std::string normalizedMoveString = MoveToMoveString(mv);
    ApplyMove(normalizedMoveString);  // stringa canonica, non quella del viewer

    std::cout << BuildGameString() << "\n";
    WriteOk();
}

void Engine::CommandPass()
{
    // Il pass è valido solo se GetValidMoves restituisce solo PassMove
    std::vector<Move> validMoves;
    m_board->GetValidMoves(validMoves);

    bool passIsValid = false;
    for (const Move& vm : validMoves)
    {
        if (vm == PassMove)
        {
            passIsValid = true;
            break;
        }
    }

    if (!passIsValid)
    {
        WriteInvalidMove("Non puoi passare quando hai mosse disponibili.");
        return;
    }

    ApplyMove(PassMoveString);

    std::cout << BuildGameString() << "\n";
    WriteOk();
}

void Engine::CommandUndo(const std::string& param)
{
    // Quante mosse annullare (default 1)
    int movesToUndo = 1;
    if (!param.empty())
    {
        try { movesToUndo = std::stoi(param); }
        catch (...) { WriteError("Parametro undo non valido: " + param); return; }
    }

    if (movesToUndo < 1 || movesToUndo > (int)m_moveHistory.size())
    {
        WriteError(ErrorMessage_UnableToUndo);
        return;
    }

    // Strategia: salva il tipo di partita e le mosse rimanenti,
    // poi ricostruisce la partita dall'inizio rigiocandole.
    GameType gt = m_board->gameType;
    std::vector<std::string> remaining(
        m_moveHistory.begin(),
        m_moveHistory.end() - movesToUndo
    );

    ResetGame();
    m_board = new Board(gt);

    if (!remaining.empty())
    {
        // m_board->StartGame(); // ridondante: il primo ApplyMove chiama ApplyTurnEffects() → InProgress
        for (const std::string& ms : remaining)
        {
            if (!ApplyMove(ms))
            {
                WriteError("Errore durante la ricostruzione dopo undo.");
                ResetGame();
                m_board = new Board(gt);
                return;
            }
        }
    }

    std::cout << BuildGameString() << "\n";
    WriteOk();
}

void Engine::CommandOptions(const std::string& param)
{
    // Nessuna opzione configurabile per ora.
    (void)param;
    WriteOk();
}
void Engine::CommandFeatures()
{
    GNNGraph graph = BoardEncoder::encode(*m_board);
    std::cout << GNNGraphToJson(graph) << "\n";
    WriteOk();
}

// =============================================================================
// LOGICA DI GIOCO INTERNA
// =============================================================================

bool Engine::ApplyMove(const std::string& moveString)
{
    if (moveString == PassMoveString)
    {
        m_board->ApplyMove(PassMove);
        m_moveHistory.push_back(moveString);
        return true;
    }

    Move mv = MoveStringToMove(moveString);
    if (mv.Piece == PieceName::INVALID) return false;

    m_board->ApplyMove(mv);
    m_moveHistory.push_back(moveString);
    return true;
}

// =============================================================================
// CONVERSIONE MOVESTRING <-> MOVE
// =============================================================================
//Move -> string
std::string Engine::MoveToMoveString(const Move& move) const
{
    // Caso speciale: PassMove
    if (move == PassMove) return PassMoveString;

    std::ostringstream ss;
    ss << GetEnumString(move.Piece);

    // Verifica se la board era vuota prima di questo piazzamento
    // (cioè: l'unico pezzo in gioco è quello che stiamo piazzando ora,
    //  oppure la board è ancora completamente vuota)
    bool boardIsEmpty = true;
    for (int i = 0; i < NumPieceNames; ++i)
    {
        PieceName p = static_cast<PieceName>(i);
        if (p == move.Piece) continue;
        if (m_board->PieceInPlay(p)) { boardIsEmpty = false; break; }
    }

    // Primo piazzamento: solo il nome del pezzo, senza posizione relativa
    if (boardIsEmpty && move.Source == NullIndex)
        return ss.str();

    // Tutti gli altri casi: aggiungi la posizione relativa
    std::string relPos = PositionToRelativeString(move.Destination);
    if (!relPos.empty())
        ss << " " << relPos;

    return ss.str();
}

std::string Engine::PositionToRelativeString(Index pos) const
{
    // Cerca un pezzo adiacente alla destinazione e costruisce la notazione UHP.
    //
    // Mappa direzione (DA pos VERSO il pezzo) -> separatore UHP:
    //
    //   Pezzo in direzione Right(0)     -> "PieceName-"   (separatore DOPO)
    //   Pezzo in direzione DownRight(1) -> "PieceName\"   (separatore DOPO)
    //   Pezzo in direzione DownLeft(2)  -> "/PieceName"   (separatore PRIMA)
    //   Pezzo in direzione Left(3)      -> "-PieceName"   (separatore PRIMA)
    //   Pezzo in direzione UpLeft(4)    -> "\PieceName"   (separatore PRIMA)
    //   Pezzo in direzione UpRight(5)   -> "PieceName/"   (separatore DOPO)
    
    // Destinazione con un pezzo sopra: notazione "PieceName" senza separatore
    PieceName onTop = m_board->GetPieceAt(pos);
    if (onTop != PieceName::INVALID)
    {
        return GetEnumString(onTop); // restituisce "wQ" senza separatore
    }
    static const char separators[6] = { '-', '\\', '/', '-', '\\', '/' };
    static const bool afterPiece[6] = { true, true, false, false, false, true };
    // Priorità di scelta del pezzo di riferimento: Right(0) > DownRight(1) > ... > UpRight(5)
    // Questa priorità è fissa e garantisce che MoveToMoveString sia deterministica:
    // la stessa (pezzo, destinazione) produce sempre la stessa stringa UHP.
    for (int d = 0; d < 6; ++d)
    {
        Index neighborPos = GetNeighborAt(pos, static_cast<Direction>(d));
        if (!IsValidIndex(neighborPos)) continue;

        PieceName np = m_board->GetPieceAt(neighborPos);
        if (np == PieceName::INVALID) continue;

        // Trovato un pezzo adiacente: costruisci la notazione
        std::string pieceName = GetEnumString(np);
        char sep = separators[static_cast<int>(Opposite(static_cast<Direction>(d)))];

        std::ostringstream ss;
        if (afterPiece[static_cast<int>(Opposite(static_cast<Direction>(d)))])
            ss << pieceName << sep;   // es. "wS1-"
        else
            ss << sep << pieceName;   // es. "-wS1"

        return ss.str();
    }

    return ""; // Nessun pezzo adiacente (non dovrebbe succedere per mosse valide)
}
//string -> Move
Move Engine::MoveStringToMove(const std::string& moveString) const
{
    Move invalid = {PieceName::INVALID, NullIndex, NullIndex};

    if (moveString.empty()) return invalid;
    if (moveString == PassMoveString) return PassMove;

    // Usa il parser già esistente in Move.cpp
    bool isPass;
    PieceName startPiece;
    char beforeSep;
    PieceName endPiece;
    char afterSep;
    //Ricorda: i parametri sono passati per riferimento e valorizzati da TryNormalizeMoveString, che restituisce true se la conversione è riuscita, false altrimenti. Se la conversione fallisce, restituiamo una mossa INVALID.
    if (!TryNormalizeMoveString(moveString, isPass, startPiece, beforeSep, endPiece, afterSep))
        return invalid;

    if (isPass) return PassMove;
    if (startPiece == PieceName::INVALID) return invalid;

    // Nessun pezzo di riferimento: primo piazzamento al centro della board
    if (endPiece == PieceName::INVALID)
        return Move{startPiece, NullIndex, BoardCenter};

    // Trova la posizione del pezzo di riferimento
    Index refPos = m_board->GetPosition(endPiece);
    if (refPos == NullIndex) return invalid;

    // Calcola la destinazione dai separatori.
    // La logica è l'inverso di PositionToRelativeString:
    //   "ref-"  -> dest è a DESTRA di ref      (Direction::Right)
    //   "ref/"  -> dest è in alto-DESTRA       (Direction::UpRight)
    //   "ref\"  -> dest è in BASSO-destra     (Direction::DownRight)
    //   "-ref"  -> dest è a SINISTRA di ref     (Direction::Left)
    //   "/ref"  -> dest è in basso-SINISTRA      (Direction::DownLeft)
    //   "\ref"  -> dest è in ALTO-Sinistra       (Direction::UpLeft)
    //   nessun separatore -> Beetle che sale sopra (dest == refPos)
    Index dest = NullIndex;

    if (beforeSep == '\0' && afterSep == '\0')
        dest = refPos;                                              // Beetle sopra
    else if (afterSep == '-')
        dest = GetNeighborAt(refPos, Direction::Right);
    else if (afterSep == '/')
        dest = GetNeighborAt(refPos, Direction::UpRight);
    else if (afterSep == '\\')
        dest = GetNeighborAt(refPos, Direction::DownRight);
    else if (beforeSep == '-')
        dest = GetNeighborAt(refPos, Direction::Left);
    else if (beforeSep == '/')
        dest = GetNeighborAt(refPos, Direction::DownLeft);
    else if (beforeSep == '\\')
        dest = GetNeighborAt(refPos, Direction::UpLeft);

    if (!IsValidIndex(dest)) return invalid;

    return Move{startPiece, m_board->GetPosition(startPiece), dest};
}

// =============================================================================
// COSTRUZIONE GAMESTRING UHP
// =============================================================================

//STAMPA LA GAMESTRING COMPLETA PER LO STATO CORRENTE DELLA BOARD, INCLUSO LO STORICO MOSSE. IL FORMATO UHP È:
// "GameTypeString;GameStateString;TurnString[;Move1;Move2
std::string Engine::BuildGameString() const
{
    if (m_board == nullptr) return "";

    std::ostringstream ss;
    ss << BuildGameTypeString() << ";";
    ss << GetEnumString(m_board->GetBoardState()) << ";";
    ss << BuildTurnString();

    for (const std::string& ms : m_moveHistory)
        ss << ";" << ms;

    return ss.str();
}

std::string Engine::BuildTurnString() const
{
    // currentTurn parte da 1 e aumenta ad ogni mossa (bianco e nero).
    // Il "round" UHP (il numero tra parentesi quadre) è:
    //   round = (currentTurn + 1) / 2
    //
    // Esempi:
    //   currentTurn=1 (primo turno bianco)  -> White[1]
    //   currentTurn=2 (primo turno nero)    -> Black[1]
    //   currentTurn=3 (secondo turno bianco)-> White[2]
    //   currentTurn=4 (secondo turno nero)  -> Black[2]
    int round = (m_board->GetCurrentTurn()/2) + 1;

    std::ostringstream ss;
    ss << GetEnumString(m_board->currentColor) << "[" << round << "]";
    return ss.str();
}

std::string Engine::BuildGameTypeString() const
{
    return GetEnumString(m_board->gameType);
}

// =============================================================================
// UTILITY
// =============================================================================

void Engine::WriteError(const std::string& message) const
{
    std::cout << ErrString << " " << message << "\n";
    WriteOk();
}

void Engine::WriteInvalidMove(const std::string& message) const
{
    std::cout << InvalidMoveString << " " << message << "\n";
    WriteOk();
}

void Engine::WriteOk() const
{
    std::cout << OkString << "\n";
    std::cout.flush(); // Fondamentale: il viewer aspetta l'ok prima di procedere
}

std::string Engine::Trim(const std::string& s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::vector<std::string> Engine::Split(const std::string& s)
{
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string token;
    while (iss >> token)
        tokens.push_back(token);
    return tokens;
}

void Engine::ResetGame()
{
    delete m_board;
    m_board = nullptr;
    m_moveHistory.clear();
}

} // namespace HiveGotThis
