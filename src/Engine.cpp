#include "Engine.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <stdexcept>

namespace HiveGotThis
{

// =============================================================================
// COSTRUTTORE / DISTRUTTORE
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
    // UHP richiede che al momento dell'avvio, il motore stampi automaticamente
    // l'output del comando "info" senza aspettare che il viewer lo chieda.
    CommandInfo();

    std::string line;
    while (std::getline(std::cin, line))
    {
        line = Trim(line);

        // Ignora righe vuote
        if (line.empty()) continue;

        // Splitta la riga in tokens: il primo è il comando, il resto sono parametri
        std::vector<std::string> tokens = Split(line);
        const std::string& command = tokens[0];

        // Ricostruisce i parametri (tutto ciò che viene dopo il primo token)
        std::string param = "";
        if (tokens.size() > 1)
        {
            // Ricostruisce la stringa dei parametri unendo i token con spazio
            std::ostringstream ss;
            for (size_t i = 1; i < tokens.size(); ++i)
            {
                if (i > 1) ss << " ";
                ss << tokens[i];
            }
            param = ss.str();
        }

        // Dispatch del comando
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
            // validmoves non accetta parametri
            if (m_board == nullptr)
            {
                WriteError(ErrorMessage_NoGameInProgress);
            }
            else if (GameIsOver(m_board->boardState))
            {
                WriteError(ErrorMessage_GameIsOver);
            }
            else
            {
                CommandValidMoves();
            }
        }
        else if (command == CommandString_BestMove)
        {
            if (m_board == nullptr)
            {
                WriteError(ErrorMessage_NoGameInProgress);
            }
            else if (GameIsOver(m_board->boardState))
            {
                WriteError(ErrorMessage_GameIsOver);
            }
            else
            {
                CommandBestMove(param);
            }
        }
        else if (command == CommandString_Play)
        {
            if (m_board == nullptr)
            {
                WriteError(ErrorMessage_NoGameInProgress);
            }
            else if (GameIsOver(m_board->boardState))
            {
                WriteError(ErrorMessage_GameIsOver);
            }
            else
            {
                CommandPlay(param);
            }
        }
        else if (command == CommandString_Pass)
        {
            if (m_board == nullptr)
            {
                WriteError(ErrorMessage_NoGameInProgress);
            }
            else if (GameIsOver(m_board->boardState))
            {
                WriteError(ErrorMessage_GameIsOver);
            }
            else
            {
                CommandPass();
            }
        }
        else if (command == CommandString_Undo)
        {
            if (m_board == nullptr)
            {
                WriteError(ErrorMessage_NoGameInProgress);
            }
            else
            {
                CommandUndo(param);
            }
        }
        else if (command == CommandString_Options)
        {
            CommandOptions(param);
        }
        else if (command == CommandString_Exit)
        {
            // Termina il loop (e il programma)
            break;
        }
        else
        {
            // Comando sconosciuto
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
    std::cout << IdString << "\n";
    std::cout << CapabilitiesString << "\n";
    WriteOk();
}

void Engine::CommandNewGame(const std::string& param)
{
    // Azzera lo stato precedente
    ResetGame();

    GameType gt = GameType::Base; // Default: partita base senza espansioni

    if (!param.empty())
    {
        // Il parametro può essere:
        //   1. Una GameTypeString (es. "Base+MLP")
        //   2. Una GameString completa (es. "Base;InProgress;White[3];wS1;bG1 -wS1")
        //
        // Distinguiamo i due casi cercando il carattere ';'
        if (param.find(';') != std::string::npos)
        {
            // --- CASO 2: GameString completa ---
            // Splittiamo per ';'
            std::vector<std::string> parts;
            std::istringstream iss(param);
            std::string token;
            while (std::getline(iss, token, ';'))
            {
                parts.push_back(Trim(token));
            }

            if (parts.size() < 3)
            {
                WriteError("GameString non valida.");
                return;
            }

            // parts[0] = GameTypeString
            gt = GetGameTypeValue(parts[0].c_str());
            if (gt == GameType::INVALID)
            {
                WriteError("GameTypeString non riconosciuta: " + parts[0]);
                return;
            }

            m_board = new Board(gt);

            // parts[1] = GameStateString (non usato direttamente, la ricalcoliamo)
            // parts[2] = TurnString (non usato direttamente, lo ricaviamo dalle mosse)

            // parts[3..N] = storia delle mosse da riprodurre
            for (size_t i = 3; i < parts.size(); ++i)
            {
                const std::string& ms = parts[i];
                if (ms.empty()) continue;

                // Gioca la mossa come se arrivasse dal comando "play"
                // (riutilizziamo la logica interna senza stampare output intermedi)
                if (ms == PassMoveString)
                {
                    // Mossa di passaggio: cambia semplicemente il turno
                    m_board->currentColor = (m_board->currentColor == Color::White) ? Color::Black : Color::White;
                    if (m_board->currentColor == Color::White)
                        m_board->currentTurn++;
                    m_board->ToggleTurnHash();
                    m_moveHistory.push_back(ms);
                }
                else
                {
                    Move mv = MoveStringToMove(ms);
                    if (mv.Piece == PieceName::INVALID)
                    {
                        WriteError("Mossa non valida nello storico: " + ms);
                        ResetGame();
                        return;
                    }
                    m_board->MovePiece(mv.Piece, mv.Destination);

                    // Aggiorna turno
                    Color prev = m_board->currentColor;
                    m_board->currentColor = (prev == Color::White) ? Color::Black : Color::White;
                    if (m_board->currentColor == Color::White)
                        m_board->currentTurn++;
                    m_board->ToggleTurnHash();
                    m_board->boardState = BoardState::InProgress;
                    m_moveHistory.push_back(ms);
                }
            }

            // Ricalcola lo stato finale
            UpdateBoardState();
        }
        else
        {
            // --- CASO 1: GameTypeString ---
            gt = GetGameTypeValue(param.c_str());
            if (gt == GameType::INVALID)
            {
                WriteError("GameTypeString non riconosciuta: " + param);
                return;
            }
            m_board = new Board(gt);
        }
    }
    else
    {
        // Nessun parametro: partita base standard
        m_board = new Board(gt);
    }

    // Stampa la GameString risultante
    std::cout << BuildGameString() << "\n";
    WriteOk();
}

void Engine::CommandValidMoves()
{
    // Genera tutte le mosse valide
    std::vector<Move> moves;
    GenerateValidMoves(moves);

    if (moves.empty())
    {
        // Nessuna mossa disponibile (dovrebbe essere solo "pass")
        std::cout << PassMoveString << "\n";
    }
    else
    {
        // Formato UHP: mosse separate da ';' su una sola riga
        for (size_t i = 0; i < moves.size(); ++i)
        {
            if (i > 0) std::cout << ";";
            std::cout << MoveToMoveString(moves[i]);
        }
        std::cout << "\n";
    }

    WriteOk();
}

void Engine::CommandBestMove(const std::string& param)
{
    // Genera tutte le mosse valide
    std::vector<Move> moves;
    GenerateValidMoves(moves);

    if (moves.empty())
    {
        // Nessuna mossa: passa
        std::cout << PassMoveString << "\n";
        WriteOk();
        return;
    }

    // --- SELEZIONE CASUALE ---
    // Per ora il motore sceglie una mossa a caso tra quelle disponibili.
    // In futuro qui andrà l'algoritmo di ricerca (Minimax, MCTS, ecc.)
    //
    // Usiamo il tempo corrente come seed per avere casualità diversa ad ogni esecuzione.
    static std::mt19937 rng(
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count())
    );
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    const Move& chosen = moves[dist(rng)];

    // Il parametro indica il tipo di limite:
    //   "time hh:mm:ss"  -> limite temporale (per ora ignorato, risponde subito)
    //   "depth N"        -> limite di profondità (per ora ignorato)
    // In entrambi i casi restituiamo la mossa casuale.
    std::cout << MoveToMoveString(chosen) << "\n";
    WriteOk();
}

void Engine::CommandPlay(const std::string& moveString)
{
    if (moveString.empty())
    {
        WriteInvalidMove(InvalidMoveMessage_Generic);
        return;
    }

    // Gestione speciale del "pass" tramite il comando "play pass"
    if (moveString == PassMoveString)
    {
        CommandPass();
        return;
    }

    // Converte la MoveString UHP in una mossa interna
    Move mv = MoveStringToMove(moveString);

    if (mv.Piece == PieceName::INVALID)
    {
        WriteInvalidMove("Impossibile interpretare la mossa: " + moveString);
        return;
    }

    // Verifica che il pezzo appartenga al giocatore corrente
    if (GetColor(mv.Piece) != m_board->currentColor)
    {
        WriteInvalidMove("Non è il turno di quel pezzo.");
        return;
    }

    // Verifica che la mossa sia nelle mosse valide
    // (semplice validazione: la destinazione deve essere libera o valida per piazzamento/movimento)
    std::vector<Move> validMoves;
    GenerateValidMoves(validMoves);
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

    // Esegui la mossa
    m_board->MovePiece(mv.Piece, mv.Destination);

    // Aggiorna il turno
    Color prev = m_board->currentColor;
    m_board->currentColor = (prev == Color::White) ? Color::Black : Color::White;
    if (m_board->currentColor == Color::White)
        m_board->currentTurn++;
    m_board->ToggleTurnHash();
    m_board->boardState = BoardState::InProgress;

    // Salva la mossa nello storico
    m_moveHistory.push_back(moveString);

    // Controlla se la partita è finita
    UpdateBoardState();

    // Stampa la GameString aggiornata
    std::cout << BuildGameString() << "\n";
    WriteOk();
}

void Engine::CommandPass()
{
    // Una mossa di passaggio è valida solo se il giocatore non ha altre mosse
    std::vector<Move> validMoves;
    GenerateValidMoves(validMoves);

    if (!validMoves.empty())
    {
        WriteInvalidMove("Non puoi passare quando hai mosse disponibili.");
        return;
    }

    // Applica il passaggio: cambia solo il turno
    Color prev = m_board->currentColor;
    m_board->currentColor = (prev == Color::White) ? Color::Black : Color::White;
    if (m_board->currentColor == Color::White)
        m_board->currentTurn++;
    m_board->ToggleTurnHash();

    // Salva nello storico
    m_moveHistory.push_back(PassMoveString);

    // Stampa la GameString aggiornata
    std::cout << BuildGameString() << "\n";
    WriteOk();
}

void Engine::CommandUndo(const std::string& param)
{
    // Quante mosse annullare (default: 1)
    int movesToUndo = 1;
    if (!param.empty())
    {
        try
        {
            movesToUndo = std::stoi(param);
        }
        catch (...)
        {
            WriteError("Parametro undo non valido: " + param);
            return;
        }
    }

    if (movesToUndo < 1 || movesToUndo > (int)m_moveHistory.size())
    {
        WriteError(ErrorMessage_UnableToUndo);
        return;
    }

    // Strategia: ricostruiamo la partita dall'inizio rigiocando le prime
    // (N - movesToUndo) mosse. È l'approccio più semplice e robusto.
    
    // Salva il tipo di partita e lo storico ridotto
    GameType gt = m_board->gameType;
    std::vector<std::string> remainingHistory(
        m_moveHistory.begin(),
        m_moveHistory.end() - movesToUndo
    );

    // Ricostruisce la GameString da riprodurre
    std::ostringstream gs;
    gs << GetEnumString(gt);
    // Aggiungiamo le mosse rimanenti
    for (const std::string& ms : remainingHistory)
    {
        gs << ";" << ms;
    }

    // Resetta e rigioca
    ResetGame();

    if (remainingHistory.empty())
    {
        // Nessuna mossa da riprodurre: partita appena iniziata
        m_board = new Board(gt);
    }
    else
    {
        // Costruiamo una GameString con stato fittizio (lo ricalcoleremo)
        // formato: "GameType;InProgress;White[1];mossa1;mossa2;..."
        // Prima troviamo il TurnString corretto ripercorrendo le mosse
        // In realtà CommandNewGame accetta una GameString: costruiamola
        std::ostringstream fullGs;
        fullGs << GetEnumString(gt) << ";InProgress;White[1]";
        for (const std::string& ms : remainingHistory)
        {
            fullGs << ";" << ms;
        }
        CommandNewGame(fullGs.str());
        // CommandNewGame scrive già output: dobbiamo invece solo ricostruire silenziosamente.
        // Tuttavia per semplicità di implementazione, accettiamo questo comportamento
        // (stamperà una GameString intermedia prima dell'ok finale).
        // TODO: separare la logica di newgame in una funzione privata silenziosa.
        return; // CommandNewGame ha già scritto ok
    }

    std::cout << BuildGameString() << "\n";
    WriteOk();
}

void Engine::CommandOptions(const std::string& param)
{
    // Per ora il motore non espone nessuna opzione configurabile.
    // Il comando "options" senza parametri restituisce semplicemente "ok".
    // In futuro si potranno aggiungere opzioni tipo profondità di ricerca, ecc.
    (void)param; // Parametro non usato (evita warning del compilatore)
    WriteOk();
}

// =============================================================================
// GENERAZIONE MOSSE
// =============================================================================

void Engine::GenerateValidMoves(std::vector<Move>& moves) const
{
    moves.clear();

    if (m_board == nullptr) return;

    Color color = m_board->currentColor;

    // Genera mosse di piazzamento (da mano a board)
    GeneratePlacementMoves(moves);

    // Genera mosse di spostamento (da board a board)
    // Solo se la regina del giocatore corrente è già in gioco
    // (non puoi muovere pezzi prima di aver piazzato la regina)
    PieceName queen = (color == Color::White) ? PieceName::wQ : PieceName::bQ;
    if (m_board->PieceInPlay(queen))
    {
        GenerateMovementMoves(moves);
    }
}

void Engine::GeneratePlacementMoves(std::vector<Move>& moves) const
{
    Color color = m_board->currentColor;
    int turn = m_board->currentTurn;

    // Se è il primo turno (board vuota), il bianco può piazzare qualsiasi pezzo
    // direttamente al centro. Il nero deve piazzare adiacente al bianco.
    bool boardEmpty = true;
    for (int i = 0; i < NumPieceNames; ++i)
    {
        if (m_board->PieceInPlay(static_cast<PieceName>(i)))
        {
            boardEmpty = false;
            break;
        }
    }

    // Deve essere giocata la regina entro il 4° turno del giocatore?
    bool mustPlayQueen = MustPlaceQueen(color);

    // Raccoglie i pezzi disponibili in mano per questo colore
    // (abilitati per il tipo di partita corrente)
    std::vector<PieceName> availablePieces;
    for (int i = 0; i < NumPieceNames; ++i)
    {
        PieceName p = static_cast<PieceName>(i);
        if (GetColor(p) != color) continue;
        if (!m_board->PieceInHand(p)) continue;
        if (!PieceNameIsEnabledForGameType(p, m_board->gameType)) continue;

        // Se deve giocare la regina, può piazzare SOLO la regina
        if (mustPlayQueen && GetBugType(p) != BugType::QueenBee) continue;

        availablePieces.push_back(p);
    }

    if (availablePieces.empty()) return;

    if (boardEmpty)
    {
        // Primo piazzamento: solo al centro
        for (PieceName p : availablePieces)
        {
            moves.push_back(Move{p, NullIndex, BoardCenter});
        }
        return;
    }

    // Secondo piazzamento: il nero deve stare adiacente al primo pezzo bianco,
    // ma NON ci sono restrizioni sul toccare pezzi avversari (perché è il primo turno nero).
    // Dal 3° turno in poi: non si può piazzare adiacente a pezzi avversari.
    bool isSecondPlacement = (PiecesInPlay(color) == 0);

    // Raccogli tutte le posizioni vuote adiacenti a pezzi già sulla board
    // e che rispettano il vincolo del colore
    std::vector<bool> candidateChecked(BoardSize, false);
    std::vector<Index> candidates;

    for (int i = 0; i < NumPieceNames; ++i)
    {
        PieceName p = static_cast<PieceName>(i);
        if (!m_board->PieceInPlay(p)) continue;
        if (!m_board->PieceIsOnTop(p)) continue;

        Index pos = m_board->GetPosition(p);

        // Esamina i 6 vicini
        for (Direction d = Direction::Right; d < Direction::NumDirections;
             d = static_cast<Direction>(static_cast<uint8_t>(d) + 1))
        {
            Index neighbor = GetNeighborAt(pos, d);
            if (!IsValidIndex(neighbor)) continue;
            if (m_board->HasPieceAt(neighbor)) continue;
            if (candidateChecked[neighbor]) continue;

            candidateChecked[neighbor] = true;

            if (isSecondPlacement)
            {
                // Il secondo piazzamento (primo del nero) non ha restrizioni di colore
                candidates.push_back(neighbor);
            }
            else
            {
                // Dal 3° turno: posizione valida solo se adiacente al proprio colore
                // e NON adiacente al colore avversario
                if (IsValidPlacementPosition(neighbor, color))
                {
                    candidates.push_back(neighbor);
                }
            }
        }
    }

    // Crea una mossa per ogni combinazione (pezzo disponibile) x (posizione candidata)
    for (PieceName p : availablePieces)
    {
        for (Index dest : candidates)
        {
            moves.push_back(Move{p, NullIndex, dest});
        }
    }
}

void Engine::GenerateMovementMoves(std::vector<Move>& moves) const
{
    // TODO: implementare la generazione di mosse per ogni tipo di pezzo.
    //
    // Ogni tipo di pezzo ha regole di movimento diverse:
    //   - QueenBee:   si sposta di 1 posizione seguendo il bordo dell'alveare
    //   - Spider:     si sposta esattamente di 3 posizioni seguendo il bordo
    //   - Beetle:     si sposta di 1 posizione, può salire su altri pezzi
    //   - Grasshopper: salta in linea retta sopra i pezzi contigui
    //   - SoldierAnt: si sposta in qualsiasi posizione lungo il bordo dell'alveare
    //   - Mosquito:   copia il comportamento del pezzo adiacente
    //   - Ladybug:    3 passi (2 sopra l'alveare + 1 giù)
    //   - Pillbug:    si sposta di 1 + può spostare un pezzo adiacente
    //
    // Per ora questo stub non genera mosse di spostamento.
    (void)moves; // Evita warning
}

bool Engine::IsValidPlacementPosition(Index pos, Color color) const
{
    Color enemyColor = (color == Color::White) ? Color::Black : Color::White;

    bool touchesFriendly = false;
    bool touchesEnemy = false;

    for (Direction d = Direction::Right; d < Direction::NumDirections;
         d = static_cast<Direction>(static_cast<uint8_t>(d) + 1))
    {
        Index neighbor = GetNeighborAt(pos, d);
        if (!IsValidIndex(neighbor)) continue;

        PieceName np = m_board->GetPieceAt(neighbor);
        if (np == PieceName::INVALID) continue;

        Color nc = GetColor(np);
        if (nc == color) touchesFriendly = true;
        if (nc == enemyColor) touchesEnemy = true;
    }

    return touchesFriendly && !touchesEnemy;
}

bool Engine::MustPlaceQueen(Color color) const
{
    // La regina DEVE essere giocata entro il 4° turno del giocatore
    // (ovvero quando il giocatore sta per fare la sua 4ª mossa).
    // piecesInPlay == 3 significa che ha già giocato 3 pezzi, sta per giocarne il 4°.
    PieceName queen = (color == Color::White) ? PieceName::wQ : PieceName::bQ;
    if (m_board->PieceInPlay(queen)) return false; // Regina già piazzata

    int inPlay = PiecesInPlay(color);
    return (inPlay >= 3); // Al 4° turno (0-indexed: 3 pezzi già in gioco)
}

int Engine::PiecesInPlay(Color color) const
{
    int count = 0;
    for (int i = 0; i < NumPieceNames; ++i)
    {
        PieceName p = static_cast<PieceName>(i);
        if (GetColor(p) == color && m_board->PieceInPlay(p))
            count++;
    }
    return count;
}

// =============================================================================
// CONVERSIONE POSIZIONE <-> MOVESTRING UHP
// =============================================================================

std::string Engine::MoveToMoveString(const Move& move) const
{
    std::ostringstream ss;
    ss << GetEnumString(move.Piece);

    // Se la destinazione è il centro e non c'è nient'altro sulla board
    // (primo piazzamento), non aggiungiamo la posizione relativa
    bool boardWasEmpty = true;
    for (int i = 0; i < NumPieceNames; ++i)
    {
        PieceName p = static_cast<PieceName>(i);
        if (p == move.Piece) continue;
        if (m_board->PieceInPlay(p))
        {
            boardWasEmpty = false;
            break;
        }
    }

    if (boardWasEmpty && move.Source == NullIndex)
    {
        // Primo piazzamento: solo il nome del pezzo
        return ss.str();
    }

    // Calcola la posizione relativa rispetto ai pezzi sulla board
    std::string relPos = PositionToRelativeString(move.Destination);
    if (!relPos.empty())
    {
        ss << " " << relPos;
    }

    return ss.str();
}

std::string Engine::PositionToRelativeString(Index pos) const
{
    // Cerca un pezzo adiacente alla destinazione e restituisce la notazione relativa UHP:
    //   Right (0)      -> "PieceName-"  (il pezzo è a sinistra della destinazione)
    //   DownRight (1)  -> "PieceName/"  (a sinistra-sopra)
    //   DownLeft (2)   -> "\PieceName"  (a destra-sopra)  <- separatore PRIMA
    //   Left (3)       -> "-PieceName"  <- separatore PRIMA
    //   UpLeft (4)     -> "/PieceName"  <- separatore PRIMA
    //   UpRight (5)    -> "PieceName\"  (a sinistra-sotto)
    //
    // La notazione UHP segue la convenzione BoardSpace:
    //   pezzo- = destinazione è a destra del pezzo di riferimento  (Right)
    //   pezzo/ = destinazione è in basso-destra del pezzo           (DownRight)
    //   pezzo\ = destinazione è in basso-sinistra                   (DownLeft) ... 
    //   -pezzo = destinazione è a sinistra del pezzo                (Left)
    //   /pezzo = destinazione è in alto-sinistra                    (UpLeft)
    //   \pezzo = destinazione è in alto-destra                      (UpRight)
    //
    // Mappa: direzione DA destinazione VERSO il pezzo di riferimento
    //   Se pezzo è in direzione Right rispetto a dest -> separatore "-" DOPO il pezzo
    //   Se pezzo è in direzione DownRight             -> "/" DOPO
    //   Se pezzo è in direzione DownLeft              -> "\" DOPO
    //   Se pezzo è in direzione Left                  -> "-" PRIMA
    //   Se pezzo è in direzione UpLeft                -> "/" PRIMA
    //   Se pezzo è in direzione UpRight               -> "\" PRIMA

    // Separatori per ciascuna direzione (in cui si trova il pezzo di riferimento)
    // Indice = static_cast<int>(Direction)
    // true = separatore DOPO il nome del pezzo (es. "wS1-")
    // false = separatore PRIMA del nome del pezzo (es. "-wS1")
    static const char separators[6] = { '-', '/', '\\', '-', '/', '\\' };
    static const bool afterPiece[6] = { true, true, true, false, false, false };

    for (int d = 0; d < 6; ++d)
    {
        Index neighborPos = GetNeighborAt(pos, static_cast<Direction>(d));
        if (!IsValidIndex(neighborPos)) continue;

        PieceName np = m_board->GetPieceAt(neighborPos);
        if (np == PieceName::INVALID) continue;

        // Trovato un pezzo adiacente: costruisci la stringa relativa
        std::string pieceName = GetEnumString(np);
        char sep = separators[d];

        std::ostringstream ss;
        if (afterPiece[d])
        {
            // Es. "wS1-"
            ss << pieceName << sep;
        }
        else
        {
            // Es. "-wS1"
            ss << sep << pieceName;
        }
        return ss.str();
    }

    // Nessun pezzo adiacente trovato (non dovrebbe succedere per mosse valide)
    return "";
}

Move Engine::MoveStringToMove(const std::string& moveString) const
{
    Move invalid = {PieceName::INVALID, NullIndex, NullIndex};

    if (moveString.empty()) return invalid;

    // Caso speciale: "pass"
    if (moveString == PassMoveString) return PassMove;

    // Usiamo TryNormalizeMoveString per parsare
    bool isPass;
    PieceName startPiece;
    char beforeSep;
    PieceName endPiece;
    char afterSep;

    if (!TryNormalizeMoveString(moveString, isPass, startPiece, beforeSep, endPiece, afterSep))
    {
        return invalid;
    }

    if (isPass) return PassMove;
    if (startPiece == PieceName::INVALID) return invalid;

    // Se non c'è un pezzo di riferimento: primo piazzamento al centro
    if (endPiece == PieceName::INVALID)
    {
        return Move{startPiece, NullIndex, BoardCenter};
    }

    // Trova la posizione del pezzo di riferimento sulla board
    Index refPos = m_board->GetPosition(endPiece);
    if (refPos == NullIndex)
    {
        return invalid; // Il pezzo di riferimento non è sulla board
    }

    // Calcola la destinazione in base ai separatori
    // La logica è l'inverso di PositionToRelativeString:
    //
    //  "refPiece-" -> destinazione è a DESTRA del refPiece  (Direction::Right)
    //  "refPiece/" -> destinazione è in BASSO-DESTRA         (Direction::DownRight)
    //  "refPiece\" -> destinazione è in BASSO-SINISTRA       (Direction::DownLeft)
    //  "-refPiece" -> destinazione è a SINISTRA              (Direction::Left)
    //  "/refPiece" -> destinazione è in ALTO-SINISTRA        (Direction::UpLeft)
    //  "\refPiece" -> destinazione è in ALTO-DESTRA          (Direction::UpRight)
    //
    //  Caso speciale senza separatori: il beetle sale sopra il refPiece
    //  -> destinazione == refPos

    Index dest = NullIndex;

    if (beforeSep == '\0' && afterSep == '\0')
    {
        // Beetle che sale sopra: la destinazione è la stessa posizione del pezzo di riferimento
        dest = refPos;
    }
    else if (afterSep == '-')
    {
        dest = GetNeighborAt(refPos, Direction::Right);
    }
    else if (afterSep == '/')
    {
        dest = GetNeighborAt(refPos, Direction::DownRight);
    }
    else if (afterSep == '\\')
    {
        dest = GetNeighborAt(refPos, Direction::DownLeft);
    }
    else if (beforeSep == '-')
    {
        dest = GetNeighborAt(refPos, Direction::Left);
    }
    else if (beforeSep == '/')
    {
        dest = GetNeighborAt(refPos, Direction::UpLeft);
    }
    else if (beforeSep == '\\')
    {
        dest = GetNeighborAt(refPos, Direction::UpRight);
    }

    if (!IsValidIndex(dest)) return invalid;

    return Move{startPiece, m_board->GetPosition(startPiece), dest};
}

// =============================================================================
// COSTRUZIONE GAMESTRING UHP
// =============================================================================

std::string Engine::BuildGameString() const
{
    if (m_board == nullptr) return "";

    std::ostringstream ss;

    // GameTypeString;GameStateString;TurnString
    ss << BuildGameTypeString() << ";";
    ss << GetEnumString(m_board->boardState) << ";";
    ss << BuildTurnString();

    // Aggiunge lo storico delle mosse
    for (const std::string& ms : m_moveHistory)
    {
        ss << ";" << ms;
    }

    return ss.str();
}

std::string Engine::BuildTurnString() const
{
    // Formato: "White[3]" o "Black[2]"
    std::ostringstream ss;
    ss << GetEnumString(m_board->currentColor)
       << "[" << m_board->currentTurn << "]";
    return ss.str();
}

std::string Engine::BuildGameTypeString() const
{
    // Converte il GameType in stringa UHP
    // GetEnumString non è definita per GameType nei file forniti,
    // quindi lo facciamo qui manualmente
    switch (m_board->gameType)
    {
        case GameType::Base:    return "Base";
        case GameType::BaseM:   return "Base+M";
        case GameType::BaseL:   return "Base+L";
        case GameType::BaseP:   return "Base+P";
        case GameType::BaseML:  return "Base+ML";
        case GameType::BaseMP:  return "Base+MP";
        case GameType::BaseLP:  return "Base+LP";
        case GameType::BaseMLP: return "Base+MLP";
        default:                return "Base";
    }
}

// =============================================================================
// AGGIORNAMENTO STATO PARTITA
// =============================================================================

void Engine::UpdateBoardState()
{
    if (m_board == nullptr) return;

    // Se la partita non è ancora iniziata o è già finita, non aggiorniamo
    if (m_board->boardState == BoardState::NotStarted) return;
    if (GameIsOver(m_board->boardState)) return;

    // Controlla se le regine sono circondate
    bool whiteQueenSurrounded = false;
    bool blackQueenSurrounded = false;

    if (m_board->PieceInPlay(PieceName::wQ))
    {
        whiteQueenSurrounded = (CountQueenNeighbors(Color::White) == 6);
    }
    if (m_board->PieceInPlay(PieceName::bQ))
    {
        blackQueenSurrounded = (CountQueenNeighbors(Color::Black) == 6);
    }

    if (whiteQueenSurrounded && blackQueenSurrounded)
    {
        m_board->boardState = BoardState::Draw;
    }
    else if (whiteQueenSurrounded)
    {
        m_board->boardState = BoardState::BlackWins;
    }
    else if (blackQueenSurrounded)
    {
        m_board->boardState = BoardState::WhiteWins;
    }
    // Altrimenti la partita continua
}

int Engine::CountQueenNeighbors(Color color) const
{
    PieceName queen = (color == Color::White) ? PieceName::wQ : PieceName::bQ;
    if (!m_board->PieceInPlay(queen)) return 0;

    Index queenPos = m_board->GetPosition(queen);
    int count = 0;

    for (Direction d = Direction::Right; d < Direction::NumDirections;
         d = static_cast<Direction>(static_cast<uint8_t>(d) + 1))
    {
        Index neighbor = GetNeighborAt(queenPos, d);
        if (IsValidIndex(neighbor) && m_board->HasPieceAt(neighbor))
        {
            count++;
        }
    }

    return count;
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
    {
        tokens.push_back(token);
    }
    return tokens;
}

void Engine::ResetGame()
{
    delete m_board;
    m_board = nullptr;
    m_moveHistory.clear();
}

} // namespace HiveGotThis