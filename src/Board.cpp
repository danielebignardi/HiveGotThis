#include "Board.h"
#include <iterator>
#include <algorithm>
#include <unordered_set>
#include <random>
#include <iostream>

namespace HiveGotThis
{  

// - - - - - - - - - - COSTRUTTORE- - - - - - - - - -

Board::Board(GameType gameType)
{
    this->gameType = gameType;

    // Riempie tutto l'array 'cells' con INVALID
    std::fill(std::begin(cells), std::end(cells), PieceName::INVALID);
    
    // Riempie tutto 'below' con INVALID
    std::fill(std::begin(below), std::end(below), PieceName::INVALID);

    // Riempie tutto 'piecesPositions' con NullIndex (all'inizio tutte in mano)
    std::fill(std::begin(piecesPositions), std::end(piecesPositions), NullIndex);

    // GESTIONE ZOBRIST
    currentHash = 0;                        // Board vuota = Hash 0
    hashHistory.clear();
    hashHistory.reserve(100);               // Ottimizzazione: pre-alloca spazio
    hashHistory.push_back(currentHash);     // Aggiungi lo stato iniziale (vuoto) allo storico

    // Azzera le altezze delle pile
    std::fill(std::begin(stackHeight), std::end(stackHeight), 0);

    // Azzera i pezzi che questo turno non possono essere mossi
    memset(cannotBeMoved, 0, sizeof(cannotBeMoved));
}



// - - - - - - - - - - FUNZIONI DI MODIFICA - - - - - - - - - -

PieceName Board::PopAt(Index position)
{
    assert(IsValidIndex(position) && "PopAt chiamato con indice invalido!");

    if (HasPieceAt(position))
    {
        PieceName topPiece = GetPieceAt(position);  // Identifica il pezzo in cima

        cells[position] = below[topPiece];          // Il pezzo sotto diventa il nuovo top
        below[topPiece] = PieceName::INVALID;       // Il pezzo rimosso non ha più nulla sotto

        piecesPositions[topPiece] = NullIndex;      // Il pezzo non è più sulla board

        // Aggiorna altezza
        stackHeight[position]--;
        int h = stackHeight[position];

        // Aggiorna hash
        currentHash ^= (ZobristTable[topPiece][position] ^ ZobristLevel[topPiece][h]);

        return topPiece;
    }
    else
    {
        return PieceName::INVALID;                  // Non c'è nulla da rimuovere
    }
}

void Board::PushAt(PieceName pieceName, Index position)
{
    assert(IsValidIndex(position) && "PushAt chiamato con indice invalido!");
    
    // Aggiorna hash (prima di modificare la board in quanto h è l'altezza a cui inseriremo il pezzo)
    int h = stackHeight[position];                                                      // Altezza attuale dove andrà il pezzo 
    currentHash ^= (ZobristTable[pieceName][position] ^ ZobristLevel[pieceName][h]);    // XOR combinato: Posizione univoca ^ Livello univoco

    below[pieceName] = cells[position];         // Il vecchio top diventa il pezzo sotto
    cells[position] = pieceName;                // Il nuovo pezzo diventa il top
    
    piecesPositions[pieceName] = position;      // Il vecchio top diventa il pezzo sotto

    // Aggiorna altezza
    stackHeight[position]++;
}

void Board::MovePiece(PieceName pieceName, Index newPosition)
{    
    assert(IsValidIndex(newPosition) && "MovePiece chiamato con indice invalido!");

    if (PieceInPlay(pieceName))
    {
        PopAt(piecesPositions[pieceName]);      // Il pezzo è sulla board, quindi lo rimuoviamo dalla vecchia posizione
    }
    
    PushAt(pieceName, newPosition);
}



void Board::ApplyTurnEffects(PieceName movedPiece)
{
    // Il pezzo mosso/piazzato non può essere spostato dal Pillbug nel turno successivo
    memset(cannotBeMoved, 0, sizeof(cannotBeMoved));
    cannotBeMoved[movedPiece] = true;

    currentColor = static_cast<Color>((static_cast<int>(currentColor) + 1) % 2);
    currentTurn++;
}



// - - - - - - - - - - ZOBRIST HASHING - - - - - - - - - -

uint64_t Board::ZobristTable[NumPieceNames][BoardSize];
uint64_t Board::ZobristLevel[NumPieceNames][8];
uint64_t Board::ZobristBlackTurn;

void Board::InitializeZobristTable()
{
    // Usa un generatore a 64-bit di alta qualità
    std::mt19937_64 gen(12345); // Seed fisso per riproducibilità
    std::uniform_int_distribution<uint64_t> dist;

    for (int p = 0; p < NumPieceNames; ++p)
    {
        // Riempi tabella Posizione
        for (int i = 0; i < BoardSize; ++i)
        {
            ZobristTable[p][i] = dist(gen);
        }
        // Riempi tabella Livelli (fino ad altezza 7)
        for (int h = 0; h < 8; ++h)
        {
            ZobristLevel[p][h] = dist(gen);
        }
    }
    
    ZobristBlackTurn = dist(gen);
}



// - - - - - - - - - - FUNZIONI DI CONTROLLO DELLO STATO - - - - - - - - - -

bool Board::CanMoveWithoutBreakingHive(PieceName piece) const
{
    Index pos = piecesPositions[piece];
    
    // Il pezzo non è sulla board (è in mano) -> si può muovere
    if (PieceInHand(piece)) return true; 

    // Il pezzo è coperto -> non si può muovere
    if (GetPieceAt(pos) != piece) return false;

    // Se sotto al pezzo ce ne è un altro, si può spostare 
    if (below[piece] != PieceName::INVALID) return true;

    // Unico pezzo in questa posizione (altezza 1)
    // Bisogna simulare la rimozione e vedere se tutto resta connesso
    // Si dice a IsOneHive di considerare pos come se fosse vuota
    return IsOneHive(pos);
}

bool Board::IsOneHive(Index ignorePos) const
{
    // PREPARAZIONE BFS
    bool visitedPieces[NumPieceNames] = { false };              // Pezzi già visitati
    static std::vector<Index> queue;                            // Nodi nel BFS
    queue.clear();
    queue.reserve(NumPieceNames);
    
    // CONTEGGIO DEI NODI
    int nodesFound = 0;
    int nodesExpected = 0;
    Index startNode = NullIndex;

    // Conta i nodi attesi (ovvero il numero di posizioni occupate esclusa ignorePos) e trova startNode
    for (int i = 0; i < NumPieceNames; ++i) {
        Index pos = piecesPositions[i];
        
        // Se non in gioco o è nella casella ignorata
        if (pos == NullIndex || pos == ignorePos) continue;

        // Se è il pezzo in cima alla pila
        if (PieceIsOnTop(static_cast<PieceName>(i))) {
            nodesExpected++;
            if (startNode == NullIndex) startNode = pos;
        }
    }

    if (nodesExpected == 0) return true;

    // BFS
    queue.push_back(startNode);
    
    // Segniamo come visitato il pezzo che si trova a startNode
    PieceName startPiece = cells[startNode];
    visitedPieces[startPiece] = true;
    nodesFound++;

    // Loop BFS (usando indice 'head' invece di pop)
    int head = 0;
    while(head < (int)queue.size()) {
        Index currentPos = queue[head++]; // Leggi e avanza cursore

        // Controlla i 6 vicini
        for (int offset : NeighborOffsets) {
            Index neighborPos = currentPos + offset;

            // Controlla che il vicino sia una cella valida
            if (!IsValidIndex(neighborPos)) continue;

            // Ignora la casella "rimossa"
            if (neighborPos == ignorePos) continue;

            // Leggi chi c'è
            PieceName neighborPiece = cells[neighborPos];

            // Se vuoto, salta
            if (neighborPiece == PieceName::INVALID) continue;

            // Se già visitato, salta
            if (visitedPieces[neighborPiece]) continue;

            // Trovato nuovo pezzo connesso!
            visitedPieces[neighborPiece] = true; // Segna come visto
            queue.push_back(neighborPos);        // Aggiungi alla coda
            nodesFound++;
        }
    }

    // VERDETTO
    return nodesFound == nodesExpected;
}



bool Board::IsQueenSurrounded(Color color) const
{
    PieceName queen = (color == Color::White) ? PieceName::wQ : PieceName::bQ;

    if (PieceInHand(queen)) return false;

    Index pos = piecesPositions[queen];

    for (int offset : NeighborOffsets)
    {
        Index neighbor = pos + offset;
        if (!IsValidIndex(neighbor) || !HasPieceAt(neighbor)) return false;
    }

    return true;
}



// - - - - - - - - - - FUNZIONI DI UTILITÀ PER LA GENERAZIONE DELLE MOSSE - - - - - - - - - -

bool Board::CanSlide(Index pos, Direction dir, SlideMode mode, Index ignorePos) const
{
    uint8_t d = static_cast<uint8_t>(dir);
    Index gate1 = pos + SlideGates[d][0];
    Index gate2 = pos + SlideGates[d][1];

    if (mode == SlideMode::Beetle)
    {
        Index dest    = pos + NeighborOffsets[d];
        // Il Beetle sale sopra dest → la quota effettiva di arrivo è stackHeight[dest]+1
        int maxHeight = std::max((int)stackHeight[pos], (int)stackHeight[dest] + 1);
        int minGate   = std::min((int)stackHeight[gate1], (int)stackHeight[gate2]);
        if (minGate >= maxHeight) return false;

        // Beetle a terra (pos h=1, dest vuota): almeno un gate occupato, stessa regola di Ground mode.
        // Un Beetle elevato (h>1) può scendere su celle vuote anche con gate vuoti — è già sopra di essi.
        if (stackHeight[pos] == 1 && stackHeight[dest] == 0 &&
            stackHeight[gate1] == 0 && stackHeight[gate2] == 0)
            return false;

        return true;
    }

    if (mode == SlideMode::LadyBug)
    {
        // La Ladybug è fisicamente sopra il pezzo a pos, quindi altezza effettiva = stackHeight[pos] + 1
        Index dest    = pos + NeighborOffsets[d];
        int maxHeight = std::max((int)stackHeight[pos] + 1, (int)stackHeight[dest]);
        int minGate   = std::min((int)stackHeight[gate1], (int)stackHeight[gate2]);
        return minGate < maxHeight;
    }

    // SlideMode::Ground — pezzi a terra
    bool g1 = HasPieceAt(gate1) && (gate1 != ignorePos);
    bool g2 = HasPieceAt(gate2) && (gate2 != ignorePos);
    if (g1 && g2) return false;
    return g1 || g2;
}

void Board::GetOccupiedNeighbors(Index pos, std::vector<Index>& result) const
{
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = pos + NeighborOffsets[i];
        if (IsValidIndex(neighbor) && HasPieceAt(neighbor))
            result.push_back(neighbor);
    }
}

void Board::GetEmptyNeighbors(Index pos, std::vector<Index>& result) const
{
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = pos + NeighborOffsets[i];
        if (IsValidIndex(neighbor) && !HasPieceAt(neighbor))
            result.push_back(neighbor);
    }
}

void Board::GetOneSlideSteps(Index from, SlideMode mode, bool visited[], std::vector<Index>& result, Index ignorePos) const
{
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = from + NeighborOffsets[i];
        if (!IsValidIndex(neighbor)) continue;
        if (mode == SlideMode::Ground && HasPieceAt(neighbor)) continue;    // se siamo in modalità Ground allora saltiamo le celle su cui c'è già un pezzo
        if (visited[neighbor]) continue;
        if (!CanSlide(from, static_cast<Direction>(i), mode, ignorePos)) continue;
        result.push_back(neighbor);
    }
}



// - - - - - - - - - - GENERAZIONE DELLE MOSSE VALIDE - - - - - - - - - -

void Board::GetQueenBeeMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    static bool visited[BoardSize];         // static così viene allocato solo una volta
    memset(visited, 0, sizeof(visited));
    visited[pos] = true;

    std::vector<Index> steps;
    GetOneSlideSteps(pos, SlideMode::Ground, visited, steps);

    for (Index dest : steps)
    {
        Move move = {piece, pos, dest};
        moves.push_back(move);
    }
}

void Board::GetBeetleMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    static bool visited[BoardSize];
    memset(visited, 0, sizeof(visited));
    visited[pos] = true;

    std::vector<Index> steps;
    GetOneSlideSteps(pos, SlideMode::Beetle, visited, steps);

    for (Index dest : steps)
    {
        Move move = {piece, pos, dest};
        moves.push_back(move);
    }
}

void Board::GetGrasshopperMoves(PieceName piece, std::vector<Move>& moves) const
{
    // one hive rule
    if (!CanMoveWithoutBreakingHive(piece)) return;    

    Index pos = piecesPositions[piece];

    for (int i = 0; i < 6; i++)
    {
        Index neighbor = pos + NeighborOffsets[i];

        if (!IsValidIndex(neighbor)) continue;    // fuori dalla board
        if (!HasPieceAt(neighbor)) continue;      // deve saltare oltre un pezzo

        Index current = neighbor;
        while (IsValidIndex(current) && HasPieceAt(current)) {    // va oltre tutti i pezzi nella direzione considerata
            current = current + NeighborOffsets[i];
        }

        if (IsValidIndex(current))
        {
            Move move = {piece, pos, current};
            moves.push_back(move);
        }
    }
}

void Board::GetSpiderMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    // celle già visitate
    static bool visited[BoardSize];
    memset(visited, 0, sizeof(visited));

    // indici raggiungibili ad ognuno dei 3 passi
    std::vector<Index> steps1, steps2, steps3;

    // inizio: indici raggiungibili dopo un passo
    visited[pos] = true;
    GetOneSlideSteps(pos, SlideMode::Ground, visited, steps1, pos);

    for (Index step1 : steps1)
    {
        visited[step1] = true;

        steps2.clear();
        GetOneSlideSteps(step1, SlideMode::Ground, visited, steps2, pos);

        for (Index step2 : steps2)
        {
            visited[step2] = true;

            steps3.clear();
            GetOneSlideSteps(step2, SlideMode::Ground, visited, steps3, pos);

            for (Index step3 : steps3)
            {
                Move move = {piece, pos, step3};
                moves.push_back(move);
            }

            visited[step2] = false;
        }

        visited[step1] = false;
    }
}

void Board::GetSoldierAntMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    // Posizioni già visitate
    static bool visited[BoardSize];
    memset(visited, 0, sizeof(visited));
    visited[pos] = true;

    // Posizioni visitabili, iniziamo a riempirle con i vicini
    std::vector<Index> queue;
    GetOneSlideSteps(pos, SlideMode::Ground, visited, queue, pos);
    for (Index cell : queue) visited[cell] = true;

    int head = 0;
    while (head < (int)queue.size()) {
        Index current = queue[head++];
        int sizeBefore = queue.size();
        GetOneSlideSteps(current, SlideMode::Ground, visited, queue, pos);

        // Segna subito come visitati i nuovi nodi aggiunti
        for (int i = sizeBefore; i < (int)queue.size(); i++)
            visited[queue[i]] = true;
    }

    for (Index cell : queue)
    {
        Move move = {piece, pos, cell};
        moves.push_back(move);
    }
}

void Board::GetLadybugMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    // Passo 1: celle occupate adiacenti alla posizione originale
    std::vector<Index> steps1;
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = pos + NeighborOffsets[i];
        if (!IsValidIndex(neighbor)) continue;
        if (!HasPieceAt(neighbor)) continue;
        if (!CanSlide(pos, static_cast<Direction>(i), SlideMode::Beetle)) continue;
        steps1.push_back(neighbor);
    }

    // Passo 2: celle occupate adiacenti a steps1
    std::vector<Index> steps2;
    for (Index step1 : steps1)
    {
        for (int i = 0; i < 6; i++)
        {
            Index neighbor = step1 + NeighborOffsets[i];
            if (!IsValidIndex(neighbor)) continue;
            if (!HasPieceAt(neighbor)) continue;
            if (neighbor == pos) continue;
            if (!CanSlide(step1, static_cast<Direction>(i), SlideMode::Beetle)) continue;
            steps2.push_back(neighbor);
        }
    }

    // Passo 3: scende su celle vuote adiacenti a steps2
    for (Index step2 : steps2)
    {
        for (int i = 0; i < 6; i++)
        {
            Index neighbor = step2 + NeighborOffsets[i];
            if (!IsValidIndex(neighbor)) continue;
            if (HasPieceAt(neighbor)) continue;
            if (neighbor == pos) continue;
            if (!CanSlide(step2, static_cast<Direction>(i), SlideMode::LadyBug)) continue;

            Move move = {piece, pos, neighbor};
            moves.push_back(move);
        }
    }
}

void Board::GetPillbugMoves(PieceName piece, std::vector<Move>& moves) const
{
    Index pos = piecesPositions[piece];

    // Movimento 1: slide di 1 passo come la Queen Bee (richiede che il Pillbug non rompa l'hive)
    if (CanMoveWithoutBreakingHive(piece))
    {
        for (int i = 0; i < 6; i++)
        {
            Index neighbor = pos + NeighborOffsets[i];
            if (!IsValidIndex(neighbor)) continue;
            if (HasPieceAt(neighbor)) continue;
            if (!CanSlide(pos, static_cast<Direction>(i), SlideMode::Ground)) continue;
            moves.push_back({piece, pos, neighbor});
        }
    }

    // Movimento 2: sposta un pezzo adiacente sopra il Pillbug e depositalo
    // (la special ability funziona anche se il Pillbug è pinnato)
    for (int i = 0; i < 6; i++)
    {
        Index neighborPos = pos + NeighborOffsets[i];
        if (!IsValidIndex(neighborPos)) continue;
        if (!HasPieceAt(neighborPos)) continue;

        PieceName neighborPiece = GetPieceAt(neighborPos);

        // Il pezzo non deve essere in uno stack
        if (stackHeight[neighborPos] > 1) continue;

        // Il pezzo non deve essere nella lista cannotBeMoved
        if (cannotBeMoved[neighborPiece]) continue;

        // Il pezzo deve potersi muovere senza rompere l'hive
        if (!CanMoveWithoutBreakingHive(neighborPiece)) continue;

        // Gate rule con altezze per la raccolta (neighborPos → pos)
        // La direzione corretta è l'opposta di i: il vicino si muove VERSO il Pillbug
        if (!CanSlide(neighborPos, static_cast<Direction>((i + 3) % 6), SlideMode::Beetle)) continue;

        // Deposita il pezzo in ogni cella vuota adiacente al Pillbug
        for (int j = 0; j < 6; j++)
        {
            Index dest = pos + NeighborOffsets[j];
            if (!IsValidIndex(dest)) continue;
            if (HasPieceAt(dest)) continue;
            if (dest == neighborPos) continue;

            // Gate rule con altezze per il deposito (pos → dest)
            // Il Pillbug porta un pezzo sopra di sé: altezza effettiva = stackHeight[pos]+1 → LadyBug
            if (!CanSlide(pos, static_cast<Direction>(j), SlideMode::LadyBug)) continue;

            moves.push_back({neighborPiece, neighborPos, dest});
        }
    }
}

void Board::GetMosquitoMoves(PieceName piece, std::vector<Move>& moves) const
{
    Index pos = piecesPositions[piece];

    // Caso speciale: se la Mosquito è sopra un pezzo, si muove solo come Beetle
    if (stackHeight[pos] > 1)
    {
        if (!CanMoveWithoutBreakingHive(piece)) return;
        GetBeetleMoves(piece, moves);
        return;
    }

    // La Mosquito può copiare la special ability del Pillbug anche se è pinnata,
    // perché il Pillbug non richiede di muoversi fisicamente.
    // Tutti gli altri tipi richiedono che il pezzo possa muoversi senza rompere l'hive.
    bool canMove = CanMoveWithoutBreakingHive(piece);

    // Tipi di insetti adiacenti già processati (per evitare duplicati)
    bool processedTypes[static_cast<int>(BugType::NumBugTypes)] = { false };

    for (int i = 0; i < 6; i++)
    {
        Index neighborPos = pos + NeighborOffsets[i];
        if (!IsValidIndex(neighborPos)) continue;
        if (!HasPieceAt(neighborPos)) continue;

        PieceName neighborPiece = GetPieceAt(neighborPos);
        BugType type = GetBugType(neighborPiece);

        // La Mosquito non può copiare se stessa o un'altra Mosquito
        if (type == BugType::Mosquito) continue;

        // Evita di processare due volte lo stesso tipo
        if (processedTypes[static_cast<int>(type)]) continue;
        processedTypes[static_cast<int>(type)] = true;

        // Pillbug: la special ability funziona anche quando pinnata
        if (type == BugType::Pillbug) { GetPillbugMoves(piece, moves); continue; }

        // Gli altri tipi richiedono che la Mosquito possa muoversi
        if (!canMove) continue;

        switch (type)
        {
            case BugType::QueenBee:    GetQueenBeeMoves(piece, moves);    break;
            case BugType::Beetle:      GetBeetleMoves(piece, moves);      break;
            case BugType::Grasshopper: GetGrasshopperMoves(piece, moves); break;
            case BugType::Spider:      GetSpiderMoves(piece, moves);      break;
            case BugType::SoldierAnt:  GetSoldierAntMoves(piece, moves);  break;
            case BugType::Ladybug:     GetLadybugMoves(piece, moves);     break;
            default: break;
        }
    }
}

void Board::GetValidMoves(std::vector<Move>& moves) const
{
    moves.clear();

    // Gioco non in corso, nessuna mossa
    if (!GameInProgress(boardState)) return;

    // Regola della Regina: al quarto turno deve essere piazzata
    // Bianco gioca ai turni 0,2,4,6 — quarto turno = 6
    // Nero gioca ai turni 1,3,5,6 — quarto turno = 7
    PieceName queen = (currentColor == Color::White) ? PieceName::wQ : PieceName::bQ;   // chi è la regina in questione a seconda del turno
    bool queenInPlay = PieceInPlay(queen);
    int queenDeadline = (currentColor == Color::White) ? 6 : 7;             // turno entro il quale la regina in questione deve essere piazzata
    bool mustPlaceQueen = !queenInPlay && (currentTurn >= queenDeadline); 

    std::vector<Index> positions;
    GetValidPlacements(currentColor, positions);

    if (mustPlaceQueen)
    {
        for (Index dest : positions)
            moves.push_back({queen, NullIndex, dest});
        return;
    }

    // Piazzamento: combina ogni posizione valida con ogni pezzo in mano
    for (Index dest : positions)
    {
        for (int p = 0; p < NumPieceNames; p++)
        {
            PieceName piece = static_cast<PieceName>(p);
            if (GetColor(piece) != currentColor) continue;
            if (!PieceInHand(piece)) continue;
            if (!PieceNameIsEnabledForGameType(piece, gameType)) continue;

            // Regola: la queenBee non può essere piazzata al primo turno del proprio colore
            if (piece == queen && currentTurn <= 1) continue;

            // Regola Mzinga: un pezzo numerato non può essere piazzato se il precedente è ancora in mano
            // (es. non si può piazzare WS2 se WS1 non è ancora sulla board)
            if (piece > 0) {
                PieceName prev = static_cast<PieceName>(piece - 1);
                if (GetColor(prev) == currentColor &&
                    GetBugType(prev) == GetBugType(piece) &&
                    PieceInHand(prev)) continue;
            }

            moves.push_back({piece, NullIndex, dest});
        }
    }

    // Movimento: per ogni pezzo in gioco del colore corrente
    // La Regina deve essere in gioco per poter muovere altri pezzi
    if (queenInPlay)
    {
        for (int p = 0; p < NumPieceNames; p++)
        {
            PieceName piece = static_cast<PieceName>(p);
            if (GetColor(piece) != currentColor) continue;
            if (!PieceInPlay(piece)) continue;
            if (!PieceIsOnTop(piece)) continue;
            if (cannotBeMoved[piece]) continue;

            BugType type = GetBugType(piece);
            switch (type)
            {
                case BugType::QueenBee:    GetQueenBeeMoves(piece, moves);    break;
                case BugType::Beetle:      GetBeetleMoves(piece, moves);      break;
                case BugType::Grasshopper: GetGrasshopperMoves(piece, moves); break;
                case BugType::Spider:      GetSpiderMoves(piece, moves);      break;
                case BugType::SoldierAnt:  GetSoldierAntMoves(piece, moves);  break;
                case BugType::Ladybug:     GetLadybugMoves(piece, moves);     break;
                case BugType::Pillbug:     GetPillbugMoves(piece, moves);     break;
                case BugType::Mosquito:    GetMosquitoMoves(piece, moves);    break;
                default: break;
            }
        }
    }

    // DEDUPLICAZIONE: alcune funzioni (Ladybug, Mosquito+Pillbug) possono generare la stessa mossa più volte.
    // Sort + unique richiede che i duplicati siano adiacenti.
    {
        std::unordered_set<size_t> seen;
        auto it = std::remove_if(moves.begin(), moves.end(), [&](const Move& m) {
            return !seen.insert(hash(m)).second;
        });
        moves.erase(it, moves.end());
    }

    // Se non ci sono mosse valide, l'unica opzione è passare
    if (moves.empty())
    {
        moves.push_back(PassMove);
    }
}

bool Board::CanPlaceAt(Index pos, Color myColor, int currentTurn) const
{
    // 1. Posizione deve essere Vuota 
    if (HasPieceAt(pos)) return false;

    // 2. CASI SPECIALI (Turno 0 e 1)
    if (currentTurn == 0) return pos == BoardCenter; // Turno 0: il Bianco mette solo al centro

    if (currentTurn == 1) {
        // Turno 1: il Nero DEVE toccare il Bianco.
        // Appena troviamo un vicino qualsiasi (che per forza è il Bianco), usciamo con true.
        for (int offset : NeighborOffsets) {
            Index neighbor = pos + offset;
            if (IsValidIndex(neighbor) && HasPieceAt(neighbor)) return true;
        }
        return false;
    }

    // 3. REGOLA GENERALE (Dal turno 3 in poi): Devo toccare almeno un mio pezzo e NESSUN pezzo avversario
    bool foundFriend = false;

    for (int offset : NeighborOffsets) {
        Index neighbor = pos + offset;

        // Bounds check: ignora celle fuori dalla board
        if (!IsValidIndex(neighbor)) continue;

        PieceName neighborPiece = GetPieceAt(neighbor);
        
        if (neighborPiece != PieceName::INVALID) {
            // Appena tocchiamo un pezzo nemico, la mossa è illegale!
            if (GetColor(neighborPiece) != myColor) {
                return false; 
            }
            // Se non è nemico, è per forza amico. Ce lo segniamo.
            foundFriend = true;
        }
    }

    return foundFriend;
}

// Mette in positions gli indici delle celle in cui il giocatore corrente può posizionare un pezzo

void Board::GetValidPlacements(Color color, std::vector<Index>& positions) const
{
    // Turno 0: board vuota, l'unica posizione valida è BoardCenter
    if (currentTurn == 0)
    {
        positions.push_back(BoardCenter);
        return;
    }

    bool candidateAdded[BoardSize];
    memset(candidateAdded, 0, sizeof(candidateAdded));

    // logica: per ogni pezzo sulla board, vediamo quali tra i suoi 6 vicini sono liberi e quelli liberi saranno validi
    for (int p = 0; p < NumPieceNames; p++)
    {
        if (PieceInHand(static_cast<PieceName>(p))) continue; // cerchiamo i pezzi già sulla board

        Index pos = piecesPositions[p]; // posizione del pezzo considerato

        for (int i = 0; i < 6; i++)
        {
            Index neighbor = pos + NeighborOffsets[i];
            if (!IsValidIndex(neighbor)) continue;
            if (candidateAdded[neighbor]) continue;
            if (!CanPlaceAt(neighbor, color, currentTurn)) continue;

            candidateAdded[neighbor] = true;
            positions.push_back(neighbor);
        }
    }
}

} // namespace HiveGotThis