#include "Board.h"
#include <iterator>
#include <algorithm>
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



// - - - - - - - - - - FUNZIONI DI UTILITÀ PER LA GENERAZIONE DELLE MOSSE - - - - - - - - - -

// SARA 27 FEB
// FREEDOM OF MOVEMENT RULE
bool Board::CanSlide(Index pos, Direction dir, SlideMode mode) const
{
    uint8_t d = static_cast<uint8_t>(dir);
    Index gate1 = pos + SlideGates[d][0];
    Index gate2 = pos + SlideGates[d][1];

    if (mode == SlideMode::Beetle)
    {
        Index dest    = pos + NeighborOffsets[d];
        int maxHeight = std::max((int)stackHeight[pos], (int)stackHeight[dest]);
        int minGate   = std::min((int)stackHeight[gate1], (int)stackHeight[gate2]);
        return minGate < maxHeight;
    }

    // SlideMode::Ground — pezzi a terra
    bool g1 = HasPieceAt(gate1);
    bool g2 = HasPieceAt(gate2);
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

void Board::GetOneSlideSteps(Index from, SlideMode mode, bool visited[], std::vector<Index>& result) const
{
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = from + NeighborOffsets[i];
        if (!IsValidIndex(neighbor)) continue;
        if (mode == SlideMode::Ground && HasPieceAt(neighbor)) continue;    // se siamo in modalità Ground allora saltiamo le celle su cui c'è già un pezzo
        if (visited[neighbor]) continue;
        if (!CanSlide(from, static_cast<Direction>(i), mode)) continue;
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
    GetOneSlideSteps(pos, SlideMode::Ground, visited, steps1);

    for (Index step1 : steps1)      
    {
        visited[step1] = true;

        steps2.clear();                     
        GetOneSlideSteps(step1, SlideMode::Ground, visited, steps2);

        for (Index step2 : steps2)
        {
            visited[step2] = true;

            steps3.clear();
            GetOneSlideSteps(step2, SlideMode::Ground, visited, steps3);

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
    GetOneSlideSteps(pos, SlideMode::Ground, visited, queue);
    for (Index cell : queue) visited[cell] = true;

    int head = 0;
    while (head < (int)queue.size()) {
        Index current = queue[head++];
        int sizeBefore = queue.size();
        GetOneSlideSteps(current, SlideMode::Ground, visited, queue);

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
            if (!CanSlide(step2, static_cast<Direction>(i), SlideMode::Ground)) continue;

            Move move = {piece, pos, neighbor};
            moves.push_back(move);
        }
    }
}

void Board::GetPillbugMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    // Movimento 1: slide di 1 passo come la Queen Bee
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = pos + NeighborOffsets[i];
        if (!IsValidIndex(neighbor)) continue;
        if (HasPieceAt(neighbor)) continue;
        if (!CanSlide(pos, static_cast<Direction>(i), SlideMode::Ground)) continue;
        moves.push_back({piece, pos, neighbor});
    }

    // Movimento 2: sposta un pezzo adiacente sopra il Pillbug e depositalo
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
        if (!CanSlide(neighborPos, static_cast<Direction>(i), SlideMode::Beetle)) continue;

        // Deposita il pezzo in ogni cella vuota adiacente al Pillbug
        for (int j = 0; j < 6; j++)
        {
            Index dest = pos + NeighborOffsets[j];
            if (!IsValidIndex(dest)) continue;
            if (HasPieceAt(dest)) continue;
            if (dest == neighborPos) continue;

            // Gate rule con altezze per il deposito (pos → dest)
            if (!CanSlide(pos, static_cast<Direction>(j), SlideMode::Beetle)) continue;

            moves.push_back({neighborPiece, neighborPos, dest});
        }
    }
}

void Board::GetMosquitoMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    // Caso speciale: se la Mosquito è sopra un pezzo, si muove solo come Beetle
    if (stackHeight[pos] > 1)
    {
        GetBeetleMoves(piece, moves);
        return;
    }

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

        switch (type)
        {
            case BugType::QueenBee:    GetQueenBeeMoves(piece, moves);    break;
            case BugType::Beetle:      GetBeetleMoves(piece, moves);      break;
            case BugType::Grasshopper: GetGrasshopperMoves(piece, moves); break;
            case BugType::Spider:      GetSpiderMoves(piece, moves);      break;
            case BugType::SoldierAnt:  GetSoldierAntMoves(piece, moves);  break;
            case BugType::Ladybug:     GetLadybugMoves(piece, moves);     break;
            case BugType::Pillbug:     GetPillbugMoves(piece, moves);     break;
            default: break;
        }
    }
    // Se processedTypes è tutto false, non è stata generata nessuna mossa <-> adiacente solo a Mosquito, moves resta vuoto
}

void Board::GetValidMoves(std::vector<Move>& moves) const
{
    moves.clear();

    // Gioco non in corso, nessuna mossa
    if (!GameInProgress(boardState)) return;

    PieceName queen = (currentColor == Color::White) ? PieceName::wQ : PieceName::bQ;
    bool queenInPlay = PieceInPlay(queen);

    // Regola della Regina: al quarto turno deve essere piazzata
    // Bianco gioca ai turni 1,3,5,7 — quarto turno = 7
    // Nero gioca ai turni 2,4,6,8 — quarto turno = 8
    int queenDeadline = (currentColor == Color::White) ? 7 : 8;
    bool mustPlaceQueen = !queenInPlay && (currentTurn >= queenDeadline);

    if (mustPlaceQueen)
    {
        std::vector<Move> placements;
        GetValidPlacements(currentColor, placements);
        for (Move& placement : placements)
        {
            Move move = {queen, NullIndex, placement.Destination};
            moves.push_back(move);
        }
        return;
    }

    // Piazzamento: combina ogni posizione valida con ogni pezzo in mano
    std::vector<Move> placements;
    GetValidPlacements(currentColor, placements);
    for (Move& placement : placements)
    {
        for (int p = 0; p < NumPieceNames; p++)
        {
            PieceName piece = static_cast<PieceName>(p);
            if (GetColor(piece) != currentColor) continue;
            if (!PieceInHand(piece)) continue;
            if (!PieceNameIsEnabledForGameType(piece, gameType)) continue;

            Move move = {piece, NullIndex, placement.Destination};
            moves.push_back(move);
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

    // Se non ci sono mosse valide, l'unica opzione è passare
    if (moves.empty())
    {
        moves.push_back(PassMove);
    }
}

// SARA 27 FEB
bool Board::CanPlaceAt(Index pos, Color myColor, int currentTurn) const
{
    // 1. Posizione deve essere Vuota 
    if (HasPieceAt(pos)) return false;

    // 2. CASI SPECIALI (Turno 1 e 2)
    if (currentTurn <= 2) {
        if (currentTurn == 1) return true; // Turno 1: metti dove vuoi
        
        // Turno 2: il Nero DEVE toccare il Bianco.
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

void Board::GetValidPlacements(Color color, std::vector<Move>& moves) const
{
    bool candidateAdded[BoardSize];
    memset(candidateAdded, 0, sizeof(candidateAdded));

    for (int p = 0; p < NumPieceNames; p++)
    {
        if (PieceInHand(static_cast<PieceName>(p))) continue;               // cerchiamo i pezzi già sulla board

        Index pos = piecesPositions[p];

        for (int i = 0; i < 6; i++)
        {
            Index neighbor = pos + NeighborOffsets[i];
            if (!IsValidIndex(neighbor)) continue;
            if (candidateAdded[neighbor]) continue;
            if (!CanPlaceAt(neighbor, color, currentTurn)) continue;

            candidateAdded[neighbor] = true;
            moves.push_back({PieceName::INVALID, NullIndex, neighbor});
        }
    }
}

} // namespace HiveGotThis