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

int Board::GetDirection(Index from, Index to) const
{
    for (int i = 0; i < 6; i++)
    {
        if (from + NeighborOffsets[i] == to) return i;
    }
    return -1;
}

bool Board::CanSlide(Index from, Index to) const
{
    int dir = GetDirection(from, to);
    assert(dir != -1 && "CanSlide chiamato con celle non adiacenti!");

    bool leftOccupied  = IsValidIndex(from + NeighborOffsets[(dir + 5) % 6]) && HasPieceAt(from + NeighborOffsets[(dir + 5) % 6]);
    bool rightOccupied = IsValidIndex(from + NeighborOffsets[(dir + 1) % 6]) && HasPieceAt(from + NeighborOffsets[(dir + 1) % 6]);

    if (leftOccupied && rightOccupied) return false;
    return leftOccupied || rightOccupied;
}

bool Board::CanSlideWithHeight(Index from, Index to) const
{
    int dir = GetDirection(from, to);
    assert(dir != -1 && "CanSlideWithHeight chiamato con celle non adiacenti!");

    Index left  = from + NeighborOffsets[(dir + 5) % 6];
    Index right = from + NeighborOffsets[(dir + 1) % 6];

    int hLeft  = IsValidIndex(left)  ? stackHeight[left]  : 0;
    int hRight = IsValidIndex(right) ? stackHeight[right] : 0;
    int hFrom  = stackHeight[from];
    int hTo    = stackHeight[to];

    return (std::min(hLeft, hRight) < std::max(hFrom, hTo));
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

void Board::GetOneSlideSteps(Index from, bool visited[], std::vector<Index>& result) const
{
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = from + NeighborOffsets[i];

        if (!IsValidIndex(neighbor)) continue;
        if (HasPieceAt(neighbor)) continue;
        if (visited[neighbor]) continue;
        if (!CanSlide(from, neighbor)) continue;

        result.push_back(neighbor);
    }
}




// - - - - - - - - - - GENERAZIONE DELLE MOSSE VALIDE - - - - - - - - - -

void Board::GetQueenBeeMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    bool visited[BoardSize];
    memset(visited, 0, sizeof(visited));
    visited[pos] = true;

    std::vector<Index> steps;
    GetOneSlideSteps(pos, visited, steps);

    for (Index dest : steps)
    {
        Move move = {piece, pos, dest};
        moves.push_back(move);
    }
}

void Board::GetBeetleMoves(PieceName piece, std::vector<Move>& moves) const
{
    // one hive rule
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    // stesse mosse della QueenBee
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = pos + NeighborOffsets[i];

        if (!IsValidIndex(neighbor)) continue;      // fuori dalla board
        
        if (!HasPieceAt(neighbor))                  // destinazione vuota -> valgono le stesse regole della QueenBee
        {
            if (CanSlide(pos, neighbor))
            {
                Move move = {piece, pos, neighbor};
                moves.push_back(move);
            }
        }   
        else
        {
            if (CanSlideWithHeight(pos, neighbor))
            {
                Move move = {piece, pos, neighbor};
                moves.push_back(move);
            }
        }
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
    // one hive rule
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    // celle già visitate
    bool visited[BoardSize];
    memset(visited, 0, sizeof(visited));

    // indici raggiungibili ad ognuno dei 3 passi
    std::vector<Index> steps1;
    std::vector<Index> steps2;
    std::vector<Index> steps3;

    // inizio: indici raggiungibili dopo un passo
    visited[pos] = true;
    steps1.clear();
    GetOneSlideSteps(pos, visited, steps1);

    for (Index step1 : steps1)      
    {
        visited[step1] = true;

        steps2.clear();                     
        GetOneSlideSteps(step1, visited, steps2);

        for (Index step2 : steps2)
        {
            visited[step2] = true;

            steps3.clear();
            GetOneSlideSteps(step2, visited, steps3);

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
    // one hive rule
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    // posizioni già visitate
    bool visited[BoardSize];
    memset(visited, 0, sizeof(visited));
    visited[pos] = true;

    // posizioni visitabili
    std::vector<Index> queue;
    queue.clear();
    GetOneSlideSteps(pos, visited, queue);

    int head = 0;
    while (head < (int)queue.size()) {
        visited[queue[head]] = true;
        GetOneSlideSteps(queue[head], visited, queue);
        head++;
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
        if (!CanSlideWithHeight(pos, neighbor)) continue;
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
            if (!CanSlideWithHeight(step1, neighbor)) continue;
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
            if (!CanSlideWithHeight(step2, neighbor)) continue;

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
        if (!CanSlide(pos, neighbor)) continue;

        Move move = {piece, pos, neighbor};
        moves.push_back(move);
    }

    // Movimento 2: sposta un pezzo adiacente
    std::vector<Index> emptyNeighbors;
    GetEmptyNeighbors(pos, emptyNeighbors);

    for (int i = 0; i < 6; i++)
    {
        Index neighborPos = pos + NeighborOffsets[i];
        if (!IsValidIndex(neighborPos)) continue;
        if (!HasPieceAt(neighborPos)) continue;

        PieceName neighborPiece = GetPieceAt(neighborPos);

        // Il pezzo deve essere in cima alla sua pila (stackHeight == 1)
        if (stackHeight[neighborPos] > 1) continue;

        // Il pezzo non deve essere nella lista cannotBeMoved
        if (cannotBeMoved[neighborPiece]) continue;

        // Il pezzo deve potersi muovere senza rompere l'hive
        if (!CanMoveWithoutBreakingHive(neighborPiece)) continue;

        // Gate rule con altezze per la raccolta (neighborPos → pos)
        if (!CanSlideWithHeight(neighborPos, pos)) continue;

        // Deposita il pezzo in ogni cella vuota adiacente alla Pillbug
        for (Index dest : emptyNeighbors)
        {
            // Non può tornare nella sua posizione originale
            if (dest == neighborPos) continue;

            // Gate rule con altezze per il deposito (pos → dest)
            if (!CanSlideWithHeight(pos, dest)) continue;

            Move move = {neighborPiece, neighborPos, dest};
            moves.push_back(move);
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
    // Se processedTypes è tutto false, non è stata generata nessuna mossa
    // (adiacente solo a Mosquito) — nessuna azione necessaria, moves resta vuoto
}

void Board::GetValidPlacements(Color color, std::vector<Move>& moves) const
{
    if (currentTurn == 0)
    {
        Move move = {PieceName::INVALID, NullIndex, BoardCenter};
        moves.push_back(move);
        return;
    }

    // Step 1: trova tutte le celle vuote adiacenti all'hive
    bool candidateAdded[BoardSize];
    memset(candidateAdded, 0, sizeof(candidateAdded));
    std::vector<Index> candidates;

    for (int p = 0; p < NumPieceNames; p++)
    {
        if (PieceInHand(static_cast<PieceName>(p))) continue;
        Index pos = piecesPositions[p];

        for (int i = 0; i < 6; i++)
        {
            Index neighbor = pos + NeighborOffsets[i];
            if (!IsValidIndex(neighbor)) continue;
            if (HasPieceAt(neighbor)) continue;
            if (candidateAdded[neighbor]) continue;
            candidateAdded[neighbor] = true;
            candidates.push_back(neighbor);
        }
    }

    // Step 2: filtra le celle valide
    for (Index candidate : candidates)
    {
        bool adjacentToFriend = false;
        bool adjacentToEnemy = false;

        for (int j = 0; j < 6; j++)
        {
            Index n = candidate + NeighborOffsets[j];
            if (!IsValidIndex(n)) continue;
            if (!HasPieceAt(n)) continue;
            if (GetColor(GetPieceAt(n)) == color)
                adjacentToFriend = true;
            else
                adjacentToEnemy = true;
        }

        if (currentTurn == 1)
        {
            // Black turno 1: qualsiasi cella adiacente all'hive va bene
            Move move = {PieceName::INVALID, NullIndex, candidate};
            moves.push_back(move);
        }
        else
        {
            // Dal turno 2: adiacente ad amico, non adiacente a nemico
            if (adjacentToFriend && !adjacentToEnemy)
            {
                Move move = {PieceName::INVALID, NullIndex, candidate};
                moves.push_back(move);
            }
        }
    }
}

void Board::GetValidMoves(std::vector<Move>& moves) const
{
    moves.clear();

    // Se la partita non è in corso, non ci sono mosse valide
    if (!GameInProgress(boardState)) return;

    Color color = currentColor;

    // Determina se la Regina del colore corrente è in gioco
    PieceName queen = (color == Color::White) ? PieceName::wQ : PieceName::bQ;
    bool queenInPlay = PieceInPlay(queen);

    // Regola della Regina: al quarto turno deve essere piazzata
    // White gioca ai turni 0,2,4,6 — il quarto turno di White è il turno 6
    // Black gioca ai turni 1,3,5,7 — il quarto turno di Black è il turno 7
    int queenDeadline = (color == Color::White) ? 6 : 7;
    bool mustPlaceQueen = !queenInPlay && (currentTurn >= queenDeadline);

    if (mustPlaceQueen)
    {
        // L'unica mossa valida è piazzare la Regina
        std::vector<Move> placements;
        GetValidPlacements(color, placements);
        for (Move& placement : placements)
        {
            Move move = {queen, NullIndex, placement.Destination};
            moves.push_back(move);
        }
        return;
    }

    // Piazzamento: per ogni pezzo in mano del colore corrente
    std::vector<Move> placements;
    GetValidPlacements(color, placements);
    for (Move& placement : placements)
    {
        for (int p = 0; p < NumPieceNames; p++)
        {
            PieceName piece = static_cast<PieceName>(p);
            if (GetColor(piece) != color) continue;
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
            if (GetColor(piece) != color) continue;
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

}