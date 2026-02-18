#include "Board.h"
#include <iterator>
#include <algorithm>
#include <random>

namespace HiveGotThis
{  

Board::Board(GameType gameType)
{
    this->gameType = gameType;

    // Riempie tutto l'array 'cells' con INVALID
    std::fill(std::begin(cells), std::end(cells), PieceName::INVALID);
    
    // Riempie tutto 'below' con INVALID
    std::fill(std::begin(below), std::end(below), PieceName::INVALID);

    // Riempie tutto 'piecesPositions' con NullIndex
    std::fill(std::begin(piecesPositions), std::end(piecesPositions), NullIndex);

    //GESTIONE ZOBRIST
    currentHash = 0; // Board vuota = Hash 0
    hashHistory.clear();
    hashHistory.reserve(100); // Ottimizzazione: pre-alloca spazio
    
    // Aggiungi lo stato iniziale (vuoto) allo storico
    hashHistory.push_back(currentHash);

    // Azzera le altezze delle pile
    std::fill(std::begin(stackHeight), std::end(stackHeight), 0);
}

PieceName Board::PopAt(Index position)
{
    if (HasPieceAt(position))
    {
        PieceName topPiece = GetPieceAt(position);

        piecesPositions[topPiece] = NullIndex; // Il pezzo non è più sulla board

        cells[position] = below[topPiece]; //aggiornato il pezzo in cima alla posizione position
        
        below[topPiece] = PieceName::INVALID; // Il pezzo non ha più nulla sotto di lui

        // AGGIORNA ALTEZZA 
        stackHeight[position]--;
        int h = stackHeight[position];

        // C. AGGIORNA HASH (Rimuovi)
        // Usiamo lo stesso XOR usato in PushAt per annullare l'effetto
        currentHash ^= (ZobristTable[topPiece][position] ^ ZobristLevel[topPiece][h]);

        return topPiece;
    }
    else
    {
        return PieceName::INVALID; // Non c'è nulla da rimuovere
    }
}

void Board::PushAt(PieceName pieceName, Index position)
{
    assert(IsValidIndex(position) && "PushAt chiamato con indice invalido!");
    
    // A. CALCOLO HASH ROBUSTO
    // Recuperiamo l'altezza attuale dove andrà il pezzo
    int h = stackHeight[position];

    // XOR combinato: Posizione univoca ^ Livello univoco
    // Questo "garantisce" che (A sotto, B sopra) sia diverso da (B sotto, A sopra)
    currentHash ^= (ZobristTable[pieceName][position] ^ ZobristLevel[pieceName][h]);

    // Aggiorna il pezzo in cima alla posizione position
    PieceName currentTop = GetPieceAt(position);
    cells[position] = pieceName;

    // Aggiorna la catena di pezzi sotto la posizione
    below[pieceName] = currentTop; // Il nuovo pezzo ora ha come "sotto" il vecchio top (che potrebbe essere INVALID se non c'era nulla)

    // Aggiorna la posizione del pezzo
    piecesPositions[pieceName] = position;

    // AGGIORNA ALTEZZA
    stackHeight[position]++;
}

void Board::MovePiece(PieceName pieceName, Index newPosition)
{    
    assert(IsValidIndex(newPosition) && "MovePiece chiamato con indice invalido!");

    if (PieceInPlay(pieceName))
    {
        // Il pezzo è sulla board, quindi lo rimuoviamo dalla vecchia posizione
        PopAt(piecesPositions[pieceName]);
    }
    
    PushAt(pieceName, newPosition);
}

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

bool Board::CanMoveWithoutBreakingHive(PieceName piece) const
{
    Index pos = piecesPositions[piece];
    
    // CASO 1: Il pezzo non è sulla board (è in mano)
    if (PieceInHand(piece)) return true; 

    // 2. CHECK FONDAMENTALE: Sono coperto?
    // Se GetPieceAt(pos) non sono io, significa che c'è qualcuno sopra di me!
    // Non posso muovermi, quindi la domanda "rompo l'alveare?" è irrilevante.
    // Ritorniamo false (o gestiamo l'errore, ma false blocca la mossa).
    if (GetPieceAt(pos) != piece) {
        assert(GetPieceAt(pos) == piece && "Si è provata CanMoveWithoutBreakingHive su un pezzo coperto");
        return false; 
    }

    // Sono in cima, ma c'è qualcuno sotto?
    // Se sono appoggiato su un altro pezzo, posso spostarmi senza rompere nulla.
    if (below[piece] != PieceName::INVALID) {
        return true; 
    }

    // 4. CHECK BASE: Sono l'unico pezzo in questa casella (a terra).
    // Devo simulare la rimozione e vedere se tutto resta connesso.
    // Invece di togliere il pezzo e verificare che l'hive non sia rotto basta dire a IsOneHive di considerare pos come se fosse vuota
    return IsOneHive(pos);
}

bool Board::IsOneHive(Index ignorePos) const
{
    // OTTIMIZZAZIONE 1: Array visited indicizzato per Pezzo (O(1))
    // Usiamo l'ID del pezzo come indice. Inizializzato a false.
    bool visitedPieces[NumPieceNames] = { false };
    
    // OTTIMIZZAZIONE 2: Vector pre-allocato come coda
    // Riserviamo spazio per tutti i pezzi possibili per evitare allocazioni
    static std::vector<Index> queue; // static per non riallocare ogni volta (opzionale ma veloce)
    queue.clear();
    queue.reserve(NumPieceNames);

    int nodesFound = 0;
    int nodesExpected = 0;
    Index startNode = NullIndex;

    // 1. Conta i nodi attesi (ovvero il numero di posizioni occupate esclusa ignorePos) e trova startNode
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

    // 2. Setup BFS
    queue.push_back(startNode);
    
    // Segniamo come visitato il pezzo che si trova a startNode
    PieceName startPiece = cells[startNode];
    visitedPieces[startPiece] = true;
    nodesFound++;

    // 3. Loop BFS (usando indice 'head' invece di pop)
    int head = 0;
    while(head < queue.size()) {
        Index currentPos = queue[head++]; // Leggi e avanza cursore

        // Controlla i 6 vicini
        for (int offset : NeighborOffsets) {
            Index neighborPos = currentPos + offset;

            // Ignora la casella "rimossa"
            if (neighborPos == ignorePos) continue;

            // Leggi chi c'è (O(1))
            PieceName neighborPiece = cells[neighborPos];

            // Se vuoto, salta
            if (neighborPiece == PieceName::INVALID) continue;

            // OTTIMIZZAZIONE 1 in azione: Controllo istantaneo
            if (visitedPieces[neighborPiece]) continue;

            // Trovato nuovo pezzo connesso!
            visitedPieces[neighborPiece] = true; // Segna come visto
            queue.push_back(neighborPos);        // Aggiungi alla coda
            nodesFound++;
        }
    }

    return nodesFound == nodesExpected;
}


}