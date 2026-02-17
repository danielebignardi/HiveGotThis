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


}