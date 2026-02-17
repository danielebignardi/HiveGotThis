#include "Board.h"
#include <iterator>
#include <algorithm>

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
}

PieceName Board::PopAt(Index position)
{
    if (HasPieceAt(position))
    {
        PieceName topPiece = GetPieceAt(position);

        piecesPositions[topPiece] = NullIndex; // Il pezzo non è più sulla board

        cells[position] = below[topPiece]; //aggiornato il pezzo in cima alla posizione position
        
        below[topPiece] = PieceName::INVALID; // Il pezzo non ha più nulla sotto di lui

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
    
    // Aggiorna il pezzo in cima alla posizione position
    PieceName currentTop = GetPieceAt(position);
    cells[position] = pieceName;

    // Aggiorna la catena di pezzi sotto la posizione
    below[pieceName] = currentTop; // Il nuovo pezzo ora ha come "sotto" il vecchio top (che potrebbe essere INVALID se non c'era nulla)

    // Aggiorna la posizione del pezzo
    piecesPositions[pieceName] = position;

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


}