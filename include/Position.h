#ifndef POSITION_H
#define POSITION_H

#include <cmath>
#include <cstdlib>
#include "Constants.h"
#include "Enums.h"

namespace HiveGotThis
{

// -----------------------------------------------------------------------------
// 1. IL TIPO "INDEX" (Il cuore dell'ottimizzazione)
// -----------------------------------------------------------------------------

// Usiamo un intero a 16 bit. 
// Leggero, veloce da copiare, e supporta indici fino a 32.767 (noi ne usiamo 16.384).
using Index = int16_t;

// Valore sentinella per indicare "Nessuna Posizione"
constexpr Index NullIndex = -1;

//Indice della posizione centrale della board, la partita dovra' iniziare da questa posizione
//BoardSize/2 mi porta all'inizio della riga centrale, BoardWidth/2 mi porta a meta' riga
constexpr Index BoardCenter = (BoardSize / 2) + (BoardWidth / 2);

/*
   Mappatura Direzioni -> Matematica 1D (Axial Coordinates flattened)
   
   Supponendo BOARD_WIDTH = 128:
   
   Right (0)      -> +1
   DownRight (1)  -> +128  (Vai alla riga sotto)
   DownLeft (2)   -> +127  (Riga sotto, uno a sinistra)
   Left (3)       -> -1
   UpLeft (4)     -> -128  (Vai alla riga sopra)
   UpRight (5)    -> -127  (Riga sopra, uno a destra)
*/

constexpr Index NeighborOffsets[6] = {
    1,                  // Direction::Right (0)
    BoardWidth,        // Direction::DownRight (1)
    BoardWidth - 1,    // Direction::DownLeft (2)
    -1,                 // Direction::Left (3)
    -BoardWidth,       // Direction::UpLeft (4)
    -BoardWidth + 1    // Direction::UpRight (5)
};

// Restituisce l'indice della cella vicina in una data direzione.
// VELOCITÀ: 1 Somma + 1 Accesso Array. (Quasi istantaneo)
inline Index GetNeighborAt(Index idx, Direction dir)
{
    return idx + NeighborOffsets[static_cast<uint8_t>(dir)];
}

// Verifica se l'indice è dentro l'array (Safety check)
inline bool IsValidIndex(Index idx)
{
    return idx >= 0 && idx < BoardSize;
}

//rispetto a MzingaCPP: non è necessario fare l'overloading di == e != in quanto basta conforntare direttamente gli indici (interi a 16 bit)
//NeighborOffsets corrisponde a NeighborDeltas
//non è necessario implementare la funzione di hash in quanto nel nostro caso la posizione è "gia' un hash"

}

#endif