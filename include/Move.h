#ifndef MOVE_H
#define MOVE_H

#include <string>

#include "Enums.h"
#include "Position.h"

namespace HiveGotThis
{
struct Move
{
    PieceName Piece;
    Index Source;
    Index Destination;
};

static const Move PassMove{PieceName::INVALID, NullIndex, NullIndex};

bool operator==(Move const &lhs, Move const &rhs);
bool operator!=(Move const &lhs, Move const &rhs);

size_t hash(Move const &move);

// 1. Funzione "pulitrice" 
bool TryNormalizeMoveString(const std::string &moveString, std::string &result);

// 2. Funzione "smontatrice" 
bool TryNormalizeMoveString(const std::string &moveString, bool &isPass, PieceName &startPiece, 
                            char &beforeSeperator, PieceName &endPiece, char &afterSeperator);

// 3. Funzione "assemblatrice" 
std::string BuildMoveString(bool isPass, PieceName startPiece, char beforeSeperator, 
                            PieceName endPiece, char afterSeperator);

std::ostream& operator<<(std::ostream& os, const Move& move);

} // LA PARENTESI DEL NAMESPACE DEVE STARE SOPRA L'ENDIF!
#endif