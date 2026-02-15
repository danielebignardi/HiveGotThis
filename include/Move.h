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


//TODO: Queste funzioni sono un po' contorte, ma l'UHP è un formato di stringa molto specifico e non voglio scrivere un parser completo. L'obiettivo è solo convertire tra la rappresentazione interna e quella stringa, e pulire un po' le stringhe in ingresso.
//LA VERSIONE CORRENTE è COPIATA E INCOLLATA DA MZINGA, NON SO SE SIA CORRETTA, FORSE AVREBBE PIU' SENSO IMPLEMENTARNE UNA VERSIONE PIU' SEMPLICE (ASSUMENDO COME GIUSTA SOLO IL FORMATO PRECISO DI UHP, E RIFIUTANDO TUTTO IL RESTO)
std::string BuildMoveString(bool &isPass, PieceName &startPiece, char &beforeSeperator, PieceName &endPiece,
                            char &afterSeperator);

bool TryNormalizeMoveString(std::string const &moveString, std::string &result);
bool TryNormalizeMoveString(std::string const &moveString, bool &isPass, PieceName &startPiece, char &beforeSeperator,
                            PieceName &endPiece, char &afterSeperator);
}

#endif