#include <cstring>
#include <ostream>
#include <sstream>
#include <string>

#include "Constants.h"
#include "Move.h"

namespace HiveGotThis
{

std::ostream& operator<<(std::ostream& os, const Move& move)
{
    os << GetEnumString(move.Piece) << ":" << move.Source << "->" << move.Destination;
    return os;
}

bool operator==(Move const &lhs, Move const &rhs)
{
    return lhs.Piece == rhs.Piece && lhs.Source == rhs.Source && lhs.Destination == rhs.Destination;
}

bool operator!=(Move const &lhs, Move const &rhs)
{
    return !(lhs == rhs);
}

size_t hash(Move const &move)
{
    /* STRATEGIA: BIT PACKING
       Invece di calcolare hash complessi, appendiamo i tre dati in un unico numero a 64 bit.
       
       Layout dei bit nel size_t finale:
       [ ...Vuoto... ] [ Destination (16 bit) ] [ Source (16 bit) ] [ Piece (8 bit) ]
       63           40 39                    24 23               8 7              0
    */
    size_t h = 0;

    // 1. Inseriamo il Pezzo nei primi 8 bit (0-7)
    // Castiamo a size_t per lavorare su 64 bit
    h |= static_cast<size_t>(move.Piece);

    // 2. Inseriamo Source nei successivi 16 bit (8-23)
    // Nota: Castiamo prima a uint16_t per evitare problemi se l'indice è -1 (NullIndex)
    h |= (static_cast<size_t>(static_cast<uint16_t>(move.Source)) << 8);

    // 3. Inseriamo Destination nei successivi 16 bit (24-39)
    h |= (static_cast<size_t>(static_cast<uint16_t>(move.Destination)) << 24);

    return h;
}

std::string BuildMoveString(bool &isPass, PieceName &startPiece, char &beforeSeperator, PieceName &endPiece, char &afterSeperator)
{
    if (isPass)
    {
        return PassMoveString;
    }

    std::ostringstream resultStream;

    resultStream << GetEnumString(startPiece);

    if (endPiece != PieceName::INVALID)
    {
        resultStream << " ";
        if (beforeSeperator != '\0')
        {
            resultStream << beforeSeperator << GetEnumString(endPiece);
        }
        else if (afterSeperator != '\0')
        {
            resultStream << GetEnumString(endPiece) << afterSeperator;
        }
        else
        {
            resultStream << GetEnumString(endPiece);
        }
    }

    return resultStream.str();
}

bool TryNormalizeMoveString(std::string const &moveString, std::string &result)
{
    bool isPass;
    PieceName startPiece;
    char beforeSeperator;
    PieceName endPiece;
    char afterSeperator;

    if (TryNormalizeMoveString(moveString, isPass, startPiece, beforeSeperator, endPiece, afterSeperator))
    {
        result = BuildMoveString(isPass, startPiece, beforeSeperator, endPiece, afterSeperator);
        return true;
    }

    result = "";
    return false;
}

bool TryNormalizeMoveString(std::string const &moveString, bool &isPass, PieceName &startPiece, char &beforeSeperator, PieceName &endPiece, char &afterSeperator)
{
    isPass = false;
    startPiece = PieceName::INVALID;
    beforeSeperator = '\0';
    endPiece = PieceName::INVALID;
    afterSeperator = '\0';

    std::ostringstream piece1;
    std::ostringstream piece2;

    int itemsFound = 0;
    for (int i = 0; i < moveString.length(); i++)
    {
        if (itemsFound == 0 && moveString[i] != ' ')
        {
            // Start of piece1, save and bump
            piece1 << moveString[i];
            itemsFound++;
        }
        else if (itemsFound == 1)
        {
            if (moveString[i] != ' ')
            {
                // Still part of piece1
                piece1 << moveString[i];
            }
            else
            {
                // Skip, looking for beforeSeperator now
                itemsFound++;
            }
        }
        else if (itemsFound == 2)
        {
            if (moveString[i] != ' ')
            {
                // First non-space
                if (moveString[i] == '-' || moveString[i] == '/' || moveString[i] == '\\')
                {
                    // beforeSeparator found
                    beforeSeperator = moveString[i];
                }
                else
                {
                    // Start of piece2
                    piece2 << moveString[i];
                }
                itemsFound++;
            }
        }
        else if (itemsFound == 3)
        {
            if (moveString[i] != ' ')
            {
                if (moveString[i] == '-' || moveString[i] == '/' || moveString[i] == '\\')
                {
                    // afterSeperator found
                    afterSeperator = moveString[i];
                    break;
                }
                else
                {
                    // Still of piece2
                    piece2 << moveString[i];
                }
            }
            else
            {
                break;
            }
        }
    }

    std::string piece1Str = piece1.str();

    if (strcmp(piece1Str.c_str(), PassMoveString) == 0)
    {
        isPass = true;
        startPiece = PieceName::INVALID;
        beforeSeperator = '\0';
        endPiece = PieceName::INVALID;
        afterSeperator = '\0';
        return true;
    }

    startPiece = GetPieceNameValue(piece1Str.c_str());

    if (startPiece != PieceName::INVALID)
    {
        std::string piece2Str = piece2.str();
        endPiece = GetPieceNameValue(piece2Str.c_str());

        if ((strcmp(piece2Str.c_str(), "") == 0 && beforeSeperator == '\0' && afterSeperator == '\0')
            || endPiece != PieceName::INVALID)
        {
            return true;
        }
    }

    isPass = false;
    startPiece = PieceName::INVALID;
    beforeSeperator = '\0';
    endPiece = PieceName::INVALID;
    afterSeperator = '\0';
    return false;
}
}