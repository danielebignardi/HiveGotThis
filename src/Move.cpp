#include <cctype>
#include <cstring>
#include <ostream>
#include <sstream>
#include <string>

#include "Constants.h"
#include "Move.h"

namespace HiveGotThis
{

// Canonicalizza il case di un token-pezzo UHP: colore (1° char) minuscolo,
// lettera del bug (2° char) maiuscola, eventuale numero invariato.
// Es. "wq" -> "wQ", "ws1" -> "wS1", "WB1" -> "wB1".
static std::string CanonicalizePieceToken(const std::string& token)
{
    std::string r = token;
    if (!r.empty())
        r[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(r[0])));
    if (r.size() > 1)
        r[1] = static_cast<char>(std::toupper(static_cast<unsigned char>(r[1])));
    return r;
}

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
    size_t h = 0;
    h |= static_cast<size_t>(move.Piece);
    h |= (static_cast<size_t>(static_cast<uint16_t>(move.Source)) << 8);
    h |= (static_cast<size_t>(static_cast<uint16_t>(move.Destination)) << 24);
    return h;
}

// NUOVA VERSIONE VELOCE INSERITA CORRETTAMENTE
std::string BuildMoveString(bool isPass, PieceName startPiece, char beforeSeperator, 
                            PieceName endPiece, char afterSeperator)
{
    if (isPass) return PassMoveString;

    std::string result = GetEnumString(startPiece);

    if (endPiece != PieceName::INVALID)
    {
        result += " ";
        std::string endPieceStr = GetEnumString(endPiece);

        if (beforeSeperator != '\0')
        {
            result += beforeSeperator;
            result += endPieceStr;
        }
        else if (afterSeperator != '\0')
        {
            result += endPieceStr;
            result += afterSeperator;
        }
        else
        {
            result += endPieceStr;
        }
    }
    return result;
}

bool TryNormalizeMoveString(const std::string &moveString, std::string &result)
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

bool TryNormalizeMoveString(const std::string &moveString, bool &isPass, PieceName &startPiece, char &beforeSeperator,
                            PieceName &endPiece, char &afterSeperator)
{
    isPass = false;
    startPiece = PieceName::INVALID;
    beforeSeperator = '\0';
    endPiece = PieceName::INVALID;
    afterSeperator = '\0';

    if (moveString.empty()) return false;

    std::string piece1 = "";
    std::string piece2 = "";
    size_t i = 0;

    while (i < moveString.length() && moveString[i] == ' ') i++;

    while (i < moveString.length() && moveString[i] != ' ') {
        piece1 += moveString[i];
        i++;
    }

    if (piece1 == PassMoveString) {
        isPass = true;
        return true;
    }

    while (i < moveString.length() && moveString[i] == ' ') i++;

    if (i < moveString.length()) {
        char c = moveString[i];
        if (c == '-' || c == '/' || c == '\\') {
            beforeSeperator = c;
            i++;
        }
    }

    // Salta eventuali spazi tra il separatore e il pezzo di riferimento
    // (es. "wB1 - wQ", "wq /  ws1"): l'UHP canonico non li ha, ma il
    // normalizzatore deve accettare input "sporco".
    while (i < moveString.length() && moveString[i] == ' ') i++;

    while (i < moveString.length() && moveString[i] != ' ' &&
           moveString[i] != '-' && moveString[i] != '/' && moveString[i] != '\\') {
        piece2 += moveString[i];
        i++;
    }

    // Salta eventuali spazi prima di un separatore in coda (es. "wQ wS1 -")
    while (i < moveString.length() && moveString[i] == ' ') i++;

    if (i < moveString.length()) {
        char c = moveString[i];
        if (c == '-' || c == '/' || c == '\\') {
            afterSeperator = c;
        }
    }

    startPiece = GetPieceNameValue(CanonicalizePieceToken(piece1).c_str());
    if (startPiece == PieceName::INVALID) return false;

    if (!piece2.empty()) {
        endPiece = GetPieceNameValue(CanonicalizePieceToken(piece2).c_str());
        if (endPiece == PieceName::INVALID) return false;
    }

    return true;
}

} // FINE DEL NAMESPACE