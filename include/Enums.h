#ifndef ENUMS_H
#define ENUMS_H

#include <cstdint>
#include <string>


namespace HiveGotThis
{

enum class Color : uint8_t
{
    White = 0,
    Black = 1,
    NumColors = 2,
    INVALID = 255
};

enum class BoardState : uint8_t
{
    NotStarted = 0,
    InProgress,
    Draw,
    WhiteWins,
    BlackWins,
    INVALID = 255
};

enum class GameType : uint8_t
{
    Base = 0,
    BaseM,   // + Mosquito
    BaseL,   // + Ladybug
    BaseP,   // + Pillbug
    BaseML,
    BaseMP,
    BaseLP,
    BaseMLP,
    INVALID = 255
};

enum class BugType : uint8_t
{
    QueenBee = 0,
    Spider,
    Beetle,
    Grasshopper,
    SoldierAnt,
    Mosquito,
    Ladybug,
    Pillbug,
    NumBugTypes,
    INVALID = 255
};

//attenzione: questo è enum vecchia scuola e non enum class, perché voglio poterlo usare come indice in array senza cast
enum PieceName : uint8_t
{
    // --- WHITE PIECES (0 - 13) ---
    wQ = 0,
    wS1, wS2,
    wB1, wB2,
    wG1, wG2, wG3,
    wA1, wA2, wA3,
    wM, wL, wP,

    // --- BLACK PIECES (14 - 27) ---
    bQ,
    bS1, bS2,
    bB1, bB2,
    bG1, bG2, bG3,
    bA1, bA2, bA3,
    bM, bL, bP,

    NumPieceNames,
    INVALID = 255
};

enum class Direction : uint8_t
{
    Right = 0,      // +Q
    DownRight = 1,  // +R
    DownLeft = 2,   // -Q +R
    Left = 3,       // -Q
    UpLeft = 4,     // -R
    UpRight = 5,    // +Q -R
    
    NumDirections = 6,
    
    INVALID = 255
};

enum class SlideMode : uint8_t
{
    Ground,
    Beetle,
    LadyBug  // come Beetle ma con stackHeight[pos]+1 (usato per la discesa al passo 3)
};


std::string GetEnumString(BoardState const &value);
std::string GetEnumString(Color const &value);
std::string GetEnumString(PieceName const &value);
std::string GetEnumString(GameType const &value);

PieceName GetPieceNameValue(const char *str);
GameType GetGameTypeValue(const char *str);

bool GameInProgress(BoardState const &value);
bool GameIsOver(BoardState const &value);

// Definite inline in header: sono chiamate milioni di volte nei loop di generazione
// mosse; tenerle qui permette al compilatore di inlinearle (niente call cross-TU).
inline Color GetColor(PieceName p)
{
    return (p < PieceName::bQ) ? Color::White : Color::Black;
}

inline BugType GetBugType(PieceName p)
{
    uint8_t id = p;

    // Se è un pezzo nero, sottraiamo 14 per trattarlo come bianco
    if (id >= 14 && id < 28)
    {
        id -= 14;
    }

    switch (id)
    {
        case PieceName::wQ:  return BugType::QueenBee;
        case PieceName::wS1:
        case PieceName::wS2: return BugType::Spider;
        case PieceName::wB1:
        case PieceName::wB2: return BugType::Beetle;
        case PieceName::wG1:
        case PieceName::wG2:
        case PieceName::wG3: return BugType::Grasshopper;
        case PieceName::wA1:
        case PieceName::wA2:
        case PieceName::wA3: return BugType::SoldierAnt;
        case PieceName::wM:  return BugType::Mosquito;
        case PieceName::wL:  return BugType::Ladybug;
        case PieceName::wP:  return BugType::Pillbug;
        default:             return BugType::INVALID;
    }
}

// Controlla se il pezzo è legale per il tipo di partita corrente
inline bool PieceNameIsEnabledForGameType(PieceName p, GameType gt)
{
    switch (GetBugType(p))
    {
        case BugType::Mosquito:
            return (gt == GameType::BaseM || gt == GameType::BaseML || gt == GameType::BaseMP || gt == GameType::BaseMLP);
        case BugType::Ladybug:
            return (gt == GameType::BaseL || gt == GameType::BaseML || gt == GameType::BaseLP || gt == GameType::BaseMLP);
        case BugType::Pillbug:
            return (gt == GameType::BaseP || gt == GameType::BaseMP || gt == GameType::BaseLP || gt == GameType::BaseMLP);
        case BugType::INVALID:
            return false;
        default:
            return true; // I pezzi base sono sempre ammessi
    }
}

bool IsPieceNameValid(PieceName p);

 /*
 // Calcola la direzione a SINISTRA (Antiorario)
// Formula: (Dir + 5) % 6
inline Direction LeftOf(Direction d)
{
    if (d >= Direction::NumDirections) return d;
    return static_cast<Direction>((static_cast<uint8_t>(d) + 5) % 6);
}

// Calcola la direzione a DESTRA (Orario)
// Formula: (Dir + 1) % 6
inline Direction RightOf(Direction d)
{
    if (d >= Direction::NumDirections) return d;
    return static_cast<Direction>((static_cast<uint8_t>(d) + 1) % 6);
}
 */

// Calcola la direzione OPPOSTA
// Formula: (Dir + 3) % 6
inline Direction Opposite(Direction d)
{
    if (d >= Direction::NumDirections) return d;
    return static_cast<Direction>((static_cast<uint8_t>(d) + 3) % 6);
}



}

#endif