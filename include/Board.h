#include "Position.h"
#include "Move.h"
#include "Enums.h" 
#include "Constants.h"

namespace HiveGotThis
{  

class Board
{
    private:
        // Rappresentazione interna della board (Array di PieceName), praticamente una matrice BoardWidth x BoardWidth appiattita in un array monodimensionale.
        PieceName cells[BoardSize];

        PieceName below[NumPieceNames]; // Per ogni pezzo, quale pezzo è sotto di esso (o INVALID se nessuno)

        Index piecePositions[NumPieceNames]; // Per ogni pezzo, la sua posizione attuale (NullIndex se non sulla board)

        GameType gameType = GameType::Base;
        BoardState boardState = BoardState::NotStarted;
        Color currentColor = Color::White;
        int currentTurn = 0;

    public:
        Board(GameType gameType);
        //CATEGORIA 1: funzioni di stato

        BoardState GetBoardState();
        int GetCurrentTurn();

        std::string GetGameString();
    //    std::shared_ptr<MoveSet> GetValidMoves();
    
};   

}