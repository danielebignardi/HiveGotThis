#ifndef BOARD_H
#define BOARD_H

#include "Position.h"
#include "Move.h"
#include "Enums.h" 
#include "Constants.h"

#include <cassert>
#include <vector>

namespace HiveGotThis
{  

class Board
{
    public:

        // - - - - - - - - - - RAPPRESENTAZIONE BOARD - - - - - - - - - -
        
        // Array di 16 384 (128 x 128) PieceNames
        PieceName cells[BoardSize];

        // Per ogni pezzo, quale pezzo è sotto di esso (INVALID se nessuno)
        PieceName below[NumPieceNames];

        // Per ogni pezzo, la sua posizione attuale (NullIndex se non sulla board)
        Index piecesPositions[NumPieceNames];



        // - - - - - - - - - - INFORMAZIONI GIOCO - - - - - - - - - -

        // Tipo del gioco
        GameType gameType = GameType::Base;
        
        // Stato del gioco
        BoardState boardState = BoardState::NotStarted;
        
        // Colore del giocatore attuale. Inizia il bianco
        Color currentColor = Color::White;

        // Il turno aumenta di uno a ogni mossa. Turno pari: bianco, turno dispari: nero
        int currentTurn = 0;

        // Pezzi che per questo turno non possono essere mossi
        bool cannotBeMoved[NumPieceNames];


        // - - - - - - - - - - FUNZIONI DI STATO - - - - - - - - - -

        // Restituisce la posizione del pezzo (ovvero un Index), o NullIndex se non è sulla board
        Index GetPosition(PieceName pieceName) const;
        
        // Restituisce il pezzo in cima se c'è, altrimenti INVALID
        PieceName GetPieceAt(Index position) const;

        // true se c'è un pezzo alla determinata posizione
        bool HasPieceAt(Index position) const;
        
        // true se il pezzo è in mano (non sulla board)
        bool PieceInHand(PieceName pieceName) const;
        
        // true se il pezzo è in gioco (sulla board)
        bool PieceInPlay(PieceName pieceName) const;
        
        // true se il pezzo è in cima alla pila nella sua posizione (utile per verificare se un pezzo è libero di muoversi)
        bool PieceIsOnTop(PieceName pieceName) const;



        // - - - - - - - - - - FUNZIONI DI MODIFICA - - - - - - - - - -

        // Rimuove e restituisce il pezzo in cima alla posizione, o INVALID se non c'è nulla
        PieceName PopAt(Index position);

        // Pusha il pezzo sulla pila in position. Non controlla consistenza.
        void PushAt(PieceName pieceName, Index position);
        
        // Muove un pezzo da dov'è a newPosition (PopAt + PushAt)
        void MovePiece(PieceName pieceName, Index newPosition);
        


        // - - - - - - - - - - FUNZIONI DI CONTROLLO DELLO STATO - - - - - - - - - -

        // true se il pezzo può spostarsi anche temporaneamente dalla sua posizione attuale senza rompere l'hive
        bool CanMoveWithoutBreakingHive(PieceName piece) const;
        
        // true se la configurazione attuale della board è un solo hive (condizione che deve essere garantita)
        bool IsOneHive(Index ignorePos) const;



        // - - - - - - - - - - FUNZIONI DI UTILITÀ PER LA GENERAZIONE DELLE MOSSE - - - - - - - - - -

        // Restituisce gli indici dei vicini occupati di una posizione
        void GetOccupiedNeighbors(Index pos, std::vector<Index>& result) const;
        
        // Restituisce gli indici dei vicini vuoti di una posizione
        void GetEmptyNeighbors(Index pos, std::vector<Index>& result) const;
        
        // Restituisce la direzione da from a to (-1 se non adiacenti)
        int GetDirection(Index from, Index to) const;
        
        // Sliding piano → piano: gate rule + contact rule
        bool CanSlide(Index from, Index to) const;
        
        // Sliding con altezze: solo gate rule con altezze (Beetle, Ladybug)
        bool CanSlideWithHeight(Index from, Index to) const;
        
        // Espande un singolo passo di sliding da from, escludendo celle già visitate
        void GetOneSlideSteps(Index from, bool visited[], std::vector<Index>& result) const;



        // - - - - - - - - - - GENERAZIONE DELLE MOSSE VALIDE - - - - - - - - - -

        // Genera tutte le mosse valide per il colore corrente
        void GetValidMoves(std::vector<Move>& moves) const;

        // Piazzamento: celle vuote adiacenti all'hive, non adiacenti a pezzi avversari
        void GetValidPlacements(Color color, std::vector<Move>& moves) const;

        // Movimenti per tipo di insetto
        void GetQueenBeeMoves(PieceName piece, std::vector<Move>& moves) const;     // Slide 1 passo
        void GetBeetleMoves(PieceName piece, std::vector<Move>& moves) const;       // 1 passo, può salire su altri pezzi
        void GetGrasshopperMoves(PieceName piece, std::vector<Move>& moves) const;  // Salta in linea retta
        void GetSpiderMoves(PieceName piece, std::vector<Move>& moves) const;       // Slide esattamente 3 passi
        void GetSoldierAntMoves(PieceName piece, std::vector<Move>& moves) const;   // Slide su tutto il perimetro
        void GetLadybugMoves(PieceName piece, std::vector<Move>& moves) const;      // 2 passi sopra hive + 1 giù
        void GetPillbugMoves(PieceName piece, std::vector<Move>& moves) const;      // Sposta un pezzo adiacente
        void GetMosquitoMoves(PieceName piece, std::vector<Move>& moves) const;     // Copia il movimento di un vicino



        // - - - - - - - - - - COSTRUTTORE- - - - - - - - - -
        
        // Costruttore
        Board(GameType gameType);


        
        // - - - - - - - - - - ZOBRIST HASHING - - - - - - - - - -
        // Zobrist Hashing: modo per rappresentare lo stato completo di una board di gioco con un numero a 64 bit

        // Altezza della pila in ogni casella (necessario per la robustezza Zobrist)
        // 0 = vuoto, 1 = un pezzo, 2 = due pezzi, ...
        int8_t stackHeight[BoardSize];

        // L' "impronta digitale" attuale della board
        uint64_t currentHash; 

        // Tabella A: continene numeri casuali
        // [Pezzo][Posizione] -> è una tabella di 14 x 16 384 interi a 64 bit
        static uint64_t ZobristTable[NumPieceNames][BoardSize];

        // Tabella B: Identifica il pezzo a una certa altezza (chi e a che piano)
        // 8 è un limite sicuro (in Hive raramente si superano i 3 piani).
        // [Pezzo][Altezza] -> è una tabella di 14 x 8 interi a 64 bit
        static uint64_t ZobristLevel[NumPieceNames][8];
        
        // Hash per indicare di chi è il turno (Black to move)
        static uint64_t ZobristBlackTurn; 
        
        // Lo storico degli hash (per la regola della tripla ripetizione)
        std::vector<uint64_t> hashHistory;



        // - - - - - - - - - - METODI PUBBLICI GENERALI - - - - - - - - - -

        // Restituisce lo stato della board
        BoardState GetBoardState();
        
        // Restituisce il turno attuale
        int GetCurrentTurn();

        // Inizializza le tabelle dello Zobrist Hashing
        static void InitializeZobristTable(); 

        // Restituisce l'hash attuale
        uint64_t GetHash();

        // Modifica l'hash attuale con solo il turno
        void ToggleTurnHash() { currentHash ^= ZobristBlackTurn; }
};   



// - - - - - - - - - - DEFINIZIONE (INLINE) DI FUNZIONI DI STATO - - - - - - - - - -

inline Index Board::GetPosition(PieceName pieceName) const 
{
    return piecesPositions[pieceName];
}

inline PieceName Board::GetPieceAt(Index position) const
{
    assert(IsValidIndex(position) && "GetPieceAt chiamato con indice invalido!");
    return cells[position];
}

inline bool Board::HasPieceAt(Index position) const
{
    assert(IsValidIndex(position) && "HasPieceAt chiamato con indice invalido!");
    return cells[position] != PieceName::INVALID;
}

inline bool Board::PieceInHand(PieceName pieceName) const
{
    return piecesPositions[pieceName] == NullIndex;
}

inline bool Board::PieceInPlay(PieceName pieceName) const
{
    return piecesPositions[pieceName] != NullIndex;
}

inline bool Board::PieceIsOnTop(PieceName pieceName) const
{
    if (!PieceInPlay(pieceName))
    {
        return false;
    }
    Index pos = piecesPositions[pieceName];
    return cells[pos] == pieceName;
}



// - - - - - - - - - - DEFINIZIONE (INLINE) DI METODI PUBBLICI GENERALI - - - - - - - - - -

inline BoardState Board::GetBoardState()
{
    return boardState;
}

inline int Board::GetCurrentTurn()
{
    return currentTurn;
}

inline uint64_t Board::GetHash()
{
    return currentHash;
}

} // namespace HiveGotThis

#endif // BOARD_H