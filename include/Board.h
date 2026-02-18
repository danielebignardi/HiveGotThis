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
        // Rappresentazione interna della board (Array di PieceName), praticamente una matrice BoardWidth x BoardWidth appiattita in un array monodimensionale.
        PieceName cells[BoardSize];

        PieceName below[NumPieceNames]; // Per ogni pezzo, quale pezzo è sotto di esso (o INVALID se nessuno)

        Index piecesPositions[NumPieceNames]; // Per ogni pezzo, la sua posizione attuale (NullIndex se non sulla board)

        GameType gameType = GameType::Base;
        BoardState boardState = BoardState::NotStarted;
        Color currentColor = Color::White;
        int currentTurn = 0;

        // --- ZOBRIST HASHING ---
        // 3. Altezza della pila in ogni casella (Necessario per la robustezza Zobrist)
        //    0 = vuoto, 1 = un pezzo, 2 = due pezzi (uno sopra l'altro), ecc.
        int8_t stackHeight[BoardSize];
        // 1. La "Impronta Digitale" attuale della board
        uint64_t currentHash; 
        // 2. La tabella dei numeri casuali (Statica perché è uguale per tutte le board che creo nell'esecuzione del programma)
        //    [Pezzo][Posizione]
        static uint64_t ZobristTable[NumPieceNames][BoardSize];
        // Tabella B: Identifica il pezzo a una certa altezza (Chi e a che piano)
        // 8 è un limite sicuro (in Hive raramente si superano i 3 piani)
        static uint64_t ZobristLevel[NumPieceNames][8];
        // 3. Hash per indicare di chi è il turno (Black to move)
        static uint64_t ZobristBlackTurn; 
        // 4. Lo Storico degli Hash (per la regola della tripla ripetizione)
        std::vector<uint64_t> hashHistory;

        /*
            FUNZIONI DI STATO (non fanno check dell'input a runtime, si assume che chi le chiama stia facendo la cosa giusta)
        */
        Index GetPosition(PieceName pieceName) const; // Restituisce la posizione del pezzo (ovvero un Index), o NullIndex se non è sulla board
        PieceName GetPieceAt(Index position) const; //return il pezzo in cima se c'è, altrimenti INVALID
        bool HasPieceAt(Index position) const;
        bool PieceInHand(PieceName pieceName) const; //true se il pezzo è in mano (non sulla board), false altrimenti
        bool PieceInPlay(PieceName pieceName) const; //true se il pezzo è in gioco (sulla board), false altrimenti
        bool PieceIsOnTop(PieceName pieceName) const; //true se il pezzo è in cima alla pila nella sua posizione, false altrimenti (utile per verificare se un pezzo è libero di muoversi)


        /*
            FUNZIONI DI MODIFICA (senza controlli di legalità della mossa)
        */
        PieceName PopAt(Index position); // Rimuove e restituisce il pezzo in cima alla posizione, o INVALID se non c'è nulla da rimuovere (usato internamente quando si muove un pezzo, per rimuoverlo dalla vecchia posizione)
        void PushAt(PieceName pieceName, Index position); // Pusha il pezzo sulla pila in position (usato internamente quando si muove o si piazza un pezzo). Non si preoccupa se il pezzo prima era da qualche parte, quindi potenzialmente si possono generare inconsistenze
        //ATTENZIONE:  In move non si effettuano controlli di validità, si assume che la mossa sia legale e che il pezzo sia effettivamente in cima a una pila o in mano.
        void MovePiece(PieceName pieceName, Index newPosition); // Muove un pezzo da dov'è a newPosition, gestendo tutto (rimozione dalla vecchia posizione, piazzamento nella nuova posizione).

        bool CanMoveWithoutBreakingHive(PieceName piece) const;
        bool IsOneHive(Index ignorePos) const;

        /*
            ALTRO
        */

    public:
        Board(GameType gameType); // Costruttore che inizializza la board vuota (tutti PieceName::INVALID) e tutti i pezzi in mano (NullIndex) e nessun pezzo sotto a nessun altro (tutti PieceName::INVALID)

        BoardState GetBoardState();
        int GetCurrentTurn();

        // --- ZOBRIST HASHING ---
        // Funzione statica da chiamare UNA VOLTA nel main per inizializzare i numeri casuali
        static void InitializeZobristTable(); 
        // Getter per l'hash attuale (utile per l'AI)
        uint64_t GetHash();
        // Helper per cambiare il turno nell'hash
        void ToggleTurnHash() { currentHash ^= ZobristBlackTurn; }
};   


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
        return false; // Se il pezzo non è in gioco, non può essere in cima
    }

    Index pos = piecesPositions[pieceName];
    return cells[pos] == pieceName; // Il pezzo è in cima se è quello che si trova nella cella della sua posizione
}

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
