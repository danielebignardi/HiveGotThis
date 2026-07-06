#ifndef BOARD_H
#define BOARD_H

#include "Position.h"
#include "Move.h"
#include "Enums.h" 
#include "Constants.h"

#include <cassert>
#include <vector>
#include <cstring>

namespace HiveGotThis
{

// Stato salvato prima di ApplyMoveSavingUndo, necessario a UndoMove per ripristinare la board.
struct MoveUndo
{
    BoardState prevBoardState;
    bool       prevCannotBeMoved[NumPieceNames];
};

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

        // Colore del giocatore attuale. Inizia il bianco
        Color currentColor = Color::White;

        // Pezzi che per questo turno non possono essere mossi
        bool cannotBeMoved[NumPieceNames];


        // - - - - - - - - - - FUNZIONI DI STATO - - - - - - - - - -

        // Restituisce la posizione del pezzo (ovvero un Index), o NullIndex se non è sulla board
        Index GetPosition(PieceName pieceName) const;
        
        // Restituisce il pezzo in cima se c'è, altrimenti INVALID
        PieceName GetPieceAt(Index position) const;

        // Restituisce il pezzo direttamente sotto a piece nella pila (INVALID se nessuno)
        PieceName GetPieceUnder(PieceName piece) const;

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

        // Da chiamare dopo ogni mossa/piazzamento: aggiorna cannotBeMoved, currentTurn, currentColor.
        void ApplyMove(const Move& move);

        // Come ApplyMove ma salva in undo le informazioni necessarie a invertirla.
        void ApplyMoveSavingUndo(const Move& move, MoveUndo& undo);

        // Inverte una mossa applicata con ApplyMoveSavingUndo, ripristinando lo stato precedente.
        void UndoMove(const Move& move, const MoveUndo& undo);

        // Imposta boardState = InProgress (da chiamare all'avvio di una partita)
        void StartGame();
        


        // - - - - - - - - - - FUNZIONI DI CONTROLLO DELLO STATO - - - - - - - - - -

        // true se il pezzo può spostarsi anche temporaneamente dalla sua posizione attuale senza rompere l'hive
        bool CanMoveWithoutBreakingHive(PieceName piece) const;

        // true se la configurazione attuale della board è un solo hive (condizione che deve essere garantita)
        bool IsOneHive(Index ignorePos) const;

        // Ricalcola, se necessario, quali pezzi sono pinnati (vedi pinnedByHive) con una
        // sola DFS dei punti di articolazione. Idempotente: usa la cache su hiveVersion.
        void EnsureHiveAnalysisCurrent() const;

        // true se la regina del colore indicato è sulla board ed è completamente circondata (tutti e 6 i vicini occupati)
        bool IsQueenSurrounded(Color color) const;

        // Aggiorna boardState in base allo stato delle regine (vittoria/pareggio/in corso)
        void UpdateBoardState();



        // - - - - - - - - - - FUNZIONI DI UTILITÀ PER LA GENERAZIONE DELLE MOSSE - - - - - - - - - -

        // Restituisce gli indici dei vicini occupati di una posizione
        void GetOccupiedNeighbors(Index pos, std::vector<Index>& result) const;
        
        // Restituisce gli indici dei vicini vuoti di una posizione
        void GetEmptyNeighbors(Index pos, std::vector<Index>& result) const;

        // true se un pezzo si può spostare dalla posizione pos in direzione dir, muovendosi sullo stesso piano (Ground) o spostandosi di piano (Beetle)
        // ignorePos: posizione da trattare come vuota durante il controllo dei gate (usato per ignorare la posizione di partenza del pezzo in movimento)
        bool CanSlide(Index pos, Direction dir, SlideMode mode, Index ignorePos = NullIndex) const;

        // SARA 27 FEB
        //PER FUNZIONE CAN SLIDE: Pre-calcoliamo le "slide gates" per ogni direzione, ovvero le due posizioni che devono essere libere per poter scivolare in quella direzione
        // Lookup table: Direction -> {gate_offset_1, gate_offset_2}
        static constexpr Index SlideGates[6][2] = {
            { NeighborOffsets[5], NeighborOffsets[1] }, // Right (0)
            { NeighborOffsets[0], NeighborOffsets[2] }, // DownRight (1)
            { NeighborOffsets[1], NeighborOffsets[3] }, // DownLeft (2)
            { NeighborOffsets[2], NeighborOffsets[4] }, // Left (3)
            { NeighborOffsets[3], NeighborOffsets[5] }, // UpLeft (4)
            { NeighborOffsets[4], NeighborOffsets[0] }  // UpRight (5)
        };

        // Restituisce gli indici dei vicini verso i quali si può spostare, eccetto quelli già visitati
        // (le celle visitate sono lette dall'array epoca 'visited' file-local di Board.cpp)
        // ignorePos: propagato a CanSlide per ignorare la posizione di partenza del pezzo in movimento
        void GetOneSlideSteps(Index from, SlideMode mode, std::vector<Index>& result, Index ignorePos = NullIndex) const;



        // - - - - - - - - - - GENERAZIONE DELLE MOSSE VALIDE - - - - - - - - - -

        // Genera tutte le mosse valide per il colore corrente
        void GetValidMoves(std::vector<Move>& moves) const;

        // Piazzamento: al turno corrente, true se è possibile piazzare in una determinata posizione
        bool CanPlaceAt(Index pos, Color myColor, int currentTurn) const;

        // Piazzamento: celle vuote adiacenti all'hive, non adiacenti a pezzi avversari
        void GetValidPlacements(Color color, std::vector<Index>& positions) const;

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
        BoardState GetBoardState() const;

        // Restituisce il turno attuale
        int GetCurrentTurn() const;

        // Inizializza le tabelle dello Zobrist Hashing
        static void InitializeZobristTable();

        // Restituisce l'hash attuale
        uint64_t GetHash() const;

        // Modifica l'hash attuale con solo il turno
        void ToggleTurnHash() { currentHash ^= ZobristBlackTurn; }

    private:

        // - - - - - - - - - - CAMPI PRIVATI - - - - - - - - - -

        // Stato del gioco
        BoardState boardState = BoardState::NotStarted;

        // Il turno aumenta di uno a ogni mossa
        int currentTurn = 0;

        // L' "impronta digitale" attuale della board
        uint64_t currentHash;

        // - - - - - - - - - - ANALISI HIVE (PEZZI PINNATI) - - - - - - - - - -

        // Versione della struttura dell'hive: incrementata a ogni cambio di occupazione
        // (PushAt/PopAt). Serve a invalidare la cache pinnedByHive.
        uint64_t hiveVersion = 0;

        // Versione per cui pinnedByHive è valido (sentinella iniziale = "mai calcolato").
        mutable uint64_t hiveAnalyzedVersion = static_cast<uint64_t>(-1);

        // Per ogni pezzo: true se è l'unico nella sua casella e quella casella è un punto
        // di articolazione dell'hive. Valido dopo EnsureHiveAnalysisCurrent.
        mutable bool pinnedByHive[NumPieceNames];

        // DFS di Tarjan: riempie pinnedByHive a partire da una cella radice.
        void ArticulationDFS(Index u, Index parent, int& timer) const;

        // - - - - - - - - - - HELPER PRIVATI - - - - - - - - - -

        void ApplyTurnEffects();
        void ToggleColor();
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

inline PieceName Board::GetPieceUnder(PieceName piece) const
{
    return below[piece];
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

inline BoardState Board::GetBoardState() const
{
    return boardState;
}

inline int Board::GetCurrentTurn() const
{
    return currentTurn;
}

inline uint64_t Board::GetHash() const
{
    return currentHash;
}

} // namespace HiveGotThis

#endif // BOARD_H