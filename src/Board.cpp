#include "Board.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <random>

namespace HiveGotThis
{

// Insieme delle celle "visitate" da una traversata, senza azzeramento per chiamata.
// Invece di un array bool ripulito con memset (16 KB) a ogni chiamata, ogni cella
// memorizza l'EPOCA dell'ulno a visita: la cella è visitata sse visited[i] == visitEpoch.
// Per ricominciare da zero basta incrementare visitEpoch, che invalida in blocco e in
// O(1) tutti i timbri precedenti. Idioma d'uso:
//   - inizio traversata: BeginVisit();         (rimette tutto a "non visitato")
//   - marca cella i:     visited[i] = visitEpoch;
//   - smarca cella i:    visited[i] = 0;        (0 != visitEpoch, che è sempre >= 1)
//   - test cella i:      visited[i] == visitEpoch
// L'unico memset reale avviene al wrap del contatore a 32 bit (~4 miliardi di chiamate).
// Lo scratch e' thread-local: una sola traversata attiva per worker, e i
// generatori che lo usano non si annidano mai a vicenda.
static thread_local uint32_t visited[BoardSize];
static thread_local uint32_t visitEpoch = 0;

static inline void BeginVisit()
{
    if (++visitEpoch == 0) { memset(visited, 0, sizeof(visited)); visitEpoch = 1; }
}

// Scratch per la DFS dei punti di articolazione (vedi EnsureHiveAnalysisCurrent).
// Non vanno azzerati: disc/low di una cella si leggono solo dopo averla visitata
// nell'epoca corrente, quando sono già stati scritti.
static thread_local int dfsDisc[BoardSize];
static thread_local int dfsLow[BoardSize];

// Una Move è identificata univocamente da (Piece, Destination): la Source dipende dallo
// stato del pezzo (NullIndex se in mano, posizione attuale altrimenti) e non aggiunge
// informazione. Deduplichiamo con un bitmap statico [NumPieceNames][BoardSize] ripristinato
// solo nelle celle marcate, evitando il costo di un unordered_set a ogni chiamata.
static void DedupMoves(std::vector<Move>& moves)
{
    thread_local bool seen[NumPieceNames][BoardSize];

    size_t write = 0;
    for (const Move& m : moves)
    {
        // PassMove ha Destination invalida: arriva solo quando moves era vuoto,
        // quindi non collide con nulla. La lasciamo passare senza marcarla.
        if (!IsValidIndex(m.Destination)) { moves[write++] = m; continue; }

        bool& bit = seen[m.Piece][m.Destination];
        if (bit) continue;
        bit = true;
        moves[write++] = m;
    }
    // Reset mirato: tocchiamo solo le celle che abbiamo davvero marcato.
    for (size_t i = 0; i < write; ++i)
    {
        const Move& m = moves[i];
        if (IsValidIndex(m.Destination)) seen[m.Piece][m.Destination] = false;
    }
    moves.resize(write);
}

// - - - - - - - - - - COSTRUTTORE- - - - - - - - - -

Board::Board(GameType gameType)
{
    this->gameType = gameType;

    // Riempie tutto l'array 'cells' con INVALID
    std::fill(std::begin(cells), std::end(cells), PieceName::INVALID);
    
    // Riempie tutto 'below' con INVALID
    std::fill(std::begin(below), std::end(below), PieceName::INVALID);

    // Riempie tutto 'piecesPositions' con NullIndex (all'inizio tutte in mano)
    std::fill(std::begin(piecesPositions), std::end(piecesPositions), NullIndex);

    // GESTIONE ZOBRIST
    currentHash = 0;                        // Board vuota = Hash 0
    hashHistory.clear();
    hashHistory.reserve(100);               // Ottimizzazione: pre-alloca spazio
    hashHistory.push_back(currentHash);     // Aggiungi lo stato iniziale (vuoto) allo storico

    // Azzera le altezze delle pile
    std::fill(std::begin(stackHeight), std::end(stackHeight), 0);

    // Azzera i pezzi che questo turno non possono essere mossi
    memset(cannotBeMoved, 0, sizeof(cannotBeMoved));
}



// - - - - - - - - - - FUNZIONI DI MODIFICA - - - - - - - - - -

PieceName Board::PopAt(Index position)
{
    assert(IsValidIndex(position) && "PopAt chiamato con indice invalido!");

    if (HasPieceAt(position))
    {
        PieceName topPiece = GetPieceAt(position);  // Identifica il pezzo in cima

        cells[position] = below[topPiece];          // Il pezzo sotto diventa il nuovo top
        below[topPiece] = PieceName::INVALID;       // Il pezzo rimosso non ha più nulla sotto

        piecesPositions[topPiece] = NullIndex;      // Il pezzo non è più sulla board

        // Aggiorna altezza
        stackHeight[position]--;
        int h = stackHeight[position];

        // Aggiorna hash
        currentHash ^= (ZobristTable[topPiece][position] ^ ZobristLevel[topPiece][h]);

        // L'occupazione è cambiata: l'analisi dei pezzi pinnati va ricalcolata.
        hiveVersion++;

        return topPiece;
    }
    else
    {
        return PieceName::INVALID;                  // Non c'è nulla da rimuovere
    }
}

void Board::PushAt(PieceName pieceName, Index position)
{
    assert(IsValidIndex(position) && "PushAt chiamato con indice invalido!");
    
    // Aggiorna hash (prima di modificare la board in quanto h è l'altezza a cui inseriremo il pezzo)
    int h = stackHeight[position];                                                      // Altezza attuale dove andrà il pezzo 
    currentHash ^= (ZobristTable[pieceName][position] ^ ZobristLevel[pieceName][h]);    // XOR combinato: Posizione univoca ^ Livello univoco

    below[pieceName] = cells[position];         // Il vecchio top diventa il pezzo sotto
    cells[position] = pieceName;                // Il nuovo pezzo diventa il top
    
    piecesPositions[pieceName] = position;      // Il vecchio top diventa il pezzo sotto

    // Aggiorna altezza
    stackHeight[position]++;

    // L'occupazione è cambiata: l'analisi dei pezzi pinnati va ricalcolata.
    hiveVersion++;
}

void Board::MovePiece(PieceName pieceName, Index newPosition)
{    
    assert(IsValidIndex(newPosition) && "MovePiece chiamato con indice invalido!");

    if (PieceInPlay(pieceName))
    {
        PopAt(piecesPositions[pieceName]);      // Il pezzo è sulla board, quindi lo rimuoviamo dalla vecchia posizione
    }
    
    PushAt(pieceName, newPosition);
}

void Board::ApplyMove(const Move& move)
{
    // Azzera cannotBeMoved all'inizio di ogni mossa.
    // Questo array viene impostato da GetPillbugMoves per bloccare i pezzi
    // che il Pillbug ha spostato (non possono muoversi nel turno successivo).
    // Va resettato qui perché è un vincolo che dura un solo turno.
    
    memset(cannotBeMoved, 0, sizeof(cannotBeMoved));

    if (move == PassMove)
    {
        // Mossa di passaggio: cambia solo il turno, nessun pezzo si sposta
        ApplyTurnEffects();
        UpdateBoardState();
        return;
    }

    cannotBeMoved[move.Piece] = true;
    MovePiece(move.Piece, move.Destination);
    ApplyTurnEffects();
    UpdateBoardState();
}

void Board::ApplyMoveSavingUndo(const Move& move, MoveUndo& undo)
{
    undo.prevBoardState = boardState;
    memcpy(undo.prevCannotBeMoved, cannotBeMoved, sizeof(cannotBeMoved));
    ApplyMove(move);
}

void Board::UndoMove(const Move& move, const MoveUndo& undo)
{
    // Inverte ApplyTurnEffects: hashHistory, hash di turno, turno, colore, boardState.
    hashHistory.pop_back();
    ToggleTurnHash();
    currentTurn--;
    ToggleColor();

    // Inverte lo spostamento del pezzo. Per ipotesi XOR è auto-inversa, quindi richiamare
    // PopAt/PushAt sulle stesse celle/altezze ripristina anche currentHash sul piano dei pezzi.
    if (move != PassMove)
    {
        if (move.Source == NullIndex)
        {
            // Era un piazzamento: rimuovi il pezzo dalla destinazione, torna in mano.
            PopAt(move.Destination);
        }
        else
        {
            // Era un movimento (o un lancio del Pillbug): riporta il pezzo all'origine.
            MovePiece(move.Piece, move.Source);
        }
    }

    // Ripristina i vincoli e lo stato salvati prima dell'ApplyMove.
    memcpy(cannotBeMoved, undo.prevCannotBeMoved, sizeof(cannotBeMoved));
    boardState = undo.prevBoardState;
}

void Board::ApplyTurnEffects() {
    ToggleColor();
    currentTurn++;
    ToggleTurnHash();
    boardState = BoardState::InProgress;
    hashHistory.push_back(currentHash);
}

void Board::ToggleColor() {
    currentColor = (currentColor == Color::White) ? Color::Black : Color::White;
}

void Board::StartGame() {
    boardState = BoardState::InProgress;
}



// - - - - - - - - - - ZOBRIST HASHING - - - - - - - - - -

uint64_t Board::ZobristTable[NumPieceNames][BoardSize];
uint64_t Board::ZobristLevel[NumPieceNames][8];
uint64_t Board::ZobristBlackTurn;

void Board::InitializeZobristTable()
{
    // Usa un generatore a 64-bit di alta qualità
    std::mt19937_64 gen(12345); // Seed fisso per riproducibilità
    std::uniform_int_distribution<uint64_t> dist;

    for (int p = 0; p < NumPieceNames; ++p)
    {
        // Riempi tabella Posizione
        for (int i = 0; i < BoardSize; ++i)
        {
            ZobristTable[p][i] = dist(gen);
        }
        // Riempi tabella Livelli (fino ad altezza 7)
        for (int h = 0; h < 8; ++h)
        {
            ZobristLevel[p][h] = dist(gen);
        }
    }
    
    ZobristBlackTurn = dist(gen);
}



// - - - - - - - - - - FUNZIONI DI CONTROLLO DELLO STATO - - - - - - - - - -

bool Board::CanMoveWithoutBreakingHive(PieceName piece) const
{
    // In mano: nessun vincolo, si potrà piazzare.
    if (PieceInHand(piece)) return true;

    Index pos = piecesPositions[piece];

    // Coperto da un altro pezzo (es. uno scarabeo sopra): pinnato.
    if (GetPieceAt(pos) != piece) return false;

    // Ha un pezzo sotto: rimuoverlo non rompe l'hive (la casella resta occupata).
    if (below[piece] != PieceName::INVALID) return true;

    // Unico pezzo nella casella: rompe l'hive sse la sua casella è un punto di
    // articolazione del grafo delle celle occupate (rimuoverla disconnette i pezzi
    // restanti). Invece di una BFS per ogni pezzo (costo O(pezzi^2) per posizione),
    // calcoliamo tutti i punti di articolazione in una sola DFS e li mettiamo in cache.
    EnsureHiveAnalysisCurrent();
    return !pinnedByHive[piece];
}

// Aggiorna pinnedByHive[] per la posizione corrente, se non già valido.
// Una sola DFS di Tarjan trova tutti i punti di articolazione del grafo delle celle
// occupate (i nodi sono le celle, gli archi l'adiacenza esagonale): un pezzo è pinnato
// se è l'unico nella sua casella (altezza 1) e quella casella è un punto di articolazione.
// Gestisce correttamente i cicli dell'hive, dove il test "locale" sui vicini fallirebbe.
// Il risultato vale finché l'occupazione non cambia (cache su hiveVersion).
void Board::EnsureHiveAnalysisCurrent() const
{
    if (hiveAnalyzedVersion == hiveVersion) return;
    hiveAnalyzedVersion = hiveVersion;

    memset(pinnedByHive, 0, sizeof(pinnedByHive));

    // Una cella occupata qualsiasi come radice: l'hive è connesso per invariante.
    Index root = NullIndex;
    for (int p = 0; p < NumPieceNames; ++p)
        if (piecesPositions[p] != NullIndex) { root = piecesPositions[p]; break; }
    if (root == NullIndex) return;   // board vuota

    BeginVisit();
    int timer = 0;
    ArticulationDFS(root, NullIndex, timer);
}

// DFS di Tarjan per i punti di articolazione. disc[u] = ordine di scoperta di u;
// low[u] = disc minima raggiungibile dal sottoalbero di u tramite un solo arco
// all'indietro. u (non radice) è articolazione se ha un figlio v con low[v] >= disc[u];
// la radice lo è se ha più di un figlio nella DFS.
void Board::ArticulationDFS(Index u, Index parent, int& timer) const
{
    visited[u] = visitEpoch;
    dfsDisc[u] = dfsLow[u] = ++timer;
    int children = 0;

    for (int i = 0; i < 6; ++i)
    {
        Index v = u + NeighborOffsets[i];
        if (!IsValidIndex(v) || !HasPieceAt(v) || v == parent) continue;

        if (visited[v] == visitEpoch)
        {
            // Arco all'indietro: v è già nello stack della DFS.
            if (dfsDisc[v] < dfsLow[u]) dfsLow[u] = dfsDisc[v];
            continue;
        }

        ++children;
        ArticulationDFS(v, u, timer);
        if (dfsLow[v] < dfsLow[u]) dfsLow[u] = dfsLow[v];

        // u taglia il sottoalbero di v dal resto dell'hive: è un punto di articolazione.
        // Pinniamo solo se nella casella c'è un unico pezzo (altezza 1).
        if (parent != NullIndex && dfsLow[v] >= dfsDisc[u] && stackHeight[u] == 1)
            pinnedByHive[cells[u]] = true;
    }

    if (parent == NullIndex && children > 1 && stackHeight[u] == 1)
        pinnedByHive[cells[u]] = true;
}

// true se le celle occupate, escludendo 'ignorePos', formano un unico blocco connesso.
// Usato per la regola dell'hive: simula la rimozione di un pezzo e controlla la connessione.
bool Board::IsOneHive(Index ignorePos) const
{
    bool visitedPieces[NumPieceNames] = { false };
    thread_local std::vector<Index> queue;
    queue.clear();
    queue.reserve(NumPieceNames);

    // Conta i pezzi attesi (in cima alla pila, esclusa ignorePos) e scegli una radice.
    int   nodesExpected = 0;
    Index startNode     = NullIndex;
    for (int i = 0; i < NumPieceNames; ++i)
    {
        Index pos = piecesPositions[i];
        if (pos == NullIndex || pos == ignorePos) continue;
        if (PieceIsOnTop(static_cast<PieceName>(i)))
        {
            nodesExpected++;
            if (startNode == NullIndex) startNode = pos;
        }
    }

    if (nodesExpected == 0) return true;   // niente da connettere

    // BFS dalla radice attraverso i vicini occupati (cursore 'head' invece di pop).
    queue.push_back(startNode);
    visitedPieces[cells[startNode]] = true;
    int nodesFound = 1;

    int head = 0;
    while (head < (int)queue.size())
    {
        Index currentPos = queue[head++];
        for (int offset : NeighborOffsets)
        {
            Index neighborPos = currentPos + offset;
            if (!IsValidIndex(neighborPos) || neighborPos == ignorePos) continue;

            PieceName neighborPiece = cells[neighborPos];
            if (neighborPiece == PieceName::INVALID) continue;   // cella vuota
            if (visitedPieces[neighborPiece]) continue;          // già contato

            visitedPieces[neighborPiece] = true;
            queue.push_back(neighborPos);
            nodesFound++;
        }
    }

    // Connesso sse abbiamo raggiunto tutti i pezzi attesi.
    return nodesFound == nodesExpected;
}

bool Board::IsQueenSurrounded(Color color) const
{
    PieceName queen = (color == Color::White) ? PieceName::wQ : PieceName::bQ;

    if (PieceInHand(queen)) return false;

    Index pos = piecesPositions[queen];

    for (int offset : NeighborOffsets)
    {
        Index neighbor = pos + offset;
        if (!IsValidIndex(neighbor) || !HasPieceAt(neighbor)) return false;
    }

    return true;
}

void Board::UpdateBoardState()
{
    if (boardState == BoardState::NotStarted) return;
    if (GameIsOver(boardState)) return;

    bool whiteQueenSurrounded = PieceInPlay(PieceName::wQ) && IsQueenSurrounded(Color::White);
    bool blackQueenSurrounded = PieceInPlay(PieceName::bQ) && IsQueenSurrounded(Color::Black);

    if (whiteQueenSurrounded && blackQueenSurrounded)
        boardState = BoardState::Draw;
    else if (whiteQueenSurrounded)
        boardState = BoardState::BlackWins;
    else if (blackQueenSurrounded)
        boardState = BoardState::WhiteWins;
    else
    {
        int count = 0;
        for (uint64_t h : hashHistory)
            if (h == currentHash)
            {
                count++;
                if (count >= 3)
                {
                    boardState = BoardState::Draw;
                    break;
                }
            }
    }
}



// - - - - - - - - - - FUNZIONI DI UTILITÀ PER LA GENERAZIONE DELLE MOSSE - - - - - - - - - -

bool Board::CanSlide(Index pos, Direction dir, SlideMode mode, Index ignorePos) const
{
    uint8_t d = static_cast<uint8_t>(dir);
    Index gate1 = pos + SlideGates[d][0];
    Index gate2 = pos + SlideGates[d][1];

    if (mode == SlideMode::Beetle)
    {
        Index dest    = pos + NeighborOffsets[d];
        // Il Beetle sale sopra dest → la quota effettiva di arrivo è stackHeight[dest]+1
        int maxHeight = std::max((int)stackHeight[pos], (int)stackHeight[dest] + 1);
        int minGate   = std::min((int)stackHeight[gate1], (int)stackHeight[gate2]);
        if (minGate >= maxHeight) return false;

        // Beetle a terra (pos h=1, dest vuota): almeno un gate occupato, stessa regola di Ground mode.
        // Un Beetle elevato (h>1) può scendere su celle vuote anche con gate vuoti — è già sopra di essi.
        if (stackHeight[pos] == 1 && stackHeight[dest] == 0 &&
            stackHeight[gate1] == 0 && stackHeight[gate2] == 0)
            return false;

        return true;
    }

    if (mode == SlideMode::LadyBug)
    {
        // La Ladybug è fisicamente sopra il pezzo a pos, quindi altezza effettiva = stackHeight[pos] + 1
        Index dest    = pos + NeighborOffsets[d];
        int maxHeight = std::max((int)stackHeight[pos] + 1, (int)stackHeight[dest]);
        int minGate   = std::min((int)stackHeight[gate1], (int)stackHeight[gate2]);
        return minGate < maxHeight;
    }

    // SlideMode::Ground — pezzi a terra
    bool g1 = HasPieceAt(gate1) && (gate1 != ignorePos);
    bool g2 = HasPieceAt(gate2) && (gate2 != ignorePos);
    if (g1 && g2) return false;
    return g1 || g2;
}

void Board::GetOccupiedNeighbors(Index pos, std::vector<Index>& result) const
{
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = pos + NeighborOffsets[i];
        if (IsValidIndex(neighbor) && HasPieceAt(neighbor))
            result.push_back(neighbor);
    }
}

void Board::GetEmptyNeighbors(Index pos, std::vector<Index>& result) const
{
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = pos + NeighborOffsets[i];
        if (IsValidIndex(neighbor) && !HasPieceAt(neighbor))
            result.push_back(neighbor);
    }
}

void Board::GetOneSlideSteps(Index from, SlideMode mode, std::vector<Index>& result, Index ignorePos) const
{
    for (int i = 0; i < 6; i++)
    {
        Index neighbor = from + NeighborOffsets[i];
        if (!IsValidIndex(neighbor)) continue;
        if (mode == SlideMode::Ground && HasPieceAt(neighbor)) continue;    // se siamo in modalità Ground allora saltiamo le celle su cui c'è già un pezzo
        if (visited[neighbor] == visitEpoch) continue;
        if (!CanSlide(from, static_cast<Direction>(i), mode, ignorePos)) continue;
        result.push_back(neighbor);
    }
}



// - - - - - - - - - - GENERAZIONE DELLE MOSSE VALIDE - - - - - - - - - -

void Board::GetQueenBeeMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    BeginVisit();
    visited[pos] = visitEpoch;

    thread_local std::vector<Index> steps;  // buffer riusato per worker
    steps.clear();
    GetOneSlideSteps(pos, SlideMode::Ground, steps);

    for (Index dest : steps)
    {
        Move move = {piece, pos, dest};
        moves.push_back(move);
    }
}

void Board::GetBeetleMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    BeginVisit();
    visited[pos] = visitEpoch;

    thread_local std::vector<Index> steps;  // buffer riusato per worker
    steps.clear();
    GetOneSlideSteps(pos, SlideMode::Beetle, steps);

    for (Index dest : steps)
    {
        Move move = {piece, pos, dest};
        moves.push_back(move);
    }
}

void Board::GetGrasshopperMoves(PieceName piece, std::vector<Move>& moves) const
{
    // one hive rule
    if (!CanMoveWithoutBreakingHive(piece)) return;    

    Index pos = piecesPositions[piece];

    for (int i = 0; i < 6; i++)
    {
        Index neighbor = pos + NeighborOffsets[i];

        if (!IsValidIndex(neighbor)) continue;    // fuori dalla board
        if (!HasPieceAt(neighbor)) continue;      // deve saltare oltre un pezzo

        Index current = neighbor;
        while (IsValidIndex(current) && HasPieceAt(current)) {    // va oltre tutti i pezzi nella direzione considerata
            current = current + NeighborOffsets[i];
        }

        if (IsValidIndex(current))
        {
            Move move = {piece, pos, current};
            moves.push_back(move);
        }
    }
}

void Board::GetSpiderMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    BeginVisit();

    // Indici raggiungibili a ciascuno dei 3 passi (buffer riusati fra le chiamate).
    thread_local std::vector<Index> steps1, steps2, steps3;
    steps1.clear();

    // inizio: indici raggiungibili dopo un passo
    visited[pos] = visitEpoch;
    GetOneSlideSteps(pos, SlideMode::Ground, steps1, pos);

    for (Index step1 : steps1)
    {
        visited[step1] = visitEpoch;

        steps2.clear();
        GetOneSlideSteps(step1, SlideMode::Ground, steps2, pos);

        for (Index step2 : steps2)
        {
            visited[step2] = visitEpoch;

            steps3.clear();
            GetOneSlideSteps(step2, SlideMode::Ground, steps3, pos);

            for (Index step3 : steps3)
            {
                Move move = {piece, pos, step3};
                moves.push_back(move);
            }

            visited[step2] = 0;     // backtrack: la cella torna libera per altri rami
        }

        visited[step1] = 0;
    }
}

void Board::GetSoldierAntMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    BeginVisit();
    visited[pos] = visitEpoch;

    // Frontiera della BFS: partiamo dai vicini scivolabili (buffer riusato).
    thread_local std::vector<Index> queue;
    queue.clear();
    GetOneSlideSteps(pos, SlideMode::Ground, queue, pos);
    for (Index cell : queue) visited[cell] = visitEpoch;

    int head = 0;
    while (head < (int)queue.size()) {
        Index current = queue[head++];
        int sizeBefore = queue.size();
        GetOneSlideSteps(current, SlideMode::Ground, queue, pos);

        // Segna subito come visitati i nuovi nodi aggiunti
        for (int i = sizeBefore; i < (int)queue.size(); i++)
            visited[queue[i]] = visitEpoch;
    }

    for (Index cell : queue)
    {
        Move move = {piece, pos, cell};
        moves.push_back(move);
    }
}

void Board::GetLadybugMoves(PieceName piece, std::vector<Move>& moves) const
{
    if (!CanMoveWithoutBreakingHive(piece)) return;

    Index pos = piecesPositions[piece];

    // La Ladybug fa esattamente: 2 passi sopra l'hive + 1 passo che scende su una
    // cella vuota. Più step1 diversi possono portare allo stesso step2, e più step2
    // diversi possono portare alla stessa cella d'arrivo: deduplichiamo entrambi i
    // livelli con due bitmap statici, ripristinati solo nelle celle toccate.
    thread_local bool step2Seen[BoardSize];
    thread_local bool destSeen [BoardSize];

    // Passo 1: i 6 vicini occupati. Sono già distinti, niente dedup.
    Index steps1[6];
    int   nSteps1 = 0;
    for (int i = 0; i < 6; ++i)
    {
        Index n = pos + NeighborOffsets[i];
        if (!IsValidIndex(n)) continue;
        if (!HasPieceAt(n)) continue;
        if (!CanSlide(pos, static_cast<Direction>(i), SlideMode::Beetle)) continue;
        steps1[nSteps1++] = n;
    }

    // Passo 2: celle occupate adiacenti a un step1, deduplicate.
    Index steps2[6 * 6];
    int   nSteps2 = 0;
    for (int k = 0; k < nSteps1; ++k)
    {
        Index step1 = steps1[k];
        for (int i = 0; i < 6; ++i)
        {
            Index n = step1 + NeighborOffsets[i];
            if (!IsValidIndex(n) || !HasPieceAt(n) || n == pos) continue;
            if (step2Seen[n]) continue;
            if (!CanSlide(step1, static_cast<Direction>(i), SlideMode::Beetle)) continue;
            step2Seen[n] = true;
            steps2[nSteps2++] = n;
        }
    }

    // Passo 3: discesa su cella vuota adiacente a uno step2, deduplicata.
    Index destsPushed[6 * 6];
    int   nPushed = 0;
    for (int k = 0; k < nSteps2; ++k)
    {
        Index step2 = steps2[k];
        for (int i = 0; i < 6; ++i)
        {
            Index n = step2 + NeighborOffsets[i];
            if (!IsValidIndex(n) || HasPieceAt(n) || n == pos) continue;
            if (destSeen[n]) continue;
            if (!CanSlide(step2, static_cast<Direction>(i), SlideMode::LadyBug)) continue;
            destSeen[n] = true;
            destsPushed[nPushed++] = n;
            moves.push_back({piece, pos, n});
        }
    }

    // Reset mirato dei due bitmap.
    for (int k = 0; k < nSteps2; ++k) step2Seen[steps2[k]]       = false;
    for (int k = 0; k < nPushed; ++k) destSeen [destsPushed[k]]  = false;
}

void Board::GetPillbugMoves(PieceName piece, std::vector<Move>& moves) const
{
    Index pos = piecesPositions[piece];

    // Movimento 1: slide di 1 passo come la Queen Bee (richiede che il Pillbug non rompa l'hive)
    if (CanMoveWithoutBreakingHive(piece))
    {
        for (int i = 0; i < 6; i++)
        {
            Index neighbor = pos + NeighborOffsets[i];
            if (!IsValidIndex(neighbor)) continue;
            if (HasPieceAt(neighbor)) continue;
            if (!CanSlide(pos, static_cast<Direction>(i), SlideMode::Ground)) continue;
            moves.push_back({piece, pos, neighbor});
        }
    }

    // Movimento 2: sposta un pezzo adiacente sopra il Pillbug e depositalo
    // (la special ability funziona anche se il Pillbug è pinnato)
    for (int i = 0; i < 6; i++)
    {
        Index neighborPos = pos + NeighborOffsets[i];
        if (!IsValidIndex(neighborPos)) continue;
        if (!HasPieceAt(neighborPos)) continue;

        PieceName neighborPiece = GetPieceAt(neighborPos);

        // Il pezzo non deve essere in uno stack
        if (stackHeight[neighborPos] > 1) continue;

        // Il pezzo non deve essere nella lista cannotBeMoved
        if (cannotBeMoved[neighborPiece]) continue;

        // Il pezzo deve potersi muovere senza rompere l'hive
        if (!CanMoveWithoutBreakingHive(neighborPiece)) continue;

        // Gate rule con altezze per la raccolta (neighborPos → pos)
        // La direzione corretta è l'opposta di i: il vicino si muove VERSO il Pillbug
        if (!CanSlide(neighborPos, static_cast<Direction>((i + 3) % 6), SlideMode::Beetle)) continue;

        // Deposita il pezzo in ogni cella vuota adiacente al Pillbug
        for (int j = 0; j < 6; j++)
        {
            Index dest = pos + NeighborOffsets[j];
            if (!IsValidIndex(dest)) continue;
            if (HasPieceAt(dest)) continue;
            if (dest == neighborPos) continue;

            // Gate rule con altezze per il deposito (pos → dest)
            // Il Pillbug porta un pezzo sopra di sé: altezza effettiva = stackHeight[pos]+1 → LadyBug
            if (!CanSlide(pos, static_cast<Direction>(j), SlideMode::LadyBug)) continue;

            moves.push_back({neighborPiece, neighborPos, dest});
        }
    }
}

void Board::GetMosquitoMoves(PieceName piece, std::vector<Move>& moves) const
{
    Index pos = piecesPositions[piece];

    // Caso speciale: se la Mosquito è sopra un pezzo, si muove solo come Beetle
    if (stackHeight[pos] > 1)
    {
        if (!CanMoveWithoutBreakingHive(piece)) return;
        GetBeetleMoves(piece, moves);
        return;
    }

    // La Mosquito può copiare la special ability del Pillbug anche se è pinnata,
    // perché il Pillbug non richiede di muoversi fisicamente.
    // Tutti gli altri tipi richiedono che il pezzo possa muoversi senza rompere l'hive.
    bool canMove = CanMoveWithoutBreakingHive(piece);

    // Tipi di insetti adiacenti già processati (per evitare duplicati)
    bool processedTypes[static_cast<int>(BugType::NumBugTypes)] = { false };

    // I generatori che copieremo possono produrre la stessa cella d'arrivo: QueenBee,
    // Pillbug (slide-1), Beetle (su cella vuota), Spider, Grasshopper, Ladybug.
    // Ricordiamo dove inizia la coda delle mosse della Mosquito per deduplicarle in fondo.
    size_t tailStart = moves.size();

    for (int i = 0; i < 6; ++i)
    {
        Index neighborPos = pos + NeighborOffsets[i];
        if (!IsValidIndex(neighborPos) || !HasPieceAt(neighborPos)) continue;

        BugType type = GetBugType(GetPieceAt(neighborPos));

        // La Mosquito non copia se stessa o un'altra Mosquito.
        if (type == BugType::Mosquito) continue;
        // Stesso tipo già processato da un altro vicino: salta.
        if (processedTypes[static_cast<int>(type)]) continue;
        processedTypes[static_cast<int>(type)] = true;

        // Pillbug: la special ability funziona anche se la Mosquito è pinnata.
        if (type == BugType::Pillbug) { GetPillbugMoves(piece, moves); continue; }

        // Tutti gli altri tipi richiedono che la Mosquito si possa muovere.
        if (!canMove) continue;

        switch (type)
        {
            case BugType::QueenBee:    GetQueenBeeMoves(piece, moves);    break;
            case BugType::Beetle:      GetBeetleMoves(piece, moves);      break;
            case BugType::Grasshopper: GetGrasshopperMoves(piece, moves); break;
            case BugType::Spider:      GetSpiderMoves(piece, moves);      break;
            case BugType::SoldierAnt:  GetSoldierAntMoves(piece, moves);  break;
            case BugType::Ladybug:     GetLadybugMoves(piece, moves);     break;
            default: break;
        }
    }

    // Dedup in-place dei soli self-move della Mosquito (Piece == piece && Source == pos).
    // I lanci generati da GetPillbugMoves hanno Piece diverso e passano invariati.
    thread_local bool destSeen[BoardSize];
    size_t write = tailStart;
    for (size_t r = tailStart; r < moves.size(); ++r)
    {
        const Move& m = moves[r];
        const bool isSelfMove = (m.Piece == piece) && (m.Source == pos);
        if (isSelfMove)
        {
            if (destSeen[m.Destination]) continue;
            destSeen[m.Destination] = true;
        }
        moves[write++] = m;
    }
    // Reset mirato del bitmap.
    for (size_t r = tailStart; r < write; ++r)
    {
        const Move& m = moves[r];
        if (m.Piece == piece && m.Source == pos) destSeen[m.Destination] = false;
    }
    moves.resize(write);
}

void Board::GetValidMoves(std::vector<Move>& moves) const
{
    moves.clear();

    // Gioco non in corso, nessuna mossa
    if (!GameInProgress(boardState)) return;

    // Regola della Regina: al quarto turno deve essere piazzata
    // Bianco gioca ai turni 0,2,4,6 — quarto turno = 6
    // Nero gioca ai turni 1,3,5,6 — quarto turno = 7
    PieceName queen = (currentColor == Color::White) ? PieceName::wQ : PieceName::bQ;   // chi è la regina in questione a seconda del turno
    bool queenInPlay = PieceInPlay(queen);
    int queenDeadline = (currentColor == Color::White) ? 6 : 7;             // turno entro il quale la regina in questione deve essere piazzata
    bool mustPlaceQueen = !queenInPlay && (currentTurn >= queenDeadline); 

    thread_local std::vector<Index> positions; // buffer riusato per worker
    positions.clear();
    GetValidPlacements(currentColor, positions);

    if (mustPlaceQueen)
    {
        for (Index dest : positions)
            moves.push_back({queen, NullIndex, dest});
        return;
    }

    // Quali pezzi in mano sono piazzabili NON dipende dalla posizione di destinazione:
    // lo calcoliamo una volta sola, poi per ogni cella valida emettiamo la lista.
    // (Prima questo filtro O(NumPieceNames) girava per OGNI destinazione.)
    thread_local PieceName placeable[NumPieceNames];
    int nPlaceable = 0;
    for (int p = 0; p < NumPieceNames; p++)
    {
        PieceName piece = static_cast<PieceName>(p);
        if (GetColor(piece) != currentColor) continue;
        if (!PieceInHand(piece)) continue;
        if (!PieceNameIsEnabledForGameType(piece, gameType)) continue;

        // Regola: la queenBee non può essere piazzata al primo turno del proprio colore
        if (piece == queen && currentTurn <= 1) continue;

        // Regola Mzinga: un pezzo numerato non può essere piazzato se il precedente è ancora in mano
        // (es. non si può piazzare WS2 se WS1 non è ancora sulla board)
        if (piece > 0) {
            PieceName prev = static_cast<PieceName>(piece - 1);
            if (GetColor(prev) == currentColor &&
                GetBugType(prev) == GetBugType(piece) &&
                PieceInHand(prev)) continue;
        }

        placeable[nPlaceable++] = piece;
    }

    // Piazzamento: combina ogni posizione valida con ogni pezzo piazzabile
    for (Index dest : positions)
        for (int k = 0; k < nPlaceable; k++)
            moves.push_back({placeable[k], NullIndex, dest});

    // Movimento: per ogni pezzo in gioco del colore corrente
    // La Regina deve essere in gioco per poter muovere altri pezzi
    if (queenInPlay)
    {
        for (int p = 0; p < NumPieceNames; p++)
        {
            PieceName piece = static_cast<PieceName>(p);
            if (GetColor(piece) != currentColor) continue;
            if (!PieceInPlay(piece)) continue;
            if (!PieceIsOnTop(piece)) continue;
            if (cannotBeMoved[piece]) continue;

            BugType type = GetBugType(piece);
            switch (type)
            {
                case BugType::QueenBee:    GetQueenBeeMoves(piece, moves);    break;
                case BugType::Beetle:      GetBeetleMoves(piece, moves);      break;
                case BugType::Grasshopper: GetGrasshopperMoves(piece, moves); break;
                case BugType::Spider:      GetSpiderMoves(piece, moves);      break;
                case BugType::SoldierAnt:  GetSoldierAntMoves(piece, moves);  break;
                case BugType::Ladybug:     GetLadybugMoves(piece, moves);     break;
                case BugType::Pillbug:     GetPillbugMoves(piece, moves);     break;
                case BugType::Mosquito:    GetMosquitoMoves(piece, moves);    break;
                default: break;
            }
        }
    }

    // Rimuove i duplicati residui (lancio del Pillbug che coincide con la mossa
    // normale del pezzo lanciato, due Pillbug-attori che lanciano lo stesso pezzo
    // nella stessa cella, ecc.).
    DedupMoves(moves);

    // Se non ci sono mosse valide, l'unica opzione è passare
    if (moves.empty())
    {
        moves.push_back(PassMove);
    }
}

bool Board::CanPlaceAt(Index pos, Color myColor, int currentTurn) const
{
    // 1. Posizione deve essere Vuota 
    if (HasPieceAt(pos)) return false;

    // 2. CASI SPECIALI (Turno 0 e 1)
    if (currentTurn == 0) return pos == BoardCenter; // Turno 0: il Bianco mette solo al centro

    if (currentTurn == 1) {
        // Turno 1: il Nero DEVE toccare il Bianco.
        // Appena troviamo un vicino qualsiasi (che per forza è il Bianco), usciamo con true.
        for (int offset : NeighborOffsets) {
            Index neighbor = pos + offset;
            if (IsValidIndex(neighbor) && HasPieceAt(neighbor)) return true;
        }
        return false;
    }

    // 3. REGOLA GENERALE (Dal turno 3 in poi): Devo toccare almeno un mio pezzo e NESSUN pezzo avversario
    bool foundFriend = false;

    for (int offset : NeighborOffsets) {
        Index neighbor = pos + offset;

        // Bounds check: ignora celle fuori dalla board
        if (!IsValidIndex(neighbor)) continue;

        PieceName neighborPiece = GetPieceAt(neighbor);
        
        if (neighborPiece != PieceName::INVALID) {
            // Appena tocchiamo un pezzo nemico, la mossa è illegale!
            if (GetColor(neighborPiece) != myColor) {
                return false; 
            }
            // Se non è nemico, è per forza amico. Ce lo segniamo.
            foundFriend = true;
        }
    }

    return foundFriend;
}

void Board::GetValidPlacements(Color color, std::vector<Index>& positions) const
{
    // Turno 0: board vuota, l'unica posizione valida è BoardCenter
    if (currentTurn == 0)
    {
        positions.push_back(BoardCenter);
        return;
    }

    // visited marca le celle candidate già aggiunte, per non duplicarle.
    BeginVisit();

    // Per ogni pezzo sulla board, le sue celle vuote adiacenti sono candidate al piazzamento.
    for (int p = 0; p < NumPieceNames; p++)
    {
        if (PieceInHand(static_cast<PieceName>(p))) continue; // solo pezzi già sulla board

        Index pos = piecesPositions[p];

        for (int i = 0; i < 6; i++)
        {
            Index neighbor = pos + NeighborOffsets[i];
            if (!IsValidIndex(neighbor)) continue;
            if (visited[neighbor] == visitEpoch) continue;
            if (!CanPlaceAt(neighbor, color, currentTurn)) continue;

            visited[neighbor] = visitEpoch;
            positions.push_back(neighbor);
        }
    }
}

} // namespace HiveGotThis
