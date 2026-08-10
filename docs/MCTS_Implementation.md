# Implementazione della MCTS

Questo documento descrive la Monte Carlo Tree Search usata da HiveGotThis e
l'architettura con cui vengono generate piu' partite di self-play in parallelo.
La descrizione segue il codice corrente.

I file principali sono:

- `include/MCTS.h` e `src/MCTS.cpp`: albero, selezione, solver e transposition
  table.
- `include/NeuralEvaluator.h` e `src/NeuralEvaluator.cpp`: inferenza
  TorchScript su CPU o CUDA e batching tra partite.
- `src/selfplay_main.cpp`: worker delle partite e scrittura del dataset JSONL.
- `src/Board.cpp` e `src/Evaluation.cpp`: generazione delle mosse e buffer
  temporanei resi sicuri per l'esecuzione concorrente.

## 1. API pubbliche

La classe `MCTS` espone tre modalita' di ricerca:

```cpp
Move MCTS::Search(const Board& board, int timeLimitMs);
Move MCTS::SearchIterations(const Board& board, int maxIterations);
std::vector<PolicyTarget> MCTS::SearchPolicyTargets(
    const Board& board,
    int maxIterations);
```

`Search` usa un limite di tempo ed e' l'entry point naturale per il motore UHP.
`SearchIterations` usa un numero fisso di iterazioni. `SearchPolicyTargets`
esegue la stessa ricerca a iterazioni fisse, ma restituisce visite e
distribuzione `pi` per addestrare la policy head durante il self-play.

Prima della ricerca deve essere registrato un evaluator:

```cpp
MCTS::SetValueNetwork(&evaluator);
```

La MCTS non possiede l'evaluator. Il chiamante deve mantenerlo vivo per tutta
la durata delle ricerche.

## 2. Struttura dell'albero

Un `MCTSNode` non contiene una copia della board. Conserva:

- la mossa che ha prodotto il nodo;
- puntatore al padre e lista dei figli;
- `visitCount` e `totalValue`;
- stato di espansione e terminalita';
- risultato certo del solver;
- prior della policy e punteggio euristico;
- mosse non ancora espanse e relative prior.

All'inizio di ogni iterazione viene copiata soltanto la board della radice:

```cpp
Board board = rootBoard;
```

Durante la selezione le mosse del percorso vengono riapplicate alla copia.
Questo riduce la memoria dell'albero: una ricerca puo' creare molti nodi senza
duplicare l'intero stato di gioco in ciascuno.

## 3. Convenzione dei valori

Tutti i valori seguono la convenzione negamax `side-to-move`:

- `+1`: posizione vincente per il giocatore che deve muovere nel nodo;
- `-1`: posizione perdente;
- `0`: posizione neutrale o patta.

`totalValue` accumula valori dalla prospettiva del giocatore di turno nel nodo.
Durante il backup il segno viene invertito a ogni livello:

```cpp
current->visitCount++;
current->totalValue += value;
value = -value;
current = current->parent;
```

La value network usa la stessa convenzione, quindi il suo output `[-1, 1]` puo'
essere propagato direttamente.

## 4. Una iterazione MCTS

`MCTS::RunIteration` esegue sempre una sola iterazione completa:

```text
selezione -> espansione -> valutazione -> backpropagation
```

Non vengono raccolte piu' foglie dello stesso albero, non viene applicata
virtual loss e non esistono thread concorrenti dentro una singola MCTS.

### 4.1 Selezione

La ricerca parte dalla radice e scende finche' il nodo e':

- completamente espanso;
- non terminale;
- non risolto dal solver.

Per ogni figlio viene calcolato `MCTSNode::SelectionScore`.

Il termine di sfruttamento converte il valore del figlio nella prospettiva del
genitore:

```text
childValue  = child.totalValue / child.visitCount
exploitation = (1 - childValue) / 2
```

Se la policy head ha prodotto una prior valida viene usato un termine PUCT:

```text
exploration =
    C * policyPrior * sqrt(parentVisits) / (childVisits + 1)
```

In assenza di prior si usa il fallback UCB1:

```text
exploration =
    C * sqrt(log(parentVisits + 1) / childVisits)
```

Infine viene aggiunto il progressive bias euristico:

```text
bias =
    PROGRESSIVE_BIAS_WEIGHT * heuristicScore / (childVisits + 1)
```

Con i valori correnti:

```text
EXPLORATION_C = 1.0
PROGRESSIVE_BIAS_WEIGHT = 5.0
```

Il bias aiuta nelle prime visite e decade progressivamente. Le statistiche
della ricerca diventano quindi dominanti quando il nodo viene visitato spesso.

Un risultato provato ha precedenza assoluta: una perdita certa per il
giocatore del figlio e' una vittoria per il genitore e riceve il punteggio
massimo; una vittoria certa dell'avversario riceve il minimo.

### 4.2 Espansione

Al primo accesso a un nodo:

1. viene controllato `BoardState`;
2. se la partita e' terminata, il nodo viene marcato terminale;
3. altrimenti vengono generate tutte le mosse legali;
4. le mosse vengono ordinate con `EvaluateMove`;
5. se esiste `forward_policy`, vengono calcolate le prior della policy.

Quando le prior sono disponibili, mosse e prior vengono riordinate insieme.
L'ordinamento e' stabile: a parita' di prior resta valido l'ordine euristico
precedente.

Una iterazione crea un solo figlio:

```cpp
Move move = node->unexpandedMoves.back();
node->unexpandedMoves.pop_back();
```

Il nodo diventa `isExpanded` soltanto quando non restano mosse da trasformare
in figli. Il nuovo stato viene controllato immediatamente per rilevare una
vittoria o una patta prodotta dalla mossa.

### 4.3 Valutazione della foglia

Il valore della foglia viene scelto in questo ordine:

1. risultato certo terminale o risolto;
2. valore presente nella transposition table;
3. inferenza della value network.

Su un cache miss:

```cpp
value = evaluator->EvaluateBoard(board);
```

In una normale ricerca singola l'evaluator esegue direttamente il forward.
Nel self-play parallelo la chiamata puo' fermarsi in attesa del coordinatore
cross-game descritto nella sezione 8.

### 4.4 Backpropagation

Il valore viene propagato dalla foglia alla radice, incrementando visite e
somma dei valori. L'inversione di segno a ogni passaggio mantiene ogni statistica
nella prospettiva corretta.

Se la foglia ha un risultato certo, dopo il backup viene invocato `TrySolve`
sul padre.

## 5. Policy target per il training

`SearchPolicyTargets` restituisce un elemento per ogni mossa legale:

```cpp
struct PolicyTarget
{
    Move move;
    int visitCount;
    double pi;
    int8_t provenResult;
};
```

La distribuzione target e':

```text
pi(a) = N(a) / sum_b N(b)
```

dove `N(a)` e' il numero di visite del figlio associato alla mossa `a`.

Nel self-play:

- nei primi `tempPlies` la mossa viene campionata proporzionalmente alle visite;
- successivamente viene scelta la mossa piu' visitata;
- una vittoria provata viene giocata subito;
- una sconfitta provata viene evitata finche' esiste un'alternativa.

## 6. Solver dei risultati certi

`provenResult` usa la prospettiva del giocatore che deve muovere nel nodo:

- `0`: risultato non dimostrato;
- `+1`: vittoria forzata;
- `-1`: sconfitta forzata.

Poiche' un figlio appartiene alla prospettiva dell'avversario:

- basta un figlio con `provenResult == -1` per dimostrare la vittoria del padre;
- il padre e' una sconfitta soltanto se e' completamente espanso e tutti i
  figli hanno `provenResult == +1`.

La vittoria puo' quindi essere propagata appena viene trovata una linea
vincente. Per dimostrare una sconfitta devono invece essere escluse tutte le
alternative.

Quando la radice viene risolta, la ricerca termina anticipatamente.

Prima di costruire l'albero, `IsInstantWin` controlla inoltre se una mossa
legale circonda immediatamente la regina avversaria. In quel caso la mossa
viene restituita senza avviare la ricerca.

## 7. Transposition table

La transposition table memorizza:

```cpp
struct TTEntry
{
    uint64_t hash;
    double value;
    int16_t depth;
    bool isExact;
};
```

La chiave e' l'hash Zobrist della board. L'hash comprende il giocatore di
turno, quindi il valore `side-to-move` resta coerente con la posizione.

La tabella:

- e' un array di dimensione potenza di due;
- usa `hash & mask` come indice;
- verifica sempre l'hash completo per distinguere le collisioni;
- preferisce la nuova entry quando la sua profondita' e' almeno quella
  memorizzata.

Nel self-play parallelo la TT e' `thread_local`. Ogni worker possiede una
tabella indipendente da `2^19` entry, circa 12 MB:

```text
worker 0 -> TT 0
worker 1 -> TT 1
worker 2 -> TT 2
...
```

La tabella resta viva tra i turni e tra le partite assegnate allo stesso
worker. Non servono mutex e non ci sono data race. La memoria totale cresce
pero' con il numero di worker:

```text
memoria TT approssimativa = worker * 12 MB
```

## 8. Parallelizzazione del self-play

L'obiettivo del self-play e' massimizzare le partite generate per unita' di
tempo. Il parallelismo e' applicato tra partite indipendenti.

### 8.1 Architettura

```text
Game worker 0: Board + MCTS + TT + RNG --\
Game worker 1: Board + MCTS + TT + RNG ----> coda value --> batch GPU
Game worker 2: Board + MCTS + TT + RNG ----> condiviso  --> risultati
Game worker N: Board + MCTS + TT + RNG --/

writer principale <--- risultati completi ordinati per game_id
```

Ogni game worker:

1. ottiene un indice tramite il contatore atomico `nextGame`;
2. crea board, RNG e record della propria partita;
3. chiama `SearchPolicyTargets` a ogni ply;
4. consegna il risultato completo al thread principale.

Il thread principale scrive i risultati in ordine di indice. Le partite
possono terminare in ordine diverso, ma nel JSONL i `game_id` restano ordinati
e ogni partita viene scritta come blocco completo.

### 8.2 Ownership e sincronizzazione

| Risorsa | Ownership | Protezione |
|---|---|---|
| Board e albero MCTS | una partita | nessuna condivisione |
| RNG | una partita | nessuna condivisione |
| Transposition table | un worker | `thread_local` |
| Statistiche MCTS | un worker | `thread_local` |
| Modello TorchScript | condiviso | `m_moduleMutex` |
| Code value e policy | condivise | mutex + condition variable separate |
| Output JSONL | thread principale | risultati consegnati tramite mutex |

Anche i buffer temporanei riutilizzati da `Board.cpp` e `Evaluation.cpp` sono
`thread_local`. Questo mantiene il beneficio del riuso delle allocazioni senza
permettere a due partite di scrivere nello stesso scratch buffer.

### 8.3 Batching tra partite

`TorchScriptValueEvaluator::EnableCrossGameBatching` avvia un inference worker
per il value head e, se il modello la espone, uno per la policy head. Quando
una MCTS incontra un cache miss value:

1. codifica la board in `GNNGraph`;
2. inserisce una `ValueRequest` nella coda;
3. conserva una `future<float>`;
4. si blocca in attesa del risultato.

Il thread `ValueBatchWorker`:

1. aspetta la prima richiesta;
2. raccoglie fino a `inferenceBatch` richieste;
3. attende al massimo `batchWaitUs` per riempire il batch;
4. chiama una sola volta `EvaluateGraphs`;
5. completa le promise associate alle richieste.

Ogni partita puo' avere al massimo una value request in volo, perche' la sua
MCTS e' sequenziale. Di conseguenza:

```text
batch massimo realmente riempibile <= numero di worker attivi
```

Questo batch contiene foglie di alberi diversi. Non altera la selezione della
singola MCTS e non richiede virtual loss.

### 8.4 Accesso alla policy head

La policy head viene usata durante l'espansione tramite
`EvaluateMovePriors`. Il forward e' protetto dallo stesso mutex del modello,
quindi e' sicuro tra thread.

Quando il batching cross-game e' attivo, le richieste policy vengono codificate
dal worker MCTS e aggregate in una coda separata. `PolicyBatchWorker` concatena
grafi e feature delle mosse, usa `move_batch` per associare ogni mossa alla
posizione corretta ed esegue un solo `forward_policy`. La softmax viene poi
applicata separatamente alle mosse di ciascuna posizione, quindi una partita
non influenza le prior delle altre.

### 8.5 Perche' non parallelizzare il singolo albero

Parallelizzare un solo albero richiederebbe:

- sincronizzazione dei nodi e delle liste dei figli;
- aggiornamenti atomici delle statistiche;
- virtual loss per evitare selezioni duplicate;
- gestione di risultati ottenuti su uno stato dell'albero gia' superato.

Nel self-play esistono gia' molte partite indipendenti. Usarle come unita' di
parallelismo offre batch naturali alla GPU e non modifica la semantica della
ricerca. Per questo il progetto mantiene ogni MCTS sequenziale.

## 9. Configurazione di SelfPlay

La sintassi completa e':

```text
SelfPlay <model.pt> <output.jsonl>
         [iterations] [maxPlies] [seed] [tempPlies]
         [game_id] [numGames] [device]
         [workers] [inferenceBatch] [batchWaitUs]
```

Esempio:

```powershell
.\build\SelfPlay.exe model.pt data\partite.jsonl `
    400 200 1 10 0 32 cuda 16 16 2000
```

Questo comando usa:

- 32 partite totali;
- 16 partite concorrenti;
- CUDA;
- batch value e policy fino a 16 richieste ciascuno;
- attesa massima del batch di 2000 microsecondi.

I default sono:

- `workers = min(numGames, hardware_concurrency)`;
- `inferenceBatch = workers`;
- `batchWaitUs = 2000`;
- `device = auto`.

`device=auto` usa CUDA quando disponibile e altrimenti la CPU. CUDA richiede
una distribuzione libtorch compilata con supporto CUDA.

Per riproducibilita' bit-a-bit usare:

```text
workers=1
inferenceBatch=1
```

Batch GPU diversi possono produrre minime differenze floating point e, in una
posizione al limite, modificare una scelta.

## 10. Scelta dei parametri

Indicazioni iniziali:

- non impostare piu' worker del numero di partite;
- non impostare `inferenceBatch` sopra `workers`;
- considerare circa 12 MB di TT per worker;
- provare batch 8, 16 e 32 su GPU;
- ridurre `batchWaitUs` se la latenza e' piu' importante del throughput;
- aumentarlo moderatamente se i batch restano spesso parziali.

La configurazione migliore deve essere misurata in partite/ora, non soltanto
in tempo medio del forward. Un numero eccessivo di worker puo' aumentare
contesa, memoria e latenza senza migliorare il throughput.

## 11. Test e diagnostica

I test rilevanti sono:

- `TestPerft`: correttezza e prestazioni della generazione delle mosse;
- `TestParallelBoard`: genera alberi Perft completi su otto thread e verifica
  che i conteggi siano identici;
- `TestTournamentBenchmark`: misura iterazioni, cache hit e tempo della rete
  nella ricerca a tempo;
- smoke test di `SelfPlay`: controlla JSONL, numero di partite e risultati.

Il Perft completo `Base+MLP` attualmente produce:

| Profondita' | Nodi |
|---:|---:|
| 1 | 7 |
| 2 | 294 |
| 3 | 6.678 |
| 4 | 151.686 |
| 5 | 5.427.108 |
| 6 | 192.353.904 |

Questi valori sono un riferimento utile dopo modifiche alla concorrenza o ai
buffer temporanei della board.

## 12. Riferimento rapido

| Concetto | Simbolo principale |
|---|---|
| Iterazione sequenziale | `MCTS::RunIteration` |
| Selezione PUCT/UCB1 | `MCTSNode::SelectionScore` |
| Target policy | `MCTS::SearchPolicyTargets` |
| Solver | `MCTS::TrySolve` |
| Vittoria immediata | `MCTS::IsInstantWin` |
| TT per worker | `MCTS::GetPersistentTranspositionTable` |
| Attivazione batch cross-game | `TorchScriptValueEvaluator::EnableCrossGameBatching` |
| Inference worker | `TorchScriptValueEvaluator::ValueBatchWorker` |
| Orchestrazione partite | `selfplay_main.cpp` |
| Test thread safety | `tests/TestParallelBoard.cpp` |
