# La value network: pipeline completa (self-play → training → export → motore)

Questo documento spiega tutta la parte del progetto che riguarda la rete
neurale di valutazione: come è fatta, come si generano i dati per
addestrarla, come si addestra, come si esporta per il C++ e come il motore
la usa in partita. Per il significato delle singole feature (cosa c'è dentro
`x`, `edge_attr`, `u`) vedi `Hive_GNN_Spec.md`; per compilare ed eseguire
vedi `guida_compilazione/Come_Compilare_Ed_Eseguire.md`.

## 1. Il quadro d'insieme

Python e C++ **non si parlano mai a runtime**: niente socket, niente
interprete Python embedded. Il ponte è un **file**. Python definisce e
addestra la rete, poi la esporta (architettura + pesi) in un unico file
TorchScript autocontenuto (`hive_value_gnn.pt`); il C++ lo carica una volta
all'avvio e lo esegue direttamente con libtorch. Il lavoro di Python finisce
nel momento in cui il file è scritto.

Il ciclo completo:

```
[C++] SelfPlay ──► dati JSONL ──► [Python] training ──► checkpoint pesi
                                                              │
[C++] motore più forte ◄── hive_value_gnn.pt ◄── [Python] export
        │
        └──► (si ripete: nuove partite di self-play con la rete migliorata)
```

Un principio fondamentale attraversa tutto il design: **le regole di Hive e
l'encoder vivono solo nel C++**. Il calcolo delle feature richiede vera
logica di gioco (articolazioni dell'alveare, pezzi bloccati, distanze), che
non è mai stata riscritta in Python. Per l'inferenza l'encoder gira dentro
MCTS; per il training il C++ serializza *gli stessi identici array* su file,
e Python li legge senza ricostruire nessuna board. Conseguenza: non esiste
un problema di "parità" tra due encoder, perché l'encoder è uno solo. La
validazione che conta è invece la **fedeltà dell'export** (§5).

### Il ciclo in pratica (i 4 comandi)

```bash
# 1. genera partite di self-play (qui: 20 partite, 400 iterazioni/mossa)
./build/SelfPlay hive_value_gnn.pt data/partite.jsonl 400 200 1 6 0 20

# 2. allena la rete sui dati
python3 scripts/train_hive_value_gnn.py data/partite.jsonl --output checkpoint.pt

# 3. esporta il checkpoint per il C++
python3 scripts/export_hive_value_gnn.py --weights checkpoint.pt --output hive_value_gnn.pt

# 4. il motore usa la rete nuova
./build/HiveEngine hive_value_gnn.pt
```

### I file coinvolti

| File | Ruolo |
|---|---|
| `scripts/hive_value_gnn.py` | Definizione del modello (unica, condivisa) |
| `scripts/train_hive_value_gnn.py` | Training sui dataset JSONL |
| `scripts/export_hive_value_gnn.py` | Export TorchScript + fidelity check |
| `src/selfplay_main.cpp` (`SelfPlay`) | Generazione dati di self-play |
| `src/BoardEncoder.cpp` | L'encoder board → grafo (unico, C++) |
| `src/NeuralEvaluator.cpp` | Caricamento del `.pt` e inferenza in C++ |

## 2. Il modello (`scripts/hive_value_gnn.py`)

`HiveValueGNN` è una graph neural network: 3 layer `GATv2Conv` (message
passing con attenzione sul grafo dei pezzi, `LayerNorm` + `ELU` dopo
ognuno), poi un pooling globale (concatenazione di media e massimo sui nodi)
unito alle 21 feature globali `u`, e una testa di valore (`Linear` →
`LayerNorm` → `ReLU` → `Dropout` → `Linear` → `Tanh`) che produce un singolo
numero in `[-1, 1]`.

Contratto di input/output (identico in training Python e inferenza C++):

| Tensore | Forma | Tipo | Contenuto |
|---|---|---|---|
| `x` | `[N, 18]` | float32 | feature dei pezzi sulla board (N nodi) |
| `edge_index` | `[2, E]` | int64 | archi in formato COO |
| `edge_attr` | `[E, 9]` | float32 | tipo di arco one-hot (6 direzioni piane, su, giù, self) |
| `u` | `[B, 21]` | float32 | feature globali, una riga per grafo del batch |
| `batch` | `[N]` | int64 | a quale grafo del batch appartiene ogni nodo |
| output | `[B, 1]` | float32 | valore in `[-1, 1]` |

Il valore è **dal punto di vista di chi muove** (side-to-move): +1 = chi
muove vince. Questa convenzione attraversa tutto il sistema — l'MCTS negamax
in C++, i label `z` del self-play, la loss di training — e non richiede mai
conversioni.

Le costanti di dimensione (`NODE_IN_DIM=18`, `EDGE_IN_DIM=9`,
`GLOBAL_DIM=21`) devono restare coerenti con `GNNNodeDim`/`GNNEdgeDim`/
`GNNGlobalDim` in `include/BoardEncoder.h`: **non c'è un controllo
automatico che le tenga sincronizzate**.

## 3. Generazione dati: `SelfPlay`

```bash
./build/SelfPlay <model.pt> <output.jsonl> [iterazioni] [maxPly] [seed] [plyAperturaCasuale] [game_id] [numPartite]
```

| Argomento | Default | Significato |
|---|---|---|
| `model.pt` | (obbligatorio) | Value network TorchScript, la stessa usata da `HiveEngine` |
| `output.jsonl` | (obbligatorio) | File di output; le righe vengono **aggiunte in coda** (append) |
| `iterazioni` | 400 | Iterazioni MCTS per ogni mossa (non tempo: riproducibile) |
| `maxPly` | 200 | Tetto di mosse: oltre, la partita viene troncata |
| `seed` | casuale | Seed base (stampato per ogni partita) |
| `plyAperturaCasuale` | 6 | Mosse iniziali scelte a caso invece che con MCTS |
| `game_id` | 0 | Identificativo base delle partite |
| `numPartite` | 1 | Quante partite giocare in questa esecuzione |

La partita i-esima del lotto usa `seed+i` e `game_id+i`: ogni singola
partita resta riproducibile da sola rilanciando con `numPartite=1` e i
valori corrispondenti.

### Formato dell'output

Una riga JSON per posizione (JSONL: ogni riga è un oggetto autonomo — se
un'esecuzione si interrompe, al massimo si perde l'ultima riga):

```json
{"game_id":3,"ply":12,"side_to_move":"White","z":1,"x":[...],"edge_index":[...],"edge_attr":[...],"u":[...]}
```

- **`x`, `edge_index`, `edge_attr`, `u`** — il grafo encodato, in versione
  appiattita: `x` = N×18 valori in fila, `edge_index` = `[tutte le
  sorgenti..., tutte le destinazioni...]` (lunghezza 2E), `edge_attr` = E×9,
  `u` = 21. Il training li rimodella con `view()`.
- **`z`** — il label: esito finale dal punto di vista di chi muoveva. `+1`
  se ha poi vinto, `-1` se ha perso, `0` per patta o partita troncata.
- **`game_id`, `ply`, `side_to_move`** — metadati, non input di training.
  Rendono il file verificabile: con `side_to_move` si può controllare, dal
  solo file, che in ogni partita decisiva `z` sia `+1` per il colore
  vincitore e `-1` per l'altro. Un bug nei label non fa crashare niente (la
  rete semplicemente impara male), quindi la verificabilità vale più del
  ~1% di spazio che i metadati costano.

### Scelte di design

- **Le prime mosse sono casuali** perché MCTS a parità di pesi e posizione è
  deterministico: senza rumore iniziale, N partite sarebbero la stessa
  partita ripetuta N volte.
- **I due colori condividono la transposition table** (quella persistente di
  `src/MCTS.cpp`). In torneo sarebbe irrealistico — l'avversario è un
  processo separato che non ci regala le sue valutazioni, e infatti
  `tests/TestTournamentBenchmark.cpp` usa un avversario indipendente — ma in
  self-play è corretto e vantaggioso: stessa rete, stesse valutazioni, un
  valore calcolato per un colore vale anche per l'altro.
- **Più partite nello stesso processo** (`numPartite`) evitano di ricaricare
  il modello e mantengono calda la transposition table tra partite (che
  condividono molte posizioni, soprattutto in apertura).
- **La board vuota di ply 0 è esclusa**: zero nodi non danno nulla da
  imparare, e il max-pooling su un grafo vuoto è mal definito.
- **Le mosse giocate non vengono salvate** (solo le posizioni encodate): la
  conversione mossa→notazione UHP vive dentro `Engine` e non serve al
  training.
- **Con la rete non addestrata quasi tutte le partite finiscono in patta**
  (per ripetizione o al tetto di mosse) e producono `z=0`: poco segnale. È
  transitorio — il bootstrap iniziale verrà da partite umane, e da lì il
  self-play produrrà partite sensate.
- **Nessun parallelismo**: se la generazione diventa il collo di bottiglia,
  la strada è lanciare più processi `SelfPlay` (ognuno col suo blocco di
  seed/`game_id` e il suo file), non parallelizzare la singola ricerca MCTS.

## 4. Training: `train_hive_value_gnn.py`

```bash
python3 scripts/train_hive_value_gnn.py data/*.jsonl --output checkpoint.pt \
    [--epochs 20] [--batch-size 64] [--lr 1e-3] [--val-fraction 0.1] \
    [--hidden-dim 64] [--heads 4] [--dropout 0.2] [--device cpu] [--seed 42]
```

Accetta più file JSONL insieme (self-play + future partite umane: stesso
formato, il training non distingue la provenienza). Per ogni epoca stampa la
MSE di training e, sulla validation, MSE e "segno-ok".

### Le scelte che contano

- **Split train/validation per partita, non per posizione.** Posizioni
  consecutive della stessa partita differiscono di una mossa e hanno lo
  stesso `z`: sono quasi duplicate. Dividendo per posizione, la validation
  conterrebbe copie di ciò che la rete ha visto in training e le metriche
  mentirebbero ("generalizza benissimo" quando ha solo memorizzato).
- **Loss MSE, `(z - v)²`.** Non è una scelta di comodo: il minimizzatore
  della MSE è la **media condizionata** `E[z | posizione]`. La stessa
  posizione compare nel dataset a volte con `z=+1` e a volte con `z=-1`
  (l'esito non dipende solo da lei); minimizzando la MSE la rete impara
  l'esito *atteso* — es. 70% vittorie e 30% sconfitte → valore `+0.4` — che
  è esattamente la semantica che MCTS assume quando media i valori nel
  backup. (Con la MAE, per contrasto, imparerebbe la mediana: valori secchi
  ±1 senza gradazioni di incertezza.) È lo stesso design della value head di
  AlphaZero: uscita `tanh` + errore quadratico.
- **Metrica "segno-ok"**: sulle sole posizioni decisive (`z ≠ 0`), quante
  volte il segno della predizione indovina il vincitore. Più leggibile della
  MSE: 50% = moneta, 100% = oracolo.
- **Viene salvato il checkpoint migliore su validation**, non l'ultimo: la
  train MSE continua a scendere anche quando la rete sta solo memorizzando
  (overfitting); la val MSE dice quando fermarsi davvero.

### Formato del checkpoint

`{"model_state_dict": ..., "epoch": ..., "val_mse": ..., "hidden_dim": ...,
"heads": ..., "dropout": ...}` — il `load_weights` di `hive_value_gnn.py`
(usato dall'export) accetta sia questo formato sia uno state_dict puro.
Attenzione: se cambi `--hidden-dim` o `--heads` in training, devi passare
gli stessi valori all'export, altrimenti `load_state_dict` fallisce per
shape mismatch.

## 5. Export: `export_hive_value_gnn.py`

```bash
# con pesi addestrati
python3 scripts/export_hive_value_gnn.py --weights checkpoint.pt --output hive_value_gnn.pt

# con pesi casuali (per testare la pipeline senza training)
python3 scripts/export_hive_value_gnn.py --output hive_value_gnn.pt
```

| Argomento | Default | Significato |
|---|---|---|
| `--weights` | nessuno | Checkpoint PyTorch (opzionale) |
| `--output` | `hive_value_gnn.pt` | File TorchScript prodotto |
| `--hidden-dim` | `64` | Deve combaciare col training |
| `--heads` | `4` | Deve combaciare col training |
| `--dropout` | `0.2` | Probabilità di dropout nella testa di valore |
| `--device` | `cpu` | Device su cui costruire/esportare |
| `--atol` | `1e-5` | Tolleranza del fidelity check |

Cosa fa, in breve:

1. Costruisce `HiveValueGNN` e, se c'è `--weights`, carica i pesi
   (`torch.load` con `weights_only=True`: carica solo tensori, senza il
   pickle completo che può eseguire codice arbitrario).
2. Mette il modello in `eval()` — obbligatorio: disattiva il `Dropout`,
   altrimenti l'inferenza sarebbe stocastica.
3. Compila con **`torch.jit.script`, non `trace`**: `trace` registra
   un'esecuzione con forme fisse dei tensori, sbagliato per una GNN dove N
   ed E cambiano ad ogni board; `script` compila il codice vero e preserva
   le forme dinamiche.
4. **Fidelity check**: esegue lo stesso input sul modello normale e su
   quello compilato e pretende che gli output combacino (`torch.allclose`).
   Se non combaciano si ferma con errore invece di produrre un file
   silenziosamente sbagliato. Nel design a encoder unico, questa è LA
   validazione di correttezza del ponte Python→C++.
5. Salva il file: autocontenuto (architettura compilata + pesi), da quel
   momento il C++ non ha bisogno di Python.

### Vincolo di versione importante

`torch_geometric` **deve restare `2.6.1`** (pinnato in
`scripts/requirements.txt`): dalla 2.7 una regressione nota rompe
`torch.jit.script` su `GATv2Conv` (`Could not cast value of type
Optional[Tensor] to bool`). Non è un bug di questo progetto.

## 6. Inferenza in C++ (`NeuralEvaluator`)

`TorchScriptValueEvaluator` (in `include/NeuralEvaluator.h` /
`src/NeuralEvaluator.cpp`) incapsula il modello: lo carica una volta nel
costruttore e offre `EvaluateBoard`/`EvaluateGraph` (valutazione singola,
usata dalle foglie MCTS) ed `EvaluateBoards`/`EvaluateGraphs` (valutazione
in batch: concatena più grafi in uno solo, traslando gli indici degli archi
e costruendo il vettore `batch` — lo stesso lavoro che fa il `DataLoader`
di PyTorch Geometric in training). Punti tecnici chiave, tutti già gestiti
dal codice:

- **`torch::from_blob` è zero-copy**: i tensori avvolgono direttamente la
  memoria degli array dell'encoder, senza copiarla. In cambio, i buffer
  devono restare vivi fino alla fine della `forward`.
- **`eval()` + `torch::NoGradGuard`**: niente dropout, niente autograd.
- **Tipi rigidi**: `edge_index` e `batch` in int64, il resto float32 —
  devono combaciare col training.
- **Un solo thread libtorch** (`SetTorchThreads(1)`, chiamato da
  `main.cpp`): sui grafi piccoli di Hive il thread pool interno costa più di
  quanto renda (misurato ~3x più lento coi thread di default).

L'MCTS chiama la rete solo sui cache miss della transposition table
persistente; il segno negamax e la prospettiva side-to-move sono gestiti
dall'MCTS, non dall'evaluator. Per i tempi reali (latenza per chiamata,
quota del tempo di ricerca spesa in inferenza) vedi
`tests/TestTournamentBenchmark.cpp`.

## 7. Regole d'oro (riassunto degli errori facili)

- Le dimensioni feature C++ (`BoardEncoder.h`) e Python
  (`hive_value_gnn.py`) devono combaciare: nessun controllo automatico.
- `--hidden-dim`/`--heads` uguali tra training ed export.
- `torch.jit.script`, mai `trace`. `eval()` prima di export e inferenza.
- `torch_geometric == 2.6.1` finché la regressione TorchScript non è risolta.
- Testare l'export con un modello **non addestrato** appena si tocca
  qualcosa del ponte: fa emergere i problemi di compatibilità subito, non
  dopo ore di training.

## 8. Idee già individuate per il futuro (non implementate)

- **Data augmentation D6**: nessuna feature dei nodi è direzionale, quindi
  le 12 simmetrie della board esagonale cambiano solo l'etichetta di
  direzione piana degli archi. Si possono generare in Python, direttamente
  sui tensori già serializzati, **permutando i 6 slot piani di
  `edge_attr`** (gli slot 6-8 — su/giù/self — e `x`, `u`, `edge_index`
  restano invariati). Moltiplicherebbe il dataset ×12 senza rigiocare nulla.
- **Bootstrap da partite umane**: un convertitore C++ che rigioca partite in
  notazione UHP (il parsing esiste già in `Engine`) ed emette lo stesso
  JSONL del self-play.
- **Testa WDL / cross-entropy**: alternativa alla MSE se la `tanh` saturasse
  (gradienti migliori vicino a ±1); da considerare solo se il problema si
  manifesta nei dati.
- **GPU**: il training la supporta già (`--device cuda`); per l'inferenza
  C++ va prima misurato se conviene (grafi piccoli → l'overhead di
  trasferimento può superare il guadagno; il batching di `EvaluateGraphs`
  sarebbe il prerequisito per sfruttarla).
