# La value network: pipeline completa (dati → training → export → motore)

Questo documento spiega tutta la parte del progetto che riguarda la rete
neurale di valutazione: come è fatta, come si generano i dati per
addestrarla (self-play e partite umane), come si addestra, come si esporta
per il C++ e come il motore la usa in partita. Per il significato delle singole feature (cosa c'è dentro
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
validazione che conta è invece la **fedeltà dell'export** (§6).

### Il ciclo in pratica (i 4 comandi)

```bash
# 1. genera partite di self-play (qui: 20 partite, 400 iterazioni/mossa)
./build/SelfPlay hive_value_gnn.pt data/partite.jsonl 400 200 1 10 0 20

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
| `scripts/train_colab.ipynb` | Notebook per il training su GPU (Colab) |
| `scripts/train_kaggle.ipynb` | Gemello per Kaggle (più RAM: dataset completo) |
| `scripts/export_hive_value_gnn.py` | Export TorchScript + fidelity check |
| `scripts/boardspace/boardspace_to_jsonl.py` | Converte partite umane BoardSpace in JSONL (`--policy` per i target di imitazione) |
| `scripts/boardspace/download_boardspace.sh` | Scarica l'archivio BoardSpace e lancia la conversione |
| `scripts/boardspace/convert_policy_all.sh` | Conversione policy dell'intero archivio, per anno in parallelo |
| `src/selfplay_main.cpp` (`SelfPlay`) | Generazione dati di self-play (value + target policy) |
| `src/BoardEncoder.cpp` | L'encoder board → grafo (unico, C++) |
| `src/MoveEncoder.cpp` | L'encoder mossa → 32 feature + descrizione src/dst |
| `src/NeuralEvaluator.cpp` | Caricamento del `.pt` e inferenza in C++ |
| `tools/uhp_match.py` | Harness di valutazione: match tra modelli o engine via UHP |

I **dati** invece non stanno nel repository: tutto ciò che è grande o
rigenerabile vive in `data/` (nel `.gitignore`). Vedi §4 per il layout e
per come si condividono.

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
./build/SelfPlay <model.pt> <output.jsonl> [iterazioni] [maxPly] [seed] [plyTemperatura] [game_id] [numPartite] [device] [worker] [batchInferenza] [attesaBatchUs]
```

| Argomento | Default | Significato |
|---|---|---|
| `model.pt` | (obbligatorio) | Value network TorchScript, la stessa usata da `HiveEngine` |
| `output.jsonl` | (obbligatorio) | File di output; le righe vengono **aggiunte in coda** (append) |
| `iterazioni` | 400 | Iterazioni MCTS per ogni mossa (non tempo: riproducibile) |
| `maxPly` | 200 | Tetto di mosse: oltre, la partita viene troncata |
| `seed` | casuale | Seed base (stampato per ogni partita) |
| `plyTemperatura` | 10 | Ply iniziali con mossa campionata sulle visite MCTS (τ=1) |
| `game_id` | 0 | Identificativo base delle partite |
| `numPartite` | 1 | Quante partite giocare in questa esecuzione |
| `device` | `auto` | `cpu`, `cuda`, `cuda:N`; `auto` preferisce CUDA quando disponibile |
| `worker` | core disponibili | Numero massimo di partite eseguite contemporaneamente |
| `batchInferenza` | `worker` | Richieste da partite diverse aggregate in un forward |
| `attesaBatchUs` | 2000 | Attesa massima per riempire un batch GPU, in microsecondi |

La partita i-esima usa `seed+i` e `game_id+i`. Per una riproducibilità
bit-a-bit usare `worker=1` e `batchInferenza=1`: batch GPU diversi possono
produrre minime differenze floating point e cambiare una scelta al limite.

Per usare la GPU serve una distribuzione libtorch compilata con CUDA. Per
esempio, `... 0 32 cuda 16 16 2000` genera 32 partite con 16 worker e batch
condivisi fino a 16 richieste. Il batch non miscela foglie dello stesso
albero: ogni MCTS esegue sempre una sola iterazione sequenziale alla volta.

### Formato dell'output

Una riga JSON per posizione (JSONL: ogni riga è un oggetto autonomo — se
un'esecuzione si interrompe, al massimo si perde l'ultima riga):

```json
{"game_id":3,"ply":12,"side_to_move":"White","z":1,
 "moves":[{"visits":37,"pi":0.0925,"features":[...],"src":3,"dst":[[0,2]]}, ...],
 "x":[...],"edge_index":[...],"edge_attr":[...],"u":[...]}
```

- **`x`, `edge_index`, `edge_attr`, `u`** — il grafo encodato, in versione
  appiattita: `x` = N×18 valori in fila, `edge_index` = `[tutte le
  sorgenti..., tutte le destinazioni...]` (lunghezza 2E), `edge_attr` = E×9,
  `u` = 21. Il training li rimodella con `view()`.
- **`z`** — il label: esito finale dal punto di vista di chi muoveva. `+1`
  se ha poi vinto, `-1` se ha perso, `0` per patta o partita troncata.
- **`moves`** — i target della policy head: una entry per mossa legale, con
  `pi` = frazione delle visite MCTS alla radice (la distribuzione che la
  policy impara a imitare), le 32 `features` di `MoveEncoder`, e la
  descrizione strutturale `src`/`dst` in termini di nodi del grafo (per una
  futura policy sugli embedding dei nodi, §10). Il training value-only li
  ignora; con `--policy-weight > 0` diventano la seconda loss.
- **`game_id`, `ply`, `side_to_move`** — metadati, non input di training.
  Rendono il file verificabile: con `side_to_move` si può controllare, dal
  solo file, che in ogni partita decisiva `z` sia `+1` per il colore
  vincitore e `-1` per l'altro. Un bug nei label non fa crashare niente (la
  rete semplicemente impara male), quindi la verificabilità vale più del
  ~1% di spazio che i metadati costano.

### Scelte di design

- **Nei primi `plyTemperatura` ply la mossa si campiona sulle visite** (τ=1:
  probabilità proporzionale a quanto MCTS ha esplorato ogni mossa) perché a
  parità di pesi e posizione la ricerca è deterministica: senza una fonte di
  varietà, N partite sarebbero la stessa partita ripetuta N volte. Rispetto
  alle vecchie aperture uniformi, le mosse restano *sensate* (mai una
  sconfitta provata, subito una vittoria provata) e le etichette `z` sono
  più pulite. Dal ply `plyTemperatura` in poi si gioca la più visitata.
- **I due colori condividono la transposition table** (quella persistente di
  `src/MCTS.cpp`). In torneo sarebbe irrealistico — l'avversario è un
  processo separato che non ci regala le sue valutazioni, e infatti
  `tests/TestTournamentBenchmark.cpp` usa un avversario indipendente — ma in
  self-play è corretto e vantaggioso: stessa rete, stesse valutazioni, un
  valore calcolato per un colore vale anche per l'altro.
- **Più partite nello stesso processo** (`numPartite`) condividono una sola
  copia del modello. Ogni worker ha una transposition table separata e la
  mantiene calda tra le partite che gli vengono assegnate.
- **La board vuota di ply 0 è esclusa**: zero nodi non danno nulla da
  imparare, e il max-pooling su un grafo vuoto è mal definito.
- **Le mosse compaiono come feature, non come notazione UHP**: al training
  servono `move_features` e `pi`, non la stringa della mossa (la conversione
  mossa→notazione UHP vive dentro `Engine`). La ricerca passa da
  `SearchPolicyTargets`, che costa esattamente quanto `SearchIterations` ma
  espone la distribuzione delle visite alla radice.
- **Con la rete non addestrata quasi tutte le partite finiscono in patta**
  (per ripetizione o al tetto di mosse) e producono `z=0`: poco segnale. Per
  questo il bootstrap iniziale viene dalle partite umane (§4): da lì in poi
  il self-play produce partite sensate.
- **Parallelismo tra partite**: `SelfPlay` esegue più partite concorrenti con
  una MCTS sequenziale per partita e aggrega le value request in batch
  condivisi. Architettura, ownership e parametri sono descritti in
  `docs/MCTS_Implementation.md`.

## 4. Bootstrap da partite umane: `boardspace_to_jsonl.py`

Una rete a pesi casuali gioca partite di self-play senza senso che quasi mai
finiscono: poco segnale per imparare. Il bootstrap parte quindi da **partite
umane vere**, convertite nello stesso identico formato JSONL del `SelfPlay`,
così il training non distingue nemmeno da dove vengono i dati.

### La fonte: l'archivio BoardSpace

<https://www.boardspace.net/hive/hivegames/> contiene le partite giocate
online dal 2006 a oggi, una cartella `archive-ANNO/` per anno, dentro zip
periodici (`games-Jan-2-2025.zip`, ...). La variante che ci interessa è
`hive-plm` = il nostro **Base+MLP**, ed esiste solo **dal 2013** (anno di
uscita del pillbug): gli anni precedenti producono file vuoti, innocui.

Risultato sull'archivio completo 2013-2026: **~30.000 partite convertite,
~1,58 milioni di posizioni** con label vere e ben bilanciate tra vittorie
del Bianco e del Nero (tasso di conversione 89-99% a seconda dell'epoca del
formato; il resto sono abbandoni non etichettabili e partite col pillbug
pre-revisione delle regole, che l'engine attuale giustamente rifiuta).

```bash
# scarica E converte un intervallo di anni (riavviabile: salta quanto gia' fatto)
scripts/boardspace/download_boardspace.sh 2013 2026

# solo conversione, per una cartella di zip qualsiasi:
python3 scripts/boardspace/boardspace_to_jsonl.py data/boardspace/2025 \
    --output data/boardspace_2025.jsonl --model hive_value_gnn.pt

# conversione col formato policy (--policy): ogni posizione porta anche le
# mosse legali con pi=1 su quella giocata dall'umano (imitazione, ~4x piu'
# grande). Per l'archivio intero, in parallelo per anno:
scripts/boardspace/convert_policy_all.sh
```

`download_boardspace.sh` scarica gli zip in `data/boardspace/<anno>/` (una
richiesta ogni 0.4s per gentilezza verso il server, con retry sugli errori
di rete) e produce un `data/boardspace_<anno>.jsonl` per anno; il dettaglio
delle partite scartate finisce in `data/boardspace/<anno>/conversione.log`.

Il formato SGF di BoardSpace è cambiato più volte negli anni (tre epoche
del campo variante, due dialetti per le mosse, i file vecchi non hanno il
referto: lì l'esito può venire solo dal replay). Il convertitore li
gestisce tutti; il catalogo completo delle sottigliezze è nel docstring di
`boardspace_to_jsonl.py`.

### Dove stanno i dati e come si condividono

Tutto vive in `data/`, che è nel `.gitignore`: il repository porta il
codice, i dati si rigenerano dal codice. Layout:

| Percorso | Contenuto | Peso |
|---|---|---|
| `data/boardspace/<anno>/*.zip` | Gli SGF originali: la "materia prima" | ~260 MB totali |
| `data/boardspace_<anno>.jsonl` | Il dataset convertito, pronto per il training | ~12 GB totali |
| `data/dataset_jsonl.zip` | I JSONL compressi (~44:1), per Drive/Colab | ~280 MB |

Gli **zip sorgente** sono la parte da conservare (sono piccoli, e
l'archivio online potrebbe cambiare o sparire); i JSONL si rigenerano da
essi in ~30 minuti. Il file `dataset_jsonl.zip` si crea con:

```bash
cd data && zip dataset_jsonl.zip boardspace_*.jsonl && cd -
```

Il dataset già pronto sta nella **cartella Drive condivisa del progetto**
(`HiveGotThis_colab/`), che contiene gli zip sorgente, `dataset_jsonl.zip`
e i due script Python del training: chi ha bisogno dei dati li scarica da
lì e scompatta `dataset_jsonl.zip` in `data/`, senza dover eseguire
`download_boardspace.sh` (che rifà download e conversione da zero, molto
più lenti). È la stessa cartella usata dal notebook Colab (§5).

### Come funziona

I file sono SGF nel dialetto BoardSpace; l'ultimo campo di ogni mossa è già
quasi-notazione UHP. Lo script:

1. **parsa** ogni SGF (variante, giocatori, referto, mosse) e **traduce** le
   mosse in UHP — le sottigliezze del formato scoperte sui dati veri
   (turni confermati dal `Done`, ripensamenti, pass impliciti, riferimenti
   omessi) sono documentate nel docstring dello script;
2. **rigioca** ogni partita dentro `HiveEngine` (un solo processo riusato):
   dopo ogni mossa manda `features`, il comando UHP che stampa il grafo
   della posizione encodato da `BoardEncoder` — **lo stesso encoder del
   self-play**, zero rischio di incoerenze. Il replay è anche il
   validatore: una mossa intraducibile viene rifiutata dall'engine e la
   partita scartata (con log);
3. assegna il **label z** con doppio controllo: se il replay raggiunge uno
   stato terminale fa fede il verdetto dell'engine; altrimenti (abbandoni,
   timeout) si usa il referto `RE[...]`, cercando il **nome** del vincitore
   nel testo (il referto è localizzato nella lingua del client: polacco,
   cinese, ...). Se i due verdetti esistono e non concordano, la partita
   viene scartata: potrebbe essere una traduzione silenziosamente sbagliata.

Opzioni utili: `--variant` (default `hive-plm`), `--min-plies` (default 8,
scarta partite lampo), `--game-id-start` per accodare più conversioni allo
stesso file senza collisioni di `game_id` (il training comunque distingue le
partite per coppia file+id).

### Cosa NON filtra

Le partite contro i bot di BoardSpace (Dumbot/WeakBot/...) sono la
maggioranza e vengono **tenute**: per una value network conta l'esito della
partita, non l'eleganza delle mosse. Anche le vittorie per abbandono sono
tenute: chi abbandona di solito stava perdendo davvero.

## 5. Training: `train_hive_value_gnn.py`

```bash
python3 scripts/train_hive_value_gnn.py data/*.jsonl --output checkpoint.pt \
    [--init-weights checkpoint_prec.pt] [--sample 1.0] \
    [--epochs 20] [--batch-size 64] [--lr 1e-3] [--val-fraction 0.1] \
    [--hidden-dim 64] [--heads 4] [--dropout 0.2] [--device cpu] [--seed 42]
```

Accetta più file JSONL insieme (self-play + partite umane: stesso formato,
il training non distingue la provenienza). Per ogni epoca stampa la MSE di
training e, sulla validation, MSE e "segno-ok".

Il dataset viene caricato **tutto in RAM** (~13 KB a posizione: il dataset
BoardSpace completo richiede più dei ~12 GB del Colab gratuito e il
processo viene ucciso durante il caricamento). Il rimedio è `--sample`:
tiene solo quella frazione di posizioni, scelte a caso dentro ogni partita
— essendo quasi-duplicate tra loro se ne perde poca informazione, e tutte
le partite di tutti gli anni restano rappresentate (meglio che scartare
interi anni).

**Training su GPU con Colab**: si fa con `scripts/train_colab.ipynb`. Il
notebook legge dalla cartella Drive condivisa del progetto
(`HiveGotThis_colab/`, vedi §4) i file `hive_value_gnn.py`,
`train_hive_value_gnn.py` e `dataset_jsonl.zip`, e salva il checkpoint
direttamente su Drive, a prova di disconnessione. Su Colab si fa **solo il
training**: l'export TorchScript resta in locale, dove vive l'ambiente con
`torch_geometric==2.6.1` (è il motivo per cui checkpoint ed export sono due
passi separati).

**Training su GPU con Kaggle**: `scripts/train_kaggle.ipynb` è il gemello
per Kaggle Notebooks (~29 GB di RAM anche con GPU: il dataset completo ci
sta senza `--sample`). I dati si collegano come Dataset Kaggle privato con
gli stessi tre file della cartella Drive; le istruzioni passo-passo sono
nella prima cella. Nota empirica: sul target value il dataset completo
NON migliora rispetto al 40% (converge allo stesso punto dopo ~3M di
esempi visti) — le posizioni di una stessa partita condividono la stessa
etichetta z e sono quasi-duplicate, quindi l'informazione utile è limitata
dal numero di partite, non di posizioni. Le posizioni in più tornerebbero
utili con una policy head (§10), dove ogni posizione ha un'etichetta
propria (la mossa giocata).

Con `--init-weights` il training parte dai pesi di un checkpoint precedente
invece che da pesi casuali: è il meccanismo del **ciclo a generazioni**
(`gen0` da partite umane → self-play → training con `--init-weights gen0` →
`gen1` → ...). Vengono caricati solo i pesi del modello, mentre l'optimizer
riparte fresco: i suoi momenti interni erano accumulati su dati con una
distribuzione diversa da quelli nuovi. (Il caso diverso — riprendere un
training *interrotto a metà*, che richiederebbe anche lo stato
dell'optimizer e l'epoca corrente — non è supportato: i nostri training
sono corti.)

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
Attenzione: se in training si cambiano `--hidden-dim` o `--heads`, gli
stessi valori vanno passati anche all'export, altrimenti `load_state_dict`
fallisce per shape mismatch.

## 6. Export: `export_hive_value_gnn.py`

Perché due file separati (checkpoint del training e file esportato)? Sono
artefatti diversi con scopi diversi, anche se entrambi finiscono in `.pt`:
il **checkpoint** contiene solo i pesi in formato PyTorch nativo ed è il
"sorgente" — da lì si riprende il training (`--init-weights`) o si
ri-esporta; il **file TorchScript** è il modello *compilato* per libtorch,
il "binario" — dalla compilazione non si torna indietro (non si può
riprendere il training da un TorchScript). Il training salva il checkpoint
molte volte durante un run; l'export si fa una volta sola, quando si decide
che quel modello va dato al motore.

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

## 7. Inferenza in C++ (`NeuralEvaluator`)

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

Il self-play esegue piu' partite su worker CPU indipendenti. Ogni MCTS resta
sequenziale; quando una partita richiede una valutazione si ferma, mentre il
coordinatore raccoglie le richieste delle altre partite e le invia insieme a
`EvaluateGraphs`. Non vengono usate virtual loss.

L'MCTS chiama la rete solo sui cache miss della transposition table
persistente; il segno negamax e la prospettiva side-to-move sono gestiti
dall'MCTS, non dall'evaluator. Per i tempi reali (latenza per chiamata,
quota del tempo di ricerca spesa in inferenza) vedi
`tests/TestTournamentBenchmark.cpp`.

## 8. Valutazione: `tools/uhp_match.py`

Il training misura la MSE, ma la domanda vera è un'altra: **la rete nuova
gioca meglio della vecchia?** L'unico modo onesto di rispondere è farle
giocare l'una contro l'altra. `tools/uhp_match.py` fa esattamente questo: è
un arbitro che pilota due engine via protocollo UHP e conta i risultati. È
il cancello del ciclo a generazioni: una rete viene promossa solo se batte
la precedente.

```bash
# match tra due modelli (stesso binario), 20 partite da 1 secondo a mossa
python3 tools/uhp_match.py match build/HiveEngine build/HiveEngine \
    --white-model gen1.pt --black-model gen0.pt --games 20 --time 1

# una singola partita engine contro se stesso, mossa per mossa
python3 tools/uhp_match.py selfplay build/HiveEngine --model hive_value_gnn.pt
```

### Come funziona

Lo script non conosce le regole di Hive: lancia i due engine come processi
separati (ciascuno con il proprio eventuale modello `.pt` come argomento),
chiede `bestmove` a chi tocca e applica la mossa a **entrambi** i processi
con `play`, perché ognuno mantiene la propria board interna e le due vanno
tenute sincronizzate. Lo stato terminale (`WhiteWins`, `BlackWins`, `Draw`)
arriva dal GameString di risposta; una partita che supera `--max-plies`
(default 200) conta come "non conclusa".

Due accorgimenti rendono il confronto significativo:

- **Aperture casuali** (`--opening-plies`, default 4): a parità di modello e
  profondità l'engine è deterministico, quindi senza variazione tutte le
  partite del match sarebbero identiche. Le prime mosse vengono pescate a
  caso tra le `validmoves` (riproducibile via `--seed`).
- **Partite a coppie**: ogni apertura viene giocata due volte, con i colori
  invertiti. Un'apertura sbilanciata avvantaggia così una volta l'uno e una
  volta l'altro, e in media il vantaggio si annulla.

Nel riepilogo finale la patta vale mezzo punto (convenzione standard dei
match). `--time S` (secondi a mossa) è la modalità da torneo; `--depth N`
(N×1000 iterazioni MCTS) è deterministica e quindi riproducibile, utile per
i test.

### Non solo reti: engine diversi

I due argomenti posizionali sono i **binari**, e UHP è uno standard
pubblico: al posto del secondo `build/HiveEngine` si può passare qualsiasi
engine che lo parli — ad esempio Mzinga (l'implementazione di riferimento)
o nokamute. Stesso arbitro, zero modifiche: il confronto con engine esterni
misura la forza assoluta, quello tra generazioni la forza relativa.

## 9. Regole d'oro (riassunto degli errori facili)

- Le dimensioni feature C++ (`BoardEncoder.h`) e Python
  (`hive_value_gnn.py`) devono combaciare: nessun controllo automatico.
- `--hidden-dim`/`--heads` uguali tra training ed export.
- `torch.jit.script`, mai `trace`. `eval()` prima di export e inferenza.
- `torch_geometric == 2.6.1` finché la regressione TorchScript non è risolta.
- Testare l'export con un modello **non addestrato** appena si tocca
  qualcosa del ponte: fa emergere i problemi di compatibilità subito, non
  dopo ore di training.

## 10. Idee già individuate per il futuro (non implementate)

- **Data augmentation D6**: nessuna feature dei nodi è direzionale, quindi
  le 12 simmetrie della board esagonale cambiano solo l'etichetta di
  direzione piana degli archi. Si possono generare in Python, direttamente
  sui tensori già serializzati, **permutando i 6 slot piani di
  `edge_attr`** (gli slot 6-8 — su/giù/self — e `x`, `u`, `edge_index`
  restano invariati). Moltiplicherebbe il dataset ×12 senza rigiocare nulla.
- **Policy head — IN CORSO (luglio 2026)**: l'infrastruttura è implementata.
  La rete ha una `policy_head` "move-as-input" (embedding di board + 32
  feature della mossa da `MoveEncoder` → logit, softmax sulle sole legali;
  `forward()` resta value-only, compatibile coi modelli vecchi); l'MCTS usa
  le prior come termine PUCT quando il modello esportato le contiene
  (fallback automatico all'euristica `EvaluateMove` altrimenti); i target
  arrivano da tre strade: `SelfPlay` (visite MCTS, §3), il convertitore con
  `--policy` (imitazione delle mosse umane, §4), `uhp_match dataset` (via
  UHP). Il training si attiva con `--policy-weight`. Restano da fare: il
  primo training vero, la misura del costo (una chiamata rete in più per
  espansione) e il match di verifica. Evoluzione candidata ("v2"): policy
  sugli **embedding dei nodi** invece delle 32 feature a mano — i campi
  `src`/`dst` già emessi nei dataset servono a quello, senza rigiocare
  l'archivio.
- **Testa WDL / cross-entropy**: alternativa alla MSE se la `tanh` saturasse
  (gradienti migliori vicino a ±1); da considerare solo se il problema si
  manifesta nei dati.
- **GPU**: il training la supporta già (`--device cuda`); per l'inferenza
  C++ va prima misurato se conviene (grafi piccoli → l'overhead di
  trasferimento può superare il guadagno; il batching di `EvaluateGraphs`
  sarebbe il prerequisito per sfruttarla).
