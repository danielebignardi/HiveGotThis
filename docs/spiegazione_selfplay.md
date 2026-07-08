# Self-play: generazione dati di training (`src/selfplay_main.cpp`)

L'eseguibile `SelfPlay` fa giocare il motore contro se stesso per **una
partita** e salva su file, riga per riga, tutte le posizioni incontrate
insieme all'esito finale. Questi file sono il dataset con cui, in Python, si
addestra la value network a valutare una board.

Il posto di questo pezzo nel ciclo completo di training:

```
[C++] SelfPlay  →  dati JSONL  →  [Python] training  →  pesi  →  export .pt  →  [C++] motore più forte  →  si ripete
```

L'export `.pt` è già coperto da `scripts/export_hive_value_gnn.py` (vedi
`scripts/spiegazione_export_gnn.md`); la parte di training Python verrà
aggiunta in seguito.

## Come si usa

```bash
./build/SelfPlay <model.pt> <output.jsonl> [iterazioni] [maxPly] [seed] [plyAperturaCasuale] [game_id]
```

| Argomento            | Default        | Significato                                                        |
|----------------------|----------------|--------------------------------------------------------------------|
| `model.pt`           | (obbligatorio) | Value network TorchScript, la stessa usata da `HiveEngine`          |
| `output.jsonl`       | (obbligatorio) | File di output; le righe vengono **aggiunte in coda** (append)      |
| `iterazioni`         | 400            | Iterazioni MCTS per ogni mossa (non tempo: riproducibile)           |
| `maxPly`             | 200            | Tetto di mosse: oltre, la partita viene troncata                    |
| `seed`               | casuale        | Seed del generatore casuale (stampato a fine partita, per riprodurla) |
| `plyAperturaCasuale` | 6              | Numero di mosse iniziali scelte a caso invece che con MCTS          |
| `game_id`            | 0              | Identificativo della partita, riportato in ogni riga dell'output    |

Esempio: generare un piccolo dataset di 20 partite in un unico file, con
seed e `game_id` diversi per ognuna:

```bash
for seed in $(seq 1 20); do
    ./build/SelfPlay hive_value_gnn.pt data/partite.jsonl 400 200 $seed 6 $seed
done
```

A fine partita l'eseguibile stampa su stderr un riepilogo: esito, numero di
ply, seed usato e quante posizioni sono state scritte.

## Formato dell'output

Una riga JSON per posizione (formato JSONL: ogni riga è un oggetto JSON
autonomo — se un'esecuzione si interrompe a metà, al massimo si perde
l'ultima riga, il resto del file resta valido). Esempio di riga (array
abbreviati):

```json
{"game_id":3,"ply":12,"side_to_move":"White","z":1,"x":[...],"edge_index":[...],"edge_attr":[...],"u":[...]}
```

I campi:

- **`x`, `edge_index`, `edge_attr`, `u`** — il grafo della board encodato da
  `BoardEncoder::encode`, negli stessi identici formati che la rete riceve in
  inferenza dal C++ (vedi `docs/Hive_GNN_Spec.md`): `x` = N×18 feature dei
  nodi appiattite, `edge_index` = lista COO appiattita `[src..., dst...]` di
  lunghezza 2E, `edge_attr` = E×9 one-hot appiattite, `u` = 21 feature
  globali. Python può quindi leggere il dataset **senza sapere nulla delle
  regole di Hive**: le regole vivono solo nel C++.
- **`z`** — il label di training: l'esito finale della partita **dal punto di
  vista di chi doveva muovere in quella posizione** (convenzione
  side-to-move, la stessa dell'output della rete): `+1` se chi muoveva ha poi
  vinto, `-1` se ha perso, `0` per patta o partita troncata al tetto di
  mosse. Essendo già nella prospettiva giusta, il training Python non deve
  fare nessuna conversione.
- **`game_id`, `ply`, `side_to_move`** — metadati, non input di training.
  Servono a rendere il file verificabile e diagnosticabile a posteriori: ad
  esempio, con `side_to_move` si può controllare leggendo solo il file che in
  ogni partita decisiva `z` sia `+1` per tutte le posizioni del colore
  vincitore e `-1` per l'altro. Senza, un bug nei label (che non fa crashare
  nulla: la rete semplicemente impara male) sarebbe invisibile. Costano meno
  dell'1% della dimensione di una riga.

## Come funziona internamente

1. Carica la value network, forza libtorch a 1 thread (per i grafi piccoli di
   Hive è più veloce, vedi commento in `src/main.cpp`) e la collega a MCTS.
2. Gioca la partita, sempre su `GameType` Base+MLP:
   - le prime `plyAperturaCasuale` mosse sono scelte **a caso** tra le
     legali;
   - dalle successive in poi, **entrambi i colori** usano
     `MCTS::SearchIterations` con lo stesso numero di iterazioni.
3. Prima di ogni mossa registra la posizione corrente (encoding + chi muove).
   La board vuota di ply 0 viene saltata: zero nodi non danno nulla da
   imparare, e il max-pooling su un grafo vuoto è mal definito.
4. A partita finita calcola `z` per ogni posizione registrata e scrive tutte
   le righe in append sul file di output.

### Perché le prime mosse sono casuali

MCTS con gli stessi pesi e la stessa posizione iniziale è deterministico: due
esecuzioni identiche produrrebbero **la stessa identica partita**, e un
dataset di N partite conterrebbe in realtà una sola partita ripetuta N volte.
Le mosse casuali iniziali differenziano le partite; da lì in poi il gioco è
"serio" (MCTS puro).

### Perché i due colori condividono la transposition table

In torneo condividere la cache con l'avversario sarebbe irrealistico (è un
processo separato che non ci regala le sue valutazioni — per questo
`tests/TestTournamentBenchmark.cpp` usa un avversario indipendente). In
self-play invece è **corretto e vantaggioso**: entrambi i colori usano la
stessa rete, quindi la valutazione di una board è la stessa per entrambi, e
un valore calcolato durante la ricerca del Bianco è riutilizzabile pari pari
dalla ricerca del Nero. La transposition table persistente di MCTS (vedi
`src/MCTS.cpp`) fa questo automaticamente, senza codice in più.

## Limiti noti di questa versione (scelte deliberate)

- **Una partita per esecuzione, nessun parallelismo.** Prima si valida il
  ciclo end-to-end; se la velocità di generazione diventa il collo di
  bottiglia, la strada prevista è parallelizzare le *partite* (processi
  indipendenti, come nel ciclo bash qui sopra), non la singola ricerca MCTS.
- **Con la rete non ancora addestrata quasi tutte le partite finiscono in
  patta** per la regola della configurazione ripetuta (i pezzi oscillano
  avanti e indietro) o al tetto di mosse: tutte le posizioni ricevono `z=0` e
  il dataset contiene poco segnale. È una situazione transitoria: il
  bootstrap iniziale della rete verrà fatto in Python su partite umane, e da
  lì in poi il self-play produrrà partite sensate.
- **Le mosse giocate non vengono salvate** (solo le posizioni encodate): la
  conversione mossa→notazione UHP vive dentro `Engine` e non serve al
  training. Se in futuro servirà rigiocare le partite, si potrà aggiungere.
