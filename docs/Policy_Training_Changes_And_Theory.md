# Policy Training - Modifiche Implementate e Note Teoriche

**Purpose:** documentare le modifiche fatte per supportare una policy head
allenabile, spiegare il flusso dati completo e raccogliere i concetti teorici
utili per usare correttamente la nuova pipeline.

---

## 1. Obiettivo delle modifiche

Prima il progetto supportava solo una value network:

```text
posizione Hive -> GNN -> value_head -> valore [-1, 1]
```

Questo basta per valutare le foglie dell'MCTS, ma non dice direttamente quali
mosse esplorare prima.

L'obiettivo delle modifiche e' stato aggiungere il supporto a una policy head:

```text
posizione Hive -> GNN -> value_head  -> valore posizione
                     -> policy_head -> score mosse legali
```

La policy head serve a imparare una distribuzione sulle mosse legali, usando
come target le visite MCTS:

```text
pi_mossa = visite_mossa / visite_totali
```

Questa e' la stessa idea generale usata da sistemi stile AlphaZero: la rete non
impara solo "quanto vale la posizione", ma anche "quali mosse meritano
attenzione".

---

## 2. File modificati o aggiunti

| File | Tipo | Ruolo |
|---|---|---|
| `scripts/hive_value_gnn.py` | modificato | Modello neurale condiviso: backbone GNN, value head e policy head. |
| `scripts/train_hive_value_gnn.py` | modificato | Training value-only oppure value+policy. |
| `scripts/export_hive_value_gnn.py` | modificato | Export TorchScript usando il modello condiviso. |
| `include/MoveEncoder.h` | nuovo | Interfaccia C++ per codificare una mossa in 32 feature. |
| `src/MoveEncoder.cpp` | nuovo | Implementazione stabile di `move_features`. |
| `include/MCTS.h` | modificato | Aggiunta struttura `PolicyTarget` e API per estrarre target policy. |
| `src/MCTS.cpp` | modificato | Aggiunta `SearchPolicyTargets`, che calcola visite/pi alla radice. |
| `include/Constants.h` | modificato | Nuovo comando UHP `policytargets`. |
| `include/Engine.h` | modificato | Dichiarazione di `CommandPolicyTargets`. |
| `src/Engine.cpp` | modificato | Implementazione del comando UHP `policytargets depth N`. |
| `tools/uhp_match.py` | modificato | Nuova modalita' `dataset` per generare JSONL self-play con policy. |
| `docs/Hive_Value_Policy_Head.md` | nuovo | Documento operativo sulla policy head. |

---

## 3. Modello neurale aggiornato

Il modello principale e' ancora `HiveValueGNN`, ma ora contiene:

```text
GNN backbone
value_head
policy_head
```

### Backbone GNN

Il backbone resta una GNN su grafo Hive:

```text
x          = feature nodi
edge_index = archi
edge_attr  = tipo/direzione archi
u          = feature globali
batch      = appartenenza nodo -> grafo
```

Il backbone produce un embedding globale:

```text
board_embedding = global_mean_pool(x) || global_max_pool(x) || u
```

Questa rappresentazione viene usata da entrambe le teste.

### Value Head

La value head produce:

```text
value in [-1, 1]
```

con significato:

```text
+1 = ottima per chi deve muovere
 0 = circa pari
-1 = pessima per chi deve muovere
```

Questa e' ancora l'uscita usata dal C++ attuale:

```python
model(x, edge_index, edge_attr, u, batch)
```

### Policy Head

La policy head non valuta direttamente la board. Valuta una coppia:

```text
board_embedding + move_features
```

e produce un logit per ogni mossa candidata.

```text
logit_mossa = policy_head([board_embedding || move_features])
```

Poi, durante il training, i logit delle mosse della stessa posizione vengono
normalizzati con:

```python
softmax(logits_posizione)
```

---

## 4. Perche' la policy head e' fatta come move scorer

In giochi come gli scacchi si puo' spesso usare una policy head con output
quasi fisso: da-casella/a-casella/promozione, ecc.

In Hive il numero di mosse legali e la forma delle mosse cambiano molto:

```text
piazzamenti
movimenti
pezzi sopra stack
Pillbug che muove altri pezzi
pass
```

Per questo la policy head e' stata implementata come **move scorer**:

```text
per ogni mossa legale:
    score = f(posizione, mossa)
```

Vantaggi:

```text
funziona con qualunque numero di mosse legali
non richiede un enorme spazio azione fisso
usa solo mosse gia' validate dal motore C++
e' semplice da batchare in PyTorch Geometric
```

---

## 5. Codifica stabile `move_features`

La dimensione e' fissata a:

```text
MOVE_FEATURE_DIM = 32
```

Lato C++:

```cpp
constexpr int MoveFeatureDim = 32;
std::vector<float> EncodeMoveFeatures(const Board& board, const Move& move);
```

Lato Python:

```python
MOVE_FEATURE_DIM = 32
```

Questa sincronizzazione e' importante: se C++ produce 32 feature, Python deve
costruire la rete con `move_feature_dim=32`.

### Feature incluse

Le feature coprono:

```text
tipo pezzo mosso
colore relativo
placement/movement/pass
distanza da mia regina
distanza da regina nemica
avvicinamento alle regine
adiacenze alle regine
grado locale della destinazione
altezza degli stack
instant win
turno normalizzato
regine in gioco
```

La tabella completa e' in `docs/Hive_Value_Policy_Head.md`.

### Perche' codificarle in C++

La scelta e' intenzionale: il C++ conosce gia' le regole vere di Hive e genera
le mosse legali. Se la codifica delle mosse fosse reimplementata in Python,
potrebbero nascere divergenze difficili da vedere.

Con un solo encoder C++:

```text
C++ genera board features
C++ genera move features
Python legge tensori gia' pronti
```

Questo riduce il rischio di avere dataset e inferenza runtime non allineati.

---

## 6. Nuovo comando UHP `policytargets`

E' stato aggiunto il comando:

```text
policytargets depth N
```

Internamente:

```text
1. genera le mosse legali alla radice
2. esegue MCTS per N * 1000 iterazioni
3. conta le visite dei figli della radice
4. calcola pi = visite / visite_totali
5. esporta board features + moves + move_features + pi
```

Il risultato e' un JSON compatibile con il training:

```json
{
  "x": [...],
  "edge_index": [...],
  "edge_attr": [...],
  "u": [...],
  "turn": "White",
  "turn_index": 12,
  "move_feature_dim": 32,
  "mcts_iterations": 2000,
  "best_move": "wA1 bQ-",
  "moves": [
    {
      "move": "wA1 bQ-",
      "visits": 812,
      "pi": 0.406,
      "features": [...]
    }
  ]
}
```

---

## 7. Nuova funzione MCTS per i target policy

In `MCTS` e' stata aggiunta:

```cpp
static std::vector<PolicyTarget> SearchPolicyTargets(
    const Board& rootBoard,
    int maxIterations
);
```

`PolicyTarget` contiene:

```cpp
struct PolicyTarget
{
    Move move;
    int visitCount;
    double pi;
};
```

Questa funzione e' simile a `SearchIterations`, ma invece di restituire solo la
mossa migliore restituisce la distribuzione di visite sulla radice.

Questo e' importante perche' la policy non va allenata solo sulla mossa scelta:
e' meglio allenarla sulla distribuzione MCTS completa.

Esempio:

```text
mossa A: 700 visite -> pi 0.70
mossa B: 200 visite -> pi 0.20
mossa C: 100 visite -> pi 0.10
```

Questo target contiene piu' informazione di:

```text
mossa A = giusta
tutte le altre = sbagliate
```

---

## 8. Generazione dataset self-play

Il tool `tools/uhp_match.py` ora ha una modalita':

```bash
python tools/uhp_match.py dataset <engine> --output data/selfplay_policy.jsonl
```

Esempio completo:

```bash
python tools/uhp_match.py dataset build/HiveEngine \
  --engine-arg hive_value_gnn.pt \
  --output data/selfplay_policy.jsonl \
  --games 100 \
  --depth 2 \
  --max-plies 100
```

Il default del tool e' allineato al torneo: `Base+MLP` e massimo 100 mezze
mosse. Per cambiare espansioni in esperimenti locali puoi usare
`--game-type Base`, ma i dati utili per il torneo dovrebbero usare `Base+MLP`.

Per ogni posizione:

```text
1. chiama policytargets depth N
2. salva il JSON prodotto
3. gioca best_move
4. continua fino a fine partita o max-plies
5. alla fine aggiunge z a ogni posizione
```

La label `z` viene calcolata dal risultato finale:

```text
se side_to_move della posizione vince  -> z = +1
se side_to_move della posizione perde  -> z = -1
se pareggio o ply cap                  -> z = 0
```

Il file prodotto e' JSONL:

```text
una riga = una posizione
```

---

## 9. Training value + policy

Il training value-only resta valido:

```bash
python scripts/train_hive_value_gnn.py data/selfplay.jsonl \
  --output checkpoint.pt
```

Per allenare anche la policy:

```bash
python scripts/train_hive_value_gnn.py data/selfplay_policy.jsonl \
  --policy-weight 1.0 \
  --output checkpoint_policy.pt
```

La loss totale e':

```text
loss = value_loss + policy_weight * policy_loss
```

Dove:

```text
value_loss  = MSE(value_pred, z)
policy_loss = cross entropy soft-target tra logits policy e pi MCTS
```

`policy_weight` controlla quanto pesa la policy rispetto alla value.

Valori iniziali sensati:

```text
0.5
1.0
2.0
```

Se la value peggiora troppo, abbassa `policy_weight`. Se la policy non impara,
alzalo un po'.

---

## 10. Export TorchScript

Dopo il training:

```bash
python scripts/export_hive_value_gnn.py \
  --weights checkpoint_policy.pt \
  --output hive_value_gnn.pt
```

Se hai cambiato dimensioni:

```bash
python scripts/export_hive_value_gnn.py \
  --weights checkpoint_policy.pt \
  --hidden-dim 128 \
  --heads 4 \
  --move-feature-dim 32 \
  --output hive_value_gnn.pt
```

I parametri devono combaciare con quelli del training.

---

## 11. Compatibilita' con il C++ attuale

Il C++ attuale usa ancora solo:

```text
forward(x, edge_index, edge_attr, u, batch) -> value
```

La policy head viene esportata nel modello, ma non viene ancora usata dal MCTS
per selezionare le mosse.

Questo e' voluto: permette di introdurre la policy in modo incrementale.

Stato attuale:

```text
training policy: supportato
dataset con moves/pi: supportato
export modello con policy head: supportato
uso policy come prior nel MCTS C++: non ancora integrato
```

---

## 12. Prossimo passo teorico: usare la policy nel MCTS

Oggi l'MCTS usa UCB1 + progressive bias euristico:

```text
score = exploitation + exploration + heuristic_bias
```

Con la policy head si puo' passare a una forma piu' vicina a PUCT:

```text
score = Q + C * P * sqrt(N_parent) / (1 + N_child)
```

Dove:

```text
Q = valore medio stimato del figlio
P = prior policy della mossa
N = visite
```

La policy head quindi sostituirebbe o affiancherebbe il progressive bias
euristico.

Concettualmente:

```text
prima:
    "questa mossa sembra buona secondo una euristica manuale"

dopo:
    "questa mossa sembra buona secondo cio' che la rete ha imparato da MCTS"
```

---

## 13. Note teoriche sui target `pi`

La policy non dovrebbe essere allenata solo sulla mossa finale scelta, perche'
quella sarebbe una label troppo dura.

Meglio usare le visite MCTS:

```text
pi = visite / visite_totali
```

Perche':

```text
mantiene informazione sulle alternative
esprime incertezza
premia mosse simili se entrambe buone
stabilizza il training
```

Esempio:

```text
MCTS visita:
A = 500
B = 450
C = 50
```

Una label hard direbbe:

```text
A = 1
B = 0
C = 0
```

Ma in realta' B era quasi buona quanto A. La distribuzione policy conserva
questa sfumatura:

```text
A = 0.50
B = 0.45
C = 0.05
```

---

## 14. Qualita' dei dati

La policy e' utile solo se il target MCTS e' abbastanza buono.

Dataset a `depth 1`:

```text
veloce
rumoroso
utile per test pipeline
```

Dataset a `depth 2` o `depth 3`:

```text
piu' lento
target migliori
piu' utile per training serio
```

Il consiglio pratico e':

```text
1. genera poche partite depth 1 per testare formato e training
2. poi genera dataset piu' grande depth 2+
3. confronta modelli con torneo, non solo validation loss
```

---

## 15. Verifiche eseguite

Sono stati verificati gli script Python con:

```bash
python -m py_compile tools/uhp_match.py scripts/hive_value_gnn.py scripts/train_hive_value_gnn.py scripts/export_hive_value_gnn.py
```

La build C++ locale non e' stata completata perche' CMake non trova libtorch:

```text
TorchConfig.cmake / torch-config.cmake non trovato
```

Per compilare serve configurare:

```bash
cmake -B build -S . -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build
```

---

## 16. Comandi principali

Generare dataset policy:

```bash
python tools/uhp_match.py dataset build/HiveEngine \
  --engine-arg hive_value_gnn.pt \
  --output data/selfplay_policy.jsonl \
  --games 100 \
  --depth 2
```

Allenare value + policy:

```bash
python scripts/train_hive_value_gnn.py data/selfplay_policy.jsonl \
  --policy-weight 1.0 \
  --output checkpoint_policy.pt
```

Esportare:

```bash
python scripts/export_hive_value_gnn.py \
  --weights checkpoint_policy.pt \
  --output hive_value_gnn.pt
```

Testare una singola partita self-play:

```bash
python tools/uhp_match.py selfplay build/HiveEngine \
  --engine-arg hive_value_gnn.pt \
  --time 00:00:05
```

Questo simula meglio il comando ufficiale del torneo:

```text
bestmove time 00:00:05
```

---

## 17. Riassunto mentale

Value head:

```text
Quanto vale questa posizione?
```

Policy head:

```text
Quali mosse meritano attenzione?
```

MCTS genera esperienza:

```text
visite alle mosse -> pi
risultato finale -> z
```

Training:

```text
GNN impara value da z
GNN impara policy da pi
```

Uso futuro:

```text
value guida la valutazione delle foglie
policy guida l'esplorazione dell'albero
```
