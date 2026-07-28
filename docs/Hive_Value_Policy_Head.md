# Hive GNN - Value Head e Policy Head

**Purpose:** spiegare la modifica alla rete neurale: la GNN ora mantiene la
value head gia' usata dal motore C++ e aggiunge una policy head opzionale per
imparare quali mosse sono promettenti.

---

## 1. Cosa e' stato modificato

Sono stati aggiornati tre script Python:

| File | Modifica |
|---|---|
| `scripts/hive_value_gnn.py` | La classe `HiveValueGNN` ora contiene sia `value_head` sia `policy_head`. |
| `scripts/train_hive_value_gnn.py` | Il training resta value-only di default, ma puo' allenare anche la policy se il dataset contiene target di policy. |
| `scripts/export_hive_value_gnn.py` | L'export ora importa il modello condiviso da `hive_value_gnn.py`, evitando una seconda definizione duplicata. |

La compatibilita' con il C++ e' stata mantenuta: il metodo `forward(...)`
continua a restituire solo la value.

```python
model(x, edge_index, edge_attr, u, batch) -> value
```

Questo significa che il motore C++ attuale puo' continuare a caricare il
modello TorchScript e usarlo come prima, senza modifiche immediate.

---

## 2. Value head

La value head risponde alla domanda:

```text
Quanto e' buona questa posizione per chi deve muovere?
```

Produce un singolo numero:

```text
+1 = posizione molto buona per chi deve muovere
 0 = posizione circa pari
-1 = posizione molto cattiva
```

Nel progetto attuale e' la parte usata dall'MCTS per valutare le foglie
dell'albero di ricerca.

Flusso:

```text
Board Hive
  -> BoardEncoder C++
  -> x, edge_index, edge_attr, u
  -> HiveValueGNN
  -> value_head
  -> valore [-1, 1]
```

Il training value-only richiede, per ogni posizione, il target `z`:

```json
{"game_id":"g1","x":[...],"edge_index":[...],"edge_attr":[...],"u":[...],"z":1}
```

---

## 3. Policy head

La policy head risponde a una domanda diversa:

```text
Tra le mosse legali disponibili, quali sembrano piu' promettenti?
```

Non restituisce direttamente una vittoria o una sconfitta. Restituisce uno
score per ogni mossa candidata. Poi quegli score vengono trasformati in
probabilita' con una softmax fatta sulle sole mosse legali della posizione.

Esempio concettuale:

```text
mossa A -> 0.62
mossa B -> 0.25
mossa C -> 0.13
```

Serve per guidare l'MCTS: invece di esplorare tutte le mosse quasi allo stesso
modo, il motore puo' dare piu' priorita' alle mosse che la rete considera
interessanti.

La nuova architettura e':

```text
                    -> value_head  -> valore posizione
Board -> GNN encoder
                    -> policy_head -> score mosse candidate
```

---

## 4. Come funziona tecnicamente

La GNN costruisce prima un embedding globale della board:

```text
board_embedding = mean_pool(nodi) || max_pool(nodi) || u
```

La value head riceve solo `board_embedding`.

La policy head riceve invece:

```text
board_embedding della posizione
+
move_features della mossa candidata
```

e produce un logit:

```text
policy_logit = policy_head([board_embedding || move_features])
```

Nel codice sono disponibili questi metodi:

```python
model.encode_board(x, edge_index, edge_attr, u, batch)
model.forward_policy_from_embedding(board_embedding, move_features, move_batch)
model.forward_policy(x, edge_index, edge_attr, u, batch, move_features, move_batch)
```

`move_batch` dice a quale posizione appartiene ogni mossa. Serve per batchare
piu' posizioni insieme.

Esempio:

```text
posizione 0: 3 mosse
posizione 1: 2 mosse

move_batch = [0, 0, 0, 1, 1]
```

---

## 5. Formato JSONL per allenare anche la policy

Il training value-only continua a funzionare con il vecchio formato.

Per allenare anche la policy, ogni record deve contenere anche le mosse
candidate e il target `pi`.

Formato consigliato:

```json
{
  "game_id": "g1",
  "x": [...],
  "edge_index": [...],
  "edge_attr": [...],
  "u": [...],
  "z": 1,
  "moves": [
    {"features": [...], "pi": 0.70},
    {"features": [...], "pi": 0.20},
    {"features": [...], "pi": 0.10}
  ]
}
```

Il campo `features` rappresenta la mossa in forma numerica. La dimensione
stabile e':

```text
MOVE_FEATURE_DIM = 32
```

La codifica e' implementata in `include/MoveEncoder.h` e
`src/MoveEncoder.cpp`, lato C++, quindi il generatore self-play e il futuro uso
runtime della policy condividono la stessa rappresentazione.

### Specifica `move_features`

| Idx | Feature | Significato |
|---|---|---|
| 0 | Queen | tipo pezzo mosso, one-hot |
| 1 | Ant | tipo pezzo mosso, one-hot |
| 2 | Spider | tipo pezzo mosso, one-hot |
| 3 | Grasshopper | tipo pezzo mosso, one-hot |
| 4 | Beetle | tipo pezzo mosso, one-hot |
| 5 | Mosquito | tipo pezzo mosso, one-hot |
| 6 | Ladybug | tipo pezzo mosso, one-hot |
| 7 | Pillbug | tipo pezzo mosso, one-hot |
| 8 | Relative color | +1 se il pezzo mosso e' del side-to-move, -1 se e' avversario; utile per Pillbug |
| 9 | Is placement | 1 se e' un piazzamento |
| 10 | Is movement | 1 se e' un movimento da board |
| 11 | Is pass | 1 se e' `pass` |
| 12 | Source on board | 1 se la mossa ha una sorgente |
| 13 | Destination occupied | 1 se la destinazione era occupata prima della mossa |
| 14 | Dest adjacent my queen | 1 se la destinazione tocca la mia regina |
| 15 | Dest adjacent enemy queen | 1 se la destinazione tocca la regina avversaria |
| 16 | Dest dist my queen | distanza normalizzata destinazione -> mia regina |
| 17 | Dest dist enemy queen | distanza normalizzata destinazione -> regina avversaria |
| 18 | Source dist my queen | distanza normalizzata sorgente -> mia regina, 1 se non esiste |
| 19 | Source dist enemy queen | distanza normalizzata sorgente -> regina avversaria, 1 se non esiste |
| 20 | Enemy queen approach | `source_enemy_dist - dest_enemy_dist`, clamp in [-1,1] |
| 21 | My queen approach | `source_my_dist - dest_my_dist`, clamp in [-1,1] |
| 22 | Dest planar degree | vicini occupati della destinazione / 6 |
| 23 | Source planar degree | vicini occupati della sorgente / 6 |
| 24 | Dest stack height | altezza stack destinazione prima della mossa / 4, clamp a 1 |
| 25 | Source stack height | altezza stack sorgente prima della mossa / 4, clamp a 1 |
| 26 | Instant win | 1 se la mossa circonda completamente la regina avversaria |
| 27 | Becomes adjacent enemy queen | 1 se da non adiacente diventa adiacente alla regina nemica |
| 28 | Ends adjacent my queen | 1 se finisce adiacente alla mia regina |
| 29 | Turn | turno normalizzato / 40 |
| 30 | My queen in play | 1 se la mia regina e' gia' sulla board |
| 31 | Enemy queen in play | 1 se la regina avversaria e' gia' sulla board |
 
Per una mossa `pass`, quasi tutte le feature sono 0 tranne `is_pass`,
`turn`, `my_queen_in_play` e `enemy_queen_in_play`.

Il campo `pi` e' la probabilita' target della mossa. In self-play di solito si
calcola dalle visite MCTS:

```text
pi_mossa = visite_mossa / visite_totali
```

Formato alternativo accettato:

```json
{
  "game_id": "g1",
  "x": [...],
  "edge_index": [...],
  "edge_attr": [...],
  "u": [...],
  "z": 1,
  "move_features": [[...], [...], [...]],
  "policy_target": [0.70, 0.20, 0.10]
}
```

---

## 6. Training

Training value-only, come prima:

```bash
python scripts/train_hive_value_gnn.py data/selfplay.jsonl --output checkpoint.pt
```

Training value + policy:

```bash
python scripts/train_hive_value_gnn.py data/selfplay_policy.jsonl \
  --policy-weight 1.0 \
  --output checkpoint_policy.pt
```

Se le feature mossa hanno una dimensione diversa da 32:

```bash
python scripts/train_hive_value_gnn.py data/selfplay_policy.jsonl \
  --move-feature-dim 48 \
  --policy-weight 1.0 \
  --output checkpoint_policy.pt
```

La loss totale e':

```text
loss = value_loss + policy_weight * policy_loss
```

Dove:

```text
value_loss  = MSE tra value predetta e z
policy_loss = cross entropy tra policy predetta e distribuzione pi
```

Con `--policy-weight 0.0`, default, la policy head esiste ma non viene allenata.

---

## 7. Export

Export con dimensione mossa default:

```bash
python scripts/export_hive_value_gnn.py \
  --weights checkpoint_policy.pt \
  --output hive_value_gnn.pt
```

Se in training hai usato `--move-feature-dim 48`, devi usare lo stesso valore
anche in export:

```bash
python scripts/export_hive_value_gnn.py \
  --weights checkpoint_policy.pt \
  --move-feature-dim 48 \
  --output hive_value_gnn.pt
```

Lo stesso vale per `--hidden-dim`, `--heads` e `--dropout`: i valori devono
combaciare con quelli del training.

---

## 8. Uso nel C++ oggi e domani

### Stato attuale

Il C++ continua a usare solo:

```text
forward(x, edge_index, edge_attr, u, batch) -> value
```

Quindi il motore puo' caricare il modello esportato anche se contiene una
policy head. La policy head rimane semplicemente inutilizzata lato C++.

### Integrazione futura

Per usare davvero la policy nell'MCTS, il C++ dovra':

1. generare tutte le mosse legali della posizione;
2. costruire `move_features` per ogni mossa;
3. chiamare il metodo TorchScript `forward_policy`;
4. applicare softmax sugli score delle mosse legali;
5. usare quelle probabilita' come prior nell'MCTS.

In un MCTS stile AlphaZero, la formula di selezione non usa solo UCB1 classico,
ma aggiunge una prior policy:

```text
score = Q + C * P * sqrt(N_parent) / (1 + N_child)
```

Dove:

```text
Q = valore medio stimato
P = prior della policy head per quella mossa
N = numero di visite
```

Questa parte non e' ancora integrata nel C++: per ora e' stata aggiunta la
testa neurale e il supporto al training.

---

## 9. Attenzione ai checkpoint vecchi

I checkpoint value-only vecchi possono essere caricati nella nuova rete. I pesi
della value head vengono caricati; i pesi della policy head, se assenti nel
checkpoint, partono random.

Questo e' utile per continuare da un modello gia' allenato:

```bash
python scripts/train_hive_value_gnn.py data/selfplay_policy.jsonl \
  --init-weights checkpoint_gen0.pt \
  --policy-weight 1.0 \
  --output checkpoint_policy_gen1.pt
```

In questo caso:

```text
value/backbone = inizializzati dal checkpoint vecchio
policy_head    = inizializzata random e poi allenata
```

---

## 10. Generazione self-play con `moves` e `pi`

E' stato aggiunto un comando UHP:

```text
policytargets depth N
```

Il comando esegue una ricerca MCTS alla radice, con lo stesso significato di
`bestmove depth N`:

```text
iterazioni = N * 1000
```

Poi restituisce un JSON con:

```text
x, edge_index, edge_attr, u
moves[].move
moves[].features
moves[].visits
moves[].pi
best_move
```

`pi` viene calcolato dalle visite MCTS:

```text
pi = visite_mossa / somma_visite_figli_radice
```

Il tool `tools/uhp_match.py` ha una modalita' `dataset` che usa quel comando,
gioca la `best_move`, aspetta il risultato finale e aggiunge automaticamente
`z` a ogni posizione.

Esempio:

```bash
python tools/uhp_match.py dataset build/HiveEngine \
  --engine-arg hive_value_gnn.pt \
  --output data/selfplay_policy.jsonl \
  --games 100 \
  --depth 2 \
  --max-plies 200
```

Se il motore prende il modello via variabile ambiente invece che argomento,
puoi omettere `--engine-arg`.

Ogni riga prodotta e' gia' compatibile con:

```bash
python scripts/train_hive_value_gnn.py data/selfplay_policy.jsonl \
  --policy-weight 1.0 \
  --output checkpoint_policy.pt
```

---

## 11. Prossimi passi consigliati

1. Generare un piccolo JSONL policy con `tools/uhp_match.py dataset`.
2. Fare un training breve con `--policy-weight 1.0`.
3. Esportare il modello TorchScript.
4. Solo dopo, modificare il C++ MCTS per usare la policy come prior durante la selezione.

La value head migliora la valutazione delle posizioni. La policy head serve a
rendere la ricerca piu' intelligente, perche' dice al motore dove vale la pena
guardare prima.
