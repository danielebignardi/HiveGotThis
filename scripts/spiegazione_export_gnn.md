# `export_hive_value_gnn.py`

Esporta la value network `HiveValueGNN` in un file TorchScript
(`hive_value_gnn.pt`) che il motore C++ carica con libtorch tramite
`TorchScriptValueEvaluator`. L'architettura del modello è definita in
`hive_value_gnn.py`, modulo condiviso con lo script di training
(`train_hive_value_gnn.py`): il modello è dichiarato in un posto solo. Vedi anche `docs/Hive_Python_Cpp_Interaction.md`
per il quadro completo del ponte Python ↔ C++ e `docs/Hive_GNN_Spec.md` per il
significato delle feature.

## Cosa fa, in breve

1. Costruisce `HiveValueGNN`: 3 layer `GATv2Conv` (message passing sul grafo dei
   pezzi) + una testa di valore (`Linear` → `LayerNorm` → `ReLU` → `Dropout` →
   `Linear` → `Tanh`) che restituisce un valore in `[-1, 1]`.
2. Se passi `--weights`, carica i pesi salvati; altrimenti usa i pesi
   inizializzati a caso da PyTorch (utile per testare la pipeline senza aver
   ancora addestrato nulla).
3. Mette il modello in `eval()` (disattiva il `Dropout`, obbligatorio prima di
   esportare).
4. Compila il modello con `torch.jit.script` (non `trace`: i grafi di Hive
   hanno un numero di nodi/archi variabile a seconda della posizione).
5. Verifica la **fedeltà dell'export**: esegue lo stesso input sia sul modello
   "eager" (normale) sia su quello scriptato, e controlla che i due output
   combacino (`torch.allclose`, tolleranza `--atol`). Se non combaciano,
   lo script si ferma con un errore invece di produrre un file silenziosamente
   sbagliato.
6. Salva il modulo scriptato su disco (`--output`).

Il file prodotto è autocontenuto: contiene sia l'architettura compilata sia i
pesi. Da quel momento il C++ non ha bisogno di Python per usare la rete.

## Requisiti

Versioni pinnate in `scripts/requirements.txt`:

```bash
pip install -r scripts/requirements.txt
```

`torch_geometric` **deve** restare `2.6.1`: dalla 2.7 in poi c'è una
regressione nota che rompe `torch.jit.script` su `GATv2Conv` (fallisce con
`Could not cast value of type Optional[Tensor] to bool`). Non è un bug di
questo progetto — vedi il commento in cima a `requirements.txt`.

## Uso

### Solo per testare la pipeline (nessun training richiesto)

```bash
python3 scripts/export_hive_value_gnn.py --output hive_value_gnn.pt
```

Usa pesi casuali. Se questo comando stampa `Fidelity check passed`, la
conversione Python → TorchScript funziona correttamente end-to-end — utile per
verificare l'integrazione C++ ancora prima di avere un modello addestrato.

### Con pesi addestrati

```bash
python3 scripts/export_hive_value_gnn.py --weights checkpoint.pt --output hive_value_gnn.pt
```

`checkpoint.pt` può essere sia un semplice `state_dict` sia un dizionario con
chiave `"model_state_dict"` (checkpoint di training più completo, es. con
stato dell'optimizer). Gli iperparametri (`--hidden-dim`, `--heads`,
`--dropout`) devono combaciare con quelli usati in training, altrimenti
`load_state_dict` fallisce per shape mismatch.

### Argomenti disponibili

| Argomento      | Default              | Significato                                    |
|----------------|----------------------|-------------------------------------------------|
| `--weights`    | nessuno              | Path a un checkpoint PyTorch (opzionale)         |
| `--output`     | `hive_value_gnn.pt`  | Path del file TorchScript prodotto               |
| `--hidden-dim` | `64`                 | Dimensione nascosta dei layer `GATv2Conv`        |
| `--heads`      | `4`                  | Numero di teste di attenzione                    |
| `--dropout`    | `0.2`                | Probabilità di dropout nella testa di valore     |
| `--device`     | `cpu`                | Device su cui costruire/esportare il modello     |
| `--atol`       | `1e-5`               | Tolleranza del confronto eager vs scriptato      |

## Verificare il risultato senza C++

Il file esportato si può ricaricare e testare direttamente da Python, senza
libtorch/C++, con lo stesso `torch.jit.load` che userebbe il motore:

```python
import torch
from scripts.export_hive_value_gnn import make_dummy_batch

model = torch.jit.load("hive_value_gnn.pt")
model.eval()

x, edge_index, edge_attr, u, batch = make_dummy_batch(torch.device("cpu"))
with torch.no_grad():
    out = model(x, edge_index, edge_attr, u, batch)

print(out)  # atteso: shape [B, 1], valori in [-1, 1]
```

## Note

- `make_dummy_batch` non fa parte del modello: costruisce un batch fittizio di
  2 grafi solo per testare export e fidelity check, rispettando la stessa
  convenzione usata dal C++ per `edge_index` (prima tutte le sorgenti, poi
  tutte le destinazioni).
- Se cambi l'architettura (`NODE_IN_DIM`, `EDGE_IN_DIM`, `GLOBAL_DIM` in
  `hive_value_gnn.py`, o gli iperparametri), ricordati che devono restare
  coerenti con `GNNNodeDim`, `GNNEdgeDim`, `GNNGlobalDim` definiti in
  `include/BoardEncoder.h` lato C++: non c'è un controllo automatico che li
  tenga sincronizzati.
