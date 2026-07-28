# Come compilare ed eseguire HiveEngine

Guida pratica, passo per passo. Dal branch `cpp-python` in poi, il motore
**richiede sempre** la value network (rete neurale) — non esiste più una
modalità "solo euristica".

---

## Cosa serve prima di iniziare

1. Un compilatore C++ e `cmake` (su WSL/Linux di solito bastano `sudo apt
   install build-essential cmake`, oppure `pip install cmake` se non hai
   `sudo`).
2. **libtorch**: la libreria C++ di PyTorch. Va scaricata una volta sola a
   mano — è troppo grande (~700 MB) per essere nel repository, quindi ognuno
   se la scarica sulla propria macchina.
3. Un file `.pt`: la rete neurale già esportata da Python (vedi
   `docs/spiegazione_value_network.md` per come si genera).

---

## Passo 1 — Scaricare libtorch (una tantum)

Vai su <https://pytorch.org/get-started/locally/>, sezione **"LibTorch"**, e
scegli le opzioni corrispondenti al tuo sistema (CPU va bene, non serve la
GPU, almeno per ora). Il sito ti dà un link `.zip` da scaricare.


Alla fine deve esistere la cartella `third_party/libtorch/`.

Questo passaggio va fatto **una sola volta per macchina** (non ad ogni
compilazione).

---

## Passo 2 — Compilare

Da dentro la cartella del progetto:

```bash
./build.sh
```

Lo script si occupa di configurare e compilare. Se libtorch non è al suo
posto, ti dice esattamente cosa manca invece di dare un errore incomprensibile.
Lo script dovrebbe funzionare sia su linux che su mac.

Alla fine, l'eseguibile si trova in `build/HiveEngine`.

Ogni volta che modifichi un file `.cpp`/`.h`, basta rilanciare `./build.sh`
per ricompilare solo ciò che è cambiato (non riparte da zero).

---

## Passo 3 — Procurarsi un modello `.pt`

Se non hai ancora un modello addestrato, puoi generarne uno "finto" (pesi
casuali) solo per verificare che tutto funzioni:

```bash
pip install -r scripts/requirements.txt   # solo la prima volta
python3 scripts/export_hive_value_gnn.py --output hive_value_gnn.pt
```

---

## Passo 4 — Eseguire

```bash
./build/HiveEngine hive_value_gnn.pt
```

Se vedi scritto `Value network caricata da: ...`, ha funzionato — l'engine
ora resta in ascolto sullo stdin per i comandi UHP (`info`, `newgame`,
`bestmove`, `exit`, ...).

In alternativa, puoi impostare il percorso una volta sola con una variabile
d'ambiente invece di scriverlo ogni volta:

```bash
export HIVE_VALUE_NETWORK="$(pwd)/hive_value_gnn.pt"
./build/HiveEngine
```

---

## Problemi comuni

- **`Could not find a package configuration file provided by "Torch"`**:
  `third_party/libtorch` non esiste o non è nel posto giusto. Rifai il
  Passo 1.
- **`Errore: serve il path della value network`**: non hai passato né un
  argomento né impostato `HIVE_VALUE_NETWORK`. Vedi Passo 4.
- **`Errore caricando la value network`**: il file `.pt` esiste ma non è
  valido/leggibile (rigenera con il Passo 3, o verifica il percorso).
