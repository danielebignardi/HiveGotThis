# Training su GPU a noleggio (RunPod)

Questo documento descrive come eseguire il training della rete su una GPU a
noleggio, e le migliorie al trainer fatte per renderlo conveniente. Fino ad
oggi il training girava sulle sessioni gratuite di Kaggle (12h a sessione,
30h/settimana): la GPU a noleggio toglie entrambi i limiti, ma si paga a ore
— quindi prima di pagare conviene che ogni ora renda.

## 1. Il problema misurato: la GPU era quasi ferma

I run Kaggle di fine luglio (T4) viaggiavano a **~1.000-1.400 posizioni/s**.
Per i grafi minuscoli di Hive e' un numero da collo di bottiglia CPU, non
GPU. Due cause nel trainer, entrambe risolte:

1. **`DataLoader` a processo singolo** (`num_workers=0`): collation dei
   batch e cast fp16→fp32 avvenivano nello stesso processo che lancia i
   kernel GPU. Ora `--num-workers N` (con `pin_memory` e
   `persistent_workers`) sposta la preparazione dei batch su N processi.
   Ogni worker della D6 augmentation viene reseedato per non produrre
   simmetrie correlate.
2. **Policy loss con loop Python per grafo**: per OGNI posizione del batch
   un'iterazione Python con mask e `log_softmax` dedicati. Riscritta con
   scatter segmentati (una manciata di kernel per l'intero batch);
   il loop originale e' conservato commentato nel codice e l'equivalenza
   (valore e gradienti) e' verificata da `scripts/tests/test_policy_loss.py`,
   che gira su CPU senza dataset.

## 2. Igiene di training (mutuata da bee-search)

bee-search (il vincitore dell'anno scorso, `../bee-search/trainer.py`) usa
quattro accorgimenti che al nostro trainer mancavano. Aggiunti, tutti
disattivi di default (i vecchi comandi restano riproducibili):

| Flag | Cosa fa | Valore bee-search |
|---|---|---|
| `--weight-decay` | L2 su Adam | `1e-5` |
| `--clip-grad` | clip della norma del gradiente | `1.0` |
| `--patience` | early stopping dopo N epoche senza best | 20 (noi: 5, il tempo e' denaro) |
| `--lr-schedule plateau` | dimezza il lr sui plateau (alternativa al cosine) | patience 10, factor 0.5 |

## 3. Distillazione della ricerca (il target `q`)

Il trucco principale di bee-search: la rete non impara solo dall'esito
finale della partita, ma dalla **valutazione dell'engine stesso** (per loro,
la ricerca alpha-beta a profondita' 5). L'esito `z` e' un'etichetta rumorosa
— una partita vinta contiene anche mosse pessime; la stima della ricerca e'
un segnale piu' denso e pulito.

Adattato al nostro MCTS:

- `MCTS::SearchPolicyTargets` ora espone (parametro opzionale, nessun
  cambiamento per i chiamanti esistenti) il **valore Q della radice**: la
  media pesata sulle visite dei valori dei figli, in `[-1, 1]` side-to-move.
  Se il solver ha provato la radice, vale il risultato esatto.
- Il `SelfPlay` scrive `"q"` accanto a `"z"` in ogni riga JSONL.
- Il trainer ha `--q-blend B`: il target del value diventa `(1-B)*z + B*q`
  **dove q esiste**; le posizioni senza q (tutto il dataset umano
  BoardSpace) restano su z puro, quindi i dataset misti funzionano da soli.

Nota: il dataset BoardSpace attuale non contiene q — il primo run che puo'
usare `--q-blend` e' la prossima generazione di self-play. Punto di partenza
suggerito: `--q-blend 0.5`.

## 4. La macchina: tanti core, GPU modesta

Proprio perche' il collo di bottiglia e' la preparazione dei dati su CPU, la
macchina giusta e' una **GPU di fascia media con molte vCPU** (RTX 4090 / L4
/ A5000 con ≥16 vCPU), non una A100 con 8 vCPU che costerebbe il triplo per
stare a guardare il DataLoader. Su RunPod:

1. template **PyTorch 2.x** (torch + CUDA preinstallati);
2. **volume persistente** montato su `/workspace`: checkpoint, log e dataset
   vivono li' e sopravvivono a stop/riavvio del pod (i pod spot possono
   essere reclamati: il best checkpoint su volume + `--max-hours` +
   `--patience` limitano il danno a pochi minuti di lavoro perso);
3. i prezzi 2026 di un 4090/L4 stanno nell'ordine di grandezza dei
   0,30-0,70 €/h: un run completo da ~8h costa pochi euro dei 500 € di
   budget.

## 5. Procedura

```bash
# 0. IN LOCALE, prima di noleggiare: verifica che il trainer sia sano
.venv/bin/python scripts/tests/test_policy_loss.py

# 1. sul pod: clona il repo e prepara l'ambiente
git clone https://github.com/danielebignardi/HiveGotThis && cd HiveGotThis
bash scripts/remote/setup_env.sh

# 2a. via Kaggle (se si ha accesso al dataset privato): token API
#     (kaggle.com -> Settings -> API -> Create New Token), poi
mkdir -p ~/.config/kaggle && cp kaggle.json ~/.config/kaggle/ && chmod 600 ~/.config/kaggle/kaggle.json
bash scripts/remote/fetch_data.sh

# 2b. SENZA Kaggle (usato la prima volta, 2026-08-02): il dataset si
#     rigenera in locale dagli zip SGF della cartella Drive condivisa
#     (HiveGotThis_colab/boardspace_games/ -> symlink su data/boardspace):
#       bash scripts/boardspace/convert_policy_all.sh <modello.pt> 6
#       cd data && zip boardspace_policy_jsonl.zip boardspace_policy_*.jsonl
#     poi si carica lo zip (~1 GB) sul pod e si scompatta sul volume:
#       scp -P <porta> data/boardspace_policy_jsonl.zip root@<ip>:/workspace/
#       ssh -p <porta> root@<ip> 'mkdir -p /workspace/data && unzip /workspace/boardspace_policy_jsonl.zip -d /workspace/data'

# 3. misura la baseline di throughput su UN anno (facoltativo ma
#    consigliato la prima volta: e' il confronto onesto prima/dopo)
python3 -u scripts/train_hive_value_gnn.py /workspace/data/**/boardspace_policy_2025.jsonl \
    --device cuda --batch-size 256 --epochs 1 --policy-weight 1.0            # workers 0
python3 -u scripts/train_hive_value_gnn.py /workspace/data/**/boardspace_policy_2025.jsonl \
    --device cuda --batch-size 256 --epochs 1 --policy-weight 1.0 --num-workers 14
# (in un'altra shell: nvidia-smi dmon -s u   per l'utilizzo GPU)

# 4. lancia il run vero (tmux: sopravvive alla disconnessione)
bash scripts/remote/run_training.sh 8        # 8 = ore di budget
tmux attach -t training                       # Ctrl-B D per staccarsi

# 5. IN LOCALE, a fine run: recupera il checkpoint e fai l'export
bash scripts/remote/fetch_checkpoint.sh root@<ip> <porta-ssh>
python3 scripts/export_hive_value_gnn.py \
    --weights checkpoints/checkpoint_gen0_policy_aug.pt --output hive_value_gnn.pt
```

L'export TorchScript resta **in locale** (ambiente pinnato
`torch_geometric==2.6.1`, vedi `scripts/requirements.txt`): il pod produce
solo il checkpoint dei pesi.

## 6. Risultati misurati (primo run, 2026-08-03)

Macchina: vast.ai, RTX 5090 + EPYC 48 core, 0,49 $/h. Dataset: 1.583.975
posizioni rigenerate dagli SGF direttamente sul pod (fase 3-4 della pipeline,
~1h20 tutto compreso; conteggio in linea con il riferimento ~1,58M).

| Metrica | Kaggle T4 (riferimento) | RTX 5090 + fix trainer |
|---|---|---|
| throughput | ~1.000-1.400 pos/s | **~25.500 pos/s (~20x)** |
| tempo per 18 epoche | ~10 ore (cap sessione) | **~25 minuti** |
| val MSE (value) | 0.6925 | **0.6862** (0.6808 all'epoca 7) |
| val policy CE | 2.811 | 2.820 |
| segno-ok | 71.7% | 71.7-72.0% |
| costo | gratis (30h/settimana) | ~0,25 € di GPU |

Lettura: qualita' in parita' (il tetto e' l'informazione nei dati, come gia'
concluso dall'esperimento Kaggle sul dataset completo) ma iterazione **24x
piu' veloce**: la strada per il ciclo a generazioni (piano_fase5) e per gli
A/B di strategia che prima non erano abbordabili.

Nota operativa emersa sul campo: con `--num-workers` alti su Linux serve la
sharing strategy `file_system` di torch.multiprocessing (gia' nel trainer) —
senza, i batch PyG esauriscono i file descriptor ("received 0 items of
ancdata").

### A/B della stessa mattina: il multi-task aiuta il value

Stessa macchina, stesso dataset, stesso seed, una sola variabile
(`--policy-weight`):

| | value+policy (peso 1.0) | value-only (peso 0.0) |
|---|---|---|
| best val MSE | **0.6862** (epoca 15, ancora in discesa) | 0.6917 (epoca 4, poi overfitting) |
| andamento | stabile per 18 epoche | val in salita dall'epoca 5, early-stop alla 9 |

La policy head fa da **regolarizzatore** del backbone condiviso: il
multi-task non danneggia il value (dubbio lasciato aperto nei run di
luglio), lo migliora. L'early stopping (`--patience 5`) ha chiuso il run
value-only a meta' corsa, dimezzandone il costo: le due novita' del trainer
si sono ripagate al primo uso.

## 6bis. Forza di gioco misurata contro nokamute (2026-08-03)

La metrica di validation dice quanto la rete predice bene, non quanto gioca
bene: l'unica risposta onesta e' giocare. Due misure, complementari.

**Container referhive.** Il `containers/hivegotthis/Dockerfile` di referhive
era fermo a giugno (`cmake && make` senza libtorch): dall'introduzione della
value network obbligatoria quell'immagine non poteva funzionare per nessuno.
Ricostruito con libtorch dal wheel pip **CPU** (`--index-url
https://download.pytorch.org/whl/cpu`: il wheel di default, anche aarch64,
pretende CUDA al configure) e modello incluso nell'ENTRYPOINT.

**Match da 19 partite** (`tools/uhp_match.py`, entrambi i motori in Docker
per simmetria, 5 s/mossa, max 100 plies, aperture appaiate a colori
invertiti, seed 42):

| Esito | Partite |
|---|---|
| vittorie HiveGotThis | 6 |
| vittorie nokamute | 10 |
| patte | 3 |
| **punteggio** | **7,5/19 = 39,5%** |

Confronto storico, stesse condizioni: motore pre-neurale (giugno) 0/2;
gen-0 (luglio) ~30% su 10 partite; oggi **~40%**. Il miglioramento c'e' ma
e' modesto e il campione resta piccolo: nokamute e' ancora piu' forte.
Coerente con le metriche di validation (il tetto e' l'informazione nei dati
umani) e con la diagnosi di fondo: a 5 s/mossa la nostra ricerca fa molte
meno iterazioni: ~98% del tempo se ne va in inferenza.

**Rischio timeout** (emerso su referhive, dove il timeout e' sconfitta
immediata): in due partite il motore ha sforato i 5 s. Le partite erano
inquinate da altri processi sulla stessa macchina, quindi il dato non e'
pulito, ma il rischio e' reale e va misurato a parte (`TestTournamentBenchmark`
ha gia' la metrica di overshoot).

## 7. Prossimi passi (fuori da questo lavoro)

- **Generazione self-play con `q`** e primo training `--q-blend 0.5`: il
  ciclo a generazioni (docs/piano_fase5.md) con la distillazione attiva.
- **Media degli esiti sulle posizioni duplicate** (bee-search:
  `win_prob[position]`): le aperture si ripetono su migliaia di partite
  umane, mediarne la z pulirebbe il segnale di inizio partita.
- **Layer GNN residuali** come bee-search: cambierebbe l'architettura (e la
  compatibilita' dei checkpoint), da valutare su un run dedicato.
