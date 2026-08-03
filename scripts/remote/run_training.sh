#!/bin/bash
# Lancia il training sulla GPU a noleggio, in una sessione tmux che
# sopravvive alla disconnessione SSH.
#
# La ricetta e' quella collaudata su Kaggle (train_kaggle.ipynb, cella 4:
# value + policy DA ZERO, dataset intero, augmentation D6, cosine) piu' le
# migliorie del trainer: weight decay e gradient clip (igiene alla
# bee-search), early stopping (--patience: le epoche di plateau su una GPU a
# pagamento sono denaro), --num-workers (il collo di bottiglia misurato sui
# run Kaggle era il DataLoader a processo singolo: ~1.000-1.400 posizioni/s
# con la GPU quasi ferma).
#
# Riferimenti da battere (run Kaggle del 29-31 luglio, T4):
#   val MSE 0.685, val policy CE 2.81, segno-ok 71.7%
#
# Uso (dentro il pod, dalla radice del repository):
#   bash scripts/remote/run_training.sh [ore-budget] [cartella-dati] [output.pt]
# Poi:  tmux attach -t training     (Ctrl-B D per staccarsi)
#       tail -f /workspace/training.log
set -e

MAX_HOURS="${1:-8}"
DATA_DIR="${2:-/workspace/data}"
# Checkpoint e log sul volume persistente: sopravvivono al pod.
CHECKPOINT="${3:-/workspace/checkpoint_gen0_policy_aug.pt}"
LOG="/workspace/training.log"

FILES=$(find "$DATA_DIR" -name '*.jsonl' | sort | tr '\n' ' ')
[ -n "$FILES" ] || { echo "Errore: nessun jsonl in $DATA_DIR (prima: fetch_data.sh)" >&2; exit 1; }

NPROC=$(nproc)
WORKERS=$(( NPROC > 4 ? NPROC - 2 : 2 ))

echo "Dataset:    $(echo "$FILES" | wc -w) file da $DATA_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Budget:     ${MAX_HOURS}h, ${WORKERS} worker DataLoader"
echo "Log:        $LOG"

# Limite file descriptor alto: i worker del DataLoader ne consumano molti
# (vedi il commento su set_sharing_strategy nel trainer).
ulimit -n 65536 2>/dev/null || true

# python -u + tee: avanzamento in tempo reale sia in tmux sia nel log.
# La sessione tmux resta viva dopo la fine (bash finale) per leggere l'esito.
tmux new-session -d -s training \
    "ulimit -n 65536 2>/dev/null; python3 -u scripts/train_hive_value_gnn.py $FILES \
        --output '$CHECKPOINT' \
        --policy-weight 1.0 --augment --lr-schedule cosine \
        --batch-size 256 --epochs 18 \
        --weight-decay 1e-5 --clip-grad 1.0 --patience 5 \
        --num-workers $WORKERS \
        --max-hours $MAX_HOURS \
        --device cuda 2>&1 | tee '$LOG'; \
     echo; echo '== training terminato (sessione tmux ancora viva) =='; exec bash"

echo ""
echo "Training avviato nella sessione tmux 'training'."
echo "  tmux attach -t training   # guardarlo (Ctrl-B D per staccarsi)"
echo "  tail -f $LOG              # solo il log"
