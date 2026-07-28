#!/bin/bash
# Conversione dell'intero archivio BoardSpace nel formato policy (--policy):
# un processo per anno, al massimo JOBS anni in parallelo, un file di output
# per anno (data/boardspace_policy_<anno>.jsonl, stessa convenzione dei file
# value). Gli anni gia' convertiti vengono saltati: rilanciare lo script
# riprende da dove si era interrotto senza duplicare posizioni.
#
# Gli anni prima del 2013 non contengono partite hive-plm (il Pillbug esce
# nel 2013): producono file vuoti, rimossi in coda.
#
# Uso:
#   scripts/boardspace/convert_policy_all.sh [modello.pt] [processi]
#   default: weights/hive_value_gnn_gen0_colab.pt, 3 processi
#
# Il modello serve solo ad avviare HiveEngine (obbligatorio all'avvio): la
# conversione non fa ricerche, quindi i pesi non influenzano l'output.

set -u
cd "$(dirname "$0")/../.."

MODEL="${1:-weights/hive_value_gnn_gen0_colab.pt}"
JOBS="${2:-3}"
LOGDIR="data/conversione_policy_logs"
mkdir -p "$LOGDIR"

if [ ! -e "$MODEL" ]; then
    echo "Modello non trovato: $MODEL" >&2
    exit 1
fi

for year_dir in data/boardspace/*/; do
    year="$(basename "$year_dir")"
    out="data/boardspace_policy_${year}.jsonl"

    if [ -e "$out" ]; then
        echo "salto $year: $out esiste gia'"
        continue
    fi

    # Aspetta uno slot libero prima di lanciare il prossimo anno.
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n
    done

    echo "avvio $year ($(date +%H:%M:%S))"
    python3 scripts/boardspace/boardspace_to_jsonl.py "$year_dir" \
        --output "$out" --model "$MODEL" --policy \
        > "$LOGDIR/${year}.log" 2>&1 &
done
wait

# Anni senza partite hive-plm -> file vuoti: via.
find data -maxdepth 1 -name 'boardspace_policy_*.jsonl' -size 0 -delete

echo
echo "Conversione completata ($(date +%H:%M:%S)):"
total=0
for f in data/boardspace_policy_*.jsonl; do
    [ -e "$f" ] || continue
    n=$(wc -l < "$f")
    total=$((total + n))
    echo "  $(basename "$f"): $n posizioni, $(du -h "$f" | cut -f1)"
done
echo "  totale: $total posizioni"
