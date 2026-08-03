#!/bin/bash
# Scarica il dataset di training sulla macchina a noleggio.
#
# Via primaria: il Dataset Kaggle privato del progetto (gli stessi tre file
# usati dai notebook: hive_value_gnn.py, train_hive_value_gnn.py e i
# boardspace_policy_<anno>.jsonl). Serve un token API Kaggle
# (kaggle.com -> Settings -> API -> Create New Token) e l'accesso al dataset.
#
# Fallback (se Kaggle non e' accessibile): scaricare a mano lo zip dalla
# cartella Drive condivisa HiveGotThis_colab/ e scompattarlo in $DATA_DIR
# (con rclone configurato: rclone copy drive:HiveGotThis_colab/boardspace_policy_jsonl.zip .).
#
# Uso (dentro il pod):
#   mkdir -p ~/.config/kaggle && cp kaggle.json ~/.config/kaggle/ && chmod 600 ~/.config/kaggle/kaggle.json
#   bash scripts/remote/fetch_data.sh [dataset-kaggle] [cartella-dati]
set -e

KAGGLE_DATASET="${1:-danielebignardi/data-and-scripts}"
# Il volume persistente di RunPod e' montato su /workspace: i dati (e i
# checkpoint) devono vivere li', cosi' sopravvivono al riavvio del pod.
DATA_DIR="${2:-/workspace/data}"

if [ -e "$DATA_DIR" ] && [ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "Errore: $DATA_DIR esiste gia' e non e' vuota - non riscarico." >&2
    echo "Per forzare: rm -rf $DATA_DIR" >&2
    exit 1
fi
mkdir -p "$DATA_DIR"

echo "== download da Kaggle: $KAGGLE_DATASET =="
kaggle datasets download "$KAGGLE_DATASET" --path "$DATA_DIR" --unzip

echo "== contenuto =="
find "$DATA_DIR" -name '*.jsonl' | sort
N=$(find "$DATA_DIR" -name '*.jsonl' | wc -l)
echo "$N file jsonl in $DATA_DIR"
if [ "$N" -eq 0 ]; then
    echo "Errore: nessun jsonl trovato. Se il dataset contiene ancora lo zip:" >&2
    echo "  unzip $DATA_DIR/boardspace_policy_jsonl.zip -d $DATA_DIR" >&2
    exit 1
fi
