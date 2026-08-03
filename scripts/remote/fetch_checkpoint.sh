#!/bin/bash
# Recupera checkpoint e log dal pod a noleggio verso la macchina locale.
#
# Da eseguire IN LOCALE. L'host/porta SSH sono quelli mostrati da RunPod
# nella pagina del pod ("SSH over exposed TCP").
#
# Uso:
#   bash scripts/remote/fetch_checkpoint.sh root@<ip> <porta> [file-remoto]
# Esempio:
#   bash scripts/remote/fetch_checkpoint.sh root@194.26.1.99 22150
set -e

HOST="${1:?uso: fetch_checkpoint.sh utente@host porta [file-remoto]}"
PORT="${2:?manca la porta ssh}"
REMOTE_FILE="${3:-/workspace/checkpoint_gen0_policy_aug.pt}"

mkdir -p checkpoints
scp -P "$PORT" "$HOST:$REMOTE_FILE" checkpoints/
scp -P "$PORT" "$HOST:/workspace/training.log" checkpoints/ 2>/dev/null || true

echo ""
echo "Scaricato in checkpoints/. Export per il C++ (ambiente locale pinnato):"
echo "  python3 scripts/export_hive_value_gnn.py --weights checkpoints/$(basename "$REMOTE_FILE") --output hive_value_gnn.pt"
