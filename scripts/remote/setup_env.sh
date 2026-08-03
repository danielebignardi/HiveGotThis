#!/bin/bash
# Prepara l'ambiente di training su una GPU a noleggio (RunPod / vast.ai).
#
# Presuppone un template PyTorch (torch + CUDA gia' installati, come sulle
# immagini "PyTorch 2.x" di RunPod): qui si aggiunge solo cio' che manca,
# esattamente come fa la cella 2 dei notebook Colab/Kaggle. La versione di
# torch_geometric e' libera perche' sulla macchina si fa SOLO training: il
# pin ==2.6.1 riguarda l'export TorchScript, che resta in locale
# (docs/spiegazione_value_network.md par. 6).
#
# Uso (dentro il pod, dalla radice del repository):
#   bash scripts/remote/setup_env.sh
set -e

echo "== GPU =="
python3 - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("GPU disponibile:", ok, "-", torch.cuda.get_device_name(0) if ok else "NESSUNA GPU: controlla il template del pod")
assert ok, "torch.cuda.is_available() e' False"
EOF

echo "== vCPU =="
NPROC=$(nproc)
echo "core disponibili: $NPROC (per --num-workers conviene ~$((NPROC - 2)))"

echo "== dipendenze =="
pip install --quiet torch_geometric kaggle
python3 -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)"

echo ""
echo "Ambiente pronto. Prossimi passi:"
echo "  1. bash scripts/remote/fetch_data.sh      # scarica il dataset"
echo "  2. bash scripts/remote/run_training.sh    # lancia il training in tmux"
