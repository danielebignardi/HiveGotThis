"""
Training di HiveValueGNN sui dataset JSONL prodotti dal C++ (SelfPlay o,
in futuro, il convertitore di partite umane).

Uso tipico:

    python3 scripts/train_hive_value_gnn.py data/partite.jsonl --output checkpoint.pt

poi si esporta il modello per il C++ con lo script gia' esistente:

    python3 scripts/export_hive_value_gnn.py --weights checkpoint.pt --output hive_value_gnn.pt

Per il ciclo a generazioni, --init-weights fa partire il training dai pesi
della rete precedente invece che da pesi casuali (vengono caricati solo i
pesi del modello, l'optimizer riparte fresco):

    python3 scripts/train_hive_value_gnn.py data/nuove.jsonl \
        --init-weights checkpoint_gen0.pt --output checkpoint_gen1.pt

NB: se si cambiano --hidden-dim/--heads rispetto ai default, vanno passati
uguali anche all'export e alle generazioni successive, altrimenti
load_state_dict fallisce.

Ogni riga del JSONL e' una posizione: feature del grafo (x, edge_index,
edge_attr, u) gia' encodate dal C++ e label z in {-1, 0, +1} (esito finale
dal punto di vista di chi muoveva). La rete viene allenata a predire z con
loss MSE. Lo split train/validation e' fatto PER PARTITA, non per posizione:
posizioni della stessa partita sono quasi duplicate, e mescolarle tra train
e validation gonfierebbe artificialmente le metriche.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from hive_value_gnn import EDGE_IN_DIM, GLOBAL_DIM, NODE_IN_DIM, HiveValueGNN, load_weights


def record_to_data(rec: dict) -> Data:
    """Converte una riga del JSONL in un oggetto Data di PyTorch Geometric."""
    n_nodes = len(rec["x"]) // NODE_IN_DIM
    n_edges = len(rec["edge_attr"]) // EDGE_IN_DIM

    x = torch.tensor(rec["x"], dtype=torch.float32).view(n_nodes, NODE_IN_DIM)
    # Il C++ scrive edge_index appiattito [src_0..src_{E-1}, dst_0..dst_{E-1}]:
    # view(2, E) ricostruisce esattamente la forma logica COO.
    edge_index = torch.tensor(rec["edge_index"], dtype=torch.int64).view(2, n_edges)
    edge_attr = torch.tensor(rec["edge_attr"], dtype=torch.float32).view(n_edges, EDGE_IN_DIM)
    u = torch.tensor(rec["u"], dtype=torch.float32).view(1, GLOBAL_DIM)
    y = torch.tensor([[float(rec["z"])]], dtype=torch.float32)  # [1,1] -> batcha a [B,1]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, y=y)


def load_games(paths: list[str]) -> dict:
    """Legge i JSONL e raggruppa le posizioni per partita.

    La chiave e' (file, game_id): il game_id da solo non basta, perche' file
    diversi possono riusare gli stessi id.
    """
    games: dict = defaultdict(list)
    for path in paths:
        n_rows = 0
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no}: JSON malformato: {e}") from e
                games[(path, rec["game_id"])].append(record_to_data(rec))
                n_rows += 1
        # flush=True: senza, in una pipe (Colab, log) l'output arriva a blocchi
        # in ritardo e il caricamento sembra bloccato.
        print(f"  caricato {path}: {n_rows} posizioni", flush=True)
    return games


def split_by_game(games: dict, val_fraction: float, seed: int) -> tuple[list, list]:
    keys = sorted(games.keys())
    random.Random(seed).shuffle(keys)

    n_val = int(round(len(keys) * val_fraction))
    if len(keys) > 1:
        n_val = max(1, n_val)
    val_keys = set(keys[:n_val])

    train_data = [d for k in keys if k not in val_keys for d in games[k]]
    val_data = [d for k in val_keys for d in games[k]]
    return train_data, val_data


@torch.no_grad()
def evaluate(model: HiveValueGNN, loader: DataLoader, device: torch.device) -> tuple[float, float, int]:
    """Restituisce (MSE medio, accuratezza del segno sulle posizioni decisive,
    numero di posizioni decisive). "Decisive" = z != 0: li' il segno della
    predizione dice se la rete indovina il vincitore."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    sign_correct = 0
    decisive = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch)
        total_loss += torch.nn.functional.mse_loss(pred, batch.y, reduction="sum").item()
        total_count += batch.y.size(0)

        mask = batch.y.view(-1) != 0
        decisive += int(mask.sum())
        if mask.any():
            same = torch.sign(pred.view(-1)[mask]) == torch.sign(batch.y.view(-1)[mask])
            sign_correct += int(same.sum())

    mse = total_loss / max(total_count, 1)
    sign_acc = sign_correct / decisive if decisive > 0 else float("nan")
    return mse, sign_acc, decisive


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HiveValueGNN su dataset JSONL")
    parser.add_argument("data", nargs="+", help="Uno o piu' file JSONL prodotti da SelfPlay")
    parser.add_argument("--output", type=str, default="checkpoint.pt", help="Checkpoint dei pesi (il migliore su validation)")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Checkpoint da cui partire (pesi della generazione precedente) invece dei pesi casuali")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Frazione di PARTITE (non posizioni) per la validation")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Deve combaciare con l'export")
    parser.add_argument("--heads", type=int, default=4, help="Deve combaciare con l'export")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu", help="cpu oppure cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    games = load_games(args.data)
    if not games:
        print("Nessuna posizione trovata nei file dati.", file=sys.stderr)
        sys.exit(1)

    train_data, val_data = split_by_game(games, args.val_fraction, args.seed)
    n_decisive = sum(1 for g in games.values() for d in g if float(d.y) != 0.0)
    print(f"Dataset: {len(games)} partite, {len(train_data) + len(val_data)} posizioni "
          f"({n_decisive} decisive) -> train {len(train_data)}, validation {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size) if val_data else None

    model = HiveValueGNN(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout_p=args.dropout,
    ).to(device)

    # Ciclo a generazioni: si parte dai pesi della rete precedente invece che
    # da pesi casuali. Solo i pesi del modello: l'optimizer riparte fresco,
    # perche' i dati nuovi hanno una distribuzione diversa da quelli su cui i
    # suoi momenti interni erano stati accumulati.
    if args.init_weights is not None:
        load_weights(model, Path(args.init_weights), device)
        print(f"Pesi iniziali caricati da: {args.init_weights}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_batches = len(train_loader)
    progress_every = max(1, n_batches // 10)  # ~10 righe di avanzamento per epoca

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        epoch_start = time.monotonic()
        for batch_idx, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch)
            loss = torch.nn.functional.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.y.size(0)
            running_count += batch.y.size(0)

            if batch_idx % progress_every == 0 and batch_idx < n_batches:
                speed = running_count / (time.monotonic() - epoch_start)
                print(f"  epoch {epoch}: batch {batch_idx}/{n_batches}  "
                      f"MSE parziale {running_loss / running_count:.4f}  "
                      f"({speed:.0f} posizioni/s)", flush=True)
        train_mse = running_loss / max(running_count, 1)

        line = f"epoch {epoch:>3}  train MSE {train_mse:.4f}"
        current = train_mse
        if val_loader is not None:
            val_mse, sign_acc, decisive = evaluate(model, val_loader, device)
            line += f"  val MSE {val_mse:.4f}"
            if decisive > 0:
                line += f"  val segno-ok {sign_acc:.1%} (su {decisive} decisive)"
            current = val_mse

        # Salva il checkpoint migliore (su validation se c'e', altrimenti su train).
        if current < best_val:
            best_val = current
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": current,
                    "hidden_dim": args.hidden_dim,
                    "heads": args.heads,
                    "dropout": args.dropout,
                },
                output_path,
            )
            line += "  [salvato]"

        print(line, flush=True)

    print(f"\nCheckpoint migliore salvato in: {output_path} (MSE {best_val:.4f})")
    print("Per esportarlo per il C++:")
    print(f"  python3 scripts/export_hive_value_gnn.py --weights {output_path} --output hive_value_gnn.pt")


if __name__ == "__main__":
    main()
