"""
Export HiveValueGNN to TorchScript.

Uso tipico:

    python scripts/export_hive_value_gnn.py --weights weights.pt --output hive_value_gnn.pt

Per testare subito l'integrazione C++ senza training:

    python scripts/export_hive_value_gnn.py --output hive_value_gnn.pt

Lo script produce un file TorchScript autocontenuto che il C++ carica con
libtorch tramite TorchScriptValueEvaluator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

from hive_value_gnn import EDGE_IN_DIM, GLOBAL_DIM, MOVE_FEATURE_DIM, NODE_IN_DIM, HiveValueGNN, load_weights


def make_dummy_batch(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Crea un piccolo batch valido per testare script/export.

    Il batch contiene due grafi. Gli edge_index rispettano la convenzione del C++:
    prima tutte le sorgenti, poi tutte le destinazioni, forma logica [2, E].
    """

    x = torch.zeros((5, NODE_IN_DIM), dtype=torch.float32, device=device)
    x[:, 0] = 1.0
    x[:, 8] = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0], device=device)

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            [1, 0, 2, 1, 4, 3, 0, 1, 2, 3, 4],
        ],
        dtype=torch.int64,
        device=device,
    )

    edge_attr = torch.zeros((edge_index.size(1), EDGE_IN_DIM), dtype=torch.float32, device=device)
    edge_attr[0, 0] = 1.0
    edge_attr[1, 3] = 1.0
    edge_attr[2, 1] = 1.0
    edge_attr[3, 4] = 1.0
    edge_attr[4, 0] = 1.0
    edge_attr[5, 3] = 1.0
    edge_attr[6:, 8] = 1.0

    u = torch.zeros((2, GLOBAL_DIM), dtype=torch.float32, device=device)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)

    return x, edge_index, edge_attr, u, batch


def export_model(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    model = HiveValueGNN(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout_p=args.dropout,
        move_feature_dim=args.move_feature_dim,
    ).to(device)

    if args.weights is not None:
        load_weights(model, Path(args.weights), device)

    # policy_version dice al C++ quale testa di policy e' stata allenata:
    # si legge dai metadati del checkpoint (il trainer la salva), oppure si
    # forza da riga di comando. I checkpoint senza metadati restano v1.
    policy_version = args.policy_version
    if policy_version is None and args.weights is not None:
        ck = torch.load(Path(args.weights), map_location=device, weights_only=True)
        policy_version = ck.get("policy_version", 1) if isinstance(ck, dict) else 1
    model.policy_version = int(policy_version or 1)

    # eval() e' obbligatorio: Dropout deve essere disattivato sia per export
    # sia per inferenza C++.
    model.eval()

    dummy_inputs = make_dummy_batch(device)

    with torch.no_grad():
        eager_out = model(*dummy_inputs)

    scripted = torch.jit.script(model)
    scripted.eval()

    with torch.no_grad():
        scripted_out = scripted(*dummy_inputs)

    if not torch.allclose(eager_out, scripted_out, atol=args.atol):
        max_diff = (eager_out - scripted_out).abs().max().item()
        raise RuntimeError(f"Export fidelity check failed: max diff = {max_diff:.8f}")

    # Fedelta' anche della testa v2 (2 mosse finte sul primo grafo del dummy:
    # un piazzamento dalla mano e un movimento del nodo 0 con due vicini).
    x, edge_index, edge_attr, u, batch = dummy_inputs
    move_identity = torch.zeros((2, 12), device=device)
    move_identity[0, 1] = 1.0; move_identity[0, 9] = 1.0   # piazzamento di una formica
    move_identity[1, 0] = 1.0; move_identity[1, 10] = 1.0  # movimento della regina (nodo 0)
    move_src = torch.tensor([-1, 0], dtype=torch.int64, device=device)
    move_batch = torch.zeros(2, dtype=torch.int64, device=device)
    dst_node = torch.tensor([1, 1, 2], dtype=torch.int64, device=device)
    dst_slot = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
    dst_move = torch.tensor([0, 1, 1], dtype=torch.int64, device=device)
    v2_args = (x, edge_index, edge_attr, u, batch, move_identity, move_src, move_batch, dst_node, dst_slot, dst_move)
    with torch.no_grad():
        eager_v2 = model.forward_policy_v2(*v2_args)
        scripted_v2 = scripted.forward_policy_v2(*v2_args)
    if not torch.allclose(eager_v2, scripted_v2, atol=args.atol):
        raise RuntimeError("Export fidelity check failed sulla policy v2")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))

    print(f"Saved TorchScript model to: {output_path}")
    print(f"Fidelity check passed with atol={args.atol} (policy_version={model.policy_version})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HiveValueGNN to hive_value_gnn.pt")
    parser.add_argument("--weights", type=str, default=None, help="Optional PyTorch state_dict/checkpoint path.")
    parser.add_argument("--output", type=str, default="hive_value_gnn.pt", help="TorchScript output path.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--move-feature-dim", type=int, default=MOVE_FEATURE_DIM)
    parser.add_argument("--policy-version", type=int, default=None, choices=[1, 2],
                        help="Forza la versione della policy esportata; default: dai metadati del checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    export_model(parse_args())
