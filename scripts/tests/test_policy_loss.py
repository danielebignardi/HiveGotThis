"""
Test CPU (nessuna GPU, nessun dataset) di equivalenza per il trainer:

1. la policy loss vettorizzata di train_hive_value_gnn.py produce lo stesso
   valore E gli stessi gradienti del loop Python originale (conservato
   commentato nel trainer), inclusi i casi limite: grafo con target tutti a
   zero (da saltare) e grafo senza mosse;
2. il blend del target value (--q-blend) rispetta la semantica
   (1-B)*z + B*q solo dove q esiste, z puro altrove.

Uso:  .venv/bin/python scripts/tests/test_policy_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hive_value_gnn import EDGE_IN_DIM, GLOBAL_DIM, MOVE_FEATURE_DIM, NODE_IN_DIM, HiveValueGNN
from train_hive_value_gnn import batch_to_float32, policy_loss_for_batch, value_target_for_batch


def reference_policy_loss(model: HiveValueGNN, batch) -> torch.Tensor:
    """Il loop Python originale, copiato tale e quale dal trainer prima della
    vettorizzazione: e' il riferimento di correttezza."""
    if batch.move_features.numel() == 0:
        return batch.y.new_tensor(0.0)

    board_embedding = model.encode_board(batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch)
    logits = model.forward_policy_from_embedding(
        board_embedding,
        batch.move_features,
        batch.move_features_batch,
    )

    total = logits.new_tensor(0.0)
    count = 0
    for graph_id in torch.unique(batch.move_features_batch):
        mask = batch.move_features_batch == graph_id
        target = batch.policy_target[mask]
        target_sum = target.sum()
        if target.numel() == 0 or target_sum <= 0:
            continue
        target = target / target_sum
        total = total - (target * torch.nn.functional.log_softmax(logits[mask], dim=0)).sum()
        count += 1

    if count == 0:
        return logits.new_tensor(0.0)
    return total / count


def make_graph(rng: torch.Generator, n_nodes: int, n_moves: int,
               zero_targets: bool = False, z: float = 1.0,
               q: float | None = None) -> Data:
    """Un grafo sintetico con le stesse forme e gli stessi dtype dei record
    veri (x/edge_attr/move_features in float16, come record_to_data)."""
    # Anello di archi: indici sempre validi, almeno un arco per nodo.
    src = torch.arange(n_nodes, dtype=torch.int64)
    dst = (src + 1) % n_nodes
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    n_edges = edge_index.size(1)

    if n_moves > 0:
        move_features = torch.rand((n_moves, MOVE_FEATURE_DIM), generator=rng).to(torch.float16)
        if zero_targets:
            policy_target = torch.zeros(n_moves, dtype=torch.float32)
        else:
            policy_target = torch.rand(n_moves, generator=rng)
            policy_target = policy_target / policy_target.sum()
    else:
        move_features = torch.empty((0, MOVE_FEATURE_DIM), dtype=torch.float16)
        policy_target = torch.empty((0,), dtype=torch.float32)

    return Data(
        x=torch.rand((n_nodes, NODE_IN_DIM), generator=rng).to(torch.float16),
        edge_index=edge_index,
        edge_attr=torch.rand((n_edges, EDGE_IN_DIM), generator=rng).to(torch.float16),
        u=torch.rand((1, GLOBAL_DIM), generator=rng),
        y=torch.tensor([[z]], dtype=torch.float32),
        q=torch.tensor([[q if q is not None else 0.0]], dtype=torch.float32),
        has_q=torch.tensor([q is not None], dtype=torch.bool),
        move_features=move_features,
        policy_target=policy_target,
        has_policy=torch.tensor([n_moves > 0], dtype=torch.bool),
    )


def build_batch() -> Data:
    rng = torch.Generator().manual_seed(42)
    graphs = [
        make_graph(rng, n_nodes=4, n_moves=5, z=1.0, q=0.4),            # normale, con q
        make_graph(rng, n_nodes=3, n_moves=3, zero_targets=True, z=-1.0),  # target a zero: da saltare
        make_graph(rng, n_nodes=2, n_moves=0, z=0.0),                   # nessuna mossa
        make_graph(rng, n_nodes=5, n_moves=7, z=-1.0, q=-0.9),          # normale, con q
    ]
    loader = DataLoader(graphs, batch_size=len(graphs), follow_batch=["move_features"])
    return batch_to_float32(next(iter(loader)))


def grads_of(model: HiveValueGNN, loss: torch.Tensor) -> list[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    return [p.grad.clone() for p in model.parameters() if p.grad is not None]


def test_policy_loss_equivalence() -> None:
    torch.manual_seed(0)
    model = HiveValueGNN()
    model.eval()  # dropout off: il confronto deve essere deterministico
    batch = build_batch()

    loss_ref = reference_policy_loss(model, batch)
    grads_ref = grads_of(model, loss_ref)

    loss_new = policy_loss_for_batch(model, batch)
    grads_new = grads_of(model, loss_new)

    assert torch.allclose(loss_ref, loss_new, atol=1e-6), \
        f"loss diversa: riferimento {loss_ref.item():.8f}, vettorizzata {loss_new.item():.8f}"
    assert len(grads_ref) == len(grads_new)
    for g_ref, g_new in zip(grads_ref, grads_new):
        assert torch.allclose(g_ref, g_new, atol=1e-6), "gradiente diverso tra le due versioni"
    print(f"OK equivalenza policy loss: {loss_new.item():.6f} (valore e gradienti identici)")


def test_policy_loss_all_skipped() -> None:
    """Batch dove nessun grafo ha target validi: entrambe le versioni 0."""
    torch.manual_seed(0)
    model = HiveValueGNN()
    model.eval()
    rng = torch.Generator().manual_seed(7)
    graphs = [make_graph(rng, n_nodes=3, n_moves=4, zero_targets=True),
              make_graph(rng, n_nodes=2, n_moves=0)]
    loader = DataLoader(graphs, batch_size=2, follow_batch=["move_features"])
    batch = batch_to_float32(next(iter(loader)))

    loss_ref = reference_policy_loss(model, batch)
    loss_new = policy_loss_for_batch(model, batch)
    assert loss_ref.item() == 0.0 and loss_new.item() == 0.0
    print("OK caso degenere (nessun target valido): entrambe 0.0")


def test_q_blend() -> None:
    batch = build_batch()
    z = batch.y

    # B=0: target identico a z.
    assert torch.equal(value_target_for_batch(batch, 0.0), z)

    # B=0.5: blend solo dove has_q. Grafi: [z=1,q=0.4], [z=-1 no q],
    # [z=0 no q], [z=-1,q=-0.9].
    target = value_target_for_batch(batch, 0.5)
    expected = torch.tensor([[0.5 * 1.0 + 0.5 * 0.4],
                             [-1.0],
                             [0.0],
                             [0.5 * -1.0 + 0.5 * -0.9]])
    assert torch.allclose(target, expected, atol=1e-6), f"blend errato: {target.view(-1).tolist()}"

    # B=1: dove c'e' q il target e' q puro.
    target = value_target_for_batch(batch, 1.0)
    expected = torch.tensor([[0.4], [-1.0], [0.0], [-0.9]])
    assert torch.allclose(target, expected, atol=1e-6)
    print("OK q-blend: (1-B)*z + B*q dove q esiste, z puro altrove")


if __name__ == "__main__":
    test_policy_loss_equivalence()
    test_policy_loss_all_skipped()
    test_q_blend()
    print("\nTutti i test passati.")
