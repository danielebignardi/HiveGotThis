"""
Definizione della value/policy network HiveValueGNN e delle costanti di dimensione.

Questo modulo contiene SOLO il modello: viene importato sia dallo script di
training (train_hive_value_gnn.py) sia da quello di export
(export_hive_value_gnn.py), cosi' l'architettura e' definita in un posto solo.

Le dimensioni degli input devono combaciare con quelle prodotte dal C++
(BoardEncoder.h: GNNNodeDim/GNNEdgeDim/GNNGlobalDim) - vedi Hive_GNN_Spec.md.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, ReLU, Sequential, Tanh
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


NODE_IN_DIM = 18
EDGE_IN_DIM = 9
GLOBAL_DIM = 21
MOVE_FEATURE_DIM = 32


class HiveValueGNN(torch.nn.Module):
    """Value network descritta in Hive_GNN_Spec.md, con policy head opzionale.

    Input:
        x:          [N, 18] float32
        edge_index: [2, E]  int64
        edge_attr:  [E, 9]  float32
        u:          [B, 21] float32
        batch:      [N]     int64

    Output forward:
        value:      [B, 1]  float32 in [-1, 1]

    La policy head non cambia la firma di forward(), cosi' il C++ esistente
    continua a ricevere solo la value. Per usarla in training si chiama
    forward_policy(..., move_features, move_batch), dove move_batch dice a
    quale grafo/posizione appartiene ogni mossa candidata.
    """

    def __init__(
        self,
        node_in_dim: int = NODE_IN_DIM,
        edge_in_dim: int = EDGE_IN_DIM,
        u_dim: int = GLOBAL_DIM,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout_p: float = 0.2,
        move_feature_dim: int = MOVE_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.board_dim = 2 * hidden_dim + u_dim
        self.move_feature_dim = move_feature_dim

        self.conv1 = GATv2Conv(
            node_in_dim,
            hidden_dim,
            edge_dim=edge_in_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
        )
        self.norm1 = LayerNorm(hidden_dim)

        self.conv2 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            edge_dim=edge_in_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
        )
        self.norm2 = LayerNorm(hidden_dim)

        self.conv3 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            edge_dim=edge_in_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
        )
        self.norm3 = LayerNorm(hidden_dim)

        self.value_head = Sequential(
            Linear(self.board_dim, 32),
            LayerNorm(32),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(32, 1),
            Tanh(),
        )

        self.policy_head = Sequential(
            Linear(self.board_dim + move_feature_dim, 64),
            LayerNorm(64),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(64, 1),
        )

    def encode_board(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        u: Tensor,
        batch: Tensor,
    ) -> Tensor:
        x = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x = F.elu(self.norm2(self.conv2(x, edge_index, edge_attr)))
        x = F.elu(self.norm3(self.conv3(x, edge_index, edge_attr)))

        return torch.cat(
            [
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                u,
            ],
            dim=1,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        u: Tensor,
        batch: Tensor,
    ) -> Tensor:
        pooled = self.encode_board(x, edge_index, edge_attr, u, batch)
        return self.value_head(pooled)

    @torch.jit.export
    def forward_policy_from_embedding(
        self,
        board_embedding: Tensor,
        move_features: Tensor,
        move_batch: Tensor,
    ) -> Tensor:
        board_per_move = board_embedding.index_select(0, move_batch)
        policy_input = torch.cat([board_per_move, move_features], dim=1)
        return self.policy_head(policy_input).squeeze(-1)

    @torch.jit.export
    def forward_policy(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        u: Tensor,
        batch: Tensor,
        move_features: Tensor,
        move_batch: Tensor,
    ) -> Tensor:
        board_embedding = self.encode_board(x, edge_index, edge_attr, u, batch)
        return self.forward_policy_from_embedding(board_embedding, move_features, move_batch)


def load_weights(model: torch.nn.Module, weights_path: Path, device: torch.device) -> None:
    """Carica i pesi da un checkpoint: accetta sia uno state_dict puro sia il
    formato salvato dal training ({"model_state_dict": ...} + metadati).

    weights_only=True limita il caricamento a tensori e tipi semplici: piu'
    sicuro di pickle completo (che puo' eseguire codice arbitrario) e
    sufficiente per i nostri checkpoint.
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint, strict=False)
