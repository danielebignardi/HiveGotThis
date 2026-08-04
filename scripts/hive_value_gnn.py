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
from torch.nn import Dropout, Embedding, LayerNorm, Linear, ReLU, Sequential, Tanh
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


NODE_IN_DIM = 18
EDGE_IN_DIM = 9
GLOBAL_DIM = 21
MOVE_FEATURE_DIM = 32

# Policy v2: dei 32 move feature usa solo i primi 12, che sono pura IDENTITA'
# della mossa (one-hot insetto 0-7, colore 8, piazzamento 9, movimento 10,
# pass 11) e non valutazioni fatte a mano - quelle le impara la rete dagli
# embedding dei nodi.
POLICY_V2_IDENTITY_DIM = 12
POLICY_V2_SLOTS = 7  # direzioni planari 0-5 + 6 = salita sopra la pila


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

        # ------- Policy v2 (edge prediction sugli embedding dei nodi) -------
        # NB: questi moduli DEVONO restare gli ultimi dell'__init__: cosi' i
        # pesi v1 consumano il generatore casuale nello stesso ordine di prima
        # e i training v1 restano riproducibili bit a bit.
        # policy_version dice al C++ quale testa e' stata allenata davvero:
        # 1 = solo move features, 2 = testa sugli embedding. Un modello v1
        # esporta comunque il metodo forward_policy_v2 (TorchScript esporta
        # tutto), ma il C++ non lo chiamera' mai vedendo version=1.
        self.policy_version: int = 1
        # L'init di questi moduli avviene in uno stato del generatore isolato
        # (salva/ripristina): cosi' il flusso casuale visto da tutto il resto
        # (shuffle del DataLoader, dropout) resta IDENTICO a prima che la v2
        # esistesse, e i run v1 rimangono riproducibili bit a bit.
        _rng_state = torch.get_rng_state()
        self.slot_embedding = Embedding(POLICY_V2_SLOTS, 8)
        self.hand_embedding = Linear(8, hidden_dim)  # one-hot insetto -> pseudo-embedding del pezzo in mano
        self.dst_mlp = Sequential(Linear(hidden_dim + 8, hidden_dim), ReLU())
        self.policy_head_v2 = Sequential(
            Linear(self.board_dim + 2 * hidden_dim + POLICY_V2_IDENTITY_DIM, 64),
            LayerNorm(64),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(64, 1),
        )
        torch.set_rng_state(_rng_state)

    def encode_nodes(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """Il tronco GNN fino agli embedding per-nodo [N, hidden]."""
        x = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x = F.elu(self.norm2(self.conv2(x, edge_index, edge_attr)))
        x = F.elu(self.norm3(self.conv3(x, edge_index, edge_attr)))
        return x

    def pool_board(self, node_embeddings: Tensor, u: Tensor, batch: Tensor) -> Tensor:
        return torch.cat(
            [
                global_mean_pool(node_embeddings, batch),
                global_max_pool(node_embeddings, batch),
                u,
            ],
            dim=1,
        )

    def encode_board(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        u: Tensor,
        batch: Tensor,
    ) -> Tensor:
        return self.pool_board(self.encode_nodes(x, edge_index, edge_attr), u, batch)

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

    @torch.jit.export
    def forward_policy_v2(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        u: Tensor,
        batch: Tensor,
        move_identity: Tensor,  # [M, 12] identita' della mossa (one-hot insetto, colore, piazz., mov., pass)
        move_src: Tensor,       # [M] indice GLOBALE del nodo del pezzo mosso, -1 = dalla mano / pass
        move_batch: Tensor,     # [M] grafo di appartenenza di ogni mossa
        dst_node: Tensor,       # [D] indice GLOBALE del nodo vicino alla destinazione
        dst_slot: Tensor,       # [D] direzione dal vicino verso la destinazione (0-5, 6 = salita)
        dst_move: Tensor,       # [D] indice GLOBALE della mossa a cui la coppia appartiene
    ) -> Tensor:
        """Un logit per mossa dagli embedding dei nodi (policy 'edge prediction').

        La mossa e' descritta in modo strutturale: chi si muove (src) e dove
        arriva (dst = i vicini occupati della casella d'arrivo, ciascuno con
        la direzione dal vicino verso di essa). Niente feature valutative
        fatte a mano: cosa conta di una mossa lo impara la rete.
        """
        h = self.encode_nodes(x, edge_index, edge_attr)
        board = self.pool_board(h, u, batch)

        # Embedding del pezzo mosso: dal grafo se e' gia' in gioco, dal tipo
        # di insetto (one-hot -> lineare) se viene piazzato dalla mano.
        # Per il pass l'one-hot e' tutto zero: resta il bias, che la rete
        # puo' usare come "embedding del pass".
        is_hand = (move_src < 0).unsqueeze(1)
        e_src = h.index_select(0, move_src.clamp(min=0))
        e_hand = self.hand_embedding(move_identity[:, :8])
        e_src = torch.where(is_hand, e_hand, e_src)

        # Aggregazione dei vicini della destinazione: media di
        # phi(embedding_vicino || embedding_direzione) sulle coppie della mossa.
        e_pair = self.dst_mlp(torch.cat([h.index_select(0, dst_node), self.slot_embedding(dst_slot)], dim=1))
        n_moves = move_src.size(0)
        dst_sum = torch.zeros(n_moves, e_pair.size(1), dtype=e_pair.dtype, device=e_pair.device)
        dst_sum.index_add_(0, dst_move, e_pair)
        dst_count = torch.zeros(n_moves, dtype=e_pair.dtype, device=e_pair.device)
        dst_count.index_add_(0, dst_move, torch.ones_like(dst_slot, dtype=e_pair.dtype))
        dst_mean = dst_sum / dst_count.clamp(min=1.0).unsqueeze(1)

        policy_input = torch.cat(
            [board.index_select(0, move_batch), e_src, dst_mean, move_identity], dim=1
        )
        return self.policy_head_v2(policy_input).squeeze(-1)


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
