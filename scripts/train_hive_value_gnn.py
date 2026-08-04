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

import bisect

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from hive_value_gnn import EDGE_IN_DIM, GLOBAL_DIM, MOVE_FEATURE_DIM, NODE_IN_DIM, HiveValueGNN, load_weights


def read_policy_targets(rec: dict, move_feature_dim: int):
    """Legge target policy opzionali dal record JSONL.

    Formati accettati:
      {"moves": [{"features": [...], "pi": 0.7, "src": 3, "dst": [[0,2],...]}, ...]}
      {"move_features": [[...], ...], "policy_target": [...]}

    Oltre a (move_features, policy_target, has_policy) restituisce la
    descrizione strutturale per la policy v2: src (nodo del pezzo mosso,
    -1 = dalla mano/pass) e le coppie dst appiattite (nodo vicino, slot
    direzione, indice LOCALE della mossa). Indici in int16: sono piccoli
    (< 28 nodi, < 256 mosse in pratica) e il dataset intero sta in RAM.
    """
    features = []
    targets = []
    srcs = []
    dst_node = []
    dst_slot = []
    dst_move = []

    if "moves" in rec:
        for i, move in enumerate(rec["moves"]):
            features.append(move["features"])
            targets.append(move.get("pi", move.get("policy", move.get("target", 0.0))))
            srcs.append(move.get("src", -1))
            for node, slot in move.get("dst", []):
                dst_node.append(node)
                dst_slot.append(slot)
                dst_move.append(i)
    elif "move_features" in rec:
        features = rec["move_features"]
        targets = rec.get("policy_target", rec.get("policy", rec.get("pi", [])))
        srcs = [-1] * len(features)

    if not features:
        return (
            torch.empty((0, move_feature_dim), dtype=torch.float16),
            torch.empty((0,), dtype=torch.float32),
            False,
            torch.empty((0,), dtype=torch.int16),
            torch.empty((0,), dtype=torch.int16),
            torch.empty((0,), dtype=torch.int16),
            torch.empty((0,), dtype=torch.int16),
        )

    # float16: le move features sono il grosso del dataset in RAM (~60 mosse
    # x 32 float a posizione) e sono valori normalizzati in [-1, 1], dove la
    # mezza precisione basta. Il cast a float32 avviene per batch al momento
    # dell'uso (batch_to_float32).
    move_features = torch.tensor(features, dtype=torch.float16).view(-1, move_feature_dim)
    policy_target = torch.tensor(targets, dtype=torch.float32).view(-1)
    if move_features.size(0) != policy_target.size(0):
        raise ValueError("Numero di move_features diverso dal numero di target policy")

    target_sum = float(policy_target.sum())
    if target_sum > 0.0:
        policy_target = policy_target / target_sum

    return (
        move_features,
        policy_target,
        True,
        torch.tensor(srcs, dtype=torch.int16),
        torch.tensor(dst_node, dtype=torch.int16),
        torch.tensor(dst_slot, dtype=torch.int16),
        torch.tensor(dst_move, dtype=torch.int16),
    )


def record_to_data(rec: dict, move_feature_dim: int) -> Data:
    """Converte una riga del JSONL in un oggetto Data di PyTorch Geometric."""
    n_nodes = len(rec["x"]) // NODE_IN_DIM
    n_edges = len(rec["edge_attr"]) // EDGE_IN_DIM

    # x ed edge_attr in float16 come le move features: sono one-hot e valori
    # normalizzati, la mezza precisione non perde nulla di utile e dimezza
    # la RAM del dataset caricato.
    x = torch.tensor(rec["x"], dtype=torch.float16).view(n_nodes, NODE_IN_DIM)
    # Il C++ scrive edge_index appiattito [src_0..src_{E-1}, dst_0..dst_{E-1}]:
    # view(2, E) ricostruisce esattamente la forma logica COO.
    edge_index = torch.tensor(rec["edge_index"], dtype=torch.int64).view(2, n_edges)
    edge_attr = torch.tensor(rec["edge_attr"], dtype=torch.float16).view(n_edges, EDGE_IN_DIM)
    u = torch.tensor(rec["u"], dtype=torch.float32).view(1, GLOBAL_DIM)
    y = torch.tensor([[float(rec["z"])]], dtype=torch.float32)  # [1,1] -> batcha a [B,1]
    move_features, policy_target, has_policy, move_src, dst_node, dst_slot, dst_move = \
        read_policy_targets(rec, move_feature_dim)

    # NB: move_src/dst_* NON contengono 'index' nel nome apposta: il batching
    # PyG non deve auto-incrementarli (move_src usa -1 come sentinella "dalla
    # mano"); gli offset di batch si applicano a mano nella loss v2.
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,
        y=y,
        move_features=move_features,
        policy_target=policy_target,
        has_policy=torch.tensor([has_policy], dtype=torch.bool),
        move_src=move_src,
        dst_node=dst_node,
        dst_slot=dst_slot,
        dst_move=dst_move,
    )


def load_games(paths: list[str], sample: float, seed: int, move_feature_dim: int) -> dict:
    """Legge i JSONL e raggruppa le posizioni per partita.

    La chiave e' (file, game_id): il game_id da solo non basta, perche' file
    diversi possono riusare gli stessi id.

    Ogni partita viene impacchettata in un Batch PyG appena e' completa:
    un oggetto con pochi tensori grandi al posto di ~50 Data con ~8 tensori
    ciascuno. Serve per la RAM: l'overhead fisso per-tensore di PyTorch
    (circa 1 KB tra oggetto Python e storage) supererebbe i dati veri sul
    dataset completo. Le singole posizioni si riestraggono al volo con
    Batch.get_example (vedi PackedDataset).

    Con sample < 1 tiene solo quella frazione di posizioni, scelte a caso
    (deterministicamente, dato il seed). Le posizioni della stessa partita
    sono quasi-duplicate, quindi sottocampionare DENTRO le partite perde
    poca informazione e preserva la diversita' (tutte le partite restano
    rappresentate) — meglio che scartare interi anni.
    """
    rng = random.Random(seed)
    games: dict = {}
    pending_key = None
    pending: list = []

    def flush() -> None:
        nonlocal pending_key, pending
        if pending:
            if pending_key in games:
                # Righe della stessa partita non contigue nel file: si riapre
                # il pacchetto e si accoda (non succede coi convertitori
                # attuali, ma il formato non lo vieta).
                pending = games[pending_key].to_data_list() + pending
            games[pending_key] = Batch.from_data_list(pending)
        pending_key, pending = None, []

    for path in paths:
        n_rows = 0
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                if sample < 1.0 and rng.random() >= sample:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no}: JSON malformato: {e}") from e
                key = (path, rec["game_id"])
                if key != pending_key:
                    flush()
                    pending_key = key
                pending.append(record_to_data(rec, move_feature_dim))
                n_rows += 1
        flush()
        # flush=True: senza, in una pipe (Colab, log) l'output arriva a blocchi
        # in ritardo e il caricamento sembra bloccato.
        print(f"  caricato {path}: {n_rows} posizioni", flush=True)
    return games


class PackedDataset(torch.utils.data.Dataset):
    """Vista piatta per-posizione su una lista di partite impacchettate
    (Batch). __getitem__ trova la partita con una ricerca binaria sugli
    offset cumulativi e riestrae la singola posizione con get_example."""

    def __init__(self, blocks: list):
        self.blocks = blocks
        self.offsets = []
        total = 0
        for b in blocks:
            total += b.num_graphs
            self.offsets.append(total)

    def __len__(self) -> int:
        return self.offsets[-1] if self.offsets else 0

    def __getitem__(self, idx: int):
        block_i = bisect.bisect_right(self.offsets, idx)
        local_i = idx - (self.offsets[block_i - 1] if block_i > 0 else 0)
        return self.blocks[block_i].get_example(local_i)


def split_by_game(games: dict, val_fraction: float, seed: int) -> tuple[PackedDataset, PackedDataset]:
    keys = sorted(games.keys())
    random.Random(seed).shuffle(keys)

    n_val = int(round(len(keys) * val_fraction))
    if len(keys) > 1:
        n_val = max(1, n_val)
    val_keys = set(keys[:n_val])

    train_data = PackedDataset([games[k] for k in keys if k not in val_keys])
    val_data = PackedDataset([games[k] for k in keys if k in val_keys])
    return train_data, val_data


def batch_to_float32(batch):
    """I tensori grandi del dataset vivono in float16 in RAM; il modello
    lavora in float32, quindi il cast si fa una volta per batch (sul device,
    dove costa nulla) invece di tenere tutto il dataset a precisione piena."""
    batch.x = batch.x.float()
    batch.edge_attr = batch.edge_attr.float()
    batch.move_features = batch.move_features.float()
    return batch


# Le 12 simmetrie del gruppo diedrale D6 della board esagonale, come
# permutazioni delle 6 colonne "direzione planare" di edge_attr (le colonne
# 6/7/8 - sopra/sotto/self - non cambiano). Gli slot direzione del C++ sono
# in ordine ciclico (RightOf = +1, Opposite = +3), quindi:
#   rotazione di r*60 gradi:   slot d -> (d+r) % 6
#   riflessione (+rotazione):  slot d -> (r-d) % 6
# Ogni lista e' l'indice della colonna VECCHIA che finisce nella posizione
# nuova j: nuova[:, j] = vecchia[:, perm[j]]. La prima e' l'identita'.
D6_COLUMN_PERMS = [
    [(j - r) % 6 for j in range(6)] + [6, 7, 8] for r in range(6)
] + [
    [(r - j) % 6 for j in range(6)] + [6, 7, 8] for r in range(6)
]

# Mappa VALORE->VALORE degli slot per ogni simmetria (sigma tale che lo slot
# d finisce in sigma(d)): serve per ruotare i dst_slot della policy v2 in modo
# coerente con la permutazione delle colonne di edge_attr. Lo slot 6 (salita
# sopra la pila) e' invariante per qualunque simmetria planare.
D6_SLOT_MAPS = []
for _perm in D6_COLUMN_PERMS:
    _sigma = [0] * 7
    for _j in range(6):
        _sigma[_perm[_j]] = _j
    _sigma[6] = 6
    D6_SLOT_MAPS.append(torch.tensor(_sigma, dtype=torch.int16))


def _permute_dst_slots(slots: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0 or slots.numel() == 0:
        return slots
    return D6_SLOT_MAPS[k].index_select(0, slots.long())


class D6Augment(torch.utils.data.Dataset):
    """Augmentation on-the-fly: a ogni epoca ogni posizione riceve una delle
    12 simmetrie D6 scelta a caso. Tutto il resto del dato e' gia' invariante
    per costruzione (feature di nodi, globali e mosse: distanze, conteggi e
    one-hot senza direzioni assolute; z e pi non cambiano ruotando la board),
    quindi basta permutare le colonne di edge_attr. Nessun costo di RAM:
    il dataset resta uno, la variante si crea al volo nel __getitem__."""

    def __init__(self, data_list: list, seed: int):
        self.data_list = data_list
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        d = self.data_list[idx]
        k = self.rng.randrange(len(D6_COLUMN_PERMS))
        if k == 0:
            return d  # identita'
        # La simmetria lascia invariati move_features (nessuna direzione
        # assoluta) e gli indici di nodo (i nodi non cambiano posto); gli
        # UNICI valori direzionali della policy sono i dst_slot, che ruotano
        # con la stessa sigma delle colonne di edge_attr.
        return Data(
            x=d.x,
            edge_index=d.edge_index,
            edge_attr=d.edge_attr[:, D6_COLUMN_PERMS[k]],
            u=d.u,
            y=d.y,
            move_features=d.move_features,
            policy_target=d.policy_target,
            has_policy=d.has_policy,
            move_src=d.move_src,
            dst_node=d.dst_node,
            dst_slot=_permute_dst_slots(d.dst_slot, k),
            dst_move=d.dst_move,
        )


def _cross_entropy_over_positions(logits: torch.Tensor, batch) -> torch.Tensor:
    """CE media per posizione: softmax sulle sole mosse legali di ogni grafo."""
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


def policy_loss_for_batch(model: HiveValueGNN, batch, policy_version: int = 1) -> torch.Tensor:
    if batch.move_features.numel() == 0:
        return batch.y.new_tensor(0.0)

    if policy_version == 2:
        # Gli indici nel JSONL sono locali alla posizione: nel batch vanno
        # tradotti in globali. batch.ptr da' l'offset dei nodi di ogni grafo;
        # per le mosse l'offset si ricava dai conteggi per grafo.
        move_batch = batch.move_features_batch
        node_offset = batch.ptr.index_select(0, move_batch)
        src_local = batch.move_src.long()
        move_src = torch.where(src_local >= 0, src_local + node_offset, src_local)

        move_counts = torch.bincount(move_batch, minlength=batch.num_graphs)
        move_ptr = torch.cat([move_counts.new_zeros(1), move_counts.cumsum(0)[:-1]])
        dst_graph = batch.dst_node_batch
        dst_node = batch.dst_node.long() + batch.ptr.index_select(0, dst_graph)
        dst_move = batch.dst_move.long() + move_ptr.index_select(0, dst_graph)

        logits = model.forward_policy_v2(
            batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch,
            batch.move_features[:, :12].float(), move_src, move_batch,
            dst_node, batch.dst_slot.long(), dst_move,
        )
        return _cross_entropy_over_positions(logits, batch)

    board_embedding = model.encode_board(batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch)
    logits = model.forward_policy_from_embedding(
        board_embedding,
        batch.move_features,
        batch.move_features_batch,
    )
    return _cross_entropy_over_positions(logits, batch)


def batch_losses(model: HiveValueGNN, batch, policy_weight: float, policy_version: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch)
    value_loss = torch.nn.functional.mse_loss(pred, batch.y)
    policy_loss = policy_loss_for_batch(model, batch, policy_version) if policy_weight > 0.0 else pred.new_tensor(0.0)
    return value_loss + policy_weight * policy_loss, value_loss, policy_loss


@torch.no_grad()
def evaluate(model: HiveValueGNN, loader: DataLoader, device: torch.device, policy_weight: float, policy_version: int) -> tuple[float, float, int, float]:
    """Restituisce (MSE medio, accuratezza del segno sulle posizioni decisive,
    numero di posizioni decisive). "Decisive" = z != 0: li' il segno della
    predizione dice se la rete indovina il vincitore."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    sign_correct = 0
    decisive = 0
    total_policy_loss = 0.0
    policy_batches = 0

    for batch in loader:
        batch = batch_to_float32(batch.to(device))
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch)
        total_loss += torch.nn.functional.mse_loss(pred, batch.y, reduction="sum").item()
        total_count += batch.y.size(0)
        if policy_weight > 0.0 and batch.move_features.numel() > 0:
            total_policy_loss += policy_loss_for_batch(model, batch, policy_version).item()
            policy_batches += 1

        mask = batch.y.view(-1) != 0
        decisive += int(mask.sum())
        if mask.any():
            same = torch.sign(pred.view(-1)[mask]) == torch.sign(batch.y.view(-1)[mask])
            sign_correct += int(same.sum())

    mse = total_loss / max(total_count, 1)
    sign_acc = sign_correct / decisive if decisive > 0 else float("nan")
    policy_loss = total_policy_loss / policy_batches if policy_batches > 0 else float("nan")
    return mse, sign_acc, decisive, policy_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HiveValueGNN su dataset JSONL")
    parser.add_argument("data", nargs="+", help="Uno o piu' file JSONL prodotti da SelfPlay")
    parser.add_argument("--output", type=str, default="checkpoint.pt", help="Checkpoint dei pesi (il migliore su validation)")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Checkpoint da cui partire (pesi della generazione precedente) invece dei pesi casuali")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                        help="cosine: decadimento coseno del learning rate da --lr a --lr/10 lungo --epochs. Un lr costante resta troppo alto nelle epoche finali e fa oscillare la validation attorno al minimo invece di scenderci")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Frazione di PARTITE (non posizioni) per la validation")
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Frazione di posizioni da caricare (es. 0.4 se il dataset intero non sta in RAM)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Deve combaciare con l'export")
    parser.add_argument("--heads", type=int, default=4, help="Deve combaciare con l'export")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--move-feature-dim", type=int, default=MOVE_FEATURE_DIM)
    parser.add_argument("--policy-weight", type=float, default=0.0,
                        help="Peso della policy loss. Default 0 = training value-only compatibile coi dataset attuali")
    parser.add_argument("--policy-v2", action="store_true",
                        help="Policy 'edge prediction' sugli embedding dei nodi (campi src/dst del JSONL) al posto delle 32 move features fatte a mano")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Tetto di ore dall'avvio (caricamento incluso): il training si ferma PRIMA di iniziare un'epoca che non ci starebbe. Serve sulle sessioni a tempo (Kaggle: 12h, oltre le quali l'output va perso)")
    parser.add_argument("--augment", action="store_true",
                        help="Augmentation D6 on-the-fly sul train: a ogni epoca ogni posizione riceve una delle 12 simmetrie della board esagonale (la validation resta NON aumentata, per confrontare i run)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu oppure cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_start = time.monotonic()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    games = load_games(args.data, args.sample, args.seed, args.move_feature_dim)
    if not games:
        print("Nessuna posizione trovata nei file dati.", file=sys.stderr)
        sys.exit(1)

    train_data, val_data = split_by_game(games, args.val_fraction, args.seed)
    n_decisive = sum(int((b.y.view(-1) != 0).sum()) for b in games.values())
    print(f"Dataset: {len(games)} partite, {len(train_data) + len(val_data)} posizioni "
          f"({n_decisive} decisive) -> train {len(train_data)}, validation {len(val_data)}")

    policy_version = 2 if args.policy_v2 else 1
    if args.policy_v2 and args.policy_weight > 0.0:
        # La v2 richiede i campi strutturali src/dst nel JSONL: verifica su
        # un campione che ci siano (i dataset value-only o pre-policy no).
        for i in range(min(len(train_data), 200)):
            d = train_data[i]
            if d.move_features.size(0) > 0:
                if d.move_src.size(0) != d.move_features.size(0):
                    sys.exit("Errore: --policy-v2 ma il dataset non ha i campi src/dst per mossa")
                break
    train_set = D6Augment(train_data, args.seed) if args.augment else train_data
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=["move_features", "dst_node"])
    val_loader = DataLoader(val_data, batch_size=args.batch_size, follow_batch=["move_features", "dst_node"]) if val_data else None

    model = HiveValueGNN(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout_p=args.dropout,
        move_feature_dim=args.move_feature_dim,
    ).to(device)

    # Ciclo a generazioni: si parte dai pesi della rete precedente invece che
    # da pesi casuali. Solo i pesi del modello: l'optimizer riparte fresco,
    # perche' i dati nuovi hanno una distribuzione diversa da quelli su cui i
    # suoi momenti interni erano stati accumulati.
    if args.init_weights is not None:
        load_weights(model, Path(args.init_weights), device)
        print(f"Pesi iniziali caricati da: {args.init_weights}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # T_max = --epochs anche se --max-hours ferma prima: il decadimento resta
    # piu' lento del previsto, che e' il difetto meno dannoso.
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_batches = len(train_loader)
    progress_every = max(1, n_batches // 10)  # ~10 righe di avanzamento per epoca

    last_epoch_hours = 0.0
    for epoch in range(1, args.epochs + 1):
        # Guardia sul tempo di sessione: se questa epoca (stimata come la
        # precedente + 10% di margine) sforasse il tetto, meglio fermarsi
        # con il best checkpoint gia' salvato che perdere tutto al limite.
        if args.max_hours is not None:
            elapsed_hours = (time.monotonic() - run_start) / 3600.0
            if epoch > 1 and elapsed_hours + last_epoch_hours * 1.1 > args.max_hours:
                print(f"Stop per --max-hours: {elapsed_hours:.1f}h trascorse, "
                      f"la prossima epoca (~{last_epoch_hours:.1f}h) non ci starebbe.", flush=True)
                break

        model.train()
        running_loss = 0.0
        running_value_loss = 0.0
        running_policy_loss = 0.0
        running_count = 0
        epoch_start = time.monotonic()
        for batch_idx, batch in enumerate(train_loader, 1):
            batch = batch_to_float32(batch.to(device))
            optimizer.zero_grad()
            loss, value_loss, policy_loss = batch_losses(model, batch, args.policy_weight, policy_version)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.y.size(0)
            running_value_loss += value_loss.item() * batch.y.size(0)
            running_policy_loss += policy_loss.item() * batch.y.size(0)
            running_count += batch.y.size(0)

            if batch_idx % progress_every == 0 and batch_idx < n_batches:
                speed = running_count / (time.monotonic() - epoch_start)
                print(f"  epoch {epoch}: batch {batch_idx}/{n_batches}  "
                      f"value MSE parziale {running_value_loss / running_count:.4f}  "
                      f"({speed:.0f} posizioni/s)", flush=True)
        train_loss = running_loss / max(running_count, 1)
        train_mse = running_value_loss / max(running_count, 1)
        train_policy_loss = running_policy_loss / max(running_count, 1)

        line = f"epoch {epoch:>3}  train MSE {train_mse:.4f}"
        if args.policy_weight > 0.0:
            line += f"  train policy CE {train_policy_loss:.4f}  train loss {train_loss:.4f}"
        current = train_loss
        if val_loader is not None:
            val_mse, sign_acc, decisive, val_policy_loss = evaluate(model, val_loader, device, args.policy_weight, policy_version)
            line += f"  val MSE {val_mse:.4f}"
            if decisive > 0:
                line += f"  val segno-ok {sign_acc:.1%} (su {decisive} decisive)"
            if args.policy_weight > 0.0 and val_policy_loss == val_policy_loss:
                line += f"  val policy CE {val_policy_loss:.4f}"
                current = val_mse + args.policy_weight * val_policy_loss
            else:
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
                    "move_feature_dim": args.move_feature_dim,
                    "policy_weight": args.policy_weight,
                    "policy_version": policy_version,
                },
                output_path,
            )
            line += "  [salvato]"

        if scheduler is not None:
            line += f"  lr {optimizer.param_groups[0]['lr']:.2e}"
            scheduler.step()
        print(line, flush=True)
        last_epoch_hours = (time.monotonic() - epoch_start) / 3600.0

    print(f"\nCheckpoint migliore salvato in: {output_path} (MSE {best_val:.4f})")
    print("Per esportarlo per il C++:")
    print(f"  python3 scripts/export_hive_value_gnn.py --weights {output_path} --output hive_value_gnn.pt")


if __name__ == "__main__":
    main()
