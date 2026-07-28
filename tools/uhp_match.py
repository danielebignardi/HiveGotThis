#!/usr/bin/env python3
"""
Harness di valutazione: fa giocare due engine UHP uno contro l'altro e conta
i risultati. Risponde alla domanda del ciclo a generazioni: "il modello nuovo
e' piu' forte di quello vecchio?".

Ogni engine puo' caricare un modello .pt diverso (passato come argomento al
binario). Le partite si giocano a coppie: stessa apertura casuale, colori
invertiti. Cosi' un'apertura sbilanciata avvantaggia una volta l'uno e una
volta l'altro, e il confronto resta equo. Le aperture casuali servono perche'
a parita' di modello e profondita' l'engine e' deterministico: senza, tutte
le partite del match sarebbero identiche.

Uso:
  # una partita engine contro se stesso, mossa per mossa
  python3 tools/uhp_match.py selfplay build/HiveEngine --model hive_value_gnn.pt

  # match tra due modelli (stesso binario), 20 partite da 1 secondo a mossa
  python3 tools/uhp_match.py match build/HiveEngine build/HiveEngine \
      --white-model gen0.pt --black-model random.pt --games 20 --time 1

  # dataset policy da self-play: posizioni + mosse legali con pi dalle visite
  python3 tools/uhp_match.py dataset build/HiveEngine --model hive_value_gnn.pt \
      --output data/selfplay_policy.jsonl --games 100 --depth 1

Con --depth N ogni mossa usa N*1000 iterazioni MCTS (deterministico, utile
per test riproducibili); con --time S ogni mossa usa S secondi (la modalita'
da torneo). Se si danno entrambi vince --time.

Il modo dataset usa il comando UHP `policytargets depth N`: per ogni posizione
l'engine restituisce un JSON con le feature della board, le mosse legali con
move_features e pi = visite/totale della ricerca MCTS alla radice. La mossa
giocata si sceglie con --move-selection (default: campionamento da pi con
temperatura, la fonte di varieta' del self-play); a fine partita il label z
viene aggiunto a ogni posizione. L'output e' pronto per
train_hive_value_gnn.py --policy-weight.
"""
import argparse
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path


def time_param(seconds):
    # Lo standard UHP vuole HH:MM:SS zero-padded ("bestmove time 00:00:05");
    # il formato conta quando l'avversario e' un engine esterno.
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"time {h:02d}:{m:02d}:{s:02d}"


class Engine:
    def __init__(self, path, model=None):
        self.path = path
        # Etichetta per i report: il modello distingue i due engine quando il
        # binario e' lo stesso (il caso tipico: stesso MCTS, pesi diversi).
        self.label = model if model else path
        # shlex.split permette binari con argomenti, es. "path/nokamute uhp".
        args = shlex.split(path) + ([model] if model else [])
        self.p = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Consuma il banner di avvio ("id ...", capacita', "ok").
        self._drain_to_ok()

    def _drain_to_ok(self):
        while True:
            resp = self.p.stdout.readline()
            if resp == "":
                raise RuntimeError(f"engine {self.path} chiuso durante l'avvio")
            if resp.rstrip("\n") == "ok":
                return

    def cmd(self, line):
        """Invia un comando, restituisce le righe di risposta prima di 'ok'."""
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = []
        while True:
            resp = self.p.stdout.readline()
            if resp == "":
                raise RuntimeError(f"engine {self.path} chiuso inaspettatamente")
            resp = resp.rstrip("\n")
            if resp == "ok":
                return out
            out.append(resp)

    def close(self):
        try:
            self.p.stdin.write("exit\n")
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


def board_state(gamestring):
    # "Base+MLP;InProgress;White[5];..." -> "InProgress"
    parts = gamestring.split(";")
    return parts[1] if len(parts) >= 2 else "?"


def side_to_move(gamestring):
    # "...;White[5]" -> "White"
    parts = gamestring.split(";")
    return parts[2].split("[")[0] if len(parts) >= 3 else "?"


TERMINAL = {"WhiteWins", "BlackWins", "Draw"}


def play_game(white, black, variant, search_param, max_plies,
              opening_plies, opening_moves, rng, verbose=False):
    """Gioca una partita e restituisce (stato_finale, mosse).

    Le prime opening_plies mosse sono l'apertura: riprese da opening_moves se
    fornite (la seconda partita della coppia rigioca la stessa apertura),
    altrimenti scelte a caso tra le mosse valide. Dalla mossa successiva in
    poi decide bestmove.
    """
    engines = {"White": white, "Black": black}
    gs = white.cmd(f"newgame {variant}")[-1]
    black.cmd(f"newgame {variant}")
    stm = side_to_move(gs)
    moves = []

    for ply in range(max_plies):
        mover = engines[stm]
        if ply < opening_plies:
            if opening_moves is not None:
                mv = opening_moves[ply]
            else:
                mv = rng.choice(mover.cmd("validmoves")[-1].split(";"))
        else:
            mv = mover.cmd(f"bestmove {search_param}")[-1].strip()
        # La mossa va applicata a ENTRAMBI gli engine per tenerli in sincrono.
        gs = mover.cmd(f"play {mv}")[-1]
        other = black if mover is white else white
        other.cmd(f"play {mv}")
        moves.append(mv)
        state = board_state(gs)
        if verbose:
            tag = "(apertura)" if ply < opening_plies else ""
            print(f"  ply {ply + 1:>3} {stm:>5}: {mv:<18} -> {state} {tag}",
                  flush=True)
        if state in TERMINAL:
            return state, moves
        stm = side_to_move(gs)
    return "PlyCapReached", moves


def run_match(args):
    search_param = time_param(args.time) if args.time else f"depth {args.depth}"
    eng_a = Engine(args.white, args.white_model)
    eng_b = Engine(args.black, args.black_model)
    if eng_a.label == eng_b.label:
        eng_a.label += " (A)"
        eng_b.label += " (B)"

    print(f"Match: {eng_a.label} vs {eng_b.label}")
    print(f"  {args.games} partite, {search_param} a mossa, "
          f"apertura casuale di {args.opening_plies} plies, seed {args.seed}")

    rng = random.Random(args.seed)
    wins = {eng_a.label: 0, eng_b.label: 0}
    draws = 0
    caps = 0
    opening = None  # apertura condivisa dalla coppia di partite corrente

    try:
        for i in range(args.games):
            swapped = i % 2 == 1
            if not swapped:
                opening = None  # coppia nuova -> apertura nuova
            white, black = (eng_b, eng_a) if swapped else (eng_a, eng_b)

            state, moves = play_game(
                white, black, args.variant, search_param, args.max_plies,
                args.opening_plies, opening, rng, verbose=args.games == 1,
            )
            if opening is None:
                opening = moves[:args.opening_plies]

            if state == "WhiteWins":
                outcome = f"vince {white.label}"
                wins[white.label] += 1
            elif state == "BlackWins":
                outcome = f"vince {black.label}"
                wins[black.label] += 1
            elif state == "Draw":
                outcome = "patta"
                draws += 1
            else:
                outcome = "non conclusa (limite plies)"
                caps += 1
            print(f"  partita {i + 1}/{args.games}: "
                  f"White={white.label} Black={black.label} -> "
                  f"{outcome} in {len(moves)} plies", flush=True)
    finally:
        eng_a.close()
        eng_b.close()

    played = wins[eng_a.label] + wins[eng_b.label] + draws
    print(f"\nRisultato su {args.games} partite:")
    print(f"  {eng_a.label}: {wins[eng_a.label]} vittorie")
    print(f"  {eng_b.label}: {wins[eng_b.label]} vittorie")
    print(f"  patte: {draws}, non concluse: {caps}")
    if played > 0:
        # Convenzione standard dei match: una patta vale mezza vittoria.
        score = (wins[eng_a.label] + draws / 2) / played
        print(f"  score di {eng_a.label}: {score:.0%} sulle {played} concluse")


# ── Modo dataset: self-play con target di policy ─────────────────────────────

def result_to_z(result, side):
    # Stessa convenzione side-to-move di SelfPlay: +1 = chi muoveva ha vinto.
    if result == "WhiteWins":
        return 1.0 if side == "White" else -1.0
    if result == "BlackWins":
        return 1.0 if side == "Black" else -1.0
    return 0.0  # patta o partita troncata al tetto di mosse


def choose_move(sample, rng, move_selection, temperature, random_plies, ply):
    """Sceglie la mossa da giocare tra quelle restituite da policytargets."""
    moves = sample.get("moves", [])
    if not moves:
        return sample.get("best_move", "pass")

    if move_selection == "random":
        return rng.choice(moves)["move"]

    if move_selection == "best":
        return sample.get("best_move", max(moves, key=lambda m: m.get("pi", 0.0))["move"])

    # move_selection == "pi": apertura uniforme, poi campionamento con
    # temperatura sulla distribuzione delle visite (pi ∝ N^(1/tau)) - la
    # fonte di varieta' del self-play, vedi docs/piano_fase5.md.
    if ply < random_plies:
        return rng.choice(moves)["move"]

    if temperature <= 0.0:
        return sample.get("best_move", max(moves, key=lambda m: m.get("pi", 0.0))["move"])

    weights = [max(0.0, float(m.get("pi", 0.0))) ** (1.0 / temperature) for m in moves]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(moves)["move"]

    pick = rng.random() * total
    acc = 0.0
    for move, weight in zip(moves, weights):
        acc += weight
        if pick <= acc:
            return move["move"]
    return moves[-1]["move"]


def generate_dataset(args):
    search_param = f"depth {args.depth}"
    rng = random.Random(args.seed)
    total_positions = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Un solo processo engine riusato per tutte le partite: stessa rete e
    # stesse valutazioni, quindi la transposition table persistente resta
    # valida e calda tra una partita e l'altra (come in SelfPlay).
    engine = Engine(args.engine, args.model)
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            for game_id in range(args.games):
                gs = engine.cmd(f"newgame {args.variant}")[-1]
                samples = []
                moves = []

                for ply in range(args.max_plies):
                    if board_state(gs) in TERMINAL:
                        break

                    sample = json.loads(engine.cmd(f"policytargets {search_param}")[-1])
                    sample["game_id"] = f"selfplay_{game_id:06d}"
                    sample["ply"] = ply
                    sample["side_to_move"] = side_to_move(gs)
                    samples.append(sample)

                    move = choose_move(sample, rng, args.move_selection,
                                       args.temperature, args.random_plies, ply)
                    sample["played_move"] = move
                    gs = engine.cmd(f"play {move}")[-1]
                    moves.append(move)

                    if args.verbose:
                        print(f"game {game_id + 1}/{args.games} ply {ply + 1:>3}: "
                              f"{move:<18} -> {board_state(gs)}", flush=True)

                result = board_state(gs)
                if result not in TERMINAL:
                    result = "PlyCapReached"

                for sample in samples:
                    sample["result"] = result
                    sample["z"] = result_to_z(result, sample["side_to_move"])
                    f.write(json.dumps(sample, separators=(",", ":")) + "\n")

                total_positions += len(samples)
                print(f"partita {game_id + 1}/{args.games}: {result} "
                      f"in {len(moves)} plies, {len(samples)} posizioni", flush=True)
    finally:
        engine.close()

    print(f"Scritte {total_positions} posizioni in {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("selfplay", help="una partita engine contro se stesso")
    sp.add_argument("engine")
    sp.add_argument("--model", default=None, help="modello .pt per l'engine")
    sp.add_argument("--variant", default="Base+MLP")
    sp.add_argument("--depth", type=int, default=1)
    sp.add_argument("--time", type=int, default=None, help="secondi a mossa (prevale su --depth)")
    sp.add_argument("--max-plies", type=int, default=200)

    mt = sub.add_parser("match", help="match su piu' partite tra due engine/modelli")
    mt.add_argument("white", help="binario del primo engine")
    mt.add_argument("black", help="binario del secondo engine")
    mt.add_argument("--white-model", default=None, help="modello .pt del primo engine")
    mt.add_argument("--black-model", default=None, help="modello .pt del secondo engine")
    mt.add_argument("--games", type=int, default=1)
    mt.add_argument("--variant", default="Base+MLP")
    mt.add_argument("--depth", type=int, default=1)
    mt.add_argument("--time", type=int, default=None, help="secondi a mossa (prevale su --depth)")
    mt.add_argument("--opening-plies", type=int, default=4,
                    help="mosse iniziali casuali per differenziare le partite")
    mt.add_argument("--max-plies", type=int, default=200)
    mt.add_argument("--seed", type=int, default=42)

    ds = sub.add_parser("dataset", help="self-play che scrive un JSONL con target di policy")
    ds.add_argument("engine")
    ds.add_argument("--model", default=None, help="modello .pt per l'engine")
    ds.add_argument("--output", required=True)
    ds.add_argument("--games", type=int, default=1)
    ds.add_argument("--variant", default="Base+MLP")
    ds.add_argument("--depth", type=int, default=1,
                    help="budget di ricerca per policytargets (N*1000 iterazioni)")
    ds.add_argument("--max-plies", type=int, default=200)
    ds.add_argument("--move-selection", choices=["random", "pi", "best"], default="pi",
                    help="mossa giocata: pi = campiona dalle visite MCTS (default), "
                         "best = sempre best_move, random = uniforme tra le legali")
    ds.add_argument("--temperature", type=float, default=1.0,
                    help="solo con pi: temperatura di campionamento. 0 = sempre best_move")
    ds.add_argument("--random-plies", type=int, default=8,
                    help="solo con pi: ply iniziali scelti uniformemente tra le legali")
    ds.add_argument("--seed", type=int, default=42)
    ds.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.mode == "selfplay":
        search_param = time_param(args.time) if args.time else f"depth {args.depth}"
        print(f"Self-play: {args.engine} ({search_param})")
        # Due processi distinti anche se il binario e' lo stesso: play_game
        # applica ogni mossa a entrambi per tenerli in sincrono.
        eng_w = Engine(args.engine, args.model)
        eng_b = Engine(args.engine, args.model)
        try:
            result, moves = play_game(
                eng_w, eng_b, args.variant, search_param, args.max_plies,
                opening_plies=0, opening_moves=None, rng=random.Random(),
                verbose=True,
            )
        finally:
            eng_w.close()
            eng_b.close()
        print(f"RISULTATO: {result} in {len(moves)} plies")
    elif args.mode == "dataset":
        generate_dataset(args)
    else:
        run_match(args)


if __name__ == "__main__":
    main()
