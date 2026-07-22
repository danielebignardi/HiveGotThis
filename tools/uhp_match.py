#!/usr/bin/env python3
"""
Behavioral test harness for the HiveGotThis UHP engine.

Drives one or two engine binaries through complete games over the UHP protocol
and reports the result. Used to compare the old (rootColor-fixed) MCTS against
the refactored (negamax / side-to-move) MCTS.

A correct negamax refactor must:
  * still reach decisive results (not wander until the ply cap),
  * not play suicidally (a global sign error makes the engine surround its OWN
    queen / refuse obvious wins), so the new engine must NOT get crushed by the
    old one in a head-to-head match.

Usage:
  python3 tools/uhp_match.py selfplay <engine> [--depth N] [--max-plies M]
  python3 tools/uhp_match.py match <white_engine> <black_engine> [--depth N] [--max-plies M]
  python3 tools/uhp_match.py dataset <engine> --output data.jsonl [--games G] [--depth N]
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


class Engine:
    def __init__(self, path, extra_args=None):
        self.path = path
        extra_args = extra_args or []
        self.p = subprocess.Popen(
            [path, *extra_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Drain the startup banner ("id ...", capabilities, "ok") before commands.
        self._drain_to_ok()

    def _drain_to_ok(self):
        while True:
            resp = self.p.stdout.readline()
            if resp == "":
                raise RuntimeError(f"engine {self.path} closed during startup")
            if resp.rstrip("\n") == "ok":
                return

    def cmd(self, line):
        """Send a command, return the payload lines emitted before 'ok'."""
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = []
        while True:
            resp = self.p.stdout.readline()
            if resp == "":
                raise RuntimeError(f"engine {self.path} closed unexpectedly")
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
    # "Base;InProgress;White[5]" -> "InProgress"
    parts = gamestring.split(";")
    return parts[1] if len(parts) >= 2 else "?"


def side_to_move(gamestring):
    # "...;White[5]" -> "White"
    parts = gamestring.split(";")
    return parts[2].split("[")[0] if len(parts) >= 3 else "?"


TERMINAL = {"WhiteWins", "BlackWins", "Draw"}


def play_game(white_path, black_path, depth, max_plies, game_type, bestmove_time, white_args=None, black_args=None, verbose=False):
    white = Engine(white_path, white_args)
    black = Engine(black_path, black_args)
    engines = {"White": white, "Black": black}
    moves = []
    try:
        gs = white.cmd(f"newgame {game_type}")[-1]
        black.cmd(f"newgame {game_type}")
        stm = side_to_move(gs)

        for ply in range(max_plies):
            mover = engines[stm]
            if bestmove_time is not None:
                mv = mover.cmd(f"bestmove time {bestmove_time}")[-1].strip()
            else:
                mv = mover.cmd(f"bestmove depth {depth}")[-1].strip()
            # apply to BOTH engines to keep them in sync
            gs = mover.cmd(f"play {mv}")[-1]
            other = black if mover is white else white
            other.cmd(f"play {mv}")
            moves.append(mv)
            state = board_state(gs)
            if verbose:
                print(f"  ply {ply+1:>3} {stm:>5}: {mv:<18} -> {state}")
            if state in TERMINAL:
                return state, moves
            stm = side_to_move(gs)
        return "PlyCapReached", moves
    finally:
        white.close()
        black.close()


def result_to_z(result, side):
    if result == "Draw" or result == "PlyCapReached":
        return 0.0
    if result == "WhiteWins":
        return 1.0 if side == "White" else -1.0
    if result == "BlackWins":
        return 1.0 if side == "Black" else -1.0
    return 0.0


def choose_move(sample, rng, move_selection, temperature, random_plies, ply):
    moves = sample.get("moves", [])
    if not moves:
        return sample.get("best_move", "pass")

    if move_selection == "random":
        return rng.choice(moves)["move"]

    if move_selection == "best":
        return sample.get("best_move", max(moves, key=lambda m: m.get("pi", 0.0))["move"])

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


def generate_dataset(engine_path, engine_args, output, games, depth, max_plies, game_type, move_selection, temperature, random_plies, seed, verbose=False):
    total_positions = 0
    rng = random.Random(seed)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for game_id in range(games):
            engine = Engine(engine_path, engine_args)
            samples = []
            moves = []
            try:
                gs = engine.cmd(f"newgame {game_type}")[-1]
                for ply in range(max_plies):
                    state = board_state(gs)
                    if state in TERMINAL:
                        break

                    sample = json.loads(engine.cmd(f"policytargets depth {depth}")[-1])
                    sample["game_id"] = f"selfplay_{game_id:06d}"
                    sample["ply"] = ply
                    sample["side_to_move"] = side_to_move(gs)
                    samples.append(sample)

                    move = choose_move(sample, rng, move_selection, temperature, random_plies, ply)
                    sample["played_move"] = move
                    gs = engine.cmd(f"play {move}")[-1]
                    moves.append(move)

                    if verbose:
                        print(f"game {game_id+1}/{games} ply {ply+1:>3}: {move:<18} -> {board_state(gs)}")

                    if board_state(gs) in TERMINAL:
                        break

                result = board_state(gs)
                if result not in TERMINAL:
                    result = "PlyCapReached"

                for sample in samples:
                    sample["result"] = result
                    sample["z"] = result_to_z(result, sample["side_to_move"])
                    f.write(json.dumps(sample, separators=(",", ":")) + "\n")

                total_positions += len(samples)
                print(f"game {game_id+1}/{games}: {result} in {len(moves)} plies, {len(samples)} samples")
            finally:
                engine.close()

    print(f"Wrote {total_positions} positions to {output}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("selfplay")
    sp.add_argument("engine")
    sp.add_argument("--engine-arg", action="append", default=[])
    sp.add_argument("--depth", type=int, default=1)
    sp.add_argument("--time", default=None, help="Usa bestmove time HH:MM:SS invece di depth")
    sp.add_argument("--game-type", default="Base+MLP")
    sp.add_argument("--max-plies", type=int, default=100)

    mt = sub.add_parser("match")
    mt.add_argument("white")
    mt.add_argument("black")
    mt.add_argument("--white-arg", action="append", default=[])
    mt.add_argument("--black-arg", action="append", default=[])
    mt.add_argument("--depth", type=int, default=1)
    mt.add_argument("--time", default=None, help="Usa bestmove time HH:MM:SS invece di depth")
    mt.add_argument("--game-type", default="Base+MLP")
    mt.add_argument("--max-plies", type=int, default=100)

    ds = sub.add_parser("dataset")
    ds.add_argument("engine")
    ds.add_argument("--engine-arg", action="append", default=[],
                    help="Argomento extra da passare al motore, ripetibile. Esempio: --engine-arg hive_value_gnn.pt")
    ds.add_argument("--output", required=True)
    ds.add_argument("--games", type=int, default=1)
    ds.add_argument("--depth", type=int, default=1)
    ds.add_argument("--game-type", default="Base+MLP")
    ds.add_argument("--max-plies", type=int, default=100)
    ds.add_argument("--move-selection", choices=["random", "pi", "best"], default="random",
                    help="Come scegliere la mossa giocata: random = uniforme tra legali, pi = campiona da MCTS pi, best = sempre best_move")
    ds.add_argument("--temperature", type=float, default=1.0,
                    help="Solo con --move-selection pi: temperatura di campionamento da pi. 0 = usa sempre best_move")
    ds.add_argument("--random-plies", type=int, default=8,
                    help="Solo con --move-selection pi: numero di ply iniziali scelti uniformemente tra le mosse legali")
    ds.add_argument("--seed", type=int, default=42)
    ds.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.mode == "selfplay":
        budget = f"time {args.time}" if args.time else f"depth {args.depth}"
        print(f"Self-play: {args.engine} ({args.game_type}, {budget})")
        result, moves = play_game(
            args.engine,
            args.engine,
            args.depth,
            args.max_plies,
            args.game_type,
            args.time,
            args.engine_arg,
            args.engine_arg,
            verbose=True,
        )
        print(f"RESULT: {result} in {len(moves)} plies")
    else:
        if args.mode == "dataset":
            generate_dataset(
                args.engine,
                args.engine_arg,
                args.output,
                args.games,
                args.depth,
                args.max_plies,
                args.game_type,
                args.move_selection,
                args.temperature,
                args.random_plies,
                args.seed,
                args.verbose,
            )
            return

        budget = f"time {args.time}" if args.time else f"depth {args.depth}"
        print(f"Match ({args.game_type}, {budget}):")
        print(f"  White = {args.white}")
        print(f"  Black = {args.black}")
        result, moves = play_game(
            args.white,
            args.black,
            args.depth,
            args.max_plies,
            args.game_type,
            args.time,
            args.white_arg,
            args.black_arg,
            verbose=True,
        )
        print(f"RESULT: {result} in {len(moves)} plies")


if __name__ == "__main__":
    main()
