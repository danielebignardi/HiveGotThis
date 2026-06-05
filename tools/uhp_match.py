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
"""
import argparse
import subprocess
import sys


class Engine:
    def __init__(self, path):
        self.path = path
        self.p = subprocess.Popen(
            [path],
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


def play_game(white_path, black_path, depth, max_plies, verbose=False):
    white = Engine(white_path)
    black = Engine(black_path)
    engines = {"White": white, "Black": black}
    moves = []
    try:
        gs = white.cmd("newgame Base")[-1]
        black.cmd("newgame Base")
        stm = side_to_move(gs)

        for ply in range(max_plies):
            mover = engines[stm]
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("selfplay")
    sp.add_argument("engine")
    sp.add_argument("--depth", type=int, default=1)
    sp.add_argument("--max-plies", type=int, default=200)

    mt = sub.add_parser("match")
    mt.add_argument("white")
    mt.add_argument("black")
    mt.add_argument("--depth", type=int, default=1)
    mt.add_argument("--max-plies", type=int, default=200)

    args = ap.parse_args()

    if args.mode == "selfplay":
        print(f"Self-play: {args.engine} (depth {args.depth})")
        result, moves = play_game(
            args.engine, args.engine, args.depth, args.max_plies, verbose=True
        )
        print(f"RESULT: {result} in {len(moves)} plies")
    else:
        print(f"Match (depth {args.depth}):")
        print(f"  White = {args.white}")
        print(f"  Black = {args.black}")
        result, moves = play_game(
            args.white, args.black, args.depth, args.max_plies, verbose=True
        )
        print(f"RESULT: {result} in {len(moves)} plies")


if __name__ == "__main__":
    main()
