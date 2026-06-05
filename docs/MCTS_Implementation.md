# MCTS Implementation

This document describes how Monte Carlo Tree Search (MCTS) is implemented in
HiveGotThis. It is written for someone who knows MCTS in general terms but did
not write this code.

All code lives in two files:

- `include/MCTS.h` — the `MCTSNode`, `TranspositionTable`, and `MCTS` declarations.
- `src/MCTS.cpp` — all the logic.

The public entry points are `MCTS::Search(board, timeLimitMs)` and
`MCTS::SearchIterations(board, maxIterations)` (`src/MCTS.cpp:453` and `:497`).
Both return the chosen `Move`. The engine calls them from
`Engine::CommandBestMove` in response to the UHP `bestmove` command.

One important structural note before the phases: **a tree node does not store a
board.** `MCTSNode` (`include/MCTS.h:52`) stores only the `Move` that led to it,
parent/child pointers, and statistics. The board state at any node is
reconstructed on the fly each iteration by copying the root board and replaying
moves down the path (`Board board = rootBoard;` then `board.ApplyMove(...)`,
`src/MCTS.cpp:238`, `:258`, `:320`). This keeps the tree small and is why you
will see a fresh `board` copy at the top of every iteration.

---

## 1. Baseline MCTS — the four phases

One MCTS iteration is `MCTS::RunIteration` (`src/MCTS.cpp:233`). It performs
selection → expansion → evaluation → backpropagation in that order. `Search`
calls it in a loop until the time/iteration budget runs out or the root is
solved (see §3), then picks the final move with `SelectBestMove`.

### The value convention (read this first)

We use a **negamax, side-to-move** convention. Every value in the tree is in the
range **`[-1, 1]` from the perspective of the player whose turn it is at that
node**:

- `+1` = the player to move at this node is winning,
- `-1` = losing,
- `0` = drawn / neutral.

Because the side to move alternates every ply, a value that is "good" for one
node is "bad" for its parent. We handle this by **flipping the sign at every
level during backpropagation** (negamax). As a consequence, selection never
needs a separate min/max branch — every node simply tries to maximize the
negation of its children's values.

A node's running statistics are:

- `visitCount` — how many iterations passed through this node.
- `totalValue` — the sum of backpropagated values, in node-to-move perspective.
  The node's average value is `totalValue / visitCount ∈ [-1, 1]`.

This convention was adopted deliberately so that leaf evaluation matches the
planned GNN value network, which outputs `tanh ∈ [-1, 1]` from the side-to-move
perspective. See §"Evaluation" for how the current heuristic is mapped into it.

### Selection

**Code:** the `while` loop at `src/MCTS.cpp:241-261`; the scoring formula is
`MCTSNode::UCB1` (`src/MCTS.cpp:85`).

Starting at the root, we descend by repeatedly picking the child with the
highest UCB1 score, applying that child's move to the working board as we go.
The loop continues while the current node is **fully expanded, not terminal, and
not solved**:

```cpp
while (node->isExpanded && !node->isTerminal && node->provenResult == 0)
```

The moment we reach a node that is *not* fully expanded (it still has moves to
try), terminal, or solved, the loop stops and that node becomes the target of
expansion/evaluation. (`isExpanded` only becomes true once **all** of a node's
moves have been turned into children — see Expansion — so a partially-expanded
node is treated as a leaf and gets one more child added.)

The per-child score (`UCB1`, `src/MCTS.cpp:99-106`) has three terms:

```cpp
exploitation = (1.0 - childValue) / 2.0;                              // negamax, in [0,1]
exploration  = explorationC * sqrt(logParentVisits / visitCount);     // EXPLORATION_C = 1.0
bias         = PROGRESSIVE_BIAS_WEIGHT * heuristicScore / (visitCount + 1);  // weight = 5.0
score        = exploitation + exploration + bias;
```

`childValue = totalValue / visitCount` is the child's average value in the
**child's** perspective (the opponent of the node doing the selecting). The term
`(1 - childValue) / 2` negates and re-maps it to `[0, 1]` = "win probability for
the selecting node." Re-mapping back to `[0, 1]` keeps `EXPLORATION_C` and
`PROGRESSIVE_BIAS_WEIGHT` on the same scale they had before the negamax
refactor, so those constants did not need retuning. The `exploration` and
`bias` terms are explained in §4 (progressive bias).

### Expansion

**Code:** `src/MCTS.cpp:279-338`.

Move generation is **eager per node, lazy across nodes.** The first time a node
is visited for expansion (`unexpandedMoves.empty() && children.empty()`,
`:283`), we:

1. Check whether the game is already over at this node (`GetBoardState()`,
   `:285`). If so, mark it terminal and record a proven result (see §3) — no
   children are generated.
2. Otherwise generate **all** legal moves at once into `node->unexpandedMoves`
   via `board.GetValidMoves(...)` and sort them (`OrderMoves`, see §5).

Then we create **exactly one child per iteration** (`:305-318`): pop the best
remaining move from the back of `unexpandedMoves`, build a single child node,
and descend into it. `isExpanded` flips to `true` only when `unexpandedMoves`
becomes empty (`:310-311`). So a node with N legal moves takes N iterations to
fully expand, one child at a time — classic progressive expansion, not
"generate all children immediately."

The newly created child is immediately checked for being terminal too
(`:325-336`), since the move may have just ended the game.

### Evaluation

**Code:** `src/MCTS.cpp:340-366`.

The freshly reached/created leaf needs a value. There are three sources, in
priority order:

1. **Proven result** (`:345-348`): if the node has a certain win/loss (terminal
   or solved, see §3), use `+1.0` / `-1.0` directly.
2. **Transposition table hit** (`:354-357`): if this exact position was already
   evaluated, reuse the cached value (see §2).
3. **Heuristic** (`:360-364`): otherwise call the static evaluation and store
   it in the TT.

A **terminal check always happens before heuristic evaluation** — terminality is
detected during expansion (`:285`, `:325`) and surfaces here as a non-zero
`provenResult`, so a finished game is never handed to the heuristic.

The heuristic is `EvaluateBoard` (`src/Evaluation.cpp`), which returns `[0, 1]`
from a given perspective. We call it with **`board.currentColor`** (the side to
move at the leaf) and remap to our `[-1, 1]` convention:

```cpp
value = 2.0 * EvaluateBoard(board, board.currentColor) - 1.0;
```

Note the perspective is the board's own side to move — **not** the root player.
This is what makes the value side-to-move-relative. When the GNN value network
replaces the heuristic, it will return `[-1, 1]` side-to-move directly and the
`2v - 1` wrapper disappears; this single line (`src/MCTS.cpp:363`) is the only
place leaf evaluation happens.

### Backpropagation (backup)

**Code:** the loop at `src/MCTS.cpp:368-377` (and an identical one at `:263-277`
for the solved-node shortcut).

We walk from the leaf up to the root, and at **every** node:

```cpp
current->visitCount++;
current->totalValue += value;
value = -value;            // negamax sign flip
current = current->parent;
```

The `value = -value` is the heart of the negamax convention: the leaf value is
in the leaf's perspective; one level up the perspective is the opponent's, so we
negate; and so on alternately up to the root. Each node therefore accumulates
values in *its own* side-to-move perspective, which is exactly what `UCB1` reads
back during selection. There is no separate min/max handling anywhere — the sign
flip does all of it.

(There are two copies of this loop because a node that selection lands on which
is *already* solved short-circuits expansion/evaluation and backs up its proven
value directly, `:265-277`.)

### Choosing the final move

**Code:** `MCTS::SelectBestMove` (`src/MCTS.cpp:393`).

After the search budget is exhausted, the move is chosen from the root's
children — **not** by value but by robustness, in three tiers:

1. A child that is a **proven win** for us. Careful: the root's children store
   `provenResult` in the *opponent's* perspective, so a win for us is a child
   whose mover (the opponent) **loses** → `provenResult == -1` (`:401`).
2. Otherwise the **most-visited** child, skipping any proven loss for us
   (`provenResult == 1`, `:418`). Visit count is MCTS's standard robustness
   signal: the most-explored move is the most trusted.
3. If every move is a proven loss, just play the most-visited one to delay the
   defeat (`:429-440`).

---

## 2. Addition: Transposition Table

**What:** a fixed-size cache mapping a board position to a previously computed
evaluation, so identical positions reached via different move orders are
evaluated once.

**Why:** Hive positions are frequently reachable by multiple move orders
(transpositions). Re-running the heuristic (and, later, the GNN — which is far
more expensive) for the same position wastes time. The TT turns repeated work
into an O(1) lookup.

**Where:** `class TranspositionTable` (`include/MCTS.h:25`, implemented
`src/MCTS.cpp:17-60`); used inside evaluation at `src/MCTS.cpp:351-365`.

**Keying — Zobrist hash.** Positions are keyed by the board's 64-bit Zobrist
hash (`board.GetHash()`, `src/MCTS.cpp:351`). The hash is maintained
incrementally by `Board` and **includes whose turn it is** (`ZobristBlackTurn`).
This matters for our value convention: because the side to move is part of the
key, a stored *side-to-move-relative* value is a deterministic function of the
key and is always consistent — the same key always implies the same player to
move, hence the same perspective. (This is also why the convention switch needed
no TT changes beyond the value's range.)

The table is a flat array of `2^sizePower2` entries; the index is the low bits
of the hash (`hash & m_mask`, `:38`, `:50`). Both `Search` and
`SearchIterations` create a `TranspositionTable tt(18)` — i.e. `2^18` entries —
per call (`src/MCTS.cpp:473`, `:517`). The table is local to a single search and
discarded afterward.

**Replacement policy — depth-preferring.** Two positions can hash to the same
slot (collisions, since we keep only the low bits). `Store` (`:48-60`)
overwrites the existing entry **only if the new entry's `depth` is ≥ the stored
one**:

```cpp
if (depth >= stored.depth) { ...overwrite... }
```

Here `depth` is the ply distance from the root at which the value was computed
(`src/MCTS.cpp:239`, incremented while descending). Preferring greater depth
keeps the entries that took the most work / are closest to the leaves, which are
the more valuable cached results. `Probe` (`:36-46`) returns a hit only if the
stored hash matches exactly **and** `depth > 0` — the `depth > 0` test doubles as
an "occupied slot" check, since `Clear()` zeroes the table (`:30-34`).

`TTEntry` (`include/MCTS.h:17`) also carries an `isExact` flag (set from
`node->isTerminal` at store time) to mark terminal vs. estimated values; it is
recorded for completeness and future use.

---

## 3. Addition: Proven-node solver (`TrySolve`)

**What:** an exact minimax solver layered on top of the statistical search. When
a subtree's outcome becomes *certain* (a forced win or forced loss), the solver
marks it and propagates that certainty toward the root, so the search can trust
it absolutely instead of relying on visit statistics.

**Why:** plain MCTS only ever has *estimates*. But Hive games end concretely (a
queen is surrounded), and near the end of a game whole subtrees are forced. The
solver lets the engine (a) play a guaranteed mate immediately, (b) avoid a
guaranteed loss, and (c) stop searching early once the root's result is known.
It turns the tail of the game into perfect play.

**Where:** the `provenResult` field on `MCTSNode` (`include/MCTS.h`),
`MCTS::TrySolve` (`src/MCTS.cpp:188`), the terminal detection in expansion
(`:292-295`, `:332-335`), the solver call after backup (`:381-384`), the
`±max` short-circuit in `UCB1` (`:90-94`), and the proven-result handling in
`SelectBestMove` (§1).

**`provenResult` encoding.** `0` = unknown, `+1` = the player to move at this
node has a forced win, `-1` = a forced loss. Like every other value, it is in
the node's own side-to-move perspective.

**How certainty is created.** When expansion reaches a finished game
(`GameIsOver`), the node is marked terminal and its `provenResult` is set from
the board result *relative to the node's side to move* (`src/MCTS.cpp:292-295`
and `:332-335`):

```cpp
if (childState == BoardState::WhiteWins)
    node->provenResult = (board.currentColor == Color::White) ? 1 : -1;
```

(Typically the side to move at a just-finished node is the *loser* — their queen
was just surrounded — so this is usually `-1`, but it is always computed from the
state.)

**How certainty propagates (`TrySolve`, `:188-223`).** After a leaf with a known
result is backed up, we call `TrySolve(node->parent)`. The rule is pure negamax
minimax. A node's children carry results in the **opponent's** perspective, so:

- **Forced win** (`:199-207`): if **any** child has `provenResult == -1` (the
  opponent loses in that line), the player to move here can choose it → this
  node is a forced win (`+1`). This fires **early**, before the node is fully
  expanded — one winning reply is enough.
- **Forced loss** (`:217-222`): only if **all** children are proven `+1` (every
  reply wins for the opponent) **and** the node is fully expanded
  (`node->isExpanded`) — we must be sure there is no unexplored escape before
  declaring a loss.

When a node's result is set, `TrySolve` recurses to its parent, so a single
discovered mate can cascade all the way to the root. Once `root.provenResult`
becomes non-zero, the search loop breaks early (`src/MCTS.cpp:488`, `:522`).

**How the solver steers search and move choice.** A solved node short-circuits
selection scoring in `UCB1` (`:90-94`): a child that is a loss for the opponent
(`provenResult == -1`, i.e. a win for us) returns `+DBL_MAX` so we always steer
into it; a win for the opponent returns `-DBL_MAX` so we avoid it. And
`SelectBestMove` checks proven wins/losses first (§1).

---

## 4. Addition: Progressive bias in selection

**What:** an extra term in the UCB1 score that nudges selection toward moves the
static heuristic likes, weighted so the nudge is strong when a node is young and
fades as it accumulates visits.

**Why:** standard UCB1 explores purely by visit counts, which is slow to find
good moves in a wide tree (Hive can have many legal moves). Seeding selection
with cheap heuristic knowledge focuses early exploration on plausible moves
without permanently biasing the result — once a node has real statistics, the
bias washes out.

**Where:** the `bias` term in `MCTSNode::UCB1` (`src/MCTS.cpp:104`), the
`heuristicScore` field on `MCTSNode`, the constant `PROGRESSIVE_BIAS_WEIGHT`
(`include/MCTS.h:50`, value `5.0`), and where the score is computed at child
creation (`src/MCTS.cpp:315`).

```cpp
bias = PROGRESSIVE_BIAS_WEIGHT * heuristicScore / (visitCount + 1);
```

`heuristicScore ∈ [0, 1]` is computed once, when the child is created, by
`EvaluateMove(board, move, board.currentColor)` (`:315`) — i.e. the move's
quality from the perspective of the player making it. Because the bias is added
when the **parent** evaluates the child, and that perspective is the parent's
mover, a high `heuristicScore` correctly makes a move more attractive to the
parent — consistent with the `(1 - childValue)/2` exploitation term. The
`/(visitCount + 1)` denominator makes the term decay as the node is visited, so
it dominates only early and then yields to real search statistics. This is the
standard "progressive bias" technique.

The `exploration` term alongside it,
`EXPLORATION_C * sqrt(logParentVisits / visitCount)` with `EXPLORATION_C = 1.0`
(`include/MCTS.h:99`), is textbook UCB1.

---

## 5. Addition: Move ordering

**What:** legal moves at a node are sorted by the static heuristic before being
turned into children.

**Why:** since expansion adds one child per iteration (§1, Expansion), the
*order* in which moves are expanded matters — we want to try promising moves
first so the good lines get visits (and the solver gets a chance to find mates)
sooner. It also synergizes with the solver's early-win: a winning reply found
earlier propagates certainty earlier.

**Where:** `MCTS::OrderMoves` (`src/MCTS.cpp:168`), called right after move
generation (`:300`).

```cpp
std::sort(moves.begin(), moves.end(),
    [&](const Move& a, const Move& b) {
        return EvaluateMove(board, a, perspective) < EvaluateMove(board, b, perspective);
    });
```

The sort is **ascending**, deliberately: expansion pops moves from the **back**
of the vector (`unexpandedMoves.back()`, `:307`), so sorting worst-first puts the
**best move last**, where it is expanded first. `EvaluateMove` is the same cheap
per-move heuristic used for the progressive bias, evaluated from the moving
player's perspective (`board.currentColor`).

---

## 6. Addition: Instant-win check

**What:** a fast pre-search test, done once at the root, that detects a move
winning *immediately* (one that completes the surrounding of the opponent's
queen) and plays it without searching at all.

**Why:** it is wasteful (and occasionally risky, if the budget is small) to run
a full MCTS search when a move wins on the spot. This guarantees the engine
never overlooks a one-move win and returns instantly in that case.

**Where:** `MCTS::IsInstantWin` (`src/MCTS.cpp:116`), called in the pre-search
guard of both entry points (`src/MCTS.cpp:465-469` and `:509-513`):

```cpp
for (const Move& m : moves)
    if (IsInstantWin(rootBoard, m, rootColor))
        return m;
```

`IsInstantWin` finds the opponent's queen, quickly rejects any move that does not
end adjacent to (or on top of) it, then counts how many of the queen's six
neighbors would be occupied **after** the move — correctly accounting for the
fact that moving a piece *vacates* its source square
(`board.stackHeight[move.Source] <= 1`, `:157`). If all six are occupied, the
move wins and is returned immediately.

Note this is the only place `rootColor` (the actual player to move at the search
root) is still used; the search proper is entirely side-to-move-relative.

---

## File / symbol quick reference

| Concept | Location |
| --- | --- |
| Entry points | `MCTS::Search` `src/MCTS.cpp:453`, `MCTS::SearchIterations` `:497` |
| One iteration (4 phases) | `MCTS::RunIteration` `src/MCTS.cpp:233` |
| Selection loop | `src/MCTS.cpp:241-261` |
| UCB1 score (+ bias) | `MCTSNode::UCB1` `src/MCTS.cpp:85` |
| Expansion | `src/MCTS.cpp:279-338` |
| Leaf evaluation (GNN seam) | `src/MCTS.cpp:363` |
| Backup (negamax) | `src/MCTS.cpp:368-377` (and `:263-277`) |
| Final move choice | `MCTS::SelectBestMove` `src/MCTS.cpp:393` |
| Transposition table | `include/MCTS.h:25`, `src/MCTS.cpp:17-60` |
| Solver | `MCTS::TrySolve` `src/MCTS.cpp:188`; `provenResult` field |
| Move ordering | `MCTS::OrderMoves` `src/MCTS.cpp:168` |
| Instant-win | `MCTS::IsInstantWin` `src/MCTS.cpp:116` |
| Key constants | `EXPLORATION_C=1.0`, `PROGRESSIVE_BIAS_WEIGHT=5.0`, `TIME_CHECK_INTERVAL=128` (`include/MCTS.h`) |
