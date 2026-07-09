"""
Convertitore partite umane BoardSpace -> dataset JSONL per il training.

Legge i file SGF dell'archivio https://www.boardspace.net/hive/hivegames/
(anche direttamente dagli zip giornalieri), tiene solo le partite della
variante scelta (default hive-plm = il nostro Base+MLP), le rigioca dentro
HiveEngine e scrive una riga JSONL per posizione, nello stesso identico
formato prodotto da SelfPlay:

    {"game_id":0,"ply":12,"side_to_move":"White","z":1,"x":[...],
     "edge_index":[...],"edge_attr":[...],"u":[...]}

Uso tipico (gli zip scaricati da archive-YYYY/ possono stare in una cartella):

    python3 scripts/boardspace_to_jsonl.py data/boardspace/zips \
        --output data/boardspace_2025.jsonl

Perche' il replay nell'engine invece di parsare le mosse in Python:
  1. le feature escono da BoardEncoder (comando UHP `features`), lo stesso
     encoder del SelfPlay: zero rischio di incoerenze tra dati umani e
     dati self-play;
  2. l'engine valida ogni mossa: se una partita e' corrotta o la traduzione
     e' sbagliata, `play` risponde `invalidmove` e la partita viene scartata;
  3. se il replay raggiunge uno stato terminale (regina circondata), il
     label z viene dal verdetto dell'engine, piu' affidabile del referto
     SGF (che e' testo localizzato nella lingua del client).

Per le partite finite senza stato terminale (abbandono, timeout) il vincitore
viene dal campo RE[...] cercando il nome del giocatore nel testo: funziona in
qualsiasi lingua, tranne quando il client localizza anche il nome "guest"
(caso raro, ~1%%: la partita viene scartata se nemmeno il replay decide).

Sottigliezze del formato BoardSpace scoperte sui dati veri:
  - esistono due dialetti: quello recente ("Dropb wG1 O 15 wL/", pezzi con
    nome completo, referto RE presente) e quello degli archivi vecchi
    (~2013 e prima: comandi minuscoli, pezzi SENZA colore -> l'identita'
    arriva dal token colore di move/pmove o dal pick/pickb precedente,
    salite scritte come riferimento nudo senza punto, NESSUN campo RE ->
    l'esito puo' venire solo dal replay);
  - il campo SU della variante ha attraversato tre epoche: "hive-plm"
    (fino al ~2018), "Hive-PLM" (~2019-2023), "hive-plm 2 0 101" (attuale)
    -> il confronto e' case-insensitive e si ferma a spazio o quadra;
  - un turno e' confermato solo dal comando `Done`: se il giocatore posa un
    pezzo, ci ripensa e lo riposa altrove, nel file compaiono piu' `Dropb`
    consecutivi -> conta solo l'ultimo prima del `Done`;
  - un turno fatto di SOLO `Done`, senza mossa ne' `Pass`, e' un passo
    (alcuni client lo registrano cosi') -> lo si riconosce controllando che
    quel `Done` appartenga al giocatore che deve muovere;
  - i client piu' recenti i pass forzati NON li registrano affatto -> se
    l'engine rifiuta una mossa e dichiara che l'unica mossa legale e' pass,
    il pass (obbligato dalle regole, nessuna informazione persa) viene
    inserito dal convertitore e la mossa ritentata;
  - le offerte di patta (OfferDraw/DeclineDraw/AcceptDraw) hanno un proprio
    `Done` di conferma ma non consumano il turno di gioco;
  - il riferimento `.` da solo puo' comparire anche a meta' partita: le
    coordinate di griglia (colonna/riga) sono l'informazione autorevole e il
    riferimento relativo a volte viene omesso. Tracciando le posizioni di
    griglia mossa per mossa lo si ricostruisce: cella occupata -> salita
    sopra chi sta in cima; cella vuota -> riferimento direzionale costruito
    da un vicino occupato (le 6 direzioni esagonali corrispondono a offset
    fissi di colonna/riga, vedi DOT_DIRECTIONS).

Notazione mosse BoardSpace -> UHP (l'ultimo campo di Dropb/Move e' gia'
quasi-UHP):
  - `.` da solo        -> prima pietra, oppure salita su una cella (v. sopra)
  - `wL.` (punto dopo) -> salita sopra wL: riferimento senza direzione
  - `\\\\` -> `\\` (escaping SGF), `wL1/wM1/wP1/wQ1` -> `wL/wM/wP/wQ`
    (i pezzi unici in UHP non hanno numero)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

# ── Parsing SGF ──────────────────────────────────────────────────────────────

# Una riga di gioco: "; P0[7 Dropb wG1 O 14 wL/]TM[20849]"
# Cattura anche chi la gioca (0/1): serve a riconoscere i pass impliciti.
# Lo spazio dopo il comando e' opzionale: il formato vecchio scrive "done]"
# attaccato, quello nuovo "Done ]" con lo spazio.
MOVE_LINE_RE = re.compile(r";\s*P([01])\[\d+ (\w+) ?([^\]]*)\]")

# I comandi che propongono una mossa (confermata poi dal Done). Start e'
# rumore dell'interfaccia. PMove/Pdropb sono le mosse fatte col potere del
# pillbug (anche di pezzi avversari): la notazione finale e' identica a
# Move/Dropb. I confronti sono in minuscolo: gli archivi vecchi (~2013)
# scrivono "move"/"dropb"/"done", quelli recenti "Move"/"Dropb"/"Done".
DROP_COMMANDS = {"dropb", "pdropb"}   # args: <pezzo> <col> <riga> <rif>
MOVE_COMMANDS = {"move", "pmove"}     # args: <colore> <pezzo> <col> <riga> <rif>

# Azioni di cortesia confermate da un proprio Done ma che NON consumano il
# turno di gioco (verificato sui dati: dopo un OfferDraw+Done e il relativo
# DeclineDraw+Done, chi aveva offerto muove normalmente).
NOOP_COMMANDS = {"offerdraw", "declinedraw", "acceptdraw"}

# Un nome pezzo completo in stile moderno/UHP: colore minuscolo + tipo
# maiuscolo + eventuale numero (wB1, bQ, wL1). Gli archivi vecchi invece
# scrivono il pezzo SENZA colore ("dropb s1 ...", "move W L1 ..."): li'
# il colore arriva dal token separato (move/pmove) o dal pick precedente.
FULL_PIECE_RE = re.compile(r"[wb][A-Z]\d?")


def piece_with_color(color_token: str, piece_token: str) -> str:
    """Nome UHP del pezzo dati il token colore ('W'/'b') e il token pezzo.

    Se il token pezzo e' gia' un nome completo (formato moderno) il colore
    e' ridondante; altrimenti (formato vecchio: 'L1', 'q') lo si costruisce.
    """
    if FULL_PIECE_RE.fullmatch(piece_token):
        return normalize_piece(piece_token)
    return normalize_piece(color_token.lower() + piece_token.upper())

# Parole per riconoscere una patta nel referto RE, nelle lingue viste
# nell'archivio (inglese, francese, tedesco, russo, italiano, spagnolo).
DRAW_MARKERS = ("draw", "nulle", "remis", "ничья", "patta", "tablas")

# Geometria della griglia BoardSpace: la cella (colonna, riga) ha 6 vicini
# esagonali a offset fissi. La mappa da' l'offset (dcol, driga) del TARGET
# rispetto al riferimento -> come si scrive il riferimento in UHP.
# Dedotta e verificata sui dati: es. bA1 in M12 riferito a wL in N13 e'
# "/wL", cioe' offset (-1,-1) = prefisso "/".
DOT_DIRECTIONS = {
    (+1, +1): "{}/",
    (-1, -1): "/{}",
    (+1, 0): "{}-",
    (-1, 0): "-{}",
    (0, -1): "{}\\",
    (0, +1): "\\{}",
}


def ref_from_neighbors(piece: str, cell: tuple, stacks: dict) -> str | None:
    """Costruisce il riferimento UHP di una cella vuota da un vicino occupato.

    Serve per i riferimenti '.' omessi da BoardSpace: si cerca una cella
    adiacente occupata e si descrive il target rispetto al pezzo in cima
    (saltando il pezzo che sta muovendo, che non puo' riferirsi a se stesso).
    """
    col, row = cell
    if len(col) != 1 or not row.isdigit():
        return None
    for (dc, dr), fmt in DOT_DIRECTIONS.items():
        neighbor = (chr(ord(col) - dc), str(int(row) - dr))
        for p in reversed(stacks.get(neighbor, [])):  # dalla cima al fondo
            if p != piece:
                return fmt.format(p)
    return None


def parse_sgf(text: str) -> dict:
    """Estrae variante, giocatori, referto e mosse (in notazione UHP).

    Un turno e' confermato solo dal `Done`: le azioni Dropb/Move restano "in
    sospeso" e conta l'ultima prima del Done (il giocatore puo' riposizionare
    il pezzo piu' volte prima di confermare). Per risolvere i riferimenti `.`
    (salita su una cella occupata) si tracciano le posizioni di griglia.
    """
    # Il campo puo' essere "SU[hive-plm 2 0 101]" (recente) o "SU[hive-plm]"
    # (archivi vecchi): la variante finisce al primo spazio O alla quadra.
    variant = re.search(r"SU\[([^\s\]]+)", text)
    p0 = re.search(r'P0\[id "(.*?)"\]', text)
    p1 = re.search(r'P1\[id "(.*?)"\]', text)
    result = re.search(r"RE\[(.*?)\]", text)

    moves: list[str] = []
    resigned = False
    parse_error = None
    pending = None            # (pezzo, cella, riferimento) in attesa del Done
    picked = None             # pezzo preso dall'ultimo pick/pickb (formato vecchio)
    stacks: dict = {}         # cella di griglia -> pila di pezzi (fondo->cima)
    where: dict = {}          # pezzo -> cella in cui si trova

    for player, command, args in MOVE_LINE_RE.findall(text):
        command = command.lower()
        fields = args.split()

        if command == "resign":
            resigned = True
            break

        if command == "pass":
            pending = ("pass", None, None)
            continue

        if command in NOOP_COMMANDS:
            pending = ("noop", None, None)
            continue

        if command == "pick":
            # <colore> <slot> <pezzo>: presa dalla mano. Nel formato vecchio
            # e' l'unico punto in cui il colore del pezzo e' esplicito.
            picked = piece_with_color(fields[0], fields[-1])
            continue

        if command == "pickb":
            # <col> <riga> <pezzo>: presa dalla board. Nel formato vecchio il
            # token pezzo non ha colore: l'identita' vera e' chi sta in cima
            # alla cella, che conosciamo dal tracciamento della griglia — o,
            # se la cella e' quella di una posa non ancora confermata dal
            # Done, e' quel pezzo (ripensamento: posa, riprende, riposa).
            token = fields[-1]
            cell = (fields[0], fields[1])
            if FULL_PIECE_RE.fullmatch(token):
                picked = normalize_piece(token)
            elif pending is not None and pending[1] == cell:
                picked = pending[0]
            else:
                stack = stacks.get(cell)
                picked = stack[-1] if stack else None
            continue

        if command in DROP_COMMANDS:
            # <pezzo> <col> <riga> <rif>: nel formato vecchio il token pezzo
            # e' senza colore ("s1") -> vale l'identita' del pick precedente.
            token = fields[0]
            piece = normalize_piece(token) if FULL_PIECE_RE.fullmatch(token) else picked
            if piece is None:
                parse_error = f"dropb di '{token}' senza pick precedente"
                break
            pending = (piece, (fields[1], fields[2]), fields[-1])
            continue

        if command in MOVE_COMMANDS:
            # <colore> <pezzo> <col> <riga> <rif>
            pending = (piece_with_color(fields[0], fields[1]),
                       (fields[2], fields[3]), fields[-1])
            continue

        if command != "done":
            continue  # Start e simili: rumore dell'interfaccia

        if pending is None:
            # Done senza nessuna azione prima: se e' del giocatore che deve
            # muovere (P0 ai ply pari) e' un passo implicito, altrimenti rumore.
            if int(player) == len(moves) % 2:
                moves.append("pass")
            continue

        piece, cell, ref = pending
        pending = None

        if piece == "noop":
            continue  # Done che confermava un'offerta di patta, non una mossa

        if piece == "pass":
            moves.append("pass")
            continue

        if ref == "." and moves:
            # Riferimento omesso: le coordinate di griglia sono autorevoli.
            top = next((p for p in reversed(stacks.get(cell, [])) if p != piece), None)
            if top is not None:
                moves.append(f"{piece} {top}")           # salita sulla cella
            else:
                built = ref_from_neighbors(piece, cell, stacks)
                if built is None:
                    parse_error = f"riferimento '.' non risolvibile: {piece} in {cell}"
                    break
                moves.append(f"{piece} {built}")
        else:
            moves.append(to_uhp_move(piece, ref))

        # Aggiorna il tracciamento griglia (serve solo per i riferimenti '.').
        if piece in where:
            stacks[where[piece]].remove(piece)
        stacks.setdefault(cell, []).append(piece)
        where[piece] = cell

    return {
        # In minuscolo: negli anni il campo e' stato "hive-plm", "Hive-PLM"
        # e "hive-plm 2 0 101" (la parte numerica e' gia' esclusa dalla regex).
        "variant": variant.group(1).lower() if variant else "",
        "p0": p0.group(1) if p0 else "",
        "p1": p1.group(1) if p1 else "",
        "result": result.group(1) if result else "",
        "moves": moves,
        "resigned": resigned,
        "parse_error": parse_error,
    }


def normalize_piece(piece: str) -> str:
    """wL1 -> wL: in UHP i pezzi unici (Q, L, M, P) non hanno il numero."""
    return re.sub(r"^([wb])([QLMP])1$", r"\1\2", piece)


def to_uhp_move(piece: str, ref: str) -> str:
    piece = normalize_piece(piece)
    ref = ref.replace("\\\\", "\\")  # escaping SGF del backslash

    if ref == ".":
        return piece                       # prima pietra della partita
    if ref.endswith("."):
        return f"{piece} {normalize_piece(ref[:-1])}"  # salita sopra

    # Riferimento direzionale: -X, /X, \X, X-, X/, X\
    m = re.match(r"^([-/\\]?)(\w+)([-/\\]?)$", ref)
    if m is None:
        return f"{piece} {ref}"  # lascia decidere all'engine (scartera' lui)
    return f"{piece} {m.group(1)}{normalize_piece(m.group(2))}{m.group(3)}"


def winner_from_result(game: dict) -> str | None:
    """'White', 'Black', 'Draw' oppure None se il referto non e' decifrabile.

    P0 e' sempre il primo a muovere, quindi il Bianco. Il testo di RE e'
    localizzato ma contiene il nome del vincitore: se compaiono entrambi i
    nomi (es. "guest" e "guest2") vince il match piu' lungo; se i due match
    sono pari (tipico di guest contro guest) il referto e' indecifrabile.
    """
    result = game["result"]
    hits = [(len(name), color)
            for name, color in ((game["p0"], "White"), (game["p1"], "Black"))
            if name and name in result]
    if len(hits) == 2 and hits[0][0] == hits[1][0]:
        return None  # stessi nomi (o stessa lunghezza): impossibile decidere
    if hits:
        return max(hits)[1]
    if any(marker in result.lower() for marker in DRAW_MARKERS):
        return "Draw"
    return None


# ── Dialogo con HiveEngine ───────────────────────────────────────────────────

class EngineClient:
    """Un processo HiveEngine riusato per tutte le partite (il modello viene
    caricato una volta sola)."""

    def __init__(self, binary: str, model: str | None):
        cmd = [binary] + ([model] if model else [])
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        # L'engine stampa da solo il banner info all'avvio: va solo letto,
        # senza mandare comandi, altrimenti le risposte si sfasano di uno.
        self._read_until_ok("banner di avvio")

    def send(self, command: str) -> list[str]:
        """Manda un comando e restituisce le righe di risposta prima di 'ok'."""
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        return self._read_until_ok(command)

    def _read_until_ok(self, context: str) -> list[str]:
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"HiveEngine terminato durante: {context}")
            line = line.rstrip("\n")
            if line == "ok":
                return lines
            lines.append(line)

    def close(self) -> None:
        self.proc.stdin.write("exit\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=10)


def game_state(game_string: str) -> str:
    """Secondo campo della GameString: NotStarted/InProgress/Draw/WhiteWins/BlackWins."""
    parts = game_string.split(";")
    return parts[1] if len(parts) > 1 else ""


# ── Replay di una partita ────────────────────────────────────────────────────

def convert_game(engine: EngineClient, game: dict, game_id: int,
                 out) -> tuple[int, str] | str:
    """Rigioca la partita e scrive le sue posizioni su `out`.

    Ritorna (numero posizioni, verdetto) se convertita, altrimenti una
    stringa con il motivo dello scarto.
    """
    engine.send("newgame Base+MLP")

    records = []  # (ply, side_to_move, json delle feature)
    state = "NotStarted"
    ply = 0
    inserted_passes = 0  # pass forzati consecutivi inseriti da noi
    move_index = 0
    while move_index < len(game["moves"]):
        move = game["moves"][move_index]

        # Posizione PRIMA della mossa, come nel SelfPlay. La board vuota di
        # ply 0 viene saltata (grafo senza nodi, il pooling e' mal definito).
        if ply > 0:
            features = engine.send("features")[0]
            side = "White" if ply % 2 == 0 else "Black"
            records.append((ply, side, features))

        reply = engine.send("play " + move)[0]
        if reply.startswith("invalidmove") or reply.startswith("err"):
            # I client BoardSpace piu' recenti non registrano i pass forzati.
            # Se l'engine dice che l'unica mossa legale e' pass, il pass e'
            # obbligato dalle regole: lo inseriamo e ritentiamo la mossa.
            if inserted_passes < 2 and engine.send("validmoves")[0] == "pass":
                engine.send("play pass")
                inserted_passes += 1
                ply += 1  # il pass consuma un ply (la sua posizione e' gia' registrata)
                continue
            return f"mossa rifiutata: '{move}' al ply {ply} ({reply})"

        inserted_passes = 0
        ply += 1
        move_index += 1

        state = game_state(reply)
        if state in ("WhiteWins", "BlackWins", "Draw"):
            break  # partita decisa sulla board: da qui fa fede l'engine

    # Verdetto: engine se terminale, altrimenti referto SGF. Se entrambi
    # esistono e non concordano, la traduzione della partita potrebbe essere
    # sbagliata in modo silenzioso: va segnalato.
    re_winner = winner_from_result(game)
    if state in ("WhiteWins", "BlackWins", "Draw"):
        winner = state.removesuffix("Wins")  # "White"/"Black"/"Draw"
        verdict = "engine"
        if re_winner is not None and re_winner != winner:
            return f"disaccordo engine/referto: {winner} vs {re_winner}"
    else:
        if re_winner is None:
            return "esito non decifrabile: RE ambiguo e replay non terminale"
        winner = re_winner
        verdict = "referto"

    for ply, side, features in records:
        z = 0 if winner == "Draw" else (1 if side == winner else -1)
        # I metadati vanno in testa all'oggetto {"x":...} gia' pronto,
        # subito dopo la graffa di apertura: stesso trucco del SelfPlay.
        out.write(f'{{"game_id":{game_id},"ply":{ply},"side_to_move":"{side}",'
                  f'"z":{z},{features[1:]}\n')
    return (len(records), verdict)


# ── Lettura input (file .sgf, zip, cartelle) ─────────────────────────────────

def iter_sgf_texts(paths: list[str]):
    """Genera (nome, testo) per ogni .sgf trovato nei path (anche dentro zip)."""
    for path in map(Path, paths):
        if path.is_dir():
            files = sorted(list(path.rglob("*.sgf")) + list(path.rglob("*.zip")))
        else:
            files = [path]
        for f in files:
            if f.suffix == ".zip":
                with zipfile.ZipFile(f) as z:
                    for name in sorted(z.namelist()):
                        if name.endswith(".sgf"):
                            yield f"{f.name}/{name}", z.read(name).decode("utf-8", "replace")
            else:
                yield f.name, f.read_text(encoding="utf-8", errors="replace")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Converte partite BoardSpace (SGF) in JSONL di training")
    parser.add_argument("input", nargs="+", help="File .sgf, zip di BoardSpace o cartelle che li contengono")
    parser.add_argument("--output", type=str, required=True, help="File JSONL di destinazione (append)")
    parser.add_argument("--engine", type=str, default="build/HiveEngine")
    parser.add_argument("--model", type=str, default=None,
                        help="Path del .pt per l'engine (default: variabile HIVE_VALUE_NETWORK)")
    parser.add_argument("--variant", type=str, default="hive-plm",
                        help="Variante BoardSpace da tenere (hive-plm = Base+MLP)")
    parser.add_argument("--min-plies", type=int, default=8,
                        help="Scarta partite piu' corte di cosi' (pochissimo segnale)")
    parser.add_argument("--game-id-start", type=int, default=0,
                        help="game_id della prima partita convertita (crescono di 1)")
    args = parser.parse_args()

    engine = EngineClient(args.engine, args.model)
    skipped: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    converted = positions = 0
    game_id = args.game_id_start

    with open(args.output, "a") as out:
        for name, text in iter_sgf_texts(args.input):
            game = parse_sgf(text)
            if game["variant"] != args.variant:
                skipped["altra variante"] += 1
                continue
            if game["parse_error"] is not None:
                skipped["errore di parsing"] += 1
                print(f"  scartata {name}: {game['parse_error']}", file=sys.stderr)
                continue
            if len(game["moves"]) < args.min_plies:
                skipped["troppo corta"] += 1
                continue

            outcome = convert_game(engine, game, game_id, out)
            if isinstance(outcome, str):
                skipped[outcome.split(":")[0]] += 1  # il motivo prima dei dettagli
                print(f"  scartata {name}: {outcome}", file=sys.stderr)
                continue

            n_records, verdict = outcome
            verdicts[verdict] += 1
            converted += 1
            positions += n_records
            game_id += 1
            if converted % 200 == 0:
                print(f"  ...{converted} partite convertite ({positions} posizioni)", file=sys.stderr)

    engine.close()

    print(f"\nConvertite {converted} partite ({positions} posizioni) -> {args.output}")
    print(f"Verdetto dal replay dell'engine: {verdicts['engine']}, dal referto SGF: {verdicts['referto']}")
    if skipped:
        print("Scartate:")
        for reason, count in skipped.most_common():
            print(f"  {count:>5}  {reason}")
    print(f"game_id assegnati: da {args.game_id_start} a {game_id - 1}")


if __name__ == "__main__":
    main()
