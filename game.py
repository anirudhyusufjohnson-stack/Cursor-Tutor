"""Adversarial search: minimax and alpha-beta pruning, shown on tic-tac-toe (the "Practice game").

Two players take turns. MAX (the player the AI plays for) wants the highest value, MIN (the opponent) wants the lowest.

  minimax      looks at every possible continuation and assumes both sides play perfectly
  alpha-beta   returns exactly the same value but skips branches that cannot change the answer:
               alpha = the best value MAX is already sure of, beta = the best MIN is already sure of;
               when alpha >= beta the rest of that branch is pruned
  depth limit  stops after a few moves and uses an evaluation function instead of playing to the end
               (this is how the easy and medium levels are made)

A position is a 9-character string such as "XO XO    " (index 0-8, left to right, top to bottom; space = empty).
Every search counts the positions it visits, so the algorithms can be compared."""
from dataclasses import dataclass, field

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
CENTRE_FIRST = (4, 0, 2, 6, 8, 1, 3, 5, 7)     # a good move order makes pruning work better
IN_ORDER = (0, 1, 2, 3, 4, 5, 6, 7, 8)
EMPTY = " " * 9
WIN = 10
LEVELS = {"easy": 1, "medium": 3, "hard": None}   # how many moves ahead the AI looks (None = to the end of the game)


def other(p):
    return "O" if p == "X" else "X"


def winner(b):
    for a, c, d in LINES:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return None


def is_full(b):
    return " " not in b


def is_over(b):
    return winner(b) is not None or is_full(b)


def legal_moves(b):
    return [i for i in range(9) if b[i] == " "]


def play(b, i, p):
    return b[:i] + p + b[i + 1:]


def to_move(b):
    """X always starts, so whose turn it is follows from how many marks are on the board."""
    return "X" if b.count("X") == b.count("O") else "O"


def evaluate(b, me):
    """Heuristic for an unfinished position: 1 point for a line with one of my marks and none of theirs,
    10 for two; the opponent's lines count against me."""
    score = 0
    opp = other(me)
    for line in LINES:
        cells = [b[i] for i in line]
        mine, theirs = cells.count(me), cells.count(opp)
        if mine and not theirs:
            score += 10 if mine == 2 else 1
        elif theirs and not mine:
            score -= 10 if theirs == 2 else 1
    return score


@dataclass
class Stats:
    nodes: int = 0          # positions visited
    cutoffs: int = 0        # branches skipped by alpha-beta
    algorithm: str = ""
    limit: int = None


@dataclass
class Choice:
    move: int
    value: float
    stats: Stats
    values: dict = field(default_factory=dict)   # value of each legal move (exact for minimax; bounds under pruning)


def _terminal_value(b, me, depth):
    w = winner(b)
    if w == me:
        return WIN - depth          # an earlier win is better
    if w is not None:
        return depth - WIN          # a later loss is better
    return 0


def minimax(b, turn, me, stats, depth=0, limit=None):
    """Plain minimax. Value from `me`'s point of view."""
    stats.nodes += 1
    if is_over(b):
        return _terminal_value(b, me, depth)
    if limit is not None and depth >= limit:
        return evaluate(b, me)
    if turn == me:     # MAX
        best = -10 ** 6
        for i in IN_ORDER:
            if b[i] == " ":
                best = max(best, minimax(play(b, i, turn), other(turn), me, stats, depth + 1, limit))
        return best
    best = 10 ** 6     # MIN
    for i in IN_ORDER:
        if b[i] == " ":
            best = min(best, minimax(play(b, i, turn), other(turn), me, stats, depth + 1, limit))
    return best


def alphabeta(b, turn, me, stats, alpha=-10 ** 6, beta=10 ** 6, depth=0, limit=None, order=IN_ORDER):
    """Minimax with alpha-beta pruning. Gives the same value as minimax, visiting fewer positions."""
    stats.nodes += 1
    if is_over(b):
        return _terminal_value(b, me, depth)
    if limit is not None and depth >= limit:
        return evaluate(b, me)
    if turn == me:
        value = -10 ** 6
        for i in order:
            if b[i] == " ":
                value = max(value, alphabeta(play(b, i, turn), other(turn), me, stats, alpha, beta, depth + 1, limit, order))
                if value >= beta:
                    stats.cutoffs += 1
                    return value
                alpha = max(alpha, value)
        return value
    value = 10 ** 6
    for i in order:
        if b[i] == " ":
            value = min(value, alphabeta(play(b, i, turn), other(turn), me, stats, alpha, beta, depth + 1, limit, order))
            if value <= alpha:
                stats.cutoffs += 1
                return value
            beta = min(beta, value)
    return value


ALGORITHMS = ("minimax", "alphabeta", "alphabeta+ordering")


def best_move(b, algorithm="alphabeta+ordering", limit=None, turn=None):
    """Choose the move for the player whose turn it is. Returns a Choice (move, value, stats, value of each move)."""
    turn = turn or to_move(b)
    me = turn
    stats = Stats(algorithm=algorithm, limit=limit)
    order = CENTRE_FIRST if algorithm == "alphabeta+ordering" else IN_ORDER
    best, best_v, values = None, -10 ** 6, {}
    alpha = -10 ** 6
    for i in order:
        if b[i] != " ":
            continue
        child = play(b, i, turn)
        if algorithm == "minimax":
            v = minimax(child, other(turn), me, stats, 1, limit)
        else:
            v = alphabeta(child, other(turn), me, stats, alpha, 10 ** 6, 1, limit, order)
            alpha = max(alpha, v)
        values[i] = v
        if v > best_v:
            best, best_v = i, v
    stats.nodes += 1    # the root itself
    return Choice(best, best_v, stats, values)


def ai_move(b, level="hard", algorithm="alphabeta+ordering"):
    """The practice-game opponent. easy = looks 1 move ahead, medium = 3, hard = plays perfectly."""
    return best_move(b, algorithm, LEVELS[level])


def compare(b, limit=None):
    """Run all three search methods on a position: {name: Choice}."""
    return {name: best_move(b, name, limit) for name in ALGORITHMS}


def result_text(b):
    w = winner(b)
    if w:
        return f"{w} wins"
    return "Draw" if is_full(b) else "Game on"


def winning_line(b):
    for line in LINES:
        a, c, d = line
        if b[a] != " " and b[a] == b[c] == b[d]:
            return line
    return None


class Match:
    """One game between a person and the AI (used by the practice window and the dashboard)."""

    def __init__(self, human="X", level="hard", algorithm="alphabeta+ordering"):
        self.board = EMPTY
        self.human, self.ai = human, other(human)
        self.level, self.algorithm = level, algorithm
        self.last = None             # the AI's latest Choice (so the number of positions searched can be shown)

    @property
    def turn(self):
        return to_move(self.board)

    @property
    def over(self):
        return is_over(self.board)

    @property
    def winner(self):
        return winner(self.board)

    def human_move(self, i):
        if self.over or self.turn != self.human or not (0 <= i < 9) or self.board[i] != " ":
            return False
        self.board = play(self.board, i, self.human)
        return True

    def ai_move(self):
        if self.over or self.turn != self.ai:
            return None
        self.last = ai_move(self.board, self.level, self.algorithm)
        self.board = play(self.board, self.last.move, self.ai)
        return self.last
