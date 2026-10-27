"""The practice game window (tkinter): tic-tac-toe against the minimax / alpha-beta player.

It is also a gentle way to practise pointing and clicking with the mouse."""
import tkinter as tk
from tkinter import ttk

import game


class PracticeWindow:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Practice game: tic-tac-toe")
        self.win.attributes("-topmost", True)
        self.win.resizable(False, False)
        self.score = {"win": 0, "draw": 0, "loss": 0}
        self.level = tk.StringVar(value="medium")
        self.algorithm = tk.StringVar(value="alphabeta+ordering")
        self.side = tk.StringVar(value="X")
        self.match = None
        self._build()
        self.new_game()

    def _build(self):
        w = self.win
        ttk.Label(w, text="Tic-tac-toe vs the minimax player", font=("Segoe UI", 12, "bold")).pack(pady=(8, 2))
        ttk.Label(w, text="Click a square. X always goes first.", foreground="#666").pack()
        opts = ttk.Frame(w)
        opts.pack(padx=10, pady=6, fill="x")
        ttk.Label(opts, text="Level").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opts, textvariable=self.level, values=list(game.LEVELS), width=8, state="readonly").grid(row=0, column=1, padx=4)
        ttk.Label(opts, text="Search").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Combobox(opts, textvariable=self.algorithm, values=list(game.ALGORITHMS), width=18, state="readonly").grid(row=0, column=3, padx=4)
        side = ttk.Frame(w)
        side.pack(padx=10)
        ttk.Radiobutton(side, text="I play X (I start)", variable=self.side, value="X", command=self.new_game).pack(side="left")
        ttk.Radiobutton(side, text="I play O (AI starts)", variable=self.side, value="O", command=self.new_game).pack(side="left", padx=10)

        grid = ttk.Frame(w)
        grid.pack(padx=10, pady=8)
        self.cells = []
        for i in range(9):
            b = tk.Button(grid, text=" ", width=4, height=1, font=("Segoe UI", 26, "bold"), command=lambda i=i: self.click(i))
            b.grid(row=i // 3, column=i % 3, padx=2, pady=2)
            self.cells.append(b)
        self._default_bg = self.cells[0].cget("bg")

        self.status = ttk.Label(w, text="", font=("Segoe UI", 10, "bold"))
        self.status.pack()
        self.info = ttk.Label(w, text="", foreground="#555", wraplength=330, justify="center")
        self.info.pack(padx=10, pady=(2, 4))
        self.board_score = ttk.Label(w, text="", foreground="#047857")
        self.board_score.pack()
        ttk.Button(w, text="New game", command=self.new_game).pack(pady=8)

    # ---------------------------------------------------------------- game flow
    def new_game(self):
        self.match = game.Match(self.side.get(), self.level.get(), self.algorithm.get())
        self.info.configure(text="")
        self.refresh()
        if self.match.turn == self.match.ai:
            self.win.after(200, self.ai_turn)

    def click(self, i):
        m = self.match
        if m.human_move(i):
            self.refresh()
            if not m.over:
                self.win.after(150, self.ai_turn)

    def ai_turn(self):
        m = self.match
        m.level, m.algorithm = self.level.get(), self.algorithm.get()
        c = m.ai_move()
        if c is not None:
            self.info.configure(
                text=f"I looked at {c.stats.nodes:,} positions using {c.stats.algorithm}"
                     + (f", skipping {c.stats.cutoffs:,} branches (alpha-beta pruning)." if c.stats.cutoffs else ".")
                     + (f" Looking {c.stats.limit} move{'s' if c.stats.limit != 1 else ''} ahead." if c.stats.limit else " Looking to the end of the game."))
        self.refresh()

    def refresh(self):
        m = self.match
        line = game.winning_line(m.board)
        for i, b in enumerate(self.cells):
            mark = m.board[i]
            b.configure(text=mark, state="normal" if mark == " " and not m.over else "disabled",
                        disabledforeground={"X": "#1a56db", "O": "#c2410c", " ": "#999"}[mark],
                        bg="#fde68a" if line and i in line else self._default_bg)
        if m.over:
            w = m.winner
            if w is None:
                self.status.configure(text="It's a draw.")
                self._record("draw")
            elif w == m.human:
                self.status.configure(text="You win!")
                self._record("win")
            else:
                self.status.configure(text="The AI wins.")
                self._record("loss")
        else:
            self.status.configure(text="Your move." if m.turn == m.human else "Thinking…")
        s = self.score
        self.board_score.configure(text=f"Wins {s['win']}   Draws {s['draw']}   Losses {s['loss']}")

    def _record(self, key):
        if not getattr(self.match, "_counted", False):
            self.match._counted = True
            self.score[key] += 1
