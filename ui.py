"""Floating always-on-top chat window (tkinter) that ties everything together."""
import math
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import scrolledtext, ttk

import pyautogui
import pyperclip

import db
from ai_tutor import DEFAULT_QUESTION
from cursor_context import capture_context, window_under_point
from hover import HoverWatcher
from practice_ui import PracticeWindow

DRAFT_RE = re.compile(r"<draft>(.*?)</draft>", re.S)
DASHBOARD_URL = "http://localhost:8501"


def render(text):
    """Turn <draft> tags into something friendly for the chat view."""
    text = re.sub(r"</?d?r?a?f?t?$", "", text)  # hide half-streamed tag at the very end
    return text.replace("<draft>", "\n✉ Draft message:\n").replace("</draft>", "\n")


def speak(text):
    def run():
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


class TutorApp:
    def __init__(self, cfg, tutor):
        self.cfg = cfg
        self.tutor = tutor
        self.ui_queue = queue.Queue()
        self._lock = threading.Lock()  # held while the AI is answering
        self.overlay_box = (0, 0, 0, 0)
        self.last_external_pos = tuple(pyautogui.position())
        self.last_ctx = None
        self.last_draft = ""
        self.last_auto = None  # (pos, time) of last automatic explanation
        self._dash_proc = None

        self.root = tk.Tk()
        self.root.title("Cursor Tutor")
        self.root.attributes("-topmost", True)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"410x680+{max(sw - 430, 0)}+{max(sh - 760, 0)}")
        self.auto_var = tk.BooleanVar(value=True)
        self.speak_var = tk.BooleanVar(value=False)
        self._build()
        self.root.bind("<Configure>", self._on_configure)
        self.root.after(50, self._drain)

        self.watcher = HoverWatcher(
            cfg.dwell_seconds,
            in_overlay=self.in_overlay,
            on_move=self._on_move,
            on_dwell=self.on_dwell,
            enabled=lambda: self.auto_var.get(),
        )

    # ---------- layout ----------
    def _build(self):
        ttk.Label(self.root, text="Cursor Tutor", font=("Segoe UI", 14, "bold")).pack(pady=(8, 0))
        ttk.Label(
            self.root,
            text="Hover over anything and keep still, or press "
            + self.cfg.hotkey.replace("<", "").replace(">", "").title(),
            foreground="#666",
            wraplength=370,
        ).pack()

        self.chat = scrolledtext.ScrolledText(self.root, wrap="word", state="disabled", height=20, font=("Segoe UI", 10))
        self.chat.pack(fill="both", expand=True, padx=8, pady=8)
        self.chat.tag_configure("user", foreground="#1a56db", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("label", foreground="#047857", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("tutor", foreground="#111")
        self.chat.tag_configure("sys", foreground="#888", font=("Segoe UI", 9, "italic"))

        opts = ttk.Frame(self.root)
        opts.pack(fill="x", padx=8)
        ttk.Checkbutton(opts, text="Auto-explain when I hover", variable=self.auto_var).pack(anchor="w")
        ttk.Checkbutton(opts, text="Speak answers aloud", variable=self.speak_var).pack(anchor="w")

        btns = ttk.Frame(self.root)
        btns.pack(fill="x", padx=8, pady=6)
        ttk.Button(btns, text="Explain what's under my cursor", command=self.explain_here).pack(side="left", expand=True, fill="x")
        ttk.Button(btns, text="New chat", command=self.new_chat).pack(side="left", padx=(6, 0))

        btns2 = ttk.Frame(self.root)
        btns2.pack(fill="x", padx=8)
        self.draft_btn = ttk.Button(btns2, text="Type draft for me", command=self.type_draft, state="disabled")
        self.draft_btn.pack(side="left", expand=True, fill="x")
        ttk.Button(btns2, text="Open dashboard", command=self.open_dashboard).pack(side="left", padx=(6, 0))
        btns3 = ttk.Frame(self.root)
        btns3.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Button(btns3, text="Teach this item", command=self.teach_dialog).pack(side="left", expand=True, fill="x")
        ttk.Button(btns3, text="Practice game", command=self.practice_game).pack(side="left", padx=(6, 0))

        row = ttk.Frame(self.root)
        row.pack(fill="x", padx=8, pady=8)
        self.entry = ttk.Entry(row)
        self.entry.pack(side="left", expand=True, fill="x")
        self.entry.bind("<Return>", lambda e: self.send_from_entry())
        ttk.Button(row, text="Send", command=self.send_from_entry).pack(side="left", padx=(6, 0))

        self.status = ttk.Label(self.root, text="Ready.", foreground="#666")
        self.status.pack(fill="x", padx=8, pady=(0, 6))
        self.append("Hi! Point your mouse at any app, icon or button and keep it still for a moment. "
                    "I'll explain it. You can also type questions below.\n"
                    "Everything runs on your computer: no internet, no account, nothing is uploaded.\n"
                    "I decide with logic (not a trained model): ask me \"help\" to see what I can do.\n\n", "sys")

    # ---------- thread-safe plumbing ----------
    def post(self, fn):
        self.ui_queue.put(fn)

    def _drain(self):
        try:
            while True:
                self.ui_queue.get_nowait()()
        except queue.Empty:
            pass
        self.root.after(50, self._drain)

    def _on_configure(self, event):
        if event.widget is self.root:
            x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
            self.overlay_box = (x, y, x + self.root.winfo_width(), y + self.root.winfo_height())

    def in_overlay(self, pos):
        x0, y0, x1, y1 = self.overlay_box
        return x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1

    def _on_move(self, pos):
        self.last_external_pos = pos

    # ---------- chat view helpers (UI thread only) ----------
    def _write(self, fn):
        self.chat.configure(state="normal")
        fn()
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def append(self, text, tag="tutor"):
        self._write(lambda: self.chat.insert("end-1c", text, tag))

    def set_status(self, text):
        self.status.configure(text=text)

    def begin_tutor_msg(self):
        def go():
            self.chat.insert("end-1c", "Tutor: ", "label")
            self.chat.mark_set("tstart", "end-1c")
            self.chat.mark_gravity("tstart", "left")

        self._write(go)
        self.set_status("Thinking…")

    def update_tutor_msg(self, text):
        def go():
            self.chat.delete("tstart", "end-1c")
            self.chat.insert("end-1c", render(text), "tutor")

        self._write(go)

    def _finish(self, full):
        self.append("\n\n")
        self.set_status("Ready.")
        m = DRAFT_RE.search(full)
        if m:
            self.last_draft = m.group(1).strip()
            self.draft_btn.state(["!disabled"])
        if self.speak_var.get():
            speak(re.sub(r"[*#_`]", "", DRAFT_RE.sub("", full)))

    # ---------- actions ----------
    def start_ask(self, text, capture, source):
        """Run one AI request in a background thread. Safe to call from any thread."""
        if not self._lock.acquire(blocking=False):
            self.post(lambda: self.set_status("Still thinking… try again in a moment."))
            return

        def work():
            try:
                ctx = None
                if capture:
                    ctx = capture_context(self.last_external_pos, use_uia=self.cfg.use_uia)
                    previous = self.last_ctx
                    self.last_ctx = ctx
                else:
                    previous = self.last_ctx
                self.post(self.begin_tutor_msg)
                t0 = time.time()
                trigger = "hover" if source == "hover" else "ask"   # the agent stays quiet on private windows only for hover
                full = self.tutor.ask(text, ctx, on_chunk=lambda t: self.post(lambda t=t: self.update_tutor_msg(t)),
                                      trigger=trigger)
                seconds = time.time() - t0
                self.post(lambda: self._finish(full))
                try:  # logging must never break the chat
                    db.log_interaction(source, text, full, seconds, self.tutor.last_used,
                                       ctx=ctx, last_ctx=previous, rec=self.tutor.last_recognition)
                except Exception:
                    pass
            except Exception as e:  # unexpected problem: show it instead of crashing
                self.post(lambda e=e: (self.append(f"\n⚠ {e}\n\n", "sys"), self.set_status("Ready.")))
            finally:
                self._lock.release()

        threading.Thread(target=work, daemon=True).start()

    def explain_here(self, source="button"):
        if self._lock.locked():
            self.set_status("Still thinking… try again in a moment.")
            return
        self.append("\U0001F440 Looking at what's under your cursor…\n", "sys")
        self.start_ask(DEFAULT_QUESTION, capture=True, source=source)

    def send_from_entry(self):
        text = self.entry.get().strip()
        if not text:
            return
        if self._lock.locked():
            self.set_status("Still thinking… try again in a moment.")
            return
        self.entry.delete(0, "end")
        self.append(f"You: {text}\n", "user")
        # First message with no screenshot yet -> look at the screen; later ones are plain chat.
        self.start_ask(text, capture=self.last_ctx is None, source="chat")

    def new_chat(self):
        self.tutor.reset()
        self.last_ctx = None
        self.last_draft = ""
        self.draft_btn.state(["disabled"])
        self._write(lambda: self.chat.delete("1.0", "end"))
        self.set_status("New chat started.")

    def on_dwell(self, pos):
        """Called from the hover thread when the mouse has been still for a while."""
        if self._lock.locked():
            return
        now = time.time()
        if self.last_auto and math.dist(pos, self.last_auto[0]) < 60 and now - self.last_auto[1] < 30:
            return  # already explained this spot recently
        title, proc = window_under_point(*pos)
        if self.tutor.should_pause(proc, title):    # the logical agent derives Act_Pause for private-looking windows
            self.post(lambda: self.set_status("Auto-explain paused here (looks sensitive). Use the hotkey if you want help."))
            return
        self.last_auto = (pos, now)
        self.post(lambda: self.append("\U0001F440 Looking at what's under your cursor…\n", "sys"))
        self.start_ask(DEFAULT_QUESTION, capture=True, source="hover")

    def type_draft(self):
        if not self.last_draft:
            return
        self.set_status("Click the message box in the other app now… pasting in 4 seconds.")
        self.root.after(4000, self._paste)

    def _paste(self):
        pyperclip.copy(self.last_draft)
        pyautogui.hotkey("command" if platform.system() == "Darwin" else "ctrl", "v")
        self.set_status("Pasted. Check it, then press Send in the app yourself.")

    def teach_dialog(self):
        """A small form to teach the app something new about what was under the cursor."""
        if self.last_ctx is None:
            self.set_status("Hover over something first, wait for my answer, then press Teach.")
            return
        win = tk.Toplevel(self.root)
        win.title("Teach Cursor Tutor")
        win.attributes("-topmost", True)
        win.geometry("+%d+%d" % (self.root.winfo_rootx() - 360, self.root.winfo_rooty() + 60))
        pad = {"padx": 10, "pady": 4}
        ttk.Label(win, text="I will remember the icon or text I just looked at.", foreground="#555").pack(anchor="w", **pad)
        ttk.Label(win, text="Name (what is it called?)").pack(anchor="w", **pad)
        name_var = tk.StringVar(value=self.tutor.suggested_name())
        ttk.Entry(win, textvariable=name_var, width=44).pack(**pad)
        kind_var = tk.StringVar(value="item")
        row = ttk.Frame(win)
        row.pack(anchor="w", **pad)
        ttk.Radiobutton(row, text="A button / item", variable=kind_var, value="item").pack(side="left")
        ttk.Radiobutton(row, text="A whole app", variable=kind_var, value="app").pack(side="left", padx=12)
        ttk.Label(win, text="What does it do?").pack(anchor="w", **pad)
        what = tk.Text(win, width=44, height=3, wrap="word")
        what.pack(**pad)
        ttk.Label(win, text="How do you use it? (one step per line)").pack(anchor="w", **pad)
        steps = tk.Text(win, width=44, height=5, wrap="word")
        steps.pack(**pad)

        def save():
            name = name_var.get().strip()
            desc = what.get("1.0", "end").strip()
            lines = [ln.strip() for ln in steps.get("1.0", "end").splitlines() if ln.strip()]
            if not name or not desc:
                self.set_status("Please fill in the name and what it does.")
                return
            self.tutor.teach(name, kind_var.get(), desc, lines or ["Click it."], self.last_ctx)
            self.append(f'\u2713 Learned "{name}". Hover over it again and I will explain it.\n\n', "sys")
            self.set_status("Learned.")
            win.destroy()

        ttk.Button(win, text="Save", command=save).pack(pady=10)

    def practice_game(self):
        """Tic-tac-toe against the minimax / alpha-beta player (a window of its own)."""
        PracticeWindow(self.root)

    def open_dashboard(self):
        """Start the Streamlit dashboard (free, runs locally) and open it in the browser."""
        if self._dash_proc is not None and self._dash_proc.poll() is None:
            webbrowser.open(DASHBOARD_URL)
            return
        try:
            self._dash_proc = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("dashboard.py")),
                 "--server.port", "8501", "--server.headless", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.set_status("Starting dashboard… it opens in your browser in a few seconds.")
            self.root.after(4000, lambda: webbrowser.open(DASHBOARD_URL))
        except Exception as e:
            self.set_status(f"Couldn't start dashboard: {e}")

    def start_background(self):
        self.watcher.start()
        try:
            from pynput import keyboard

            self._hotkeys = keyboard.GlobalHotKeys(
                {self.cfg.hotkey: lambda: self.post(lambda: self.explain_here("hotkey"))}
            )
            self._hotkeys.start()
        except Exception as e:
            self.append(f"(Hotkey unavailable: {e})\n", "sys")

        if True:  # tell the user right away if the text-reading part (OCR) is missing
            def check():
                ok, msg = self.tutor.check()
                if not ok:
                    self.post(lambda: self.append(f"⚠ {msg}\n\n", "sys"))

            threading.Thread(target=check, daemon=True).start()

    def run(self):
        self.start_background()
        self.root.mainloop()
        self.watcher.stop()
        if self._dash_proc is not None and self._dash_proc.poll() is None:
            self._dash_proc.terminate()
