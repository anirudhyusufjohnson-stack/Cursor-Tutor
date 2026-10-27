"""Tiny local SQLite log so the dashboard can show what the tutor has done (free, no server)."""
import base64
import io
import os
import sqlite3
import time
from pathlib import Path

from PIL import Image

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    app TEXT,
    window_title TEXT,
    ui_element TEXT,
    source TEXT,          -- hover / hotkey / button / chat
    question TEXT,
    answer TEXT,
    seconds REAL,
    model TEXT,           -- which engine answered
    thumb BLOB,
    recognized TEXT,      -- what the algorithms decided the item was ('' if unknown)
    confidence REAL,      -- 0-1
    matched INTEGER,      -- 1 if something was recognised
    methods TEXT,         -- evidence used: window, text, control, icon (comma separated)
    action TEXT,          -- what the logical agent decided: explain_item, explain_app, ask_teach, pause
    proof TEXT            -- the logic rules that fired, separated by " | "
)
"""
NEW_COLUMNS = {"action": "TEXT", "proof": "TEXT"}   # added to history files made by older versions

COLUMNS = ("id, ts, app, window_title, ui_element, source, question, answer, seconds, model, recognized, confidence, matched,"
           " methods, action, proof")


def db_path() -> Path:
    return Path(os.getenv("TUTOR_DB", str(Path(__file__).with_name("tutor_history.db"))))


def _connect():
    conn = sqlite3.connect(db_path(), timeout=10)
    conn.execute(SCHEMA)
    have = {row[1] for row in conn.execute("PRAGMA table_info(interactions)")}
    for col, kind in NEW_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE interactions ADD COLUMN {col} {kind}")
    return conn


def _thumb(b64):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        img.thumbnail((360, 360))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    except Exception:
        return None


def log_interaction(source, question, answer, seconds, engine, ctx=None, last_ctx=None, rec=None):
    """Save one explanation. `ctx` = fresh screen capture (gives a thumbnail);
    `last_ctx` = the previous capture, used to name the app for follow-up chat;
    `rec` = the Recognition result."""
    info = ctx or last_ctx
    if rec is not None and rec.app:
        app = rec.app["name"]
    else:
        app = (getattr(info, "process_name", "") if info else "") or "Unknown app"
    thumb = _thumb(ctx.images_b64[-1]) if ctx is not None and ctx.images_b64 else None
    matched = int(bool(rec is not None and rec.matched))
    methods = ",".join(rec.methods) if rec is not None and rec.methods else ("chat" if source == "chat" else "none")
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO interactions (ts, app, window_title, ui_element, source, question, answer, seconds, model, thumb,"
            " recognized, confidence, matched, methods, action, proof) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(),
                app,
                getattr(info, "window_title", "") if info else "",
                getattr(info, "ui_element", "") if info else "",
                source,
                question,
                answer,
                seconds,
                engine,
                thumb,
                (rec.name if rec is not None else "") or "",
                float(rec.confidence) if rec is not None else 0.0,
                matched,
                methods,
                getattr(rec, "action", "") if rec is not None else "",
                " | ".join(p.split("   (")[0] for p in getattr(rec, "proof", [])) if rec is not None else "",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_rows():
    """All interactions (without thumbnails) as a list of dicts, oldest first."""
    conn = _connect()
    try:
        cur = conn.execute(f"SELECT {COLUMNS} FROM interactions ORDER BY ts")
        names = [c[0] for c in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def thumb_for(row_id):
    conn = _connect()
    try:
        row = conn.execute("SELECT thumb FROM interactions WHERE id = ?", (row_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def clear_all():
    conn = _connect()
    try:
        conn.execute("DELETE FROM interactions")
        conn.commit()
    finally:
        conn.close()
