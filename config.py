"""Settings, read from environment variables / a .env file.
No API key and no internet are needed to run the app: all AI runs on your own computer.

Every setting is optional. A missing or blank value uses the default, and a value that cannot be understood
(for example DWELL_SECONDS=abc) also falls back to the default with a warning, instead of crashing the app."""
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE = Path(os.getenv("TUTOR_HOME", "").strip() or Path(__file__).resolve().parent)
KNOWLEDGE_DIR = BASE / "knowledge"   # built-in knowledge about apps and buttons (JSON, you can edit it)
LEARNED_DIR = BASE / "learned"       # things you taught the app (icons + descriptions)

LANGUAGES = ("English", "Hinglish", "Hindi")
DEFAULT_SENSITIVE = "password,bank,login,sign in,otp,wallet,payment,upi"
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


def _env(name: str, default: str) -> str:
    """The setting's text, or the default when it is missing or blank."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _words(value: str) -> tuple:
    return tuple(w.strip().lower() for w in value.split(",") if w.strip())


def _flag(name: str, default: bool) -> bool:
    """1/0, true/false, yes/no and on/off (any capitals) all work."""
    value = _env(name, "1" if default else "0").lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    warnings.warn(f"{name}={value!r} is not a yes/no value; using the default ({'on' if default else 'off'}).")
    return default


def _number(name: str, default: float, minimum: float) -> float:
    try:
        value = float(_env(name, str(default)))
    except ValueError:
        warnings.warn(f"{name} must be a number; using the default ({default}).")
        return default
    if value < minimum:
        warnings.warn(f"{name} must be at least {minimum}; using the default ({default}).")
        return default
    return value


def _language(name: str, default: str) -> str:
    """English, Hinglish or Hindi, in any capitals. Anything else falls back to English with a warning."""
    value = _env(name, default)
    for lang in LANGUAGES:
        if value.lower() == lang.lower():
            return lang
    warnings.warn(f"{name}={value!r} is not one of {', '.join(LANGUAGES)}; using {default}.")
    return default


@dataclass
class Config:
    # Language for drafted messages: English, Hinglish or Hindi.
    language: str = _language("TUTOR_LANGUAGE", "English")
    # How long the mouse must stay still before auto-explain fires (at least 0.3 s, or it would fire constantly).
    dwell_seconds: float = _number("DWELL_SECONDS", 2.0, 0.3)
    # Global hotkey that explains whatever is under the cursor right now (pynput format).
    hotkey: str = _env("TUTOR_HOTKEY", "<ctrl>+<shift>+h")
    # Auto-explain is paused over windows whose title/app contains these words.
    sensitive_words: tuple = _words(_env("SENSITIVE_WORDS", DEFAULT_SENSITIVE))
    # Windows only: also ask the OS which button/control is under the cursor.
    use_uia: bool = _flag("USE_UIA", True)
    # Read text near the cursor with the OCR neural network (needs rapidocr-onnxruntime).
    use_ocr: bool = _flag("USE_OCR", True)
    # The agent's thresholds and trust weights are not set here: they live in params.py and are tuned by tune.py
    # (saved in learned/tuned_params.json).


cfg = Config()
