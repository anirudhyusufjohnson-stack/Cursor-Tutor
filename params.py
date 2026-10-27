"""The agent's tunable settings (its "chromosome"). The genetic algorithm in genetic.py searches for good values.

Every field has a safe range. Tuned values are saved in learned/tuned_params.json and loaded at start-up."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Params:
    # --- perception thresholds: when does a sensor reading become a fact the agent TELLs its KB?
    text_dist: float = 100.0     # px: words farther than this from the cursor are not "next to the cursor"
    fuzzy_cut: float = 0.85      # how similar a misread word must be to a known name (0-1)
    ocr_min: float = 0.50        # ignore words the OCR network is less sure about than this
    icon_thr: float = 0.78       # how similar an icon must be to a taught icon
    icon_margin: float = 0.04    # how far the best icon must beat the runner-up
    # --- how much each kind of clue is trusted when ranking what the agent derived (0-1)
    w_window: float = 0.85       # program name
    w_title: float = 0.85        # window title
    w_text: float = 0.90         # words near the cursor
    w_control: float = 0.90      # button name reported by Windows
    w_icon: float = 0.80         # taught icon
    w_specific: float = 0.10     # bonus for app-specific knowledge over general knowledge
    min_conf: float = 0.45       # answers below this are treated as "I do not know"


# (low, high) for every gene, used by the genetic algorithm
RANGES = {
    "text_dist": (30.0, 160.0),
    "fuzzy_cut": (0.70, 0.95),
    "ocr_min": (0.20, 0.90),
    "icon_thr": (0.55, 0.95),
    "icon_margin": (0.0, 0.15),
    "w_window": (0.3, 0.99),
    "w_title": (0.3, 0.99),
    "w_text": (0.3, 0.99),
    "w_control": (0.3, 0.99),
    "w_icon": (0.3, 0.99),
    "w_specific": (0.0, 0.3),
    "min_conf": (0.2, 0.8),
}
GENES = list(RANGES)


def clamp(p: Params) -> Params:
    for name, (lo, hi) in RANGES.items():
        setattr(p, name, min(max(float(getattr(p, name)), lo), hi))
    return p


def to_vector(p: Params):
    return [float(getattr(p, g)) for g in GENES]


def from_vector(v) -> Params:
    return clamp(Params(**{g: float(x) for g, x in zip(GENES, v)}))


def load(path) -> Params:
    path = Path(path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {f.name for f in fields(Params)}
            return clamp(Params(**{k: v for k, v in data.get("params", data).items() if k in known}))
        except Exception:
            pass
    return Params()


def save(path, p: Params, info=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": asdict(p), "info": info or {}}, indent=2), encoding="utf-8")
