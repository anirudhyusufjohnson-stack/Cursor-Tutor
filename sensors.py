"""The agent's sensors: turn what is on the screen into facts the logical agent can TELL its knowledge base.

Two stages, so the (slow) sensing can be done once and the (fast) thresholding repeated many times,
which is what the genetic algorithm needs when it tunes the thresholds:

  sense()    ->  Raw        OCR words with distances, the icon scores, the window strings
  perceive() ->  Percepts   a list of true symbols such as Text_new_chat, using the thresholds in Params

OCR (a pretrained neural network), Canny edge detection and icon matching are SENSORS here. They do not
make decisions; they only report what is seen. All decisions are made by the logical agent."""
import difflib
from dataclasses import dataclass, field

from knowledge import norm
from ocr_engine import nearest_texts
from rules import sym
from target import crop_box, segment_target

MAX_TEXT_DISTANCE = 160     # words farther than this are never looked at
FUZZY_FLOOR = 0.70          # smallest similarity worth remembering for a misread word


def match_terms(text, terms, floor=FUZZY_FLOOR):
    """Which known names could this text be?  [(name, kind, value)], kind = exact | word | fuzzy."""
    t = norm(text)
    if not t:
        return []
    padded = f" {t} "
    n_words = len(t.split())
    out = []
    for n in terms:
        if t == n:
            out.append((n, "exact", 1.0))
        elif len(n) >= 4 and f" {n} " in padded:
            if n_words <= 3:  # a long sentence that merely contains a name is not that name
                out.append((n, "word", 0.9))
        elif len(n) >= 5 and len(t) >= 5:
            sm = difflib.SequenceMatcher(None, t, n)
            if sm.real_quick_ratio() >= floor and sm.quick_ratio() >= floor:
                r = sm.ratio()
                if r >= floor:
                    out.append((n, "fuzzy", r))
    return out


@dataclass
class TextObs:
    text: str
    conf: float
    dist: float
    matches: list


@dataclass
class Raw:
    """Everything the sensors measured, before any threshold is applied."""
    process: str = ""
    title: str = ""
    control: str = ""
    control_matches: list = field(default_factory=list)
    texts: list = field(default_factory=list)       # TextObs, nearest first
    icons: list = field(default_factory=list)       # [(score, label)] best first
    ocr_available: bool = False
    target_box: tuple = None


@dataclass
class Info:
    """Where a percept symbol came from (used for ranking and for explaining)."""
    kind: str            # window | text | control | icon
    detail: str
    sub: str = ""        # proc | title for window
    dist: float = 0.0
    conf: float = 1.0


@dataclass
class Percepts:
    symbols: list
    info: dict
    seen_text: str = ""


def sense(ctx, ocr, icons, text_terms):
    """Run the sensors on a screen capture and measure everything once."""
    raw = Raw(process=ctx.process_name or "", title=ctx.window_title or "", ocr_available=bool(ocr and ocr.available))
    if ctx.ui_element and ": " in ctx.ui_element:
        raw.control = ctx.ui_element.split(": ", 1)[1]
        raw.control_matches = match_terms(raw.control, text_terms)
    if ctx.region is not None:
        if raw.ocr_available:
            for tb, dist in nearest_texts(ocr.read(ctx.region), ctx.region_cursor, limit=8, max_distance=MAX_TEXT_DISTANCE):
                raw.texts.append(TextObs(tb.text, tb.conf, dist, match_terms(tb.text, text_terms)))
        if icons is not None:
            box, _ = segment_target(ctx.region, ctx.region_cursor)
            raw.target_box = box
            raw.icons = icons.scores(crop_box(ctx.region, box))[:3]
    return raw


def perceive(raw, compiled, params, sensitive_words=(), trigger="hover"):
    """Apply the thresholds in `params` and list the symbols that are true right now."""
    facts, info = [], {}

    def tell(s, i):
        if s not in info or (i.kind == "text" and i.dist < info[s].dist):
            info[s] = i
        if s not in facts:
            facts.append(s)

    # the window
    proc = raw.process.lower()
    owner = False
    if proc in compiled.proc_terms:
        s, a = compiled.proc_terms[proc]
        tell(s, Info("window", f"the program running here is {raw.process}", "proc"))
        owner = owner or a in compiled.window_apps
    t = f" {norm(raw.title)} "
    for kw, (s, a) in compiled.title_terms.items():
        if f" {kw} " in t:
            tell(s, Info("window", f"the window title '{raw.title[:40]}' mentions '{kw}'", "title"))
            owner = owner or a in compiled.window_apps
    facts.append("WindowKnown" if owner else "WindowUnknown")
    facts.append("Trigger_" + trigger)
    blob = norm(raw.title + " " + raw.process)
    if any(w and w in blob for w in sensitive_words):
        facts.append("Sensitive")

    def accepted(matches):
        return [n for n, kind, v in matches if kind != "fuzzy" or v >= params.fuzzy_cut]

    # words next to the cursor
    near = [o for o in raw.texts if o.conf >= params.ocr_min and o.dist <= params.text_dist]
    matched_words = set()
    for o in near:
        for term in accepted(o.matches):
            tell(sym("Text", term), Info("text", f"the word '{o.text}' is {o.dist:.0f} px from the cursor", "", o.dist, o.conf))
            matched_words.add(o.text)
    # the control name Windows reports
    for term in accepted(raw.control_matches):
        tell(sym("Ctl", term), Info("control", f"Windows says this control is called '{raw.control}'"))
    # an icon you taught
    from icons import IconStore

    hit = IconStore.decide(raw.icons, params.icon_thr, params.icon_margin)
    if hit:
        tell(sym("Icon", hit[0]), Info("icon", f"the icon matches the one you taught as '{hit[0]}'", "", 0.0, hit[1]))

    seen = next((o.text for o in near if o.text not in matched_words and len(o.text) >= 2), "")
    return Percepts(facts, info, seen)
