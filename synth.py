"""Made-up screens: used by the self-test and by the genetic-algorithm tuner. No real screen is needed.

A fake screen is a grey picture with some words and simple coloured icons drawn on it. The real OCR network and
the real icon matcher then "look" at it, exactly as they would look at a real screenshot."""
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from knowledge import norm
from sensors import sense

FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_icon(d, x, y, kind, s=64):
    if kind == "red-circle":
        d.ellipse([x, y, x + s, y + s], fill=(210, 40, 40))
        d.ellipse([x + 20, y + 20, x + s - 20, y + s - 20], fill=(255, 255, 255))
    elif kind == "blue-square":
        d.rectangle([x, y, x + s, y + s], fill=(30, 70, 200))
        d.line([x + 8, y + 8, x + s - 8, y + s - 8], fill=(255, 255, 255), width=8)
    elif kind == "green-triangle":
        d.polygon([(x + s // 2, y), (x, y + s), (x + s, y + s)], fill=(30, 160, 70))


def fake_screen(texts=(), icons=(), size=(700, 450), font_size=28, ink=(20, 20, 20)):
    """A made-up screen region: grey background, some words and some simple coloured icons."""
    img = Image.new("RGB", size, (238, 238, 238))
    d = ImageDraw.Draw(img)
    for (x, y), word in texts:
        d.text((x, y), word, font=_font(font_size), fill=ink)
    for (x, y), kind in icons:
        draw_icon(d, x, y, kind)
    return img


def ctx_for(region, cursor, process="", title="", ui=""):
    return SimpleNamespace(x=0, y=0, window_title=title, process_name=process, ui_element=ui, images_b64=[],
                           region=region, region_cursor=cursor, describe=lambda: f"{process} {title}")


# ---------------------------------------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    kind: str
    raw: object          # sensors.Raw: what the sensors measured
    expect: tuple        # (app id or None, item name or None)


def _typo(word, rng):
    """One small spelling mistake: drop, swap or replace a letter (what a hurried OCR read might look like)."""
    if len(word) < 6:
        return word
    i = rng.randrange(1, len(word) - 1)
    mode = rng.choice(["drop", "swap", "replace"])
    if mode == "drop":
        return word[:i] + word[i + 1:]
    if mode == "swap" and word[i] != word[i + 1]:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + rng.choice("aeiourstn") + word[i + 1:]


def teach_samples(tutor):
    """Teach the tutor two icons so that icon scenarios exist. Call once on a tutor that uses a scratch folder."""
    for kind, at, name, what in (("red-circle", (300, 200), "Zorb button", "kind-item"), ("blue-square", (100, 100), "Blue Studio", "kind-app")):
        r = fake_screen(icons=[(at, kind)])
        cx = (at[0] + 32, at[1] + 32)
        ctx = ctx_for(r, cx)
        if what == "kind-app":
            tutor.teach(name, "app", "Blue Studio is a made-up drawing app.", ["Open it from the desktop."], ctx)
        else:
            tutor.teach(name, "item", "The Zorb button is a made-up button.", ["Click it."], ctx)


def build_scenarios(tutor, n, seed=1):
    """n labelled made-up situations, measured once by the real sensors (so tuning is fast afterwards)."""
    rng = random.Random(seed)
    kb = tutor.kb
    apps = [a for a in kb.apps.values() if not a.get("surface") and not a.get("user")]
    native = [a for a in apps if a.get("process") and not a.get("shell") and a.get("elements")]
    web = [a for a in apps if a.get("web") and a.get("elements")]
    # names written on the desktop; "settings" is left out because the word is also a general button name (ambiguous)
    desk = [a for a in apps if a.get("aliases") and a["id"] not in ("desktop", "settings")]
    generic = [e for e in kb.generic if len(norm(e["names"][0])) >= 4 and not e.get("user")]
    junk = ["Zorblax", "Quindle", "Mirpa", "Tavrix", "Blenor", "Fosque", "Yumbit"]

    def element_word(app):
        els = [e for e in app["elements"] if len(norm(e["names"][0])) >= 4]
        return rng.choice(els)

    def place(size=(700, 450)):
        cursor = (rng.randint(180, 520), rng.randint(120, 330))
        return cursor

    def with_noise(texts, cursor, n_noise=None):
        """Add a few unrelated words far from the cursor."""
        out = list(texts)
        for _ in range(rng.randint(0, 2) if n_noise is None else n_noise):
            w = rng.choice(junk)
            for _try in range(10):
                p = (rng.randint(10, 560), rng.randint(10, 400))
                if ((p[0] + 40 - cursor[0]) ** 2 + (p[1] + 14 - cursor[1]) ** 2) ** 0.5 > 150:
                    out.append((p, w))
                    break
        return out

    def offset_word(cursor, word, dist):
        """Draw `word` so that its centre is about `dist` px from the cursor."""
        import math
        ang = rng.uniform(0, 2 * math.pi)
        w, h = 14 * len(word) * 0.9, 28
        cx, cy = cursor[0] + dist * math.cos(ang), cursor[1] + dist * math.sin(ang)
        x = max(5, min(600, int(cx - w / 2)))
        y = max(5, min(410, int(cy - h / 2)))
        return (x, y)

    kinds = ["in_app", "small", "typo", "web", "desktop", "far", "unknown", "generic", "icon", "icon_other", "decoy", "faint",
             "typo2", "blurry", "icon_scaled", "decoy_near"]
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        cursor = place()
        process = title = ""
        texts, expect, ui = [], (None, None), ""
        if kind in ("in_app", "typo", "typo2", "far", "faint", "small", "blurry"):
            app = rng.choice(native)
            el = element_word(app)
            word = el["names"][0]
            process = rng.choice(app["process"])
            title = app["name"] + rng.choice(["", " - Document1", " - Home"])
            shown = _typo(word, rng) if kind == "typo" else _typo(_typo(word, rng), rng) if kind == "typo2" else word
            dist = rng.uniform(150, 200) if kind == "far" else rng.uniform(0, 55)
            texts = [(offset_word(cursor, shown, dist), shown.title() if rng.random() < .5 else shown)]
            expect = (app["id"], None) if kind == "far" else (app["id"], norm(word))
            font, ink = {"faint": (24, (150, 150, 150)), "small": (13, (40, 40, 40)), "blurry": (16, (90, 90, 90))}.get(kind, (28, (20, 20, 20)))
            region = fake_screen(with_noise(texts, cursor), size=(700, 450), font_size=font, ink=ink)
            if kind == "blurry":
                region = region.filter(ImageFilter.GaussianBlur(0.9))
        elif kind == "web":
            app = rng.choice(web)
            el = element_word(app)
            word = el["names"][0]
            process = rng.choice(["chrome.exe", "msedge.exe"])
            title = f"{app['name']} - {rng.choice(['Inbox', 'Home', 'Watch'])}"
            texts = [(offset_word(cursor, word, rng.uniform(0, 55)), word.title())]
            expect = (app["id"], norm(word))
            region = fake_screen(with_noise(texts, cursor))
        elif kind == "desktop":
            app = rng.choice(desk)
            word = norm(app["aliases"][0]).title() if rng.random() < .5 else app["name"]
            process = "explorer.exe"
            title = rng.choice(["", "Program Manager"])
            texts = [(offset_word(cursor, word, rng.uniform(0, 40)), word)]
            expect = (app["id"], None)
            region = fake_screen(with_noise(texts, cursor))
        elif kind == "unknown":
            word = rng.choice(junk)
            texts = [(offset_word(cursor, word, rng.uniform(0, 40)), word)]
            region = fake_screen(with_noise(texts, cursor, 1))
        elif kind == "generic":
            el = rng.choice(generic)
            word = el["names"][0]
            texts = [(offset_word(cursor, word, rng.uniform(0, 40)), word.title())]
            expect = (None, norm(word))
            region = fake_screen(with_noise(texts, cursor, 1))
        elif kind == "decoy":
            app = rng.choice(desk)
            word = rng.choice(junk)
            decoy = app["name"]
            texts = [(offset_word(cursor, word, rng.uniform(0, 30)), word),
                     (offset_word(cursor, decoy, rng.uniform(125, 155)), decoy)]
            region = fake_screen(texts)
        elif kind == "decoy_near":   # unknown word under the cursor, a known app's name a bit further away
            app = rng.choice(desk)
            word = rng.choice(junk)
            texts = [(offset_word(cursor, word, rng.uniform(0, 25)), word),
                     (offset_word(cursor, app["name"], rng.uniform(95, 125)), app["name"])]
            region = fake_screen(texts)
        else:  # icon / icon_other / icon_scaled
            at = (cursor[0] - 32, cursor[1] - 32)
            if kind in ("icon", "icon_scaled"):
                which = rng.choice(["red-circle", "blue-square"])
                expect = {"red-circle": (None, "zorb button"), "blue-square": ("blue-studio", None)}[which]
            else:
                which = "green-triangle"
            region = fake_screen(icons=[(at, which)], texts=with_noise([], cursor, rng.randint(0, 1)))
            if kind == "icon":   # the same icon, but hovered (brighter) or dimmed
                region = ImageEnhance.Brightness(region).enhance(rng.choice([0.85, 1.0, 1.0, 1.2]))
            elif kind == "icon_scaled":   # a different zoom level (high-DPI screens) and a different brightness
                z = rng.choice([0.7, 0.8, 1.25, 1.4])
                cut = region.crop((cursor[0] - 90, cursor[1] - 90, cursor[0] + 90, cursor[1] + 90))
                cut = cut.resize((int(cut.width * z), int(cut.height * z)))
                region = Image.new("RGB", (700, 450), (238, 238, 238))
                region.paste(cut, (cursor[0] - cut.width // 2, cursor[1] - cut.height // 2))
                region = ImageEnhance.Brightness(region).enhance(rng.choice([0.75, 0.9, 1.15, 1.3]))
        ctx = ctx_for(region, cursor, process, title, ui)
        raw = sense(ctx, tutor.ocr, tutor.icons, tutor.agent.compiled.text_terms)
        out.append(Scenario(kind, raw, expect))
    return out
