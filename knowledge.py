"""The knowledge base: what apps and buttons are, and how to use them.

Built-in knowledge lives in the JSON files in the `knowledge/` folder (you can edit them).
Things you teach the app are saved in `learned/user_knowledge.json` and are used straight away."""
import difflib
import json
import re
from pathlib import Path


def norm(text):
    """Lower-case, keep only letters/digits (any language), single spaces."""
    text = re.sub(r"[^\w]+", " ", str(text).lower(), flags=re.UNICODE).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def name_score(text, names):
    """How well a piece of text (from OCR, a window title...) matches any of the names: 0-1."""
    t = norm(text)
    if not t:
        return 0.0
    words = t.split()
    best = 0.0
    for raw in names:
        n = norm(raw)
        if not n:
            continue
        if t == n:
            return 1.0
        if len(n) >= 4 and re.search(r"\b" + re.escape(n) + r"\b", t):
            best = max(best, 0.9 if len(words) <= 3 else 0.65)  # long sentences that merely contain the name are weaker
        elif len(n) >= 5 and len(t) >= 5:
            r = difflib.SequenceMatcher(None, t, n).ratio()
            if r >= 0.85:
                best = max(best, r * 0.95)
    return best


def slug(text):
    return re.sub(r"[^a-z0-9_-]+", "-", str(text).lower()).strip("-") or "item"


class KnowledgeBase:
    def __init__(self, knowledge_dir, learned_dir):
        self.knowledge_dir = Path(knowledge_dir)
        self.learned_dir = Path(learned_dir)
        self.user_file = self.learned_dir / "user_knowledge.json"
        self.reload()

    # ------------------------------------------------------------------ loading
    def reload(self):
        self.apps = {}
        self.generic = []
        user = {"apps": [], "elements": []}
        if self.user_file.exists():
            try:
                user = json.loads(self.user_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        for f in sorted(self.knowledge_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for app in data.get("apps", []):
                self.apps[app["id"]] = app
            self.generic.extend(data.get("elements", []))

        for app in user.get("apps", []):
            app["user"] = True
            self.apps[app["id"]] = app
        for el in user.get("elements", []):
            el["user"] = True
            owner = el.get("app")
            if owner and owner in self.apps:
                self.apps[owner].setdefault("elements", []).append(el)
            else:
                self.generic.append(el)

        for app in list(self.apps.values()):  # "extends": reuse another app's buttons and how-tos
            parent = self.apps.get(app.get("extends", ""))
            if parent:
                app["elements"] = app.get("elements", []) + [dict(e) for e in parent.get("elements", [])]
                app["howtos"] = app.get("howtos", []) + [dict(h) for h in parent.get("howtos", [])]

        for app in self.apps.values():
            for i, el in enumerate(app.get("elements", [])):
                el["_id"] = f"{app['id']}:{i}"
                el["_app"] = app["id"]
        for i, el in enumerate(self.generic):
            el["_id"] = f"generic:{i}"
            el["_app"] = None

        self.by_icon = {}
        for app in self.apps.values():
            if app.get("icon_label"):
                self.by_icon[app["icon_label"]] = ("app", app)
            for el in app.get("elements", []):
                if el.get("icon_label"):
                    self.by_icon[el["icon_label"]] = ("element", el)
        for el in self.generic:
            if el.get("icon_label"):
                self.by_icon[el["icon_label"]] = ("element", el)

    def stats(self):
        return {
            "apps": len(self.apps),
            "elements": sum(len(a.get("elements", [])) for a in self.apps.values()) + len(self.generic),
            "howtos": sum(len(a.get("howtos", [])) for a in self.apps.values()),
            "taught": sum(1 for a in self.apps.values() if a.get("user"))
            + sum(1 for e in self.generic if e.get("user"))
            + sum(1 for a in self.apps.values() for e in a.get("elements", []) if e.get("user")),
        }

    # ------------------------------------------------------------------ lookups
    def app_by_process(self, process_name):
        p = (process_name or "").lower()
        if not p:
            return None
        for app in self.apps.values():
            if p in [x.lower() for x in app.get("process", [])]:
                return app
        return None

    def apps_by_title(self, title):
        t = norm(title)
        found = []
        if not t:
            return found
        for app in self.apps.values():
            for kw in app.get("title_keywords", []):
                k = norm(kw)
                if k and re.search(r"\b" + re.escape(k) + r"\b", t):
                    found.append((app, kw))
                    break
        return found

    def apps_by_text(self, text):
        """Apps whose name or nickname matches a piece of text (e.g. a desktop icon label)."""
        out = []
        for app in self.apps.values():
            s = name_score(text, [app["name"]] + app.get("aliases", []))
            if s >= 0.8:
                out.append((app, s))
        return out

    def match_elements(self, text, app=None):
        """[(element, score)] for a piece of text, looking in the given app first, then in the general list."""
        out = []
        if app:
            for el in app.get("elements", []):
                s = name_score(text, el["names"])
                if s >= 0.8:
                    out.append((el, min(1.0, s + 0.05)))  # app-specific knowledge wins ties
        for el in self.generic:
            s = name_score(text, el["names"])
            if s >= 0.8:
                out.append((el, s))
        return out

    def howto_docs(self):
        """One searchable document per how-to, plus a 'get started' document per app."""
        docs = []
        for app in self.apps.values():
            for h in app.get("howtos", []):
                docs.append(
                    {"app_id": app["id"], "app_name": app["name"], "title": h["title"],
                     "questions": h.get("questions", []), "steps": h["steps"]}
                )
            if app.get("steps"):
                n = app["name"]
                docs.append(
                    {"app_id": app["id"], "app_name": n, "title": f"Get started with {n}",
                     "questions": [f"how do i use {n}", f"what is {n}", f"open {n}", f"start {n}", f"learn {n}"],
                     "steps": app["steps"]}
                )
        return docs

    # ------------------------------------------------------------------ teaching
    def add_user_entry(self, name, kind, what, steps, icon_label="", app_id=None):
        """Save something the user taught. kind = 'app' or 'item'."""
        data = {"apps": [], "elements": []}
        if self.user_file.exists():
            try:
                data = json.loads(self.user_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        if kind == "app":
            entry = {"id": slug(name), "name": name, "aliases": [norm(name)], "what": what,
                     "steps": steps, "icon_label": icon_label, "elements": [], "howtos": []}
            data["apps"] = [a for a in data["apps"] if a["id"] != entry["id"]] + [entry]
        else:
            entry = {"names": [norm(name)], "what": what, "steps": steps, "icon_label": icon_label}
            if app_id:
                entry["app"] = app_id
            data["elements"].append(entry)
        self.learned_dir.mkdir(parents=True, exist_ok=True)
        self.user_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.reload()
        return entry
