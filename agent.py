"""The logical agent: a knowledge-based agent in the style of a textbook "KB-AGENT".

Each time the mouse rests on something (one time step t):
  1. PERCEIVE   the sensors turn the screen into symbols               (sensors.py)
  2. TELL       the agent adds those symbols to its knowledge base as facts
  3. INFER      forward chaining derives everything that follows (which app, which item)
  4. RANK       if several items were derived, the best-supported one is chosen
  5. ASK        the agent asks its KB which action to take (explain item, explain app, pause)
                Closed-world assumption: if no action can be derived, the agent asks to be taught.
  6. REMEMBER   the app and item are kept as facts so follow-up questions know what we are talking about

Because the answer is derived by logic, the agent can always show the proof."""
from dataclasses import dataclass, field

from logic import KB
from params import Params
from rules import compile_kb
from sensors import perceive


@dataclass
class Evidence:
    kind: str       # window | text | control | icon
    detail: str
    score: float


@dataclass
class Recognition:
    app: dict = None
    element: dict = None
    app_conf: float = 0.0
    element_conf: float = 0.0
    evidence: list = field(default_factory=list)
    seen_text: str = ""            # nearest readable word that matched nothing (useful for teaching)
    target_box: tuple = None       # box of the icon/button isolated under the cursor, in region pixels
    ocr_available: bool = True
    action: str = "ask_teach"      # explain_item | explain_app | ask_teach | pause
    proof: list = field(default_factory=list)   # the rules used, as text, in the order they fired
    rules_used: list = field(default_factory=list)
    facts_told: list = field(default_factory=list)
    candidates: int = 0

    @property
    def confidence(self):
        return max(self.app_conf, self.element_conf)

    @property
    def matched(self):
        return self.app is not None or self.element is not None

    @property
    def methods(self):
        return sorted({e.kind for e in self.evidence})

    @property
    def name(self):
        if self.element:
            return self.element["names"][0].title()
        if self.app:
            return self.app["name"]
        return self.seen_text


def _safe(text):
    import re
    return re.sub(r"\W+", "_", text)


def noisy_or(scores):
    """P(at least one clue is right) when each clue is independently right with probability s."""
    p = 1.0
    for s in scores:
        p *= 1.0 - max(0.0, min(1.0, s))
    return 1.0 - p


class LogicalAgent:
    def __init__(self, kbase, params=None, sensitive_words=()):
        self.kbase = kbase
        self.params = params or Params()
        self.sensitive_words = tuple(sensitive_words)
        self.t = 0
        self.memory = set()           # facts that stay true between time steps
        self._memory_ids = {}
        self.percept_facts = set()
        self.recompile()

    def recompile(self):
        """(Re)build the rule base from the knowledge base, e.g. after the user taught something."""
        self.compiled = compile_kb(self.kbase)
        self.kb = KB()
        for r in self.compiled.rules:
            self.kb.tell_rule(r)
        self.kb.facts |= self.memory
        self.kb._closure = None

    def pauses(self, process, title, trigger="hover"):
        """Ask the KB (without changing the agent's memory) whether it should stay quiet on this window."""
        from sensors import Raw

        percepts = perceive(Raw(process=process or "", title=title or ""), self.compiled, self.params,
                            self.sensitive_words, trigger)
        kb = KB()
        for r in self.compiled.rules:
            kb.tell_rule(r)
        for f in percepts.symbols:
            kb.tell_fact(f)
        return kb.closure().holds("Act_Pause")

    # ------------------------------------------------------------------ the agent loop
    def step(self, raw, trigger="hover"):
        """One cycle of perceive -> TELL -> infer -> rank -> ASK -> remember."""
        p = self.params
        self.t += 1
        percepts = perceive(raw, self.compiled, p, self.sensitive_words, trigger)

        self.kb.retract(self.percept_facts | {"AcceptedItem", "AcceptedApp"})
        for f in percepts.symbols:
            self.kb.tell_fact(f)
        self.percept_facts = set(percepts.symbols)
        closure = self.kb.closure()

        rec = Recognition(seen_text=percepts.seen_text, target_box=raw.target_box, ocr_available=raw.ocr_available,
                          facts_told=list(percepts.symbols))
        c = self.compiled
        apps = [s for s in closure.facts if s in c.apps]
        items = [s for s in closure.facts if s in c.items]
        rec.candidates = len(apps) + len(items)
        if any(a not in c.surface_apps for a in apps):
            apps = [a for a in apps if a not in c.surface_apps]   # the bare desktop is only the background

        best_app = self._best(apps, closure, percepts.info, lambda s: self._app_bonus(s, closure))
        best_item = self._best(items, closure, percepts.info, lambda s: p.w_specific if s in c.specific_items else 0.0)

        if best_app and best_app[1] >= p.min_conf:
            rec.app, rec.app_conf = c.apps[best_app[0]], best_app[1]
            self.kb.tell_fact("AcceptedApp")
        if best_item and best_item[1] >= p.min_conf:
            rec.element, rec.element_conf = c.items[best_item[0]], best_item[1]
            self.kb.tell_fact("AcceptedItem")
            if rec.app is None and best_item[0] in c.specific_items:   # an app-specific item implies its app
                owner = c.app_by_id.get(c.items[best_item[0]]["_app"])
                if owner:
                    rec.app, rec.app_conf = owner, rec.element_conf

        # ASK the KB which action to take
        closure = self.kb.closure()
        if closure.holds("Act_Pause"):
            rec.action = "pause"
        elif closure.holds("Act_ExplainItem"):
            rec.action = "explain_item"
        elif closure.holds("Act_ExplainApp"):
            rec.action = "explain_app"
        else:
            rec.action = "ask_teach"     # closed-world assumption: what cannot be derived is not assumed

        for best in (best_app, best_item):
            if not best:
                continue
            sym_ = best[0]
            if (sym_ in c.apps and rec.app is c.apps[sym_]) or (sym_ in c.items and rec.element is c.items[sym_]):
                rec.evidence += self._evidence(sym_, closure, percepts.info)
                for s_, rule in closure.proof(sym_):
                    if rule is not None and str(rule) not in rec.rules_used:
                        rec.proof.append(f"{rule}   ({rule.note})")
                        rec.rules_used.append(str(rule))
        # REMEMBER
        self._remember(rec)
        return rec

    # ------------------------------------------------------------------ helpers
    def _app_bonus(self, sym_, closure):
        """A web app named in the window title (Gmail in Chrome) is more specific than the browser itself."""
        app = self.compiled.apps[sym_]
        if app.get("web") and any(l.startswith("Title_") for l in closure.leaves(sym_)):
            return self.params.w_specific
        return 0.0

    def _reliability(self, info):
        p = self.params
        if info.kind == "window":
            return p.w_window if info.sub == "proc" else p.w_title
        if info.kind == "text":
            nearness = 1.0 - 0.7 * min(info.dist / max(p.text_dist, 1.0), 1.0)
            return p.w_text * nearness * (0.6 + 0.4 * min(info.conf, 1.0))
        if info.kind == "control":
            return p.w_control
        return p.w_icon

    def _evidence(self, sym_, closure, info):
        return [Evidence(info[l].kind, info[l].detail, self._reliability(info[l]))
                for l in closure.leaves(sym_) if l in info]

    def _best(self, syms, closure, info, bonus):
        scored = []
        for s in syms:
            ev = self._evidence(s, closure, info)
            if not ev:
                continue
            # one vote per kind of clue (the strongest), combined with the noisy-OR rule
            by_kind = {}
            for e in ev:
                by_kind[e.kind] = max(by_kind.get(e.kind, 0.0), e.score)
            conf = noisy_or(by_kind.values())
            scored.append((conf + bonus(s), s, conf))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1], scored[0][2]

    def _remember(self, rec):
        if rec.app is not None:
            self.memory = {m for m in self.memory if not m.startswith("Mem_App_")} | {"Mem_App_" + _safe(rec.app["id"])}
            self._memory_ids["Mem_App_" + _safe(rec.app["id"])] = rec.app["id"]
        if rec.element is not None:
            self.memory = {m for m in self.memory if not m.startswith("Mem_Item_")} | {"Mem_Item_" + _safe(rec.element["_id"])}
        self.kb.facts = {f for f in self.kb.facts if not f.startswith("Mem_")} | self.memory
        self.kb._closure = None

    def remembered_app(self):
        """The app the agent last recognised, found by asking its own memory."""
        for f in self.memory:
            if f.startswith("Mem_App_"):
                app = self.compiled.app_by_id.get(self._memory_ids.get(f, ""))
                if app:
                    return app
        return None
