"""The "brain": ties the logical agent, the language rules, the route planner and the drafting together.

Everything here runs on your own computer. There is no online service, no API key and no training data.
Which algorithm does what:
  logical agent + propositional logic   decides what is under the cursor (agent.py, logic.py)
  forward chaining over word rules      decides what you are asking (nlu.py)
  BFS / DFS / uniform-cost search       plans the clicks for a how-to answer (search.py)
  genetic algorithm                     tunes the agent's settings offline (genetic.py, tune.py)
  minimax + alpha-beta pruning          plays the practice game (game.py)"""
import time

import composer
import params as params_mod
import search
from agent import LogicalAgent, Recognition
from config import KNOWLEDGE_DIR, LEARNED_DIR
from drafting import available_topics, make_draft
from icons import IconStore
from knowledge import KnowledgeBase
from nlu import Understander
from ocr_engine import OCR
from sensors import sense
from target import crop_box, segment_target

DEFAULT_QUESTION = "What is under my cursor? Explain it and teach me how to use it."
SMALL_TALK = {"greet", "thanks", "goodbye", "help"}
TUNED_FILE = "tuned_params.json"


class Tutor:
    def __init__(self, cfg, learned_dir=None, params=None):
        learned = learned_dir or LEARNED_DIR
        self.cfg = cfg
        self.learned = learned
        self.last_used = "logic + search (offline)"
        self.last_recognition = Recognition()
        self.rec = Recognition()
        self.raw = None
        self.ocr = OCR(enabled=cfg.use_ocr)
        self.kb = KnowledgeBase(KNOWLEDGE_DIR, learned)
        self.icons = IconStore(learned / "icons")
        self.params = params if params is not None else params_mod.load(learned / TUNED_FILE)   # defaults if never tuned
        self.agent = LogicalAgent(self.kb, self.params, cfg.sensitive_words)
        self.flows = search.load_flows(KNOWLEDGE_DIR / "flows.json")
        self._build_language()

    def _build_language(self):
        self.understander = Understander(self.kb.howto_docs(), KNOWLEDGE_DIR / "language.json", self.kb.apps)

    # ------------------------------------------------------------------ housekeeping
    def reset(self):
        self.rec = Recognition()
        self.agent.memory = set()
        self.agent.kb.facts = {f for f in self.agent.kb.facts if not f.startswith("Mem_")}
        self.agent.kb._closure = None

    def check(self):
        """(ok, message). The app works without OCR, but is much more limited."""
        if self.ocr.available:
            return True, "ok"
        return False, ("The text-reading part (OCR) is not installed, so I can only use window names and icons you teach me.\n"
                       "Install it with:  pip install rapidocr-onnxruntime")

    def status(self):
        s = self.kb.stats()
        return {"ocr": self.ocr.name, "icons": self.icons.count(), **s}

    # ------------------------------------------------------------------ the agent's sensors and decision
    def should_pause(self, process, title):
        """True if the logical agent derives Act_Pause for this window (it looks private)."""
        return self.agent.pauses(process, title, "hover")

    def analyze(self, ctx, trigger="ask"):
        raw = sense(ctx, self.ocr, self.icons, self.agent.compiled.text_terms)
        self.raw = raw
        rec = self.agent.step(raw, trigger)
        self.rec = self.last_recognition = rec
        return rec

    # ------------------------------------------------------------------ chat
    def ask(self, text, ctx=None, on_chunk=None, trigger="ask"):
        """Answer a question. `ctx` is a fresh CursorContext when the screen should be looked at."""
        if ctx is not None:
            self.analyze(ctx, trigger)
            if text == DEFAULT_QUESTION:
                answer = composer.pause() if self.rec.action == "pause" else composer.explain(self.rec)
                self._stream(answer, on_chunk)
                return answer
        else:  # a follow-up question: nothing new was seen, so there is no new evidence to show
            self.last_recognition = Recognition(app=self.rec.app, element=self.rec.element, app_conf=self.rec.app_conf,
                                                element_conf=self.rec.element_conf, evidence=[])
        answer = self.respond(text)
        self._stream(answer, on_chunk)
        return answer

    def understand(self, text):
        app = self.rec.app or self.agent.remembered_app()
        return self.understander.analyze(text, app["id"] if app else None)

    def respond(self, text):
        rec = self.rec
        u = self.understand(text)
        intent = u.intent
        if intent == "explain":
            return composer.explain(rec)
        if intent == "howto":
            return self._howto(u)
        if intent == "draft":
            return self._draft(text, u)
        if intent == "more":
            return composer.more(rec.app or self.agent.remembered_app())
        if intent in composer.CANNED:
            return composer.CANNED[intent]
        if u.guides:     # no clear intent, but the words match a guide
            return self._howto(u)
        return "I did not understand that.\n\n" + composer.HELP

    # ------------------------------------------------------------------ how-to answers (search)
    def plan_for(self, doc):
        """Plan the clicks for a guide: all three searches over the app's screen map, or None if there is no map."""
        found = search.goal_for_guide(self.flows, doc["app_id"], doc["title"])
        if not found:
            return None
        flow, goal = found
        return flow, search.compare(flow, flow.start, goal)

    def _howto(self, u):
        app = self.rec.app or self.agent.remembered_app()
        if not u.guides:
            return composer.no_howto(app, [])
        best, _score = u.guides[0]
        others = [d for d, _s in u.guides[1:] if d["app_id"] == best["app_id"]][:3]
        planned = self.plan_for(best)
        return composer.howto(best, others, planned)

    def _draft(self, text, u):
        msg, name = make_draft(text, self.cfg.language, topic=u.topic)
        if not msg:
            return ("I can write short messages about: " + ", ".join(available_topics()) + ".\n\n"
                    'For example: "Tell Rahul I am running late". You can also give me the exact words in quotes.')
        who = f" for {name}" if name else ""
        return (f"Here is a draft{who}:\n<draft>{msg}</draft>\n"
                'Press "Type draft for me", then click the message box in the other app. You press Send yourself.')

    @staticmethod
    def _stream(answer, on_chunk):
        if not on_chunk:
            return
        words = answer.split(" ")
        shown = ""
        for i in range(0, len(words), 3):
            shown += (" " if shown else "") + " ".join(words[i:i + 3])
            on_chunk(shown)
            time.sleep(0.02)

    # ------------------------------------------------------------------ teaching
    def suggested_name(self):
        return self.rec.seen_text or (self.rec.name if self.rec.matched else "")

    def teach(self, name, kind, what, steps, ctx, app_id=None):
        """Remember something new: save the icon under the cursor and its description, then rebuild the rule base."""
        label = ""
        if ctx is not None and ctx.region is not None:
            box, _ = segment_target(ctx.region, ctx.region_cursor)
            label = self.icons.add(name, crop_box(ctx.region, box))
        if app_id is None and kind == "item" and self.rec.app:
            app_id = self.rec.app["id"]
        self.kb.add_user_entry(name, kind, what, steps, icon_label=label, app_id=app_id)
        self.agent.recompile()        # new knowledge = new rules (TELL the KB)
        self._build_language()
        return label
