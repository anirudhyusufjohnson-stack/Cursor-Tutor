"""Understanding what the user types, using propositional logic (no statistics, no online service).

Every word the user types becomes a fact  W_<word>.  Forward chaining then climbs three levels of rules:

   W_pic  ⇒ C_photo                              (words that mean the same thing share a concept)
   C_how  ⇒ Intent_howto                         (what the user wants: explain, how-to, draft, ...)
   C_send ∧ C_photo ⇒ Guide_12                   (which how-to guide answers the question)
   C_late ⇒ Topic_late ⇒ HasTopic                (what a drafted message should say)

When several intents or guides are derived, the most specific rule wins: the one with the highest
priority, then the one with the most premises (the usual "specificity" conflict-resolution strategy of
rule-based systems). Misspelled words are first matched to the closest known word."""
import difflib
import itertools
import json
import re
from dataclasses import dataclass, field

from drafting import TEMPLATES, find_recipient
from knowledge import norm
from logic import KB, Rule

SPECIAL = {"HasRecipient", "HasQuote", "HasTopic"}


@dataclass
class Understanding:
    intent: str = "unknown"
    intent_rule: str = ""
    guides: list = field(default_factory=list)     # [(doc, score)] best first
    topic: str = ""
    recipient: str = ""
    proof: list = field(default_factory=list)      # the rules that fired for the chosen intent


class Understander:
    def __init__(self, docs, language_file, kb_apps=None):
        lang = json.loads(open(language_file, encoding="utf-8").read())
        self.groups = {c: [norm(w) for w in ws] for c, ws in lang["concepts"].items()}
        self.word_concept = {w: c for c, ws in self.groups.items() for w in ws}
        self.stop = {norm(w) for w in lang["stop_words"]}
        self.docs = docs
        self.app_words = {}
        for app in (kb_apps or {}).values():
            words = {w for n in [app["name"]] + app.get("aliases", []) for w in norm(n).split() if len(w) >= 4}
            self.app_words[app["id"]] = words
        self.vocab = set(self.word_concept)
        self.kb = KB()
        self.meta = {}                  # rule -> ("intent"|"guide"|"topic", payload, priority, specificity)
        self._build(lang["intents"])

    # ------------------------------------------------------------------ building the rule base
    def _peek(self, w):
        """The symbol for a word, without learning the word."""
        w = norm(w)
        if w in self.word_concept:
            return "C_" + self.word_concept[w]
        if w in self.groups:
            return "C_" + w
        return "W_" + w

    def word_sym(self, w):
        s = self._peek(w)
        if s.startswith("W_"):
            self.vocab.add(norm(w))
        return s

    def _add(self, premises, concl, kind, payload, priority, note):
        prem = tuple(dict.fromkeys(premises))
        rule = Rule(prem, concl, note)
        self.kb.tell_rule(rule)
        self.meta[rule] = (kind, payload, priority, len(prem))

    def _build(self, intents):
        # 1. synonyms: every word of a group implies its concept
        for c, words in self.groups.items():
            for w in dict.fromkeys(words):
                self.kb.tell_rule(Rule((f"W_{w}",), f"C_{c}", f"'{w}' means the same as '{c}'"))
        # 2. what the user wants
        for r in intents:
            prem = []
            for w in r["if"]:
                if w in SPECIAL:
                    prem.append(w)
                elif w.startswith("^"):
                    prem.append("First_" + self.word_sym(w[1:])[2:])
                else:
                    prem.append(self.word_sym(w))
            self._add(prem, "Intent_" + r["then"], "intent", r["then"], r["p"], "if " + " and ".join(r["if"]) + " then " + r["then"])
        # 3. which guide answers the question
        self._guide_words = set()
        for d in self.docs:
            if not d["title"].startswith("Get started"):
                for ph in [d["title"]] + list(d.get("questions", [])):
                    self._guide_words |= set(self._content(ph))
        for i, d in enumerate(self.docs):
            phrases = [d["title"]] + list(d.get("questions", []))
            seen = set()
            for ph in phrases:
                prem = self._content(ph)
                if 1 <= len(prem) <= 4 and tuple(sorted(prem)) not in seen:
                    seen.add(tuple(sorted(prem)))
                    self._add(prem, f"Guide_{i}", "guide", i, 0, f"the question resembles '{ph}'")
            title = self._content(d["title"])
            if len(title) >= 3:   # any two of the title's words are enough
                for pair in itertools.combinations(title, 2):
                    if tuple(sorted(pair)) not in seen:
                        seen.add(tuple(sorted(pair)))
                        self._add(list(pair), f"Guide_{i}", "guide", i, 0, f"the question uses key words of '{d['title']}'")
            # an app named in the question lets one key word of that app's guides be enough
            for w in dict.fromkeys(sum([self._content(ph) for ph in phrases], [])):
                self._add([f"Named_{d['app_id']}", w], f"Guide_{i}", "guide", i, 0,
                          f"you named {d['app_name']} and used a key word of '{d['title']}'")
            self.kb.tell_rule(Rule((f"Guide_{i}",), "Intent_howto", "a how-to guide matches"))
            self.meta[self.kb.rules[-1]] = ("intent", "howto", 2, 2)
        # an app is "named" when one of its name words is typed
        # (only distinctive words count: not a shared synonym, not a word used in the guides themselves)
        for app_id, words in self.app_words.items():
            for w in sorted(words):
                s_ = self.word_sym(w)
                if s_.startswith("W_") and s_ not in self._guide_words:
                    self.kb.tell_rule(Rule((s_,), f"Named_{app_id}", f"'{w}' names the app {app_id}"))
        # 4. what a drafted message is about
        for name, (triggers, _msgs) in TEMPLATES.items():
            for trig in triggers:
                prem = [self.word_sym(w) for w in norm(trig).split()]
                if prem:
                    self._add(prem, "Topic_" + name, "topic", name, 0, f"the words '{trig}' mean the message is about {name}")
            self.kb.tell_rule(Rule(("Topic_" + name,), "HasTopic", "a message topic was found"))

    def _content(self, phrase):
        out = []
        for w in norm(phrase).split():
            s = self.word_sym(w)
            if w in self.stop or s == "C_how":
                continue
            out.append(s)
        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------ reading a message
    def _tokens(self, text):
        out = []
        known = sorted(self.vocab)
        for w in norm(text).split():
            if w not in self.vocab and len(w) >= 4:
                near = difflib.get_close_matches(w, known, n=1, cutoff=0.8)
                if near:
                    w = near[0]
            out.append(w)
        return out

    def analyze(self, text, current_app_id=None):
        toks = self._tokens(text)
        facts = {f"W_{w}" for w in toks}
        if toks:
            facts.add("First_" + self._peek(toks[0])[2:])
        recipient = find_recipient(text)
        if recipient:
            facts.add("HasRecipient")
        if re.search(r"[\"“'].{3,300}?[\"”']\s*$|[\"“].{3,300}?[\"”]", text):
            facts.add("HasQuote")

        self.kb.facts = facts
        self.kb._closure = None
        closure = self.kb.closure()
        u = Understanding(recipient=recipient)

        # intent: most specific satisfied rule wins
        best = None
        for rule, (kind, payload, pri, spec) in self.meta.items():
            if kind == "intent" and all(p in closure.facts for p in rule.premises):
                key = (pri, spec)
                if best is None or key > best[0]:
                    best = (key, payload, rule)
        if best:
            u.intent, u.intent_rule = best[1], str(best[2])
            u.proof = [f"{r}   ({r.note})" for _, r in closure.proof("Intent_" + best[1]) if r is not None][-3:]

        # how-to guides: most specific satisfied rule per guide, plus small boosts for the app being used
        per_guide = {}
        for rule, (kind, payload, _p, spec) in self.meta.items():
            if kind == "guide" and all(p in closure.facts for p in rule.premises):
                per_guide[payload] = max(per_guide.get(payload, 0), spec)
        named = {a for a, ws in self.app_words.items() if ws & set(toks)}
        scored = []
        for i, spec in per_guide.items():
            app_id = self.docs[i]["app_id"]
            score = spec + (1.0 if app_id in named else 0.0) + (0.5 if app_id == current_app_id else 0.0)
            scored.append((score, -i, i))
        scored.sort(reverse=True)
        u.guides = [(self.docs[i], score) for score, _, i in scored[:6]]

        # topic of a drafted message
        topics = {}
        for rule, (kind, payload, _p, spec) in self.meta.items():
            if kind == "topic" and all(p in closure.facts for p in rule.premises):
                topics[payload] = max(topics.get(payload, 0), spec)
        if topics:
            order = list(TEMPLATES)
            u.topic = max(topics, key=lambda t: (topics[t], -order.index(t)))
        return u
