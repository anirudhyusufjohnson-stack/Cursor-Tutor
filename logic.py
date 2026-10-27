"""Propositional logic engine (written from scratch).

Contents
  parse / fmt            sentences such as   Proc_whatsapp & Text_new_chat => Item_new_chat
  tt_entails             model checking by truth-table enumeration (TT-ENTAILS)
  resolution_entails     proof by refutation using resolution on CNF clauses (PL-RESOLUTION)
  forward_chain          data-driven inference for Horn clauses (PL-FC-ENTAILS), with proof records
  KB                     a knowledge base you can TELL sentences/facts to and ASK queries of

Sentence syntax: symbols are words (letters, digits, underscore). Operators, highest binding first:
  ~ (not)   & (and)   | (or)   => (implies)   <=> (if and only if).   Unicode ¬ ∧ ∨ ⇒ ⇔ also work.
"""
import itertools
import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------------------------ sentences
# A sentence is a nested tuple:  ('sym', 'P')  ('not', A)  ('and', A, B)  ('or', A, B)  ('imp', A, B)  ('iff', A, B)

_TOKEN = re.compile(r"\s*(<=>|⇔|<->|=>|⇒|->|[~¬!&∧|∨()]|\w+)", re.UNICODE)
_NOT, _AND, _OR = {"~", "¬", "!"}, {"&", "∧"}, {"|", "∨"}
_IMP, _IFF = {"=>", "⇒", "->"}, {"<=>", "⇔", "<->"}


class LogicError(ValueError):
    pass


def _tokenize(text):
    pos, out = 0, []
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise LogicError(f"cannot read the sentence near: {text[pos:pos + 12]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def iff(self):
        left = self.imp()
        while self.peek() in _IFF:
            self.take()
            left = ("iff", left, self.imp())
        return left

    def imp(self):
        left = self.or_()
        if self.peek() in _IMP:
            self.take()
            return ("imp", left, self.imp())  # right associative
        return left

    def or_(self):
        left = self.and_()
        while self.peek() in _OR:
            self.take()
            left = ("or", left, self.and_())
        return left

    def and_(self):
        left = self.not_()
        while self.peek() in _AND:
            self.take()
            left = ("and", left, self.not_())
        return left

    def not_(self):
        if self.peek() in _NOT:
            self.take()
            return ("not", self.not_())
        return self.atom()

    def atom(self):
        tok = self.take()
        if tok is None:
            raise LogicError("the sentence ends too early")
        if tok == "(":
            inner = self.iff()
            if self.take() != ")":
                raise LogicError("missing )")
            return inner
        if re.fullmatch(r"\w+", tok, re.UNICODE):
            return ("sym", tok)
        raise LogicError(f"unexpected {tok!r}")


def parse(text):
    """Text -> sentence. Sentences that are already parsed are returned unchanged."""
    if isinstance(text, tuple):
        return text
    p = _Parser(_tokenize(text))
    s = p.iff()
    if p.peek() is not None:
        raise LogicError(f"unexpected {p.peek()!r}")
    return s


def fmt(s):
    """Sentence -> readable text with ¬ ∧ ∨ ⇒ ⇔."""
    op = s[0]
    if op == "sym":
        return s[1]
    if op == "not":
        inner = fmt(s[1])
        return "¬" + (inner if s[1][0] in ("sym", "not") else f"({inner})")
    sym = {"and": " ∧ ", "or": " ∨ ", "imp": " ⇒ ", "iff": " ⇔ "}[op]
    parts = []
    for child in s[1:]:
        text = fmt(child)
        parts.append(text if child[0] in ("sym", "not") else f"({text})")
    return sym.join(parts)


def symbols(s):
    if s[0] == "sym":
        return {s[1]}
    out = set()
    for child in s[1:]:
        out |= symbols(child)
    return out


def evaluate(s, model):
    """Truth value of a sentence in a model (dict symbol -> bool)."""
    op = s[0]
    if op == "sym":
        return model[s[1]]
    if op == "not":
        return not evaluate(s[1], model)
    a = evaluate(s[1], model)
    if op == "and":
        return a and evaluate(s[2], model)
    if op == "or":
        return a or evaluate(s[2], model)
    if op == "imp":
        return (not a) or evaluate(s[2], model)
    return a == evaluate(s[2], model)  # iff


# ------------------------------------------------------------------------------------ model checking
@dataclass
class TTResult:
    entails: bool
    models_checked: int
    models_where_kb_true: int
    counterexample: dict = None  # a model where the KB is true but the query is false


def tt_entails(kb, query, max_symbols=18):
    """Does KB entail query?  Enumerate every model of the symbols and check each one."""
    sentences = [parse(s) for s in kb]
    alpha = parse(query)
    syms = sorted(set().union(symbols(alpha), *[symbols(s) for s in sentences]))
    if len(syms) > max_symbols:
        raise LogicError(f"{len(syms)} symbols means {2 ** len(syms):,} models: too many for a truth table")
    checked = kb_true = 0
    for values in itertools.product([True, False], repeat=len(syms)):
        model = dict(zip(syms, values))
        checked += 1
        if all(evaluate(s, model) for s in sentences):
            kb_true += 1
            if not evaluate(alpha, model):
                return TTResult(False, checked, kb_true, model)
    return TTResult(True, checked, kb_true)


def truth_table(sentence, max_symbols=8):
    """Rows of (model, value) for one sentence, for display."""
    s = parse(sentence)
    syms = sorted(symbols(s))
    if len(syms) > max_symbols:
        raise LogicError("too many symbols to show a truth table")
    return syms, [(dict(zip(syms, v)), evaluate(s, dict(zip(syms, v))))
                  for v in itertools.product([True, False], repeat=len(syms))]


# ------------------------------------------------------------------------------------ CNF and resolution
def _eliminate(s):
    op = s[0]
    if op == "sym":
        return s
    if op == "not":
        return ("not", _eliminate(s[1]))
    a, b = _eliminate(s[1]), _eliminate(s[2])
    if op == "imp":
        return ("or", ("not", a), b)
    if op == "iff":
        return ("and", ("or", ("not", a), b), ("or", ("not", b), a))
    return (op, a, b)


def _nnf(s, negate=False):
    op = s[0]
    if op == "sym":
        return ("not", s) if negate else s
    if op == "not":
        return _nnf(s[1], not negate)
    if negate:  # De Morgan
        return ("or" if op == "and" else "and", _nnf(s[1], True), _nnf(s[2], True))
    return (op, _nnf(s[1]), _nnf(s[2]))


def _clauses(s):
    op = s[0]
    if op == "sym":
        return {frozenset([s[1]])}
    if op == "not":
        return {frozenset(["~" + s[1][1]])}
    a, b = _clauses(s[1]), _clauses(s[2])
    if op == "and":
        return a | b
    return {x | y for x in a for y in b}  # or: distribute over and


def _tautology(clause):
    return any(("~" + lit) in clause for lit in clause if not lit.startswith("~"))


def to_cnf(sentence):
    """Sentence -> set of clauses; a clause is a frozenset of literals such as 'P' and '~Q'."""
    return {c for c in _clauses(_nnf(_eliminate(parse(sentence)))) if not _tautology(c)}


def _negate(lit):
    return lit[1:] if lit.startswith("~") else "~" + lit


def _resolvents(c1, c2):
    out = []
    for lit in c1:
        if _negate(lit) in c2:
            r = (c1 - {lit}) | (c2 - {_negate(lit)})
            if not _tautology(r):
                out.append(frozenset(r))
    return out


@dataclass
class ResolutionResult:
    entails: bool
    resolvents: int          # how many new clauses were produced
    clauses: int             # clauses in the end
    gave_up: bool = False


def resolution_entails(kb, query, max_clauses=3000):
    """Proof by refutation: add NOT query to the KB; if resolution derives the empty clause, KB entails query."""
    clauses = set()
    for s in kb:
        clauses |= to_cnf(s)
    clauses |= to_cnf(("not", parse(query)))
    made = 0
    while True:
        current = list(clauses)
        new = set()
        for c1, c2 in itertools.combinations(current, 2):
            for r in _resolvents(c1, c2):
                made += 1
                if not r:
                    return ResolutionResult(True, made, len(clauses))
                new.add(r)
        if new <= clauses:
            return ResolutionResult(False, made, len(clauses))
        clauses |= new
        if len(clauses) > max_clauses:
            return ResolutionResult(False, made, len(clauses), gave_up=True)


# ------------------------------------------------------------------------------------ Horn clauses and forward chaining
@dataclass(frozen=True)
class Rule:
    premises: tuple
    conclusion: str
    note: str = field(default="", compare=False)  # plain-language meaning, used in proofs

    def __str__(self):
        return (" ∧ ".join(self.premises) + " ⇒ " if self.premises else "") + self.conclusion


def horn_from_sentence(sentence, note=""):
    """'A & B => C' or 'C' -> Rule, or None if the sentence is not a definite Horn clause."""
    s = parse(sentence)
    if s[0] == "sym":
        return Rule((), s[1], note)
    if s[0] != "imp" or s[2][0] != "sym":
        return None
    prem, stack = [], [s[1]]
    while stack:
        t = stack.pop()
        if t[0] == "and":
            stack.extend([t[2], t[1]])
        elif t[0] == "sym":
            prem.append(t[1])
        else:
            return None
    return Rule(tuple(dict.fromkeys(prem)), s[2][1], note)


@dataclass
class Closure:
    """Everything that follows from the facts, and how each thing was derived."""
    facts: set
    why: dict            # symbol -> Rule that produced it, or None for a given fact
    steps: int           # rule firings

    def holds(self, sym):
        return sym in self.facts

    def proof(self, sym):
        """Ordered derivation of `sym`: list of (symbol, rule or None). Facts come first."""
        order, seen = [], set()

        def visit(x):
            if x in seen:
                return
            seen.add(x)
            r = self.why.get(x)
            if r:
                for p in r.premises:
                    visit(p)
            order.append((x, r))

        if sym in self.facts:
            visit(sym)
        return order

    def leaves(self, sym):
        """The given facts that the derivation of `sym` rests on."""
        return [x for x, r in self.proof(sym) if r is None]


def forward_chain(rules, facts):
    """PL-FC-ENTAILS: fire every rule whose premises are all known until nothing new appears."""
    count = {i: len(r.premises) for i, r in enumerate(rules)}
    by_premise = {}
    for i, r in enumerate(rules):
        for p in r.premises:
            by_premise.setdefault(p, []).append(i)
    # sorted so that the same facts always give the same derivation (set order changes from run to run)
    why, agenda, inferred, steps = {}, sorted(facts, reverse=True), set(), 0
    for f in sorted(facts):
        why[f] = None
    for i, r in enumerate(rules):  # rules with no premises are facts
        if not r.premises and r.conclusion not in why:
            why[r.conclusion] = r
            agenda.append(r.conclusion)
    while agenda:
        p = agenda.pop()
        if p in inferred:
            continue
        inferred.add(p)
        for i in by_premise.get(p, ()):
            count[i] -= 1
            if count[i] == 0:
                steps += 1
                c = rules[i].conclusion
                if c not in why:
                    why[c] = rules[i]
                    agenda.append(c)
    return Closure(inferred, why, steps)


# ------------------------------------------------------------------------------------ knowledge base
class KB:
    """A propositional knowledge base.

    TELL sentences (rules) once, TELL facts as they are perceived, then ASK what follows.
    Rules that are Horn clauses are used by forward chaining. Any sentence can be used by the
    truth-table and resolution methods."""

    def __init__(self):
        self.rules = []         # Horn rules (definite clauses)
        self.others = []        # sentences that are not Horn clauses (kept for truth table / resolution)
        self.facts = set()
        self._closure = None

    def tell(self, sentence, note=""):
        s = parse(sentence)
        rule = horn_from_sentence(s, note)
        if rule:
            self.rules.append(rule)
        else:
            self.others.append(s)
        self._closure = None
        return rule

    def tell_rule(self, rule):
        """Add a ready-made Rule (used when a large rule set is compiled from data)."""
        self.rules.append(rule)
        self._closure = None

    def tell_fact(self, sym):
        self.facts.add(sym)
        self._closure = None

    def retract(self, syms):
        self.facts -= set(syms)
        self._closure = None

    def sentences(self):
        """Every rule and fact as a sentence."""
        out = []
        for r in self.rules:
            if not r.premises:
                out.append(("sym", r.conclusion))
                continue
            body = ("sym", r.premises[0])
            for p in r.premises[1:]:
                body = ("and", body, ("sym", p))
            out.append(("imp", body, ("sym", r.conclusion)))
        return out + list(self.others) + [("sym", f) for f in sorted(self.facts)]

    def closure(self):
        if self._closure is None:
            self._closure = forward_chain(self.rules, self.facts)
        return self._closure

    def ask(self, query, method="forward"):
        """True/False: does the KB (with its facts) entail the query?"""
        if method == "forward":
            q = parse(query)
            if q[0] != "sym":
                raise LogicError("forward chaining answers questions about one symbol")
            return self.closure().holds(q[1])
        if method == "truth_table":
            return tt_entails(self.sentences(), query).entails
        if method == "resolution":
            return resolution_entails(self.sentences(), query).entails
        raise LogicError(f"unknown method {method!r}")
