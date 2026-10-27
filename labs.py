"""The four "labs" of the dashboard. Each one lets you run one of the course algorithms and see its working:

  Logic lab     propositional logic: truth table, resolution, forward chaining, and the logical agent thinking
  Search lab    breadth-first, depth-first and uniform-cost search on an app's screen map
  Genetic lab   the genetic algorithm: what tuning achieved, and a puzzle to try it on
  Game lab      minimax and alpha-beta pruning on tic-tac-toe

Every function takes the Streamlit module `st` and what it needs; none of them needs a screen or the internet."""
import random
import time

import pandas as pd

import game
import genetic
import logic
import params as pm
import search
from agent import LogicalAgent
from sensors import Raw, TextObs, match_terms

LOGIC_PRESETS = {
    "A chain of rules": ("Proc_whatsapp => App_whatsapp\nText_new_chat & App_whatsapp => Item_new_chat\nProc_whatsapp\nText_new_chat",
                         "Item_new_chat"),
    "Needs resolution (not a Horn clause)": ("Rain | Sprinkler\nRain => Wet\nSprinkler => Wet", "Wet"),
    "Does not follow": ("P => Q\nQ => R\nQ", "P"),
}

AGENT_PRESETS = {
    "A button inside WhatsApp": ("WhatsApp.exe", "WhatsApp", "New chat, 10"),
    "A desktop icon label": ("explorer.exe", "Program Manager", "Excel, 12"),
    "The word 'Settings' inside Gmail": ("chrome.exe", "Gmail - Inbox", "Settings, 6"),
    "Something it does not know": ("", "", "Zorblax, 8"),
    "A private-looking window": ("chrome.exe", "Bank login - Chrome", "Sign in, 5"),
}


# ------------------------------------------------------------------------------------------------ logic lab
def logic_lab(st, kbase, understander, sensitive_words=()):
    st.subheader("1. Ask a knowledge base a question")
    st.caption("Three ways to answer \"does the knowledge base entail this?\": a truth table (model checking), resolution "
               "(proof by refutation) and forward chaining (only for Horn clauses). They must always agree.")
    preset = st.selectbox("Example", list(LOGIC_PRESETS), key="logic_preset")
    kb_default, q_default = LOGIC_PRESETS[preset]
    text = st.text_area("Knowledge base (one sentence per line; use ~ & | => <=>)", kb_default, key=f"logic_kb_{preset}", height=130)
    query = st.text_input("Query", q_default, key=f"logic_q_{preset}")
    if st.button("Check", key="logic_check"):
        sentences = [ln.strip() for ln in text.splitlines() if ln.strip()]
        try:
            rows = []
            tt = logic.tt_entails(sentences, query)
            rows.append({"method": "Truth table (model checking)", "entailed?": "yes" if tt.entails else "no",
                         "work done": f"{tt.models_checked} models checked, the KB is true in {tt.models_where_kb_true}"})
            rs = logic.resolution_entails(sentences, query)
            rows.append({"method": "Resolution (refutation)", "entailed?": "yes" if rs.entails else "no",
                         "work done": f"{rs.resolvents} resolvents produced, {rs.clauses} clauses"})
            kb = logic.KB()
            for s in sentences:
                kb.tell(s)
            q = logic.parse(query)
            if kb.others or q[0] != "sym":
                rows.append({"method": "Forward chaining", "entailed?": "n/a",
                             "work done": "only works when every sentence is a Horn clause and the query is one symbol"})
            else:
                closure = kb.closure()
                rows.append({"method": "Forward chaining", "entailed?": "yes" if closure.holds(q[1]) else "no",
                             "work done": f"{closure.steps} rule firings, derived {len(closure.facts)} facts"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if not tt.entails and tt.counterexample is not None:
                shown = ", ".join(f"{k}={'T' if v else 'F'}" for k, v in tt.counterexample.items())
                st.write(f"Counter-example (the KB is true but the query is false): {shown}")
            if q[0] == "sym" and not kb.others and kb.closure().holds(q[1]):
                st.write("Proof by forward chaining:")
                st.code("\n".join(f"{s}   <-   {r if r else 'given'}" for s, r in kb.closure().proof(q[1])))
            answers = {r["entailed?"] for r in rows if r["entailed?"] != "n/a"}
            (st.success if len(answers) == 1 else st.error)("All methods agree." if len(answers) == 1 else "The methods disagree!")
        except logic.LogicError as e:
            st.error(f"I could not read that: {e}")

    st.divider()
    st.subheader("2. Watch the logical agent decide what is under the cursor")
    st.caption("This is the same agent the app uses. Here the words are typed in, so no screen or OCR is needed.")
    ex = st.selectbox("Example situation", list(AGENT_PRESETS), key="agent_preset")
    proc0, title0, words0 = AGENT_PRESETS[ex]
    c1, c2, c3 = st.columns(3)
    proc = c1.text_input("Program (process name)", proc0, key=f"ag_proc_{ex}")
    title = c2.text_input("Window title", title0, key=f"ag_title_{ex}")
    words = c3.text_input("Words near the cursor: 'word, distance in px' separated by ;", words0, key=f"ag_words_{ex}")
    trigger = st.radio("How was it triggered?", ["hover", "ask"], horizontal=True, key=f"ag_trig_{ex}")
    if st.button("Run one step of the agent", key="agent_run"):
        agent = LogicalAgent(kbase, pm.Params(), sensitive_words)
        texts = []
        for part in words.split(";"):
            if not part.strip():
                continue
            w, _, d = part.rpartition(",")
            if not w:
                w, d = part, "10"
            try:
                dist = float(d)
            except ValueError:
                w, dist = part, 10.0
            texts.append(TextObs(w.strip(), 0.9, dist, match_terms(w.strip(), agent.compiled.text_terms)))
        raw = Raw(process=proc, title=title, texts=sorted(texts, key=lambda t: t.dist), ocr_available=True)
        rec = agent.step(raw, trigger)
        a, b = st.columns(2)
        a.markdown("**1. PERCEIVE and TELL** (facts added to the knowledge base)")
        a.code("\n".join(rec.facts_told) or "(nothing)")
        b.markdown("**2. INFER** (forward chaining, rules that fired)")
        b.code("\n".join(p.split("   (")[0] for p in rec.proof) or "(no rule fired)")
        st.markdown("**3. RANK** the evidence (noisy-OR of the clues)")
        if rec.evidence:
            st.dataframe(pd.DataFrame([{"clue": e.kind, "what was seen": e.detail, "trust": round(e.score, 2)} for e in rec.evidence]),
                         hide_index=True, width="stretch")
        st.markdown(f"**4. ASK** which action follows: `{rec.action}`"
                    + (f"  → {rec.name} ({rec.confidence:.0%} sure)" if rec.matched else ""))
        if rec.action == "ask_teach":
            st.info("Closed-world assumption: what the knowledge base cannot prove is not assumed, so the agent asks to be taught.")

    st.divider()
    st.subheader("3. How a typed sentence is understood")
    st.caption("Words become facts, synonym rules turn them into concepts, intent rules and guide rules fire, "
               "and the most specific rule wins. There is no trained model.")
    sentence = st.text_input("Type something", "how do i send a pic on whatsapp", key="nlu_text")
    if sentence:
        u = understander.analyze(sentence)
        st.write(f"**Intent:** `{u.intent}`" + (f"   (rule: `{u.intent_rule}`)" if u.intent_rule else ""))
        if u.proof:
            st.code("\n".join(u.proof))
        if u.guides:
            st.dataframe(pd.DataFrame([{"guide": d["title"], "app": d["app_name"], "score": round(s, 1)} for d, s in u.guides]),
                         hide_index=True, width="stretch")
        if u.topic or u.recipient:
            st.write(f"Message topic: `{u.topic or '-'}`   recipient: `{u.recipient or '-'}`")


# ------------------------------------------------------------------------------------------------ search lab
def _dot(flow, plan):
    on_path = set(zip(plan.states, plan.states[1:])) if plan and plan.found else set()
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#f6f4ee", fontname="Helvetica"];']
    for s in flow.states():
        extra = ', fillcolor="#f2b84b"' if plan and plan.found and s in plan.states else ""
        if s == flow.start:
            extra = ', fillcolor="#9bd3c9"'
        lines.append(f'"{s}" [label="{s.replace("_", " ")}"{extra}];')
    for a, outs in flow.edges.items():
        for b, action, cost in outs:
            hot = (a, b) in on_path
            label = (action if len(action) <= 22 else action[:21] + "…").replace('"', "'")
            lines.append(f'"{a}" -> "{b}" [label="{label} ({cost:g})", fontsize=9'
                         + (', color="#d97706", penwidth=2.5' if hot else ', color="#94a3b8"') + "];")
    lines.append("}")
    return "\n".join(lines)


def search_lab(st, flows):
    st.caption("An app is a graph: each screen is a state, each click or key is an action with an effort cost (a click is 1, typing 2, "
               "a file window or shortcut 3). A how-to answer is a path from the starting screen to the goal screen.")
    app = st.selectbox("App", list(flows), format_func=lambda a: flows[a].name, key="search_app")
    flow = flows[app]
    goal = st.selectbox("Goal", [g["guide"] for g in flow.goals], key=f"search_goal_{app}")
    state = next(g["state"] for g in flow.goals if g["guide"] == goal)
    plans = search.compare(flow, flow.start, state)
    rows = []
    for name, p in plans.items():
        rows.append({"algorithm": name, "route found": "yes" if p.found else "no", "steps": p.steps, "total effort": p.cost,
                     "screens expanded": p.expanded, "screens generated": p.generated, "largest frontier": p.max_frontier})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    cols = st.columns(3)
    notes = {"BFS": "Fewest steps. Optimal only when every step costs the same.",
             "DFS": "Goes deep first. Finds a route, not always a good one, but remembers little.",
             "UCS": "Cheapest total effort. Optimal for any non-negative costs. This is the route the chat shows."}
    for col, (name, p) in zip(cols, plans.items()):
        col.markdown(f"**{name}**")
        col.caption(notes[name])
        if p.found:
            col.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(p.actions, 1)))
            col.caption("Expanded in this order: " + " → ".join(s.replace("_", " ") for s in p.order))
    which = st.radio("Highlight the route found by", list(plans), index=2, horizontal=True, key="search_hl")
    st.graphviz_chart(_dot(flow, plans[which]))


# ------------------------------------------------------------------------------------------------ genetic lab
def _toy(n=8, seed=3):
    rng = random.Random(seed)
    ranges = [(0.0, 10.0)] * n
    target = [rng.uniform(1, 9) for _ in range(n)]

    def fit(v):
        return -sum(((x - t) / 10.0) ** 2 for x, t in zip(v, target))
    return fit, ranges, target


def genetic_lab(st, tuned_file):
    st.subheader("1. What tuning did to the agent")
    st.caption("The agent has 12 settings (distances, similarity cut-offs, how much each clue is trusted). `python tune.py` evolves them "
               "on made-up labelled screens and keeps the result only if it is better on screens the GA never saw.")
    info = None
    data = None
    try:
        import json
        data = json.loads(open(tuned_file, encoding="utf-8").read())
        info = data.get("info", {})
    except Exception:
        pass
    if not info:
        st.info("No tuned settings yet. Run  `python tune.py`  in the project folder (it takes a few minutes), then refresh this page.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tuning screens, default", f"{info['accuracy_default_train']:.0%}")
        c2.metric("Tuning screens, tuned", f"{info['accuracy_tuned_train']:.0%}", f"{info['accuracy_tuned_train'] - info['accuracy_default_train']:+.0%}")
        c3.metric("Unseen screens, default", f"{info['accuracy_default_held_out']:.0%}")
        c4.metric("Unseen screens, tuned", f"{info['accuracy_tuned_held_out']:.0%}",
                  f"{info['accuracy_tuned_held_out'] - info['accuracy_default_held_out']:+.0%}")
        st.caption(f"{info['scenarios']} made-up screens ({info['train']} to evolve on, {info['held_out']} held back). "
                   f"Population {info['population']}, {info['generations']} generations, {info['evaluations']} fitness evaluations. "
                   f"Random search with the same effort reached fitness {info['fitness_random_search_train']:.3f} against the GA's "
                   f"{info['fitness_tuned_train']:.3f}. The screens are synthetic, so this is evidence the method works, not a promise for every real screen.")
        hist = pd.DataFrame(info["history"])
        st.line_chart(hist.set_index("generation")[["best", "mean"]])
        default = pm.Params()
        tuned = data["params"]
        st.dataframe(pd.DataFrame([{"setting": k, "default": round(getattr(default, k), 3), "tuned": round(tuned[k], 3)} for k in pm.GENES]),
                     hide_index=True, width="stretch")

    st.divider()
    st.subheader("2. Try the genetic algorithm on a puzzle")
    st.caption("Find 8 hidden numbers between 0 and 10. The GA only sees one score (how close the guess is); it never sees the answer.")
    c1, c2, c3, c4 = st.columns(4)
    pop = c1.slider("Population", 6, 80, 30, key="ga_pop")
    gens = c2.slider("Generations", 2, 60, 25, key="ga_gens")
    mut = c3.slider("Mutation rate", 0.0, 0.5, 0.15, key="ga_mut")
    kind = c4.selectbox("Crossover", ["one_point", "uniform"], key="ga_cross")
    seed = st.number_input("Random seed", 0, 9999, 1, key="ga_seed")
    if st.button("Run the genetic algorithm", key="ga_run"):
        fit, ranges, target = _toy()
        res = genetic.evolve(fit, ranges, pop_size=pop, generations=gens, mutation_rate=mut, crossover_kind=kind, rng=random.Random(int(seed)))
        base = genetic.random_search(fit, ranges, res.evaluations, rng=random.Random(int(seed)))
        a, b, c = st.columns(3)
        a.metric("GA best score (0 = perfect)", f"{res.best_fitness:.4f}")
        b.metric("Random search, same effort", f"{base.best_fitness:.4f}")
        c.metric("Evaluations", res.evaluations)
        st.line_chart(pd.DataFrame(res.history).set_index("generation")[["best", "mean", "worst"]])
        st.dataframe(pd.DataFrame({"gene": range(1, len(target) + 1), "hidden target": [round(t, 2) for t in target],
                                   "GA's answer": [round(x, 2) for x in res.best]}), hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------------ game lab
def game_lab(st):
    st.caption("Minimax assumes both players play perfectly. Alpha-beta pruning returns the same answer but skips branches that cannot matter. "
               "Set up any position and compare how many positions each method has to look at.")
    st.write("Set the board (X always starts):")
    cells = []
    for r in range(3):
        cols = st.columns(3)
        for c in range(3):
            cells.append(cols[c].selectbox(f"square {r * 3 + c + 1}", [" ", "X", "O"], key=f"game_cell_{r * 3 + c}", label_visibility="collapsed"))
    board = "".join(cells)
    nx, no = board.count("X"), board.count("O")
    limit = st.slider("Look ahead (0 = play to the end of the game)", 0, 9, 0, key="game_limit")
    if not (nx == no or nx == no + 1):
        st.warning("That is not a position that can happen: X goes first, so X has the same number of marks as O, or one more.")
        return
    if game.is_over(board):
        st.info(f"The game is over: {game.result_text(board)}.")
        return
    turn = game.to_move(board)
    st.write(f"{turn} to move.")
    if st.button("Compare the search methods", key="game_run"):
        rows, best = [], None
        for name in game.ALGORITHMS:
            t0 = time.perf_counter()
            ch = game.best_move(board, name, limit or None)
            ms = (time.perf_counter() - t0) * 1000
            best = best or ch
            rows.append({"method": name, "best square": ch.move + 1, "value": ch.value, "positions visited": ch.stats.nodes,
                         "branches pruned": ch.stats.cutoffs, "time (ms)": round(ms, 1)})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        st.bar_chart(df.set_index("method")["positions visited"])
        if df.loc[0, "positions visited"]:
            saved = 1 - df.loc[1, "positions visited"] / df.loc[0, "positions visited"]
            st.write(f"Alpha-beta looked at {saved:.0%} fewer positions than minimax; with the centre-first move order "
                     f"{1 - df.loc[2, 'positions visited'] / df.loc[0, 'positions visited']:.0%} fewer.")
        st.caption("Value: positive is good for the player to move. With a full search it is 0 for a draw, or 10 minus the number of moves "
                   "for a win (so quicker wins score higher). With a limited look-ahead it comes from an evaluation function instead.")
        exact = game.best_move(board, "minimax", limit or None)
        st.write("Value of each move (exact, from plain minimax):")
        st.dataframe(pd.DataFrame([{"square": i + 1, "value": v} for i, v in sorted(exact.values.items())]), hide_index=True)
