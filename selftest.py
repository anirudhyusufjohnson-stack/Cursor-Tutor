"""Self-test: checks every AI part of Cursor Tutor on made-up data.      Run:  python selftest.py
No screen, internet or API key is needed. Checks that need the OCR network are skipped if it is not installed."""
import random
import sys
import tempfile
from pathlib import Path

from synth import ctx_for, fake_screen  # noqa: F401  (fake_screen / ctx_for are re-exported for convenience)


def run():
    import composer
    import game
    import genetic
    import logic
    import params as pm
    import search
    import tune
    from ai_tutor import DEFAULT_QUESTION, TUNED_FILE, Tutor
    from config import KNOWLEDGE_DIR, cfg
    from drafting import make_draft
    from icons import IconStore
    from knowledge import KnowledgeBase, name_score
    from nlu import Understander
    from ocr_engine import OCR
    from sensors import Raw, TextObs, match_terms
    from target import segment_target

    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))

    def section(title):
        print(f"\n--- {title}")

    tmp = Path(tempfile.mkdtemp())
    kb = KnowledgeBase(KNOWLEDGE_DIR, tmp)
    tutor = Tutor(cfg, learned_dir=tmp, params=pm.Params())
    agent = tutor.agent

    # ================================================================ 1. propositional logic
    section("Propositional logic: truth table, resolution, forward chaining")
    s = logic.parse("A & ~B => C | D")
    check("parser: precedence and round trip", logic.parse(logic.fmt(s)) == s and logic.fmt(logic.parse("P <=> Q")) == "P ⇔ Q",
          logic.fmt(s))
    cases = [
        (["P => Q", "Q => R", "P"], "R", True),                         # modus ponens twice
        (["A | B", "~A | C", "~B | C"], "C", True),                     # needs resolution, not a Horn KB
        (["P => Q", "Q => R", "Q"], "P", False),                        # does not follow (affirming the consequent)
        (["Rain | Sprinkler", "Rain => Wet", "Sprinkler => Wet"], "Wet", True),
        (["A <=> B", "B"], "A", True),
    ]
    agree = []
    for sentences, q, want in cases:
        tt, rs = logic.tt_entails(sentences, q).entails, logic.resolution_entails(sentences, q).entails
        agree.append(tt == rs == want)
    check("truth table and resolution agree with the textbook answers", all(agree), f"{sum(agree)}/{len(cases)}")
    tt = logic.tt_entails(["P => Q", "Q => R", "Q"], "P")
    check("a counter-example is a model where the KB is true and the query false",
          tt.counterexample is not None and all(logic.evaluate(logic.parse(x), tt.counterexample) for x in ["P => Q", "Q => R", "Q"])
          and not tt.counterexample["P"])
    rng = random.Random(4)
    ok_all, total = True, 0
    for _ in range(25):   # random definite-clause KBs: all three methods must agree
        syms = [f"S{i}" for i in range(6)]
        sentences = [f"{' & '.join(rng.sample(syms, rng.randint(1, 2)))} => {rng.choice(syms)}" for _ in range(5)]
        sentences += rng.sample(syms, 2)
        k = logic.KB()
        for x in sentences:
            k.tell(x)
        for q in syms:
            total += 1
            a = k.ask(q, "forward")
            b = logic.tt_entails(sentences, q).entails
            c = logic.resolution_entails(sentences, q).entails
            ok_all &= (a == b == c)
    check("forward chaining = truth table = resolution on 25 random Horn KBs", ok_all, f"{total} queries")
    k = logic.KB()
    for x in ["Proc_x => App_x", "Text_b & App_x => Item_b", "Proc_x", "Text_b"]:
        k.tell(x)
    proof = [str(r) for _, r in k.closure().proof("Item_b") if r and r.premises]
    check("forward chaining records a proof", proof == ["Proc_x ⇒ App_x", "Text_b ∧ App_x ⇒ Item_b"], str(proof))

    # ================================================================ 2. the logical agent
    section("Logical agent (perceive -> TELL -> infer -> ASK), using hand-made percepts, no OCR needed")
    comp = agent.compiled

    def raw_for(process="", title="", words=(), control=""):
        texts = sorted([TextObs(w, 0.9, d, match_terms(w, comp.text_terms)) for w, d in words], key=lambda t: t.dist)
        return Raw(process=process, title=title, control=control,
                   control_matches=match_terms(control, comp.text_terms) if control else [], texts=texts, ocr_available=True)

    def look(**kw):
        trig = kw.pop("trigger", "ask")
        return agent.step(raw_for(**kw), trig)

    r = look(process="WhatsApp.exe", title="WhatsApp", words=[("New chat", 10)])
    check("window + word  =>  app and button", r.app and r.app["id"] == "whatsapp" and r.element and r.element["names"][0] == "new chat"
          and r.action == "explain_item" and "Proc_whatsapp_exe ⇒ App_whatsapp" in " ".join(r.proof), f"{r.confidence:.2f} {r.action}")
    r = look(process="chrome.exe", title="Gmail - Inbox", words=[("Settings", 6)])
    check("a word never overrides the window's app (Gmail 'Settings' is not the Windows Settings app)",
          r.app and r.app["id"] == "gmail" and r.element and r.element["names"][0] == "settings", f"{r.app and r.app['name']} / {r.name}")
    r = look(process="explorer.exe", title="Program Manager", words=[("Excel", 8)])
    check("a name on the bare desktop  =>  that app", r.app and r.app["id"] == "excel" and r.action == "explain_app", f"{r.name}")
    r = look(process="", title="", words=[("Zorblax", 8)])
    check("closed-world assumption: what cannot be proved is not guessed", not r.matched and r.action == "ask_teach" and r.seen_text == "Zorblax")
    check("private-looking window: Act_Pause is derived for hover, not for an explicit request",
          agent.pauses("chrome.exe", "Bank login - Chrome", "hover") and not agent.pauses("chrome.exe", "Bank login - Chrome", "ask")
          and not agent.pauses("notepad.exe", "notes.txt - Notepad", "hover"))
    r = look(process="chrome.exe", title="Bank login", words=[("Sign in", 5)], trigger="hover")
    check("the agent's action for a private window is 'pause'", r.action == "pause", r.action)
    look(process="WhatsApp.exe", title="WhatsApp", words=[("New chat", 10)])
    remembered = agent.remembered_app()
    check("the agent remembers what it last looked at (a fact in its KB)", remembered and remembered["id"] == "whatsapp")
    r = look(process="", title="", words=[("Zorblax", 8)])
    tutor.teach("Zorblax", "item", "Zorblax is a made-up button.", ["Click it.", "Enjoy."], None)
    comp = tutor.agent.compiled
    r = tutor.agent.step(Raw(texts=[TextObs("Zorblax", 0.9, 8, match_terms("Zorblax", comp.text_terms))], ocr_available=True), "ask")
    check("teaching adds rules: the same word is now recognised", r.element and r.element["names"][0] == "zorblax" and r.action == "explain_item",
          f"{r.name} {r.confidence:.2f}")
    comp = agent.compiled

    # ================================================================ 3. language as logic
    section("Understanding what is typed (word facts -> concept rules -> intent / guide rules; most specific rule wins)")
    und = Understander(kb.howto_docs(), KNOWLEDGE_DIR / "language.json", kb.apps)
    regression = {
        "what is this button": "explain", "explain this icon": "explain", "what does this do": "explain",
        "how can i send photos": "howto", "how do i make a voice call": "howto", "steps to start a video call": "howto",
        "how to start a group": "howto", "photo kaise bhejte hain": "howto", "help me send a photo": "howto",
        "tell priya i will be late": "draft", "priya ko birthday wish bhejo": "draft", "send a thank you to dad": "draft",
        'message sam "see you at 5 at the cafe"': "draft", "write something about quantum physics": "draft",
        "hi, good morning": "greet", "hello": "greet", "namaste": "greet", "how are you doing": "greet",
        "thank you very much": "thanks", "thanks a lot": "thanks", "bye for now": "goodbye", "see you later": "goodbye",
        "what more can i do here": "more", "what else can i do": "more",
        "what are you able to do for me": "help", "what are your features": "help", "how can you assist me": "help",
        "what can you do": "help", "how do i send a message to rahul": "howto", "tell me how to send a photo": "howto",
        # a second round of phrases, written after the rules were finished (29 of 37 were right on the first run)
        "what is this thing i'm pointing at": "explain", "can you explain this to me": "explain", "what's this for": "explain",
        "how do i go about sharing a picture": "howto", "show me how to call someone": "howto", "how can i begin a video chat": "howto",
        "how to put a group together": "howto", "message kaise bhejte hain": "howto",
        "let rahul know i'm on my way": "draft", "wish anita a happy birthday": "draft", "tell mom thank you": "draft",
        "text ravi that i'll call him back": "draft", "hey there": "greet", "good evening": "greet", "thx": "thanks",
        "thanks so much": "thanks", "goodbye": "goodbye", "anything else i can do here": "more", "what else is possible": "more",
        "what are you capable of": "help", "can you help me": "help",
    }
    got = {q: und.analyze(q, "whatsapp").intent for q in regression}
    wrong = {q: (w, got[q]) for q, w in regression.items() if got[q] != w}
    check("intent rules", not wrong, f"{len(regression) - len(wrong)}/{len(regression)} correct; wrong: {wrong}")
    guide_cases = [
        ("how do i send a pic", "whatsapp", "Send a photo"), ("how to make a folder", "explorer", "Create a new folder"),
        ("share my screen", "zoom", "Share my screen"), ("make text bold", "word", "Make text bold"),
        ("how do i atach a file", "gmail", "Attach a file to an email"), ("photo kaise bhejte hain", "whatsapp", "Send a photo"),
        ("how do i save my work", "word", "Save a document"), ("how can i turn on captions", "youtube", "Turn on subtitles"),
        ("how do i connect to wifi", "settings", "Connect to Wi-Fi"), ("make a chart", "excel", "Make a chart"),
        ("delete a file", "explorer", "Delete a file"), ("how do i mute myself", "zoom", "Mute yourself"),
        ("how to add a slide", "powerpoint", "Add a slide"), ("how do i search in youtube", "youtube", "Search for a video"),
        ("my program is not responding", "taskmgr", "Close a frozen app"), ("how do i bookmark this page", "chrome", "Bookmark a page"),
        ("how can i make a new folder", "explorer", "Create a new folder"), ("copy a document", "explorer", "Copy a file"),
        ("how to send a text to someone", "whatsapp", "Send a message"), ("record a voice note", "whatsapp", "Send a voice message"),
        ("make my words thicker", "word", "Make text bold"), ("how to check my spelling", "word", "Check spelling"),
        ("how do i add up numbers in a column", "excel", "Add numbers together"), ("sort my data a to z", "excel", "Sort data"),
        ("how to turn on bluetooth", "settings", "Turn on Bluetooth"), ("make the screen brighter", "settings", "Change screen brightness"),
        ("how do i join a zoom call", "zoom", "Join a meeting"), ("speed up the video", "youtube", "Change playback speed"),
        ("how can i reply to a mail", "gmail", "Reply to an email"), ("open a private tab", "chrome", "Open a private window"),
        ("how do i start the slideshow", "powerpoint", "Start the presentation"),
    ]
    bad = []
    for q, app, title in guide_cases:
        g = und.analyze(q, app).guides
        if not g or g[0][0]["title"] != title:
            bad.append((q, g[0][0]["title"] if g else None))
    check("guide rules (typos, Hinglish, synonyms)", not bad, f"{len(guide_cases) - len(bad)}/{len(guide_cases)} correct; wrong: {bad}")
    off = [und.analyze(q).guides for q in ["what is the capital of france", "how do i make pasta", "recipe for biryani"]]
    check("unrelated questions match no guide", all(not g for g in off))
    u = und.analyze("how do i send a pic on whatsapp", None)
    check("the rule that fired can be shown", u.intent == "howto" and u.proof and "⇒" in u.proof[-1], str(u.proof[-1:]))

    # ================================================================ 4. search
    section("Search: BFS, DFS and uniform-cost search on the apps' screen maps")
    flows = search.load_flows(KNOWLEDGE_DIR / "flows.json")
    titles = {(a["id"], h["title"]) for a in kb.apps.values() for h in a.get("howtos", [])}
    goals = [(app, g["guide"], g["state"]) for app, f in flows.items() for g in f.goals]
    check("every screen-map goal is a real how-to guide", all((a, t) in titles for a, t, _ in goals), f"{len(goals)} goals")

    def best_by_relaxation(flow, weight):
        dist = {flow.start: 0.0}
        for _ in range(len(flow.states())):
            for a, outs in flow.edges.items():
                for b, _act, c in outs:
                    if a in dist and dist[a] + weight(c) < dist.get(b, 1e9):
                        dist[b] = dist[a] + weight(c)
        return dist
    all_found, ucs_opt, bfs_opt = True, True, True
    diff_bfs = diff_dfs = 0
    for app, title, state in goals:
        f = flows[app]
        plans = search.compare(f, f.start, state)
        all_found &= all(p.found for p in plans.values())
        ucs_opt &= abs(plans["UCS"].cost - best_by_relaxation(f, lambda c: c)[state]) < 1e-9
        bfs_opt &= plans["BFS"].steps == best_by_relaxation(f, lambda c: 1)[state]
        diff_bfs += plans["BFS"].cost > plans["UCS"].cost
        diff_dfs += plans["DFS"].cost > plans["UCS"].cost
    check("all three searches find a route to every goal", all_found)
    check("UCS finds the cheapest route and BFS the fewest steps (checked against an independent shortest-path calculation)",
          ucs_opt and bfs_opt)
    check("the algorithms really differ on this data (BFS is costlier than UCS somewhere, DFS too)", diff_bfs >= 1 and diff_dfs >= 1,
          f"BFS costlier on {diff_bfs} goals, DFS on {diff_dfs}")
    tutor.rec = type(tutor.rec)(app=kb.apps["explorer"], app_conf=0.9)
    ans = tutor.ask("how do i create a folder", None)
    check("a how-to answer shows the route planned by uniform-cost search",
          "Create a new folder" in ans and "uniform-cost" in ans and "fewest clicks" in ans, "")

    # ================================================================ 5. genetic algorithm
    section("Genetic algorithm")
    target = [3.0, 7.5, 1.0, 9.0, 5.0, 2.5]
    ranges = [(0.0, 10.0)] * len(target)

    def toy(v):
        return -sum(((x - t) / 10) ** 2 for x, t in zip(v, target))
    wins = 0
    gas, rss = [], []
    for seed in range(5):
        g = genetic.evolve(toy, ranges, pop_size=24, generations=20, rng=random.Random(seed))
        rs = genetic.random_search(toy, ranges, g.evaluations, rng=random.Random(seed))
        wins += g.best_fitness > rs.best_fitness
        gas.append(g.best_fitness)
        rss.append(rs.best_fitness)
    check("the GA finds the hidden numbers far better than random search with the same effort", wins == 5 and max(gas) > -0.005,
          f"GA best {max(gas):.5f}; random search best {max(rss):.5f}")
    g = genetic.evolve(toy, ranges, pop_size=20, generations=15, rng=random.Random(1), seeds=[[5.0] * 6])
    bests = [h["best"] for h in g.history]
    check("elitism: the best score never gets worse from one generation to the next", all(b2 >= b1 - 1e-12 for b1, b2 in zip(bests, bests[1:])),
          f"{bests[0]:.4f} -> {bests[-1]:.4f}")
    g1 = genetic.evolve(toy, ranges, pop_size=12, generations=8, rng=random.Random(9))
    g2 = genetic.evolve(toy, ranges, pop_size=12, generations=8, rng=random.Random(9))
    check("the same random seed gives the same result", g1.best == g2.best)
    r2 = random.Random(2)
    a, b = [0.0] * 6, [1.0] * 6
    c1, c2 = genetic.crossover(a, b, r2)
    m = genetic.mutate([5.0] * 6, ranges, 1.0, 5.0, r2)
    check("crossover mixes parents, mutation stays inside the allowed range",
          set(c1 + c2) == {0.0, 1.0} and 0.0 in c1 + c2 and all(lo <= x <= hi for x, (lo, hi) in zip(m, ranges)))
    shipped = pm.load(Path(__file__).with_name("learned") / TUNED_FILE)
    check("tuned settings load and every value is inside its allowed range",
          all(lo <= getattr(shipped, k) <= hi for k, (lo, hi) in pm.RANGES.items()))
    (tmp / TUNED_FILE).write_text('{"params": {"text_dist": 55.0}}', encoding="utf-8")
    check("the app starts with the tuned settings", Tutor(cfg, learned_dir=tmp).params.text_dist == 55.0)
    (tmp / TUNED_FILE).unlink()

    ocr = OCR(enabled=True)
    if ocr.available:
        wb = tune.make_workbench(tempfile.mkdtemp())
        _, info, _ = tune.run_tuning(n_scenarios=32, pop=8, generations=3, seed=2, log=lambda *_: None, tutor=wb)
        check("tuning pipeline on made-up screens: tuned fitness is never below the default's",
              info["fitness_tuned_train"] >= info["fitness_default_train"] - 1e-9,
              f"fitness {info['fitness_default_train']:.3f} -> {info['fitness_tuned_train']:.3f} over {info['evaluations']} evaluations")
    else:
        print("SKIP  tuning pipeline (needs the OCR network: pip install rapidocr-onnxruntime)")

    # ================================================================ 6. minimax and alpha-beta
    section("Minimax and alpha-beta pruning (tic-tac-toe)")
    c = game.compare(game.EMPTY)
    check("minimax visits 549,946 positions from an empty board (the textbook number); value 0 = a draw",
          c["minimax"].stats.nodes == 549946 and c["minimax"].value == 0)
    check("alpha-beta returns the same value with far fewer positions; move ordering helps further",
          c["alphabeta"].value == 0 and c["alphabeta"].stats.nodes < c["minimax"].stats.nodes / 10
          and c["alphabeta+ordering"].stats.nodes < c["alphabeta"].stats.nodes,
          f"{c['minimax'].stats.nodes:,} vs {c['alphabeta'].stats.nodes:,} vs {c['alphabeta+ordering'].stats.nodes:,}")
    prng = random.Random(5)
    positions = []
    while len(positions) < 60:
        b = game.EMPTY
        for _ in range(prng.randint(3, 7)):
            if game.is_over(b):
                break
            b = game.play(b, prng.choice(game.legal_moves(b)), game.to_move(b))
        if not game.is_over(b):
            positions.append(b)
    same = 0
    for b in positions:
        t = game.to_move(b)
        v0 = game.minimax(b, t, t, game.Stats())
        v1 = game.alphabeta(b, t, t, game.Stats())
        v2 = game.alphabeta(b, t, t, game.Stats(), order=game.CENTRE_FIRST)
        same += (v0 == v1 == v2)
    check("alpha-beta value = minimax value on 60 random positions", same == len(positions), f"{same}/{len(positions)}")

    def never_loses(side):
        games = losses = 0
        stack = [game.EMPTY]
        while stack:
            b = stack.pop()
            if game.is_over(b):
                games += 1
                losses += game.winner(b) == game.other(side)
            elif game.to_move(b) == side:
                stack.append(game.play(b, game.ai_move(b, "hard").move, side))
            else:
                stack.extend(game.play(b, i, game.other(side)) for i in game.legal_moves(b))
        return games, losses
    (g_o, l_o), (g_x, l_x) = never_loses("O"), never_loses("X")
    check("the 'hard' player never loses, against every possible human move, as O and as X", l_o == 0 and l_x == 0,
          f"{g_o + g_x} complete games checked")
    # X has two in a row at the top. X to move wins at square 3 (index 2); O to move must block there.
    check("the hard player takes a win when it can, and blocks when it must",
          game.ai_move("XX O O   ", "hard").move == 2 and game.ai_move("XX  O    ", "hard").move == 2)

    def human_can_beat(level):
        stack = [game.EMPTY]
        while stack:
            b = stack.pop()
            if game.is_over(b):
                if game.winner(b) == "X":
                    return True
            elif game.to_move(b) == "O":
                stack.append(game.play(b, game.ai_move(b, level).move, "O"))
            else:
                stack.extend(game.play(b, i, "X") for i in game.legal_moves(b))
        return False
    check("easy and medium players can be beaten, hard cannot (the levels differ)", human_can_beat("easy") and human_can_beat("medium")
          and not human_can_beat("hard"))
    m = game.Match("X", "hard")
    ok = m.human_move(4) and not m.human_move(4) and m.ai_move() is not None and not m.human_move(99)
    check("a game between you and the AI follows the rules", ok and m.board.count("X") == 1 and m.board.count("O") == 1)

    # ================================================================ 7. sensors
    section("Sensors (not decision makers): edge detection, icon matching, OCR, fuzzy matching")
    check("fuzzy text matching",
          name_score("New Chat", ["new chat"]) == 1.0 and name_score("Save", ["sav"]) == 0.0 and name_score("Whatsap", ["whatsapp"]) >= 0.85)
    region = fake_screen(icons=[((300, 200), "red-circle")])
    box, how = segment_target(region, (332, 232))
    iw = (min(box[2], 364) - max(box[0], 300)) * (min(box[3], 264) - max(box[1], 200))
    union = (box[2] - box[0]) * (box[3] - box[1]) + 64 * 64 - iw
    check("edge detection + contours isolate the icon", how == "contour" and iw / union > 0.6, f"{how}, overlap {iw / union:.2f}")
    store = IconStore(tmp / "icons")
    for kind, at in (("red-circle", (300, 200)), ("blue-square", (100, 100))):
        rr = fake_screen(icons=[(at, kind)])
        bb, _ = segment_target(rr, (at[0] + 32, at[1] + 32))
        store.add(kind, rr.crop(bb))
    seen = []
    for kind, at in (("red-circle", (140, 260)), ("blue-square", (480, 90))):
        rr = fake_screen(texts=[((30, 30), "Hello")], icons=[(at, kind)])
        bb, _ = segment_target(rr, (at[0] + 32, at[1] + 32))
        seen.append(store.classify(rr.crop(bb)))
    check("icon matcher recognises taught icons in a new place", seen[0] and seen[0][0] == "red-circle" and seen[1] and seen[1][0] == "blue-square")
    rr = fake_screen(icons=[((200, 200), "green-triangle")])
    bb, _ = segment_target(rr, (232, 232))
    check("icon matcher says nothing for an icon it never saw", store.classify(rr.crop(bb)) is None)
    if ocr.available:
        words = ocr.read(fake_screen(texts=[((80, 60), "New chat"), ((300, 300), "Settings")]))
        found = {w.text.lower() for w in words}
        check(f"OCR reads text ({ocr.name})", "new chat" in found and "settings" in found, str(sorted(found)))
    else:
        print("SKIP  OCR not installed (pip install rapidocr-onnxruntime); end-to-end screen tests skipped")

    # ================================================================ 8. end to end on made-up screens
    if ocr.available:
        section("End to end: made-up screen -> sensors -> logical agent -> answer")
        t2 = Tutor(cfg, learned_dir=Path(tempfile.mkdtemp()), params=pm.Params())
        ctx = ctx_for(fake_screen(texts=[((250, 200), "New chat")]), (290, 215), "WhatsApp.exe", "WhatsApp")
        ans = t2.ask(DEFAULT_QUESTION, ctx)
        rec = t2.rec
        check("button inside an app", rec.app and rec.app["id"] == "whatsapp" and rec.element and rec.element["names"][0] == "new chat"
              and "New chat" in ans and "1." in ans and "Logic used" in ans, f"conf {rec.confidence:.2f}, evidence {rec.methods}")
        ctx = ctx_for(fake_screen(texts=[((290, 270), "Excel")], icons=[((300, 190), "blue-square")]), (330, 220), "explorer.exe", "")
        t2.ask(DEFAULT_QUESTION, ctx)
        check("desktop icon recognised by its label", t2.rec.app and t2.rec.app["id"] == "excel", f"{t2.rec.name} {t2.rec.confidence:.2f}")
        ctx = ctx_for(fake_screen(texts=[((250, 200), "Settings")]), (290, 215), "chrome.exe", "Gmail - Inbox")
        t2.ask(DEFAULT_QUESTION, ctx)
        check("Gmail's 'Settings' is not the Windows Settings app", t2.rec.app and t2.rec.app["id"] == "gmail"
              and t2.rec.element and t2.rec.element["names"][0] == "settings")
        ctx = ctx_for(fake_screen(texts=[((250, 200), "Zorblax")]), (290, 215), "", "")
        ans = t2.ask(DEFAULT_QUESTION, ctx)
        check("an unknown thing is admitted, not guessed", not t2.rec.matched and "do not know" in ans.lower()
              and t2.rec.seen_text.lower().startswith("zorb"), t2.rec.seen_text)
        t2.teach("Zorblax", "item", "Zorblax is a made-up button.", ["Click it.", "Enjoy."], ctx)
        t2.ask(DEFAULT_QUESTION, ctx)
        check("teaching works on the next look", t2.rec.element and t2.rec.element["names"][0] == "zorblax",
              f"{t2.rec.name} {t2.rec.confidence:.2f} {t2.rec.methods}")
        ctx = ctx_for(fake_screen(texts=[((250, 200), "Sign in")]), (290, 215), "chrome.exe", "Bank login")
        ans = t2.ask(DEFAULT_QUESTION, ctx, trigger="hover")
        check("a private-looking window makes the agent stay quiet when hovering", t2.rec.action == "pause" and "stayed quiet" in ans)
        # the icon alone (no words) found through the whole chain
        t3 = Tutor(cfg, learned_dir=Path(tempfile.mkdtemp()), params=pm.Params())
        rr = fake_screen(icons=[((300, 200), "red-circle")])
        t3.teach("Red button", "item", "A made-up red button.", ["Click it."], ctx_for(rr, (332, 232)))
        rr2 = fake_screen(icons=[((140, 260), "red-circle")])
        t3.ask(DEFAULT_QUESTION, ctx_for(rr2, (172, 292), "", ""))
        check("a taught icon alone is enough, in a new place", t3.rec.element and t3.rec.element["names"][0] == "red button", f"{t3.rec.methods}")

    # ================================================================ 9. drafting and chat
    section("Message drafting and conversation")
    m, n = make_draft("tell rahul i am running late", "English", topic=und.analyze("tell rahul i am running late").topic)
    check("draft in English", n == "Rahul" and m and "late" in m.lower() and m.startswith("Hi Rahul,"), str(m))
    t = und.analyze("priya ko birthday wish bhejo").topic
    m, _ = make_draft("priya ko birthday wish bhejo", "Hinglish", topic=t)
    check("draft in Hinglish", bool(m) and "Janamdin" in m, str(m))
    t = und.analyze("send a thank you to dad").topic
    m, _ = make_draft("send a thank you to dad", "Hindi", topic=t)
    check("draft in Hindi", bool(m) and "धन्यवाद" in m, str(m))
    m, _ = make_draft('message sam "see you at 5 at the cafe"', "English")
    check("draft with exact words in quotes", bool(m) and "see you at 5 at the cafe" in m, str(m))
    m, _ = make_draft("write something about quantum physics", "English", topic=und.analyze("write something about quantum physics").topic)
    check("refuses what it cannot write", m is None)
    tutor.rec = type(tutor.rec)(app=kb.apps["whatsapp"], app_conf=0.9)
    a1 = tutor.ask("how do i send a photo", None)
    a2 = tutor.ask("tell rahul i am late", None)
    a3 = tutor.ask("thanks", None)
    a4 = tutor.ask("what else can i do", None)
    check("chat: how-to, draft, thanks, more", "Send a photo" in a1 and "<draft>Hi Rahul" in a2 and "welcome" in a3.lower() and "WhatsApp" in a4)
    check("the help text mentions the practice game", "Practice game" in composer.HELP)

    print(f"\n{sum(results)} of {len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(run())
