"""Turns the agent's conclusions and the knowledge base into a friendly, plain-text lesson.

This is template-based text generation: sentences are assembled from fixed patterns and the stored
descriptions and steps. It cannot say anything that is not in the knowledge base."""


def _pct(conf):
    return f"{int(round(conf * 100))}%"


def _numbered(steps):
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def _why(rec):
    """The evidence in words, then the logic rules that fired (the agent's proof)."""
    out = []
    bits = list(dict.fromkeys(e.detail for e in rec.evidence))
    if bits:
        out.append("Why I think so: " + "; ".join(bits[:4]) + ".")
    rules = [p.split("   (")[0] for p in rec.proof][:3]
    if rules:
        out.append("Logic used:\n" + "\n".join(f"  {r}" for r in rules))
    return "\n".join(out)


def _guide_titles(app, limit=4):
    return [h["title"] for h in app.get("howtos", [])][:limit]


def explain(rec, app_howtos_hint=True):
    """Lesson for what is under the cursor."""
    app, el = rec.app, rec.element
    lead = "This looks like" if rec.confidence >= 0.6 else "I think this is"
    parts = []

    if rec.action == "ask_teach" or (not el and not app):
        return unknown(rec)
    if el:
        label = el["names"][0].title()
        where = f" in {app['name']}" if app and el.get("_app") else ""
        parts.append(f'{lead} "{label}"{where} ({_pct(rec.element_conf)} sure).')
        parts.append(el["what"])
        parts.append("How to use it:\n" + _numbered(el["steps"]))
    else:
        parts.append(f"{lead} {app['name']} ({_pct(rec.app_conf)} sure).")
        parts.append(app.get("what", ""))
        if app.get("steps"):
            parts.append("To get started:\n" + _numbered(app["steps"]))

    why = _why(rec)
    if why:
        parts.append(why)
    if app_howtos_hint and app and _guide_titles(app):
        parts.append("I can also show you how to: " + "; ".join(_guide_titles(app)) + ".\nJust ask, for example: How do I "
                     + _guide_titles(app)[0].lower() + "?")
    return "\n\n".join(p for p in parts if p)


def pause():
    return ("This window looks private (its name mentions something like a password, bank or payment), "
            "so I stayed quiet. If you want my help here, press the hotkey or the \"Explain\" button.")


def unknown(rec):
    if rec.seen_text:
        return (f'I can read "{rec.seen_text}" near your cursor, but I do not know it yet. '
                "My knowledge base cannot prove what it is, and I do not guess.\n\n"
                'Press "Teach this item" and tell me what it is. I will remember it, and next time I will explain it.')
    if not rec.ocr_available:
        return ("I could not recognise anything here. My text-reading part (OCR) is not installed, so I can only use "
                "window names and icons you have taught me.\n\nInstall it with:  pip install rapidocr-onnxruntime")
    return ("I could not recognise what is under the cursor.\n\nTry moving the mouse onto a button's label, an app's name "
            'or an icon you have taught me, and keep it still. You can also press "Teach this item" to teach me something new.')


def howto(doc, others, planned=None):
    """A how-to answer. `planned` = (flow, {"BFS": Plan, "DFS": Plan, "UCS": Plan}) when the app has a screen map."""
    lead = f"How to: {doc['title']} ({doc['app_name']})"
    steps, note = doc["steps"], ""
    if planned:
        _flow, plans = planned
        best, fewest = plans["UCS"], plans["BFS"]
        if best.found:
            steps = best.actions
            note = f"Route planned by uniform-cost search: the easiest way here (effort {best.cost:g})."
            if fewest.found and fewest.steps < best.steps:
                note += (f" The route with the fewest clicks has {fewest.steps} steps, but it is harder"
                         f" (effort {fewest.cost:g}).")
    out = [lead, _numbered(steps)]
    if note:
        out.append(note)
    if others:
        out.append("Related: " + "; ".join(d["title"] for d in others))
    return "\n\n".join(out)


def no_howto(app, suggestions):
    lines = ["I do not have a guide for that yet."]
    if suggestions:
        lines.append("Here are guides that are close to what you asked:\n- " + "\n- ".join(
            f"{d['title']} ({d['app_name']})" for d in suggestions))
    if app:
        lines.append(f"You can teach me more about {app['name']} with the \"Teach this item\" button.")
    return "\n\n".join(lines)


def more(app):
    if not app:
        return "Point at an app or button first, then ask me again. I will tell you what else you can do there."
    titles = _guide_titles(app, 8)
    names = [e["names"][0].title() for e in app.get("elements", [])][:8]
    out = []
    if titles:
        out.append(f"Things I can show you in {app['name']}:\n- " + "\n- ".join(titles))
    if names:
        out.append("Buttons I know here: " + ", ".join(names) + ".")
    return "\n\n".join(out) or f"I only know the basics of {app['name']} so far."


HELP = (
    "I am Cursor Tutor. I work on your computer, with no internet needed.\n\n"
    "- Hover over an app, icon or button and keep still: I explain it.\n"
    "- Ask me how to do something, for example: How do I send a photo?\n"
    "- Ask me to write a message, for example: Tell Rahul I am late.\n"
    '- Press "Teach this item" to show me something new.\n'
    '- Press "Practice game" to play tic-tac-toe against my minimax player.'
)

CANNED = {
    "greet": "Hello! Point your mouse at something on the screen and keep it still, and I will explain it.",
    "thanks": "You are welcome! Ask me if you want to learn anything else.",
    "goodbye": "Goodbye! Come back whenever you want to learn something.",
    "help": HELP,
}
