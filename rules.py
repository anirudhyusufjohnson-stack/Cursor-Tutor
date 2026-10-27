"""Compiles the JSON knowledge base into propositional (Horn) rules for the logical agent.

Symbols the agent reasons with
  Proc_<program>      the program running under the cursor              (percept)
  Title_<word>        the window title contains a known app word        (percept)
  Text_<name>         a known name is written next to the cursor        (percept)
  Ctl_<name>          Windows says the control under the cursor is <name> (percept)
  Icon_<label>        the icon looks like one you taught                (percept)
  WindowKnown / WindowUnknown   does the window itself tell which app this is?  (percept)
  App_<id>, Item_<id> the app / the button or item being pointed at     (derived by inference)
Examples of compiled rules
  Proc_whatsapp_exe                         ⇒ App_whatsapp
  Text_new_chat ∧ App_whatsapp              ⇒ Item_whatsapp_0
  Text_excel ∧ WindowUnknown                ⇒ App_excel      (a name on the bare desktop)"""
import re
from dataclasses import dataclass, field

from knowledge import norm
from logic import Rule


def part(text):
    """Text -> a safe symbol fragment:  'New chat!' -> 'new_chat'."""
    return re.sub(r"\W+", "_", norm(text)).strip("_")


def sym(prefix, text):
    return f"{prefix}_{part(text)}"


def app_sym(app):
    return "App_" + re.sub(r"\W+", "_", app["id"])


def item_sym(el):
    return "Item_" + re.sub(r"\W+", "_", el["_id"])


@dataclass
class Compiled:
    rules: list = field(default_factory=list)
    apps: dict = field(default_factory=dict)          # App_ symbol -> app dict
    app_by_id: dict = field(default_factory=dict)     # app id -> app dict
    items: dict = field(default_factory=dict)         # Item_ symbol -> element dict
    text_terms: set = field(default_factory=set)      # normalised names the OCR text can be matched to
    title_terms: dict = field(default_factory=dict)   # normalised title word -> (Title_ symbol, App_ symbol)
    proc_terms: dict = field(default_factory=dict)    # lower-case process -> (Proc_ symbol, App_ symbol)
    window_apps: set = field(default_factory=set)     # apps whose window can identify them (not desktop/shell)
    surface_apps: set = field(default_factory=set)    # the bare desktop: anything on it is more specific
    specific_items: set = field(default_factory=set)  # items that belong to one app


ACTION_RULES = [
    ("AcceptedItem => Act_ExplainItem", "an item was recognised, so explain the item"),
    ("AcceptedApp => Act_ExplainApp", "an app was recognised, so explain the app"),
    ("Sensitive ∧ Trigger_hover => Act_Pause", "the window looks private and help was not asked for, so stay quiet"),
]


def compile_kb(kbase):
    """KnowledgeBase -> Compiled rules and vocabulary."""
    c = Compiled()

    def add(prem, concl, note):
        c.rules.append(Rule(tuple(dict.fromkeys(prem)), concl, note))

    def names_of(el):
        seen, out = set(), []
        for n in el.get("names", []):
            t = norm(n)
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    for app in kbase.apps.values():
        a = app_sym(app)
        c.apps[a] = app
        c.app_by_id[app["id"]] = app
        if app.get("surface"):
            c.surface_apps.add(a)
        elif not app.get("shell"):
            c.window_apps.add(a)
        # the window itself
        if not app.get("shell"):  # explorer.exe also draws the desktop and taskbar, so the program name alone says little
            for proc in app.get("process", []):
                p = proc.lower()
                ps = sym("Proc", p)
                c.proc_terms[p] = (ps, a)
                add([ps], a, f"the program {proc} is {app['name']}")
        for kw in app.get("title_keywords", []):
            k = norm(kw)
            if k:
                ts = sym("Title", k)
                c.title_terms[k] = (ts, a)
                add([ts], a, f"the window title mentions '{kw}', so the app is {app['name']}")
        # a name written on a bare desktop (an icon label)
        for alias in dict.fromkeys(norm(x) for x in [app["name"]] + app.get("aliases", [])):
            if alias:
                c.text_terms.add(alias)
                add([sym("Text", alias), "WindowUnknown"], a,
                    f"the name '{alias}' is written next to the cursor and no app window explains it, so it is {app['name']}")
        if app.get("icon_label"):
            add([sym("Icon", app["icon_label"])], a, f"the icon looks like the one you taught as '{app['icon_label']}'")

        for el in app.get("elements", []):
            i = item_sym(el)
            c.items[i] = el
            c.specific_items.add(i)
            for n in names_of(el):
                c.text_terms.add(n)
                add([sym("Text", n), a], i, f"the word '{n}' is next to the cursor inside {app['name']}")
                add([sym("Ctl", n), a], i, f"Windows calls this control '{n}' inside {app['name']}")
            if el.get("icon_label"):
                add([sym("Icon", el["icon_label"])], i, f"the icon looks like the one you taught as '{el['icon_label']}'")

    for el in kbase.generic:
        i = item_sym(el)
        c.items[i] = el
        for n in names_of(el):
            c.text_terms.add(n)
            add([sym("Text", n)], i, f"the word '{n}' is next to the cursor (a general button name)")
            add([sym("Ctl", n)], i, f"Windows calls this control '{n}' (a general button name)")
        if el.get("icon_label"):
            add([sym("Icon", el["icon_label"])], i, f"the icon looks like the one you taught as '{el['icon_label']}'")

    for text, note in ACTION_RULES:
        prem, concl = text.replace("∧", "&").split("=>")
        add([p.strip() for p in prem.split("&")], concl.strip(), note)
    return c
