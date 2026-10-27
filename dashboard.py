"""Cursor Tutor dashboard (Streamlit, free). Run with:  streamlit run dashboard.py"""
import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import db
import labs
import search
from config import KNOWLEDGE_DIR, LEARNED_DIR, cfg
from icons import IconStore
from knowledge import KnowledgeBase
from nlu import Understander
from ocr_engine import OCR

st.set_page_config(page_title="Cursor Tutor Dashboard", page_icon="🖱️", layout="wide")
st.title("🖱️ Cursor Tutor dashboard")
st.caption("Everything is stored on this computer. No internet, online AI or trained model is used: decisions come from "
           "logic, search, a genetic algorithm and minimax.")

with st.sidebar:
    st.header("Controls")
    auto = st.toggle("Auto-refresh every 5 seconds", value=False)
    if st.button("Refresh now"):
        st.rerun()
    st.divider()
    st.write("**AI:** propositional logic, logical agent, BFS/DFS/UCS, genetic algorithm, minimax + alpha-beta")
    st.write(f"**Draft language:** {cfg.language}")
    st.write(f"**History file:** `{db.db_path()}`")


@st.cache_resource
def load_ocr():
    return OCR(enabled=cfg.use_ocr)


kb = KnowledgeBase(KNOWLEDGE_DIR, LEARNED_DIR)


@st.cache_resource
def load_understander():
    return Understander(kb.howto_docs(), KNOWLEDGE_DIR / "language.json", kb.apps)


@st.cache_resource
def load_flows():
    return search.load_flows(KNOWLEDGE_DIR / "flows.json")


df = pd.DataFrame(db.fetch_rows())
if not df.empty:
    df["time"] = df["ts"].apply(datetime.fromtimestamp)
    df["day"] = df["time"].dt.strftime("%Y-%m-%d")
    df["looked"] = df["source"] != "chat"   # rows where the screen was actually looked at

(tab_overview, tab_history, tab_logic, tab_search, tab_genetic, tab_game, tab_teach, tab_setup) = st.tabs(
    ["Overview", "History", "Logic lab", "Search lab", "Genetic lab", "Game lab", "Things to teach", "Setup check"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    if df.empty:
        st.info("No explanations yet. Run `python main.py`, hover over something and wait a moment.")
    else:
        looks = df[df["looked"]]
        week = df[df["time"] >= datetime.now() - timedelta(days=7)]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Explanations", len(df))
        c2.metric("Apps learned", df["app"].nunique())
        c3.metric("Last 7 days", len(week))
        rate = looks["matched"].mean() * 100 if len(looks) else 0
        c4.metric("Recognition rate", f"{rate:.0f}%", help="Share of screen looks where the tutor recognised something.")
        conf = looks.loc[looks["matched"] == 1, "confidence"].mean() if len(looks) else 0
        c5.metric("Avg. confidence", f"{(conf if pd.notna(conf) else 0) * 100:.0f}%")

        left, right = st.columns(2)
        with left:
            st.subheader("Apps you asked about most")
            st.bar_chart(df["app"].value_counts().head(10))
        with right:
            st.subheader("Explanations per day")
            st.bar_chart(df.groupby("day").size())

        left, right = st.columns(2)
        with left:
            st.subheader("How I recognised things")
            parts = looks["methods"].str.split(",").explode()
            parts = parts[~parts.isin(["none", "chat", ""])]
            if len(parts):
                st.bar_chart(parts.value_counts())
                st.caption("window = program/title · text = words read by OCR · control = Windows button name · icon = taught icon")
            else:
                st.write("Nothing recognised yet.")
        with right:
            st.subheader("How you asked")
            st.bar_chart(df["source"].value_counts())
            st.caption("hover = auto-explain, hotkey/button = on demand, chat = typed question")

        if "action" in df.columns and df["action"].fillna("").ne("").any():
            st.subheader("What the logical agent decided")
            names = {"explain_item": "explain the item", "explain_app": "explain the app", "ask_teach": "ask to be taught",
                     "pause": "stay quiet (private window)"}
            st.bar_chart(looks["action"].fillna("").map(lambda a: names.get(a, a)).replace("", pd.NA).dropna().value_counts())
            st.caption("The action is derived by the knowledge base (Act_ExplainItem, Act_ExplainApp, Act_Pause); "
                       "if nothing can be proved, the agent asks to be taught instead of guessing.")

        st.subheader("Answer speed over time (seconds)")
        st.line_chart(df.set_index("time")["seconds"])

        st.subheader("Apps and last time used")
        apps = (
            df.groupby("app")
            .agg(explanations=("id", "count"), last_used=("time", "max"))
            .sort_values("explanations", ascending=False)
        )
        st.dataframe(apps, width="stretch")

# ----------------------------------------------------------------- History
with tab_history:
    if df.empty:
        st.info("Nothing here yet.")
    else:
        f1, f2 = st.columns([1, 2])
        app_choice = f1.selectbox("App", ["All"] + sorted(df["app"].unique().tolist()))
        query = f2.text_input("Search questions and answers")

        view = df.sort_values("time", ascending=False)
        if app_choice != "All":
            view = view[view["app"] == app_choice]
        if query:
            q = query.lower()
            view = view[
                view["question"].str.lower().str.contains(q, na=False)
                | view["answer"].str.lower().str.contains(q, na=False)
            ]

        st.write(f"Showing {min(len(view), 50)} of {len(view)} matching explanations")
        for _, row in view.head(50).iterrows():
            seen = row["recognized"] or "not recognised"
            label = f"{row['time']:%d %b %H:%M} · {row['app']} · {seen} ({row['confidence'] * 100:.0f}%)"
            with st.expander(label):
                a, b = st.columns([1, 2])
                thumb = db.thumb_for(int(row["id"]))
                if thumb:
                    a.image(thumb, caption="What was under the cursor", width="stretch")
                b.markdown(f"**You:** {row['question']}")
                answer = str(row["answer"]).replace("<draft>", "\n\n> ✉ Draft: ").replace("</draft>", "\n\n")
                b.text(answer)
                b.caption(f"{row['source']} · {row['seconds']:.1f} s · evidence: {row['methods']}")
                if isinstance(row.get("proof"), str) and row["proof"]:
                    b.code(row["proof"].replace(" | ", "\n"), language=None)

        st.download_button(
            "Download history as CSV",
            df.drop(columns=["day", "looked"]).to_csv(index=False),
            file_name="cursor_tutor_history.csv",
            mime="text/csv",
        )

        with st.expander("Delete all history"):
            sure = st.checkbox("Yes, delete everything. This cannot be undone.")
            if st.button("Delete history", disabled=not sure):
                db.clear_all()
                st.rerun()

# ------------------------------------------------------------------- Labs
with tab_logic:
    labs.logic_lab(st, kb, load_understander(), cfg.sensitive_words)
with tab_search:
    labs.search_lab(st, load_flows())
with tab_genetic:
    labs.genetic_lab(st, LEARNED_DIR / "tuned_params.json")
with tab_game:
    labs.game_lab(st)

# ---------------------------------------------------------- Things to teach
with tab_teach:
    st.subheader("Things the tutor did not recognise")
    if df.empty:
        st.info("Nothing yet.")
    else:
        unknown = df[(df["looked"]) & (df["matched"] == 0)]
        if unknown.empty:
            st.success("Everything you pointed at was recognised.")
        else:
            counts = (
                unknown.assign(item=unknown["recognized"].replace("", "(no readable text)"))
                .groupby(["app", "item"]).size().reset_index(name="times")
                .sort_values("times", ascending=False)
            )
            st.write("Hover over these again and press **Teach this item** in the chat window.")
            st.dataframe(counts, width="stretch", hide_index=True)

# ------------------------------------------------------------- Setup check
with tab_setup:
    st.subheader("Is everything ready?")
    ocr = load_ocr()
    if ocr.available:
        st.success(f"Text reading (OCR): {ocr.name}")
    else:
        st.warning("Text reading (OCR) is not installed. Run:  pip install rapidocr-onnxruntime")

    stats = kb.stats()
    icons = IconStore(LEARNED_DIR / "icons")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Apps known", stats["apps"])
    c2.metric("Buttons known", stats["elements"])
    c3.metric("How-to guides", stats["howtos"])
    c4.metric("Taught by you", stats["taught"])
    c5.metric("Icons learned", icons.count())

    if st.button("Run self-test (takes a few seconds)"):
        import contextlib
        import io

        import selftest

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = selftest.run()
        (st.success if code == 0 else st.error)("All checks passed" if code == 0 else "Some checks failed")
        st.code(buf.getvalue())

    st.markdown(
        """
**How it works:** the sensors (OCR, edge detection, icon matching) report what is on the screen as facts. A logical
agent adds the facts to a propositional knowledge base (`knowledge/*.json` compiled into rules) and forward chaining
derives which app and button you are pointing at. Typed questions are understood with word rules, how-to clicks are
planned with uniform-cost search, the agent's settings are tuned by a genetic algorithm, and the practice game uses
minimax with alpha-beta pruning. It only knows what is in the knowledge base, so use **Teach this item** to add more.
"""
    )

if auto:
    time.sleep(5)
    st.rerun()
