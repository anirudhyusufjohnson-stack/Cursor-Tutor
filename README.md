# Cursor Tutor: syllabus edition

Point your mouse at anything on the screen, keep it still for two seconds, and a small chat window explains what it
is and how to use it. Ask follow-up questions ("how do I send a photo?") or ask it to write a message for you
("tell Rahul I will be late").

**No online API, no API key, no account, no payment, no trained model.** Every decision is made by the algorithms
taught in the course: propositional logic, a logical agent, search (BFS, DFS, uniform-cost), a genetic algorithm, and
minimax with alpha-beta pruning. After the one-time `pip install` it works without internet.

## Which course topic does what

| Course topic | What it does in this project | Files |
|---|---|---|
| **Propositional logic** | Everything the agent knows is a sentence or a Horn rule (`Text_new_chat ∧ App_whatsapp ⇒ Item_new_chat`). The engine has a parser, model checking by truth table, CNF + resolution, and forward chaining with proofs. | `logic.py` |
| **Logical agent** | A knowledge-based agent: perceive → TELL facts → infer → ASK which action follows → remember. It decides what is under the cursor, derives `Act_ExplainItem` / `Act_ExplainApp` / `Act_Pause`, and when nothing can be proved it asks to be taught instead of guessing (closed-world assumption). | `agent.py`, `rules.py`, `sensors.py` |
| **Logic again, for language** | What you type becomes facts (`W_pic`); rules turn them into concepts, intents and the matching how-to guide; the most specific rule wins. | `nlu.py`, `knowledge/language.json` |
| **BFS / DFS / uniform-cost search** | Each app is a graph of screens; a how-to answer is a route from the start screen to the goal screen. UCS gives the easiest route (the one shown in chat), BFS the fewest clicks, DFS any route. | `search.py`, `knowledge/flows.json` |
| **Genetic algorithm** | Evolves the agent's 12 settings (distances, similarity cut-offs, how much each clue is trusted) on made-up labelled screens. | `genetic.py`, `tune.py`, `params.py` |
| **Minimax and alpha-beta pruning** | The "Practice game" (tic-tac-toe) opponent, with easy / medium / hard levels made by limiting the look-ahead. The dashboard counts the positions each method visits. | `game.py`, `practice_ui.py` |

**Sensors are not decisions.** Reading text (OCR neural network), finding an icon's outline (edge detection) and
matching a taught icon are *sensors*: they only report what is on the screen. They are not part of the course topics
and they make no decisions; every decision above comes from the algorithms in the table.

## How one hover works

```
screen --sensors--> facts: Proc_whatsapp_exe, Text_new_chat, WindowKnown ...
facts  --TELL-----> knowledge base (≈430 rules compiled from knowledge/*.json)
KB     --forward chaining--> App_whatsapp, Item_whatsapp_0           (each with a proof)
rank   --noisy-OR of the clues, nearer words count more--> best app and item
KB     --ASK-----> Act_ExplainItem   (or Act_Pause on a private window, or "teach me")
reply  --template--> lesson + the rules that were used ("Logic used: ...")
```

## Set up (once)

You need Python 3.10 or newer (Windows is the main target; the algorithms also run on Mac/Linux).

```bash
cd cursor-tutor-syllabus
python -m venv venv
venv\Scripts\activate          # Mac/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

`pip install` needs internet **one time**. Nothing is downloaded after that.

Check everything (about 20 seconds, uses made-up screens, no real screen needed):

```bash
python selftest.py
```

You should see `55 of 55 checks passed`. If the OCR lines say SKIP, run `pip install rapidocr-onnxruntime`.

## Run

```bash
python main.py
```

1. A small chat window appears and stays on top.
2. Move your mouse over anything (an app icon, a button, a menu) and keep it still for 2 seconds.
3. The tutor explains it and shows the logic it used. Type below to ask more.
4. Press **Ctrl+Shift+H** any time to explain whatever is under the cursor right now.
5. Press **Practice game** to play tic-tac-toe against the minimax player.

Things you can type:

- `how do I send a photo` / `photo kaise bhejte hain`
- `what is this?` / `what more can I do here?`
- `tell Priya I am running late` / `Rahul ko birthday wish bhejo` / `message Sam "see you at 5"`

### Teach it something new

The agent only knows what is in its knowledge base. If it says it does not know an item, hover over it and press
**Teach this item**. Give it a name and a short description. It saves the icon and your words, adds new rules to its
knowledge base, and recognises that item from then on. Taught items are stored in `learned/`.

## Dashboard (the labs)

```bash
streamlit run dashboard.py
```

| Tab | What you can do |
|---|---|
| Overview, History | Usage, which clues were used, what the agent decided, every answer with its logic proof, CSV export |
| **Logic lab** | Type your own knowledge base and query: truth table, resolution and forward chaining are run side by side. Watch the logical agent decide, step by step. See how a sentence is understood. |
| **Search lab** | Pick an app and a goal: BFS, DFS and UCS are run and compared (route, effort, screens expanded, memory), with the screen graph drawn. |
| **Genetic lab** | See what tuning achieved, and run the GA on a puzzle with your own population, generations and mutation settings. |
| **Game lab** | Set up any tic-tac-toe position and compare minimax and alpha-beta by positions visited. |
| Things to teach, Setup check | What was not recognised, and a self-test button |

## Tune the agent with the genetic algorithm

```bash
python tune.py            # or: python main.py --tune     (about 4 minutes)
```

It builds 360 made-up screens, lets the sensors measure them once, then evolves the 12 settings (population 30,
25 generations). The best settings are saved to `learned/tuned_params.json` **only if they are better on screens the
GA never saw**; otherwise the defaults are kept. A tuned file is included, and the app loads it automatically.

## Settings (optional)

Copy `.env.example` to `.env` and uncomment what you want to change: language of drafts, how long to wait before
auto-explaining, the hotkey, and so on.

## Privacy

- Screenshots are processed in memory and are not sent anywhere. Only a small thumbnail is kept in the local history
  file (`tutor_history.db`) so the dashboard can show it. Delete the history from the dashboard any time.
- Auto-explain pauses on windows whose title contains words like password, bank, login or OTP (edit
  `SENSITIVE_WORDS`). This is a rule in the knowledge base: `Sensitive ∧ Trigger_hover ⇒ Act_Pause`.

## Honest results and limits

Numbers measured while building this; please re-run them yourself (`python selftest.py`, `python tune.py`).

- **Self-test:** 55 checks, including: truth table = resolution = forward chaining on 25 random Horn knowledge bases;
  BFS/UCS routes checked against an independent shortest-path calculation; alpha-beta gives the same value as
  minimax on 60 random positions; the hard player never loses in all 613 possible games against any human.
- **Pruning:** from an empty board minimax visits 549,946 positions, alpha-beta 20,866, alpha-beta with a good move
  order 7,865. All give the same answer.
- **Genetic algorithm:** on 109 made-up screens it never saw, exact-answer accuracy went from 90% to 94% (on the 251
  screens it evolved on: 93% to 98%). Most of the gain came from learning a tighter meaning of "next to the cursor"
  (100 px down to about 43 px). The screens are synthetic, so this shows the method works, not that every real screen
  improves. The same effort of random search scored lower fitness (0.973 vs 0.983).
- **Language rules:** 30/30 intents and 16/16 guides on the phrases used while building the rules. On 37 new
  phrases written afterwards it got 29 right the first time (16/21 intents, 13/16 guides); the gaps were fixed, and
  those phrases are now part of the self-test. Expect some misses on wording nobody wrote a rule for; hand-written
  rules are less flexible than a trained model. That is the price of explainable, offline, course-topic logic.
- **Knowledge:** 17 apps, 98 buttons, 55 how-to guides, and route maps for WhatsApp, Gmail, File Explorer and Word
  (15 goals). It does not invent answers; when it cannot prove something it says so and offers to learn.
- **Not tested on a real desktop.** The checks use made-up screens. The chat window, Teach dialog and practice game
  were run on a virtual display on Linux, never on Windows. Windows-only parts (exact button names through UI
  Automation) have not been run at all. Please try it on your own screen and use **Teach this item** where it misses.
- Icon recognition only works for icons you have taught it. Answers are in English; message drafts support English,
  Hinglish and Hindi (read Hindi drafts before sending).
- The practice game is a small add-on to show minimax and alpha-beta; it is not part of explaining apps.

## Files

```
main.py            start the app   (python main.py --selftest | --tune also work)
ui.py              floating chat window, Teach dialog
practice_ui.py     the tic-tac-toe window
hover.py           detects the mouse resting on something
cursor_context.py  screenshot + window + control info under the cursor
--- sensors (report what is on the screen) ---
ocr_engine.py      OCR neural network
target.py          edges + contours to isolate the item
icons.py           icon matcher
sensors.py         turns sensor readings into symbols (facts)
--- the course algorithms ---
logic.py           propositional logic: parser, truth table, resolution, forward chaining, KB
rules.py           compiles knowledge/*.json into Horn rules
agent.py           the logical agent
nlu.py             understanding typed sentences with rules
search.py          BFS, DFS, uniform-cost search
genetic.py         genetic algorithm        tune.py   tuning on made-up screens     params.py   the settings being tuned
game.py            minimax + alpha-beta     
--- everything else ---
ai_tutor.py        ties the parts together
composer.py        builds the replies       drafting.py   message drafts
knowledge.py       knowledge-base loader, fuzzy matching, teaching
knowledge/         apps.json, generic.json, language.json, flows.json  (editable)
synth.py           made-up screens for tests and tuning
labs.py, dashboard.py   the dashboard
db.py              local history
selftest.py        55 checks
```
