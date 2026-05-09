# Bedtime Story Generator — Hippocratic AI Coding Assignment

A production-grade, multi-agent bedtime story system built on `gpt-3.5-turbo`. It takes any story request and produces an age-calibrated, engaging, quality-assured bedtime story complete with text-to-speech narration, a persistent story library, and full session logging.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your .env file (never commit this — it is in .gitignore)
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...

# 3a. Run the Streamlit web UI (recommended)
python -m streamlit run app.py

# 3b. OR run the CLI version
python main.py
```

---

## System Architecture & Block Diagram

The system is a **multi-agent pipeline** with a self-improving judge loop, streaming output, persistent story history, context-aware revision, and optional TTS narration.

```mermaid
flowchart TD
    A(["User\nEnters story request"]) --> B["Classifier Agent\nDetects genre · tone · age group · themes\nhas_scary_elements · length_hint\nJSON-output prompting · temperature 0.1"]
    B --> C["Planner Agent\nBuilds a 7-beat story arc\nChain-of-thought prompting"]
    C --> D["Storyteller Agent\nStreams full prose to UI word-by-word\nAge-calibrated vocabulary profile\nRole + constraint + few-shot + God salutation\nSafety redirect if has_scary_elements=True"]
    D --> EX{"Story below\nmin word count?"}
    EX -- "Yes: expand" --> EXP["Expand Story\nModel rewrites with more scenes\ndialogue and sensory detail\nRules already enforced in expansion"]
    EXP --> JDG
    EX -- "No: reflect" --> SR["Self-Reflection Pass\nModel self-critiques draft\nFixes rule violations silently\nSkipped when expansion ran"]
    SR --> JDG
    JDG{"LLM Judge\n6 criteria · 0–12 score\nword_count_adequacy included\nScores clamped and recalculated"}
    JDG -- "Score < 9/12\nup to 2 iterations" --> REV["Reviser Agent\nTargeted improvements only\nExpansion-aware: lifts length\nconstraint for make-it-longer requests"]
    REV --> JDG
    JDG -- "Score ≥ 9/12" --> DB["SQLite Session Store\nWAL mode · session-namespaced\nOne record per story concept\nRevisions overwrite in-place"]
    DB --> UI(["Story Card\nHTML-escaped display\nQuality bar x/12\n6-metric breakdown expander\nStory Arc expander"])
    UI --> TTS{"User wants\naudio?"}
    TTS -- "Yes" --> TTSA["TTS Narration\nOpenAI tts-1\n4 voices: Nova Shimmer Fable Echo\nAuto-chunked for stories > 4096 chars\nMP3 cached in session (LRU · max 5)"]
    UI --> FB{"User requests\nchanges?"}
    FB -- "Yes (context-aware)" --> FBREV["Context-Aware Revision\nActual story text passed as context\nClassify + Plan skipped\nExpand + Reflect + Judge applied"]
    FBREV --> DB
    FB -- "No" --> DONE(["Done ✓\nStory saved to library"])
    LOG["Session Logger\nagent.log · per-step timing\nscores · word counts · errors"] -.-> B & C & D & EXP & SR & JDG & REV & DB & TTSA & FBREV
```

---

## Prompting Strategies

| Agent | Strategy | Why |
|---|---|---|
| **Classifier** | Structured JSON output at temperature 0.1 | Downstream agents need typed, deterministic metadata |
| **Planner** | Chain-of-thought (7 explicit beats) | Forces the model to reason about plot structure before committing to prose |
| **Storyteller** | Role + age-calibrated vocabulary profile + constraint checklist + few-shot example | Role defines style; profiles enforce age-appropriate language; constraints add six engagement devices; few-shot anchors the tone. When `has_scary_elements=True`, an additional block redirects monsters/ghosts to gentle alternatives before any prose is written |
| **Judge** | Structured JSON rubric at temperature 0.1 | Enables programmatic score comparison; scores clamped to [0,2] and recalculated to prevent hallucinated totals |
| **Reviser** | Targeted revision + context injection + expansion detection | Actual previous story text passed in; "make it longer"-type requests lift the length constraint and double the target word count.  Model fixes its own draft before the Judge — skipped when expansion already enforced the rules |

---

## Features

### Core
- **Multi-agent pipeline**: Classifier → Planner → Storyteller → Expand → Self-reflect → Judge → Reviser
- **LLM Judge**: 6-criterion rubric (0–12 total), validates and clamps all scores, enforces word count
- **Story arc**: 7-beat narrative structure (Setup → Inciting Incident → Rising Action × 3 → Climax → Resolution → Reflection)
- **User revision**: Context-aware feedback loop — "make it funnier", "give the dragon a name", "make it longer"
- **Persistent library**: All stories saved to SQLite; accessible via the sidebar across sessions

### User Experience
- **Streaming output**: Story appears word-by-word in real time
- **Text-to-Speech**: Four voices (Nova, Shimmer, Fable, Echo) via OpenAI `tts-1`; stories of any length auto-chunked into ≤4000-char segments
- **Clean story card**: Only title and story shown in the main display; technical details in collapsible expanders
- **Quality breakdown**: Score bar + 6-metric grid per judge pass

---

## Age Calibration

The storyteller automatically adjusts vocabulary and complexity based on the classifier's `age_lean` output:

| Profile | Ages | Vocabulary | Sentence Length | Characters | Special |
|---|---|---|---|---|---|
| **Young** | 5–6 | 1–2 syllable words | Max 8 words | Talking animals | Repeating refrains, simple emotions |
| **Middle** | 7–8 | Up to 3 syllables | 10–12 words | Kids + animal companions | Inner thoughts, cause-and-effect |
| **Older** | 9–10 | Metaphors, similes | 12–15 words | Human protagonists | Inner conflict, rich sensory language |

---

## Quality Scoring (Judge)

Stories are scored out of **12 points** (6 criteria × 2 points each). Pass threshold: **9/12 (75%)**.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Age Appropriateness(critical, resolved on priority basis) | Scary/violent content | Minor issues | Fully safe and age-suitable |
| Narrative Completeness | Missing sections | Partial arc | Clear 7-beat beginning/middle/end |
| Engagement | None of: dialogue, sensory, suspense, refrain | Some present | All four present |
| Alignment | Ignores request | Partially matches | Faithfully addresses request |
| Emotional Resonance | No warm ending | Adequate close | Comforting close + God/gratitude salutation |
| Word Count Adequacy | Ages 5–6: < 250w / 7–8: < 350w / 9–10: < 500w | One tier below target | Meets target (400w / 600w / 800w) |

All scores are **clamped to [0, 2]** and `total_score` is **always recalculated** from the sum — the model's own `total_score` field is never trusted.

---

## Project Structure

```
.
├── app.py                    # Streamlit web UI — streaming, TTS, library sidebar
├── main.py                   # run_pipeline() + revise_pipeline() + CLI entry point
├── agents/
│   ├── classifier.py         # JSON: genre, tone, age, themes, has_scary_elements, length_hint
│   ├── planner.py            # 7-beat story arc via chain-of-thought
│   ├── storyteller.py        # Age-calibrated prose + expand_story_if_short() + self_reflect_story()
│   ├── judge.py              # 6-criterion rubric, score clamping, 0–12 total
│   └── reviser.py            # Targeted revision, expansion-aware (lifts length constraint on demand)
├── utils/
│   ├── llm.py                # call_model() + stream_model() + exponential backoff retry
│   ├── session_store.py      # SQLite WAL · save_story() · update_story() (in-place revision)
│   ├── logger.py             # Per-session structured logger → agent.log
│   └── tts.py                # OpenAI tts-1 wrapper, auto-chunking for stories > 4096 chars
├── test_edge_cases.py        # 12 automated edge-case assertions (no API calls)
├── requirements.txt          # openai>=1.0.0 · streamlit>=1.32.0 · python-dotenv>=1.0.0
├── .env.example              # Copy to .env — set OPENAI_API_KEY
├── .gitignore                # Protects .env · story_library.db · agent.log
├── agent.log                 # Auto-created at runtime · not committed to git
└── story_library.db          # Auto-created at runtime · not committed to git
```

---

## Observability & Logging

Every pipeline step writes a structured entry to `agent.log` with:
- **Timestamp** — `2026-05-08 18:23:53`
- **Level** — `INFO` / `WARNING` / `ERROR`
- **Session ID** — `[461116ff]` (first 8 chars of UUID)
- **Message** — step name, duration, word counts, scores

**Example log from a full generation:**
```
2026-05-08 18:23:53 | INFO  | [461116ff] Session started
2026-05-08 18:24:01 | INFO  | [461116ff] Classify done (4.8s): genre=adventure, tone=exciting, age=middle
2026-05-08 18:24:06 | INFO  | [461116ff] Plan done (5.1s): 291 words
2026-05-08 18:24:09 | INFO  | [461116ff] Story written (3.2s): 156 words
2026-05-08 18:24:17 | INFO  | [461116ff] Expansion done (7.9s): 156 -> 547 words
2026-05-08 18:24:17 | INFO  | [461116ff] Self-reflect SKIPPED: expansion already enforced all rules
2026-05-08 18:24:19 | INFO  | [461116ff] Judge pass 1 (2.1s): score=11/12, word_count_score=2/2, needs_revision=False
2026-05-08 18:24:19 | INFO  | [461116ff] Story accepted: score=11/12, threshold=9/12
2026-05-08 18:24:19 | INFO  | [461116ff] Pipeline complete: total=23.1s, iterations=0
2026-05-08 18:24:19 | INFO  | [461116ff] Story saved to DB: id=11, title=Anaya's Walk in the Jungle
```

---


## Deploying to Streamlit Cloud
1. Push this repo to GitHub (`.env`, `story_library.db`, and `agent.log` are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to `app.py`
3. In **App Settings > Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
4. Deploy — the app reads `st.secrets["OPENAI_API_KEY"]` automatically

> **Note:** Streamlit Cloud uses an ephemeral filesystem. `story_library.db` and `agent.log` will reset on each deployment. For persistent storage in production, swap `session_store.py` for a PostgreSQL or Supabase backend.

---

## What I Would Build Next

1. **DALL-E Illustrations** — Generate one image per story beat, turning the output into a true interactive picture book that children can look at while listening.
2. **User Rating System** — Thumbs-up/thumbs-down per story. Over time, the agent learns which genres, tones, and vocabulary levels produce the highest-rated stories for each age group — closing a true reinforcement loop.
3. **Multi-user Accounts** — Streamlit authentication so families maintain their own persistent story library.

---

## Rules Compliance Checklist

| Requirement | Status | How |
|---|---|---|
| Stories appropriate for ages 5–10 | ✅ | Three age-calibrated vocabulary profiles; Judge penalises age-inappropriate content |
| LLM judge incorporated | ✅ | `agents/judge.py` — 6 criteria, 0–12 score, score-gated revision loop |
| Block diagram provided | ✅ | Mermaid flowchart above; all agents, data flows, and feedback loops shown |
| Model unchanged (`gpt-3.5-turbo`) | ✅ | `utils/llm.py` — hardcoded in both `call_model()` and `stream_model()` |
| API key not in submission | ✅ | `.gitignore` covers `.env`; `.env.example` provided with placeholder value |
| Story arcs used | ✅ | `agents/planner.py` — 7-beat arc generated before any prose |
| User feedback / changes | ✅ | `revise_pipeline()` in `main.py`; "Request Changes" form in UI |
| Categorised generation strategy | ✅ | `agents/classifier.py` — genre, tone, age, length each tailor downstream prompts |

---

## Model

All text generation uses **`gpt-3.5-turbo`** (as required). Audio narration uses **`tts-1`** (OpenAI's text-to-speech model, separate from the story generation model and not restricted by the assignment rules).

---

*Built for the Hippocratic AI Agent Deployment Engineer coding assignment.*
