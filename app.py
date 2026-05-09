"""
app.py — Streamlit Web UI for the Bedtime Story Generator

Features:
- Streaming story output (word-by-word as it generates)
- Persistent SQLite story library with sidebar (1 record per story, revisions overwrite)
- Context-aware revision via revise_pipeline()
- Self-reflection, 7-beat arc, age-calibrated vocabulary, God salutation
- Text-to-Speech with male/female voice selection (OpenAI TTS)
- Clean story card — title and story only
- Session activity logging to agent.log
- Deployable to Streamlit Cloud
"""

import os
import uuid
import time
import html as html_module
from collections import OrderedDict
from datetime import datetime, timezone

import streamlit as st

# -- Streamlit Cloud secret injection -----------------------------------------
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

from agents.classifier import classify
from agents.planner import plan_story
from agents.storyteller import write_story_stream, self_reflect_story, expand_story_if_short
from agents.judge import judge_story
from agents.reviser import revise_story
from main import revise_pipeline, MAX_JUDGE_ITERATIONS, JUDGE_PASS_THRESHOLD
from utils.session_store import init_db, save_story, update_story, get_all_stories
from utils.tts import generate_audio, VOICES
from utils.logger import get_logger

# -- DB + session init --------------------------------------------------------
init_db()

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Bedtime Story Generator",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
)

# -- Session state ------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state._is_new_session = True   # Flag to trigger log refresh once
else:
    st.session_state._is_new_session = False
if "result" not in st.session_state:
    st.session_state.result = None
if "original_request" not in st.session_state:
    st.session_state.original_request = ""
if "current_story_id" not in st.session_state:
    st.session_state.current_story_id = None   # DB row id for current story
if "audio_cache" not in st.session_state:
    # OrderedDict used for LRU eviction: cap at 5 entries to bound memory use.
    # Each MP3 is ~100-400KB; 5 entries = max ~2MB session memory (Issue 11)
    st.session_state.audio_cache = OrderedDict()

# fresh_start=True clears agent.log and writes a new session header.
# This only fires once per browser session (when session_id is first created).
log = get_logger(st.session_state.session_id, fresh_start=st.session_state._is_new_session)

# -- CSS ----------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,700;1,400&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #0a0a0f; min-height: 100vh; }

  .page-header { text-align: center; padding: 3rem 0 1.8rem; }
  .page-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; color: #3d4a5a; margin-bottom: 0.8rem; }
  .page-title { font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 700; color: #f0ece4; line-height: 1.15; margin-bottom: 0.6rem; }
  .page-sub { font-size: 0.92rem; color: #3d4a5a; }

  .thin-rule { border: none; border-top: 1px solid #1c1f2a; margin: 1.4rem 0; }

  div[data-testid="stTextArea"] textarea {
      background-color: #0f1017 !important; border: 1px solid #1e2230 !important;
      border-radius: 8px !important; color: #c8c4bc !important;
      font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important;
      line-height: 1.6 !important; padding: 14px 16px !important; resize: none !important;
  }
  div[data-testid="stTextArea"] textarea:focus { border-color: #3d4a6b !important; box-shadow: 0 0 0 3px rgba(61,74,107,0.15) !important; }
  div[data-testid="stTextArea"] textarea::placeholder { color: #252830 !important; }
  div[data-testid="stTextArea"] label, div[data-testid="stTextInput"] label {
      font-size: 0.73rem !important; font-weight: 500 !important;
      letter-spacing: 0.08em !important; text-transform: uppercase !important; color: #3d4a5a !important;
  }

  div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
      background-color: #f0ece4 !important; color: #0a0a0f !important;
      font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
      font-size: 0.83rem !important; letter-spacing: 0.06em !important;
      border: none !important; border-radius: 6px !important;
      padding: 0.65rem 2rem !important; width: 100% !important;
      transition: background-color 0.18s ease !important;
  }
  div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover { background-color: #ddd8cf !important; }

  .story-card { background-color: #0f1017; border: 1px solid #1e2230; border-radius: 12px; padding: 2.6rem 3rem 2rem; margin-top: 1.4rem; }
  .story-card-eyebrow { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; color: #2a3045; text-align: center; margin-bottom: 1rem; }
  .story-title { font-family: 'Playfair Display', serif; font-size: 1.75rem; font-weight: 700; color: #f0ece4; text-align: center; margin-bottom: 0.4rem; line-height: 1.2; }
  .story-rule { width: 36px; height: 2px; background-color: #2a3045; margin: 1rem auto 1.6rem; border: none; }
  .story-body { font-family: 'Playfair Display', serif; font-style: italic; font-size: 1.08rem; line-height: 2.05; color: #a8a49c; }
  .story-body p { margin-bottom: 1.15em; }

  .meta-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 1rem 0 0.3rem; }
  .meta-tag { display: inline-block; background-color: #0f1017; border: 1px solid #1e2230; color: #3d4a5a; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.07em; text-transform: uppercase; padding: 3px 9px; border-radius: 4px; }

  .score-row { display: flex; align-items: center; gap: 10px; margin: 0.5rem 0 0.3rem; }
  .score-value { font-size: 0.78rem; font-weight: 600; color: #f0ece4; letter-spacing: 0.04em; }
  .score-bar-bg { flex: 1; height: 3px; background-color: #1e2230; border-radius: 2px; overflow: hidden; }
  .score-bar-fill { height: 100%; border-radius: 2px; background-color: #c8c4bc; }
  .score-label { font-size: 0.67rem; color: #2a3045; letter-spacing: 0.06em; text-transform: uppercase; }

  .section-heading { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #2a3045; margin-bottom: 0.9rem; margin-top: 1.4rem; }
  .step-status { font-size: 0.8rem; color: #2a3045; letter-spacing: 0.04em; padding: 0.4rem 0; }
  .feedback-wrapper { margin-top: 1.8rem; padding-top: 1.8rem; border-top: 1px solid #1c1f2a; }

  section[data-testid="stSidebar"] { background-color: #07070c; border-right: 1px solid #1c1f2a; }
  .lib-heading { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: #2a3045; margin-bottom: 1rem; }

  div[data-testid="stRadio"] label { font-size: 0.8rem !important; color: #a8a49c !important; }
  div[data-testid="stSelectbox"] label { font-size: 0.73rem !important; color: #3d4a5a !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }

  details summary { font-size: 0.74rem !important; font-weight: 500 !important; color: #2a3045 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
  audio { width: 100%; margin-top: 0.8rem; border-radius: 6px; }
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# -- Helpers ------------------------------------------------------------------

def _time_ago(iso_ts: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts).replace(tzinfo=timezone.utc)
        s = int((datetime.now(timezone.utc) - dt).total_seconds())
        if s < 60:    return "just now"
        if s < 3600:  return f"{s // 60}m ago"
        if s < 86400: return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"
    except Exception:
        return ""


def _extract_title_body(text: str) -> tuple[str, str]:
    """
    Extract the title (first non-blank line) and body from a story string.
    Strips markdown heading (#) and bold/italic (** * __) markers.
    Skips leading blank lines so the function is robust to model responses
    that begin with one or more newlines (Issue 4).
    """
    lines = text.strip().split("\n")
    title = "A Bedtime Story"
    title_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip().strip("*").strip("_").strip()
        if stripped:
            title = stripped
            title_idx = i
            break
    body = "\n".join(lines[title_idx + 1:]).strip()
    return title, body


def _result_from_db_row(row: dict) -> dict:
    import json
    # Issue 8: guard against NULL or corrupted metadata JSON in the DB
    try:
        meta = json.loads(row.get("metadata") or "{}")
    except (json.JSONDecodeError, TypeError):
        meta = {}
    return {
        "story": row["story_body"],
        "title": row["title"] or "A Bedtime Story",
        "full_story": row["story_body"],
        "metadata": meta,
        "arc": row.get("arc", ""),
        "judge_history": [{"total_score": row["final_score"], "scores": {}, "strengths": [], "weaknesses": []}],
        "iterations": row.get("iterations", 0),
        "request": row.get("request", ""),
    }


# -- Sidebar: Story Library ---------------------------------------------------

with st.sidebar:
    st.markdown('<div class="lib-heading">Story Library</div>', unsafe_allow_html=True)
    all_stories = get_all_stories(limit=50)

    if not all_stories:
        st.markdown(
            '<div style="font-size:0.75rem;color:#2a3045;">No stories yet. Generate one to get started.</div>',
            unsafe_allow_html=True,
        )
    else:
        for row in all_stories:
            genre = (row.get("genre") or "").replace("_", " ").title()
            when  = _time_ago(row.get("timestamp", ""))
            score = row.get("final_score", 0)
            label = row["title"][:38] + ("..." if len(row["title"]) > 38 else "")
            if st.button(label, key=f"lib_{row['id']}"):
                st.session_state.result = _result_from_db_row(row)
                st.session_state.original_request = row.get("request", "")
                st.session_state.current_story_id = row["id"]
                st.session_state.audio_cache = OrderedDict()  # Reset LRU cache
                log.info(f"Story loaded from library: id={row['id']}, title={row['title'][:40]}")
                st.rerun()
            st.caption(f"{genre}  ·  {score}/10  ·  {when}")


# -- Main area ----------------------------------------------------------------

st.markdown("""
<div class="page-header">
  <div class="page-label">Hippocratic AI &mdash; Coding Assignment</div>
  <div class="page-title">Bedtime Story Generator</div>
  <div class="page-sub">Describe the story you would like and the agent will craft it for you.</div>
</div>
<hr class="thin-rule">
""", unsafe_allow_html=True)


# -- Input form ---------------------------------------------------------------

if not st.session_state.result:
    with st.form("story_form"):
        user_request = st.text_area(
            "Story request",
            placeholder='e.g. "A story about a brave little dragon who is scared of fire"',
            height=110,
        )
        submitted = st.form_submit_button("Generate Story")

    if submitted and user_request.strip():
        status = st.empty()

        # Step 1 — Classify
        status.markdown('<div class="step-status">Classifying story request...</div>', unsafe_allow_html=True)
        try:
            metadata = classify(user_request.strip())
            log.info(f"Classified: genre={metadata.get('genre')}, age={metadata.get('age_lean')}")
        except Exception as e:
            log.error(f"Classify failed: {e}", exc_info=True)
            st.error(f"Generation failed: {e}")
            st.info("Refer to Readme.md and set .env file with an OpenAI API key. Ensure your OPENAI_API_KEY is set correctly.")
            st.stop()

        # Step 2 — Plan
        status.markdown('<div class="step-status">Building story arc...</div>', unsafe_allow_html=True)
        arc = plan_story(user_request.strip(), metadata)
        log.info(f"Arc generated: {len(arc.split())} words")

        # Step 3 — Stream story
        status.markdown('<div class="step-status">Writing story...</div>', unsafe_allow_html=True)
        try:
            full_draft = st.write_stream(write_story_stream(user_request.strip(), arc, metadata))
            log.info(f"Story streamed: {len(full_draft.split())} words")
        except Exception as e:
            log.error(f"Story streaming failed: {e}", exc_info=True)
            st.error(f"Story generation failed: {e}")
            st.stop()

        # Step 4 — Expand if short, then conditionally self-reflect
        # Self-reflection is SKIPPED when expansion ran — expansion already
        # enforces rules, and logs confirm reflection adds 0 words when called
        # after expansion (6-8s saved per story)
        status.markdown('<div class="step-status">Reviewing and expanding if needed...</div>', unsafe_allow_html=True)
        t4 = time.time()
        initial_words = len(full_draft.split())
        story_after = expand_story_if_short(full_draft, user_request.strip(), metadata)
        expanded = len(story_after.split()) > initial_words
        if expanded:
            story = story_after
            log.info(f"Expansion done ({time.time()-t4:.1f}s): {initial_words}->{len(story.split())} words")
        else:
            story = full_draft
            log.info(f"No expansion needed ({time.time()-t4:.1f}s): {initial_words} words")
            t5 = time.time()
            story = self_reflect_story(story, user_request.strip(), metadata)
            log.info(f"Self-reflect done ({time.time()-t5:.1f}s): {len(story.split())} words")

        # Step 5 — Judge loop
        status.markdown('<div class="step-status">Evaluating quality...</div>', unsafe_allow_html=True)
        judge_history = []
        iterations = 0
        for i in range(MAX_JUDGE_ITERATIONS):
            critique = judge_story(user_request.strip(), story)
            judge_history.append(critique)
            score = critique.get("total_score", 0)
            log.info(f"Judge pass {i + 1}: score={score}/10")
            if not critique.get("needs_revision", False) or score >= JUDGE_PASS_THRESHOLD:
                log.info(f"Story accepted at score {score}/10")
                break
            log.warning(f"Score {score}/10 < threshold — revising")
            status.markdown('<div class="step-status">Refining story...</div>', unsafe_allow_html=True)
            story = revise_story(user_request.strip(), story, critique)
            iterations += 1

        title, story_body = _extract_title_body(story)

        result = {
            "story": story_body,
            "title": title,
            "full_story": story,
            "metadata": metadata,
            "arc": arc,
            "judge_history": judge_history,
            "iterations": iterations,
            "request": user_request.strip(),
        }

        # Save to DB — new story gets a new row
        story_id = save_story(st.session_state.session_id, result)
        log.info(f"Story saved to DB: id={story_id}, title={title[:40]}")

        st.session_state.result = result
        st.session_state.original_request = user_request.strip()
        st.session_state.current_story_id = story_id
        st.session_state.audio_cache = OrderedDict()  # Reset LRU cache
        status.empty()
        st.rerun()

    elif submitted:
        st.warning("Please enter a story request before generating.")


# -- Story display ------------------------------------------------------------

if st.session_state.result:
    result = st.session_state.result
    meta   = result.get("metadata", {})

    # Metadata tags — HTML-escaped to prevent injection (Issue 1)
    age_map = {"young": "5-6", "middle": "7-8", "older": "9-10"}
    tags = [
        meta.get("genre", "").replace("_", " ").title(),
        meta.get("tone", "").title(),
        "Ages " + age_map.get(meta.get("age_lean", "middle"), "7-8"),
    ] + [t.title() for t in meta.get("themes", [])]
    tags = [html_module.escape(t) for t in tags if t.strip()]

    st.markdown(
        '<div class="meta-row">' +
        "".join(f'<span class="meta-tag">{t}</span>' for t in tags) +
        '</div>',
        unsafe_allow_html=True,
    )

    if result.get("judge_history"):
        score = result["judge_history"][-1].get("total_score", 0)
        iters = result.get("iterations", 0)
        iter_label = f"{iters} revision" + ("s" if iters != 1 else "")
        fill_pct = round(score / 12 * 100)
        st.markdown(f"""
        <div class="score-row">
          <span class="score-label">Quality</span>
          <div class="score-bar-bg"><div class="score-bar-fill" style="width:{fill_pct}%"></div></div>
          <span class="score-value">{score}/12</span>
          <span class="score-label">&middot; {iter_label}</span>
        </div>
        """, unsafe_allow_html=True)

    # Story card — HTML-escape all model-generated content (Issue 1)
    safe_title = html_module.escape(result['title'])
    safe_paras = "".join(
        f"<p>{html_module.escape(p.strip())}</p>"
        for p in result["story"].split("\n\n") if p.strip()
    )
    st.markdown(f"""
    <div class="story-card">
      <div class="story-card-eyebrow">Your Story</div>
      <div class="story-title">{safe_title}</div>
      <hr class="story-rule">
      <div class="story-body">{safe_paras}</div>
    </div>
    """, unsafe_allow_html=True)

    # Listen to Story (TTS)
    st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Listen to this Story</div>', unsafe_allow_html=True)

    tts_col1, tts_col2 = st.columns([1, 2])
    with tts_col1:
        gender = st.radio("Voice gender", options=list(VOICES.keys()), horizontal=False, label_visibility="collapsed")
    with tts_col2:
        voice_options = VOICES[gender]
        voice_label   = st.selectbox("Select voice", options=list(voice_options.keys()))
        selected_voice = voice_options[voice_label]

    cache_key  = (result["title"], selected_voice)
    audio_ready = cache_key in st.session_state.audio_cache

    if st.button("Play Story" if audio_ready else "Generate Audio", key="listen_btn"):
        if not audio_ready:
            with st.spinner("Generating audio narration..."):
                try:
                    narration = f"{result['title']}.\n\n{result['story']}"
                    audio_bytes = generate_audio(narration, voice=selected_voice)
                    # LRU eviction: remove oldest entry if at capacity (Issue 11)
                    _AUDIO_CACHE_MAX = 5
                    cache = st.session_state.audio_cache
                    if len(cache) >= _AUDIO_CACHE_MAX:
                        cache.popitem(last=False)
                    cache[cache_key] = audio_bytes
                    log.info(f"TTS generated: voice={selected_voice}, ~{len(narration)} chars")
                except Exception as e:
                    log.error(f"TTS failed: {e}", exc_info=True)
                    st.error(f"Audio generation failed: {e}")

        if cache_key in st.session_state.audio_cache:
            st.audio(st.session_state.audio_cache[cache_key], format="audio/mp3")

    elif audio_ready:
        st.audio(st.session_state.audio_cache[cache_key], format="audio/mp3")

    # Quality breakdown (collapsed)
    st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)
    with st.expander("Quality Breakdown", expanded=False):
        for i, critique in enumerate(result.get("judge_history", [])):
            st.markdown(f"**Pass {i + 1}** — Score: `{critique.get('total_score', 0)}/12`")
            scores = critique.get("scores", {})
            if scores:
                labels = {
                    "age_appropriateness": "Age Safe",
                    "narrative_completeness": "Structure",
                    "engagement": "Engagement",
                    "alignment": "Alignment",
                    "emotional_resonance": "Emotion",
                    "word_count_adequacy": "Length",
                }
                cols = st.columns(6)
                for col, (key, label) in zip(cols, labels.items()):
                    col.metric(label, f"{scores.get(key, 0)}/2")
            if critique.get("strengths"):
                st.caption("Strengths: " + " / ".join(critique["strengths"]))
            if critique.get("weaknesses"):
                st.caption("Weaknesses: " + " / ".join(critique["weaknesses"]))

    if result.get("arc"):
        with st.expander("Story Arc", expanded=False):
            st.code(result["arc"], language=None)

    # Request Changes
    st.markdown('<div class="feedback-wrapper"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Request Changes</div>', unsafe_allow_html=True)

    with st.form("feedback_form"):
        feedback = st.text_area(
            "What would you like to change?",
            placeholder='e.g. "Make it funnier" or "Give the dragon a name"',
            height=80,
        )
        revise_btn = st.form_submit_button("Revise Story")

    if revise_btn and feedback.strip():
        with st.spinner("Revising story..."):
            try:
                new_result = revise_pipeline(
                    original_request=st.session_state.original_request,
                    previous_story=result.get("full_story", result["story"]),
                    previous_arc=result.get("arc", ""),
                    previous_metadata=result.get("metadata", {}),
                    user_feedback=feedback.strip(),
                    session_id=st.session_state.session_id,
                )
                # UPDATE the existing record — no new sidebar entry
                if st.session_state.current_story_id:
                    update_story(st.session_state.current_story_id, new_result)
                    log.info(f"Story updated in DB: id={st.session_state.current_story_id}")
                else:
                    sid = save_story(st.session_state.session_id, new_result)
                    st.session_state.current_story_id = sid
                    log.info(f"Revision saved as new entry: id={sid}")

                st.session_state.result = new_result
                st.session_state.audio_cache = OrderedDict()  # Reset LRU cache
                st.rerun()
            except Exception as e:
                log.error(f"Revision failed: {e}", exc_info=True)
                st.error(f"Revision failed: {e}")

    # New story
    st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)
    if st.button("Start a New Story"):
        log.info("User started a new story")
        st.session_state.result = None
        st.session_state.original_request = ""
        st.session_state.current_story_id = None
        st.session_state.audio_cache = OrderedDict()  # Reset LRU cache
        st.rerun()


# -- Footer -------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;margin-top:4rem;margin-bottom:2rem;
     color:#1a1d28;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;">
  Hippocratic AI &mdash; Coding Assignment &mdash; GPT-3.5-turbo
</div>
""", unsafe_allow_html=True)
