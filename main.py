import os
from dotenv import load_dotenv
import time

load_dotenv()

from agents.classifier import classify
from agents.planner import plan_story
from agents.storyteller import write_story, self_reflect_story, expand_story_if_short
from agents.judge import judge_story
from agents.reviser import revise_story
from utils.llm import call_model  # kept for backward compatibility with original skeleton
from utils.logger import get_logger


MAX_JUDGE_ITERATIONS = 2   # Maximum improvement loops before accepting the story
JUDGE_PASS_THRESHOLD = 9   # Score out of 12 needed to skip revision (9/12 = 75%)

# CLI session uses a fixed id
_CLI_SESSION = "cli-session"


def _elapsed(start: float) -> str:
    """Return elapsed seconds as a formatted string."""
    return f"{time.time() - start:.1f}s"


def run_pipeline(
    user_request: str,
    verbose: bool = False,
    session_id: str = _CLI_SESSION,
) -> dict:
    """
    Full story generation pipeline:
      1. Classify the request                    (~5s)
      2. Plan a 7-beat story arc                 (~5s)
      3. Write the story (streaming in UI)       (~4-9s)
      4. Expand if below minimum word count      (~8s, only when needed)
      5. Self-reflect (only if expansion skipped)  (~6s, conditional)
      6. Judge -> Revise loop                    (~2s per pass)

    Latency optimisations applied:
    - Self-reflection is SKIPPED when expand_story_if_short() already ran
      (expansion already enforces rules; reflection would add ~6s for 0 gain)
    - Planner max_tokens capped at 600 (7-beat arc needs ~400 tokens)
    - Every step is timed and logged for future profiling

    Args:
        user_request: The story request string from the user.
        verbose:      If True, print step-by-step progress to stdout.
        session_id:   Session identifier for structured logging.

    Returns:
        Dict with: story, title, full_story, metadata, arc,
        judge_history, iterations, request.
    """
    log = get_logger(session_id)
    pipeline_start = time.time()

    def msg(m: str):
        if verbose:
            print(m)

    log.info(f"New request: \"{user_request[:80]}\"")

    # -- Step 1: Classify -------------------------------------------------------
    t = time.time()
    msg("\nStep 1/5 — Classifying story request...")
    metadata = classify(user_request)
    log.info(
        f"Classify done ({_elapsed(t)}): "
        f"genre={metadata.get('genre')}, tone={metadata.get('tone')}, age={metadata.get('age_lean')}"
    )
    msg(f"   Genre: {metadata.get('genre')} | Tone: {metadata.get('tone')} | Age: {metadata.get('age_lean')}")

    # -- Step 2: Plan -----------------------------------------------------------
    t = time.time()
    msg("\nStep 2/5 — Building 7-beat story arc...")
    arc = plan_story(user_request, metadata)
    log.info(f"Plan done ({_elapsed(t)}): {len(arc.split())} words")
    msg(f"   Arc generated ({len(arc.split())} words)")

    # -- Step 3: Write ----------------------------------------------------------
    t = time.time()
    msg("\nStep 3/5 — Writing the story...")
    story = write_story(user_request, arc, metadata)
    initial_words = len(story.split())
    log.info(f"Story written ({_elapsed(t)}): {initial_words} words")
    msg(f"   Story written ({initial_words} words)")

    # -- Step 4: Expand if short (conditional) ----------------------------------
    t = time.time()
    expanded = False
    story_after = expand_story_if_short(story, user_request, metadata)
    if len(story_after.split()) > initial_words:
        expanded = True
        story = story_after
        log.info(f"Expansion done ({_elapsed(t)}): {initial_words} -> {len(story.split())} words")
        msg(f"   Expanded ({initial_words} -> {len(story.split())} words)")
    else:
        log.info(f"No expansion needed ({_elapsed(t)}): {initial_words} words already sufficient")

    # -- Step 5: Self-reflect (SKIPPED if expansion ran) -----------------------
    # Rationale: expand_story_if_short() already instructs the model to follow
    # all rules when rewriting. Running self-reflection on top adds ~6s with
    # near-zero benefit — logs show 0 words changed across all three prior runs.
    if not expanded:
        t = time.time()
        msg("\nStep 4b/5 — Self-reflection pass (story was already long enough)...")
        story = self_reflect_story(story, user_request, metadata)
        log.info(f"Self-reflect done ({_elapsed(t)}): {len(story.split())} words")
        msg(f"   Reflection complete ({len(story.split())} words)")
    else:
        log.info("Self-reflect SKIPPED: expansion already enforced all rules")
        msg("   Self-reflect skipped (expansion already applied rule enforcement)")

    # -- Step 6: Judge -> Revise loop ------------------------------------------
    t = time.time()
    msg("\nStep 5/5 — Evaluating story quality...")
    judge_history = []
    iterations = 0

    for i in range(MAX_JUDGE_ITERATIONS):
        critique = judge_story(user_request, story)
        judge_history.append(critique)
        score = critique.get("total_score", 0)
        needs = critique.get("needs_revision", False)

        log.info(
            f"Judge pass {i + 1} ({_elapsed(t)}): score={score}/12, "
            f"word_count_score={critique.get('scores', {}).get('word_count_adequacy', '?')}/2, "
            f"needs_revision={needs}"
        )
        msg(f"   Iteration {i + 1}: Score {score}/12")

        if not needs or score >= JUDGE_PASS_THRESHOLD:
            log.info(f"Story accepted: score={score}/12, threshold={JUDGE_PASS_THRESHOLD}/12")
            msg(f"   Quality threshold met ({score}/12). Story accepted.")
            break

        log.warning(f"Score {score}/12 below threshold {JUDGE_PASS_THRESHOLD} — revising")
        msg(f"   Revising story ({score}/12 < {JUDGE_PASS_THRESHOLD})...")
        t_rev = time.time()
        story = revise_story(user_request, story, critique)
        iterations += 1
        log.info(f"Revision {iterations} done ({_elapsed(t_rev)}): {len(story.split())} words")
        msg(f"   Story revised ({len(story.split())} words)")
        t = time.time()  # Reset for next judge call

    log.info(f"Pipeline complete: total={_elapsed(pipeline_start)}, iterations={iterations}")

    lines = story.strip().split("\n")
    title = lines[0].strip().lstrip("#").strip().strip("*").strip("_").strip() if lines else "A Bedtime Story"
    if not title:
        title = "A Bedtime Story"
    story_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else story

    return {
        "story": story_body,
        "title": title,
        "full_story": story,
        "metadata": metadata,
        "arc": arc,
        "judge_history": judge_history,
        "iterations": iterations,
        "request": user_request,
    }


def revise_pipeline(
    original_request: str,
    previous_story: str,
    previous_arc: str,
    previous_metadata: dict,
    user_feedback: str,
    verbose: bool = False,
    session_id: str = _CLI_SESSION,
) -> dict:
    """
    Context-aware revision pipeline.

    Skips classify+plan entirely and passes the ACTUAL previous story text
    as context into the reviser. Preserves story continuity across revisions.
    Applies expand and self-reflect as a combined quality gate, then judges.

    Latency note: Revision is typically faster than fresh generation because
    classify+plan are skipped (~10s saved). Total revision time: ~20-25s.

    Args:
        original_request:   The user's original story request.
        previous_story:     The actual story text from the last generation.
        previous_arc:       The story arc used in the last generation.
        previous_metadata:  Classifier metadata from the last generation.
        user_feedback:      The user's change request in plain language.
        verbose:            Print progress to stdout.
        session_id:         Session identifier for structured logging.

    Returns:
        Same dict structure as run_pipeline().
    """
    log = get_logger(session_id)
    pipeline_start = time.time()

    def msg(m: str):
        if verbose:
            print(m)

    log.info(f"Revision requested: \"{user_feedback[:80]}\"")

    user_driven_critique = {
        "scores": {
            "age_appropriateness": 2, "narrative_completeness": 2,
            "engagement": 1, "alignment": 1, "emotional_resonance": 1,
            "word_count_adequacy": 1,
        },
        "total_score": 8,
        "strengths": ["Existing story structure and characters"],
        "weaknesses": ["User has requested specific changes"],
        "revision_instructions": (
            f"The user has provided this feedback on the current story: '{user_feedback}'. "
            "Revise the story to address this feedback precisely while preserving everything "
            "that was already working — characters, overall structure, tone, and the "
            "God/gratitude salutation at the end."
        ),
        "needs_revision": True,
    }

    # -- Apply user feedback ---------------------------------------------------
    t = time.time()
    msg("\nRevision Step 1/3 — Applying user feedback...")
    story = revise_story(original_request, previous_story, user_driven_critique)
    log.info(f"Revision applied ({_elapsed(t)}): {len(story.split())} words")
    msg(f"   Revised ({len(story.split())} words)")

    # -- Expand + selective self-reflect ---------------------------------------
    t = time.time()
    initial_words = len(story.split())
    story_after = expand_story_if_short(story, original_request, previous_metadata)
    expanded = len(story_after.split()) > initial_words
    if expanded:
        story = story_after
        log.info(f"Expansion on revision ({_elapsed(t)}): {initial_words} -> {len(story.split())} words")
    else:
        log.info(f"No expansion needed on revision ({_elapsed(t)})")
        # Run self-reflect only if not expanded (same logic as fresh pipeline)
        t = time.time()
        story = self_reflect_story(story, original_request, previous_metadata)
        log.info(f"Self-reflect on revision ({_elapsed(t)}): {len(story.split())} words")

    # -- Judge loop ------------------------------------------------------------
    t = time.time()
    msg("\nRevision Step 3/3 — Validating revised story...")
    judge_history = []
    iterations = 1

    for i in range(MAX_JUDGE_ITERATIONS):
        critique = judge_story(original_request, story)
        judge_history.append(critique)
        score = critique.get("total_score", 0)
        needs = critique.get("needs_revision", False)

        log.info(
            f"Judge pass {i + 1} (revision, {_elapsed(t)}): score={score}/12, "
            f"word_count_score={critique.get('scores', {}).get('word_count_adequacy', '?')}/2"
        )
        msg(f"   Iteration {i + 1}: Score {score}/12")

        if not needs or score >= JUDGE_PASS_THRESHOLD:
            log.info(f"Revision accepted: score={score}/12")
            msg(f"   Revision accepted ({score}/12).")
            break

        log.warning(f"Revision score {score}/12 below threshold — refining further")
        t_rev = time.time()
        story = revise_story(original_request, story, critique)
        log.info(f"Additional revision ({_elapsed(t_rev)}): {len(story.split())} words")
        iterations += 1
        t = time.time()

    log.info(f"Revision pipeline complete: total={_elapsed(pipeline_start)}, iterations={iterations}")

    lines = story.strip().split("\n")
    title = lines[0].strip().lstrip("#").strip().strip("*").strip("_").strip() if lines else "A Bedtime Story"
    if not title:
        title = "A Bedtime Story"
    story_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else story

    return {
        "story": story_body,
        "title": title,
        "full_story": story,
        "metadata": previous_metadata,
        "arc": previous_arc,
        "judge_history": judge_history,
        "iterations": iterations,
        "request": original_request,
    }


def main():
    """CLI entry point — preserves the original skeleton's interface."""
    import sys
    print("Bedtime Story Generator")
    print("=" * 50)
    user_input = input("What kind of story do you want to hear? ").strip()
    if not user_input:
        user_input = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."

    try:
        result = run_pipeline(user_input, verbose=True)
    except Exception as e:
        log = get_logger(_CLI_SESSION)
        log.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"  {result['title']}")
    print("=" * 50)
    print(result["story"])
    print("\n" + "=" * 50)

    final_score = result["judge_history"][-1]["total_score"] if result["judge_history"] else "N/A"
    print(f"Final quality score: {final_score}/12  |  Iterations: {result['iterations']}")

    print("\nWould you like any changes to the story? (Press Enter to finish)")
    feedback = input("> ").strip()
    if feedback:
        print("\nRevising based on your feedback...\n")
        result2 = revise_pipeline(
            original_request=user_input,
            previous_story=result["full_story"],
            previous_arc=result["arc"],
            previous_metadata=result["metadata"],
            user_feedback=feedback,
            verbose=True,
        )
        print("=" * 50)
        print(f"  {result2['title']}")
        print("=" * 50)
        print(result2["story"])


if __name__ == "__main__":
    main()