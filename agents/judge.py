"""
agents/judge.py

LLM Judge: evaluates a generated story on 6 criteria and returns a
structured critique with a numeric score and specific revision instructions.

Scoring rubric (each 0-2 points, total 0-12):
  1. age_appropriateness  — vocabulary safe for 5-10; no frightening content
  2. narrative_completeness — clear beginning, middle, end (7-beat arc honoured)
  3. engagement            — dialogue, sensory detail, suspense, refrain present
  4. alignment             — matches what the user asked for
  5. emotional_resonance   — leaves child feeling happy/sleepy/comforted
  6. word_count_adequacy   — story long enough for the age group

Pass threshold: 9/12 (~75%)

Reliability features (Issue 3):
  - All scores are clamped to [0, 2] after parsing
  - total_score is recalculated from the sum (never trusted from model output)
  - needs_revision is recalculated from total vs threshold
"""

import json
from utils.llm import call_model

SYSTEM = """You are an expert children's literature critic and editor.
You evaluate bedtime stories for children aged 5-10 with a structured rubric.
You are honest and precise — you do not inflate scores.
Always respond with valid JSON and nothing else."""

PROMPT_TEMPLATE = """Evaluate the following children's bedtime story against the original user request.

**Original User Request:** {request}

**Story Word Count:** {word_count} words

**Story to Evaluate:**
{story}

Score the story on EACH of the 6 criteria from 0 to 2 (0=poor, 1=adequate, 2=excellent):

1. age_appropriateness: Is the vocabulary and content safe and suitable for ages 5-10?
   Award 0 if the story contains frightening imagery, violence, or content that would genuinely
   scare a young child (ghosts as scary, monsters as threatening, etc.).
2. narrative_completeness: Does the story have a clear beginning, middle, and satisfying end?
3. engagement: Does the story include dialogue, sensory detail, a suspense moment, and a refrain?
4. alignment: Does the story faithfully address the user's original request?
5. emotional_resonance: Does the story leave a child feeling happy, comforted, or sleepy
   (should include a God/gratitude closing sentence)?
6. word_count_adequacy: Is the story long enough to be a satisfying bedtime story?
   - For ages 5-6: award 2 if >= 400 words, 1 if 250-399, 0 if < 250
   - For ages 7-8: award 2 if >= 600 words, 1 if 350-599, 0 if < 350
   - For ages 9-10: award 2 if >= 800 words, 1 if 500-799, 0 if < 500
   - If age is unclear, use the 7-8 thresholds
   - Current word count is {word_count} — apply the rule strictly, do not round up

IMPORTANT: Do NOT inflate scores. A 317-word story cannot score 2 on word_count_adequacy for ages 7-8.
Total score = sum of all 6 criteria scores (max 12).

Return ONLY this JSON structure — no explanation, no markdown:
{{
  "scores": {{
    "age_appropriateness": <0-2>,
    "narrative_completeness": <0-2>,
    "engagement": <0-2>,
    "alignment": <0-2>,
    "emotional_resonance": <0-2>,
    "word_count_adequacy": <0-2>
  }},
  "total_score": <0-12>,
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "revision_instructions": "<Specific actionable instructions. If story is short say 'The story must be significantly longer — aim for X words'. If great, say 'No major revisions needed.'>",
  "needs_revision": <true if total_score < 9, false otherwise>
}}"""

_CRITERIA = [
    "age_appropriateness",
    "narrative_completeness",
    "engagement",
    "alignment",
    "emotional_resonance",
    "word_count_adequacy",
]

_PASS_THRESHOLD = 9  # Must match JUDGE_PASS_THRESHOLD in main.py


def _validate_and_fix(critique: dict) -> dict:
    """
    Clamp all individual criterion scores to [0, 2], recalculate total_score
    from their sum, and recalculate needs_revision from the corrected total.

    This guards against the model returning scores like 3/2, -1/2, or a
    total_score that doesn't match the sum of individual scores (Issue 3).
    """
    scores = critique.get("scores", {})

    # Clamp each criterion to the valid [0, 2] integer range
    clamped = {}
    for key in _CRITERIA:
        raw = scores.get(key, 1)
        try:
            clamped[key] = max(0, min(2, int(raw)))
        except (TypeError, ValueError):
            clamped[key] = 1  # Default to adequate if unparseable

    total = sum(clamped.values())  # Always 0-12, always consistent with criteria

    critique["scores"] = clamped
    critique["total_score"] = total
    critique["needs_revision"] = total < _PASS_THRESHOLD
    return critique


def judge_story(user_request: str, story: str) -> dict:
    """
    Evaluate a story and return a validated, clamped critique dict.

    Scoring is out of 12 (6 criteria × 2 points each).
    Pass threshold: 9/12 (~75%).

    All numeric fields are validated and corrected after parsing:
    individual scores are clamped to [0,2] and total is recalculated.

    Args:
        user_request: The original user request string.
        story:        The story text to evaluate.

    Returns:
        Dict with keys: scores (clamped), total_score (recalculated),
        strengths, weaknesses, revision_instructions, needs_revision.
    """
    word_count = len(story.split())

    prompt = PROMPT_TEMPLATE.format(
        request=user_request,
        story=story,
        word_count=word_count,
    )
    raw = call_model(prompt, system=SYSTEM, temperature=0.1, max_tokens=700)

    # Strip accidental markdown code fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        critique = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: conservative scores that will trigger a revision loop
        critique = {
            "scores": {k: 1 for k in _CRITERIA},
            "total_score": 6,
            "strengths": ["Story generated successfully"],
            "weaknesses": ["Could not parse judge response — manual review needed"],
            "revision_instructions": "Improve overall quality and ensure the story is long enough.",
            "needs_revision": True,
        }

    # Always run validation to clamp and recalculate (Issue 3)
    return _validate_and_fix(critique)
