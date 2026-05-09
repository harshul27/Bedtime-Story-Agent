"""
agents/storyteller.py

Writes the full prose story from the 7-beat arc and classification metadata.

Prompting strategies used:
- Role prompting       — warm, beloved children's author persona
- Vocabulary profiles  — age-calibrated word and sentence constraints per age band
- Engagement devices   — dialogue, sensory detail, suspense beat, refrain, inner thought
- Constraint prompting — length, safety rules, tone
- Few-shot style       — one embedded style example to prime the register
- God salutation       — required gentle bedtime gratitude as final paragraph
- Self-reflection      — model critiques its own draft before the Judge sees it

Exposes both write_story() (blocking) and write_story_stream() (generator/streaming).
"""

from typing import Generator
from utils.llm import call_model, stream_model

SYSTEM = """You are a beloved children's storyteller in the tradition of Roald Dahl and A.A. Milne.
You write magical, warm, age-appropriate bedtime stories that children aged 5-10 adore.
Your writing is vivid, gentle, and full of heart — always perfectly matched to the child's age."""

FEW_SHOT_EXAMPLE = """Style reference (DO NOT copy — use only for tone and rhythm):
---
Once upon a time, in a cosy little cottage at the edge of a whispering wood, there lived a small hedgehog named Hazel.
Every evening, Hazel would sit by her window and watch the fireflies dance, wondering what lay beyond the tall oak trees.
"Someday," she would whisper to herself, "I shall find out."
---"""

# -- Age-calibrated vocabulary profiles --------------------------------------
AGE_PROFILES = {
    "young": """VOCABULARY RULES for ages 5-6 (STRICT — follow exactly):
- Use ONLY 1-2 syllable words wherever possible (e.g. "big" not "enormous", "run" not "gallop", "nice" not "delightful")
- Maximum sentence length: 8 words
- Emotions allowed: happy, sad, scared, sleepy, excited, surprised — nothing more complex
- Use a comforting repeating phrase or refrain at least twice (e.g. "And do you know what happened next?")
- Characters must be animals, toys, or very familiar objects (no abstract villains)
- Use "And then..." and "And just then..." as sentence starters for flow
- Every paragraph ends with something warm or funny""",

    "middle": """VOCABULARY RULES for ages 7-8:
- Up to 3-syllable words are fine (adventure, discover, whispered, enormous, trembling)
- Sentences can be longer but keep them clear — one idea per sentence
- Include simple inner thoughts ("She wondered if...", "He hoped that...")
- Include at least 3 exchanges of direct dialogue between characters
- Light suspense is good — a moment where things look uncertain, then resolve happily
- Use cause-and-effect ("Because the storm was coming, they decided to...")
- A small joke or funny moment is encouraged""",

    "older": """VOCABULARY RULES for ages 9-10:
- Rich vocabulary including metaphors and similes ("the sky blazed like a tangerine", "his heart sank like a stone")
- Complex sentences with subordinate clauses are welcome
- Deep character motivations — the reader should understand WHY the character wants what they want
- Inner conflict is fine — characters can doubt themselves before finding courage
- Sensory language: describe sounds, smells, textures, temperatures in vivid detail
- At least 4 dialogue exchanges with distinct character voices
- Mild mystery or tension that builds over several paragraphs before resolving""",
}

# -- Engagement devices required in every story ------------------------------
ENGAGEMENT_DEVICES = """REQUIRED ENGAGEMENT DEVICES (all stories regardless of age):
1. DIALOGUE: Include direct speech between characters (at least the minimum for the age group)
2. SENSORY MOMENT: At least one moment describing what something smells, sounds, or feels like
3. SUSPENSE BEAT: One moment where things look uncertain — then resolve happily soon after
4. REPEATING REFRAIN: A phrase or pattern the child can anticipate (e.g. a character's catchphrase or ritual)
5. INNER THOUGHT: One moment where the reader knows what the main character is thinking or feeling
6. WARM CLOSING IMAGE: The very last story paragraph (before the salutation) ends with a peaceful, sleepy visual"""

# -- God/gratitude salutation ------------------------------------------------
GOD_SALUTATION = """REQUIRED FINAL ELEMENT — BEDTIME GRATITUDE:
After the story's resolution, add one final short paragraph (2-3 sentences) as a gentle bedtime salutation.
Rules for the salutation:
- Name the main character (do not use "they" — use the character's actual name)
- Express gratitude to God in a warm, simple, non-denominational way
- Vary the phrasing based on the story's theme (e.g. thankful for friends, for a new day, for a kind heart)
- Keep it simple enough for a 5-year-old to understand
- End with a sleepy or peaceful image

Examples of good salutations (vary these — do not copy exactly):
"As [name] drifted off to sleep, a warm feeling filled her heart. She whispered softly, 'Thank you, God, for today and for all the good things in it.' And the whole world felt still and safe."
"[Name] closed his eyes and smiled. He was grateful — grateful to God for brave friends, for warm blankets, and for every single star above. Soon he was fast asleep."
"Before she fell asleep, [name] pressed her paws together and said, 'Thank you, God, for making this day so wonderful.' The moon smiled down, and everything was peaceful and quiet."
"""

PROMPT_TEMPLATE = """Write a complete, rich children's bedtime story following the arc below.

**Original Request:** {request}

**Story Arc to Follow:**
{arc}

**Story Settings:**
- Genre: {genre}
- Tone: {tone} — the story should feel {tone_description}
- Target audience: children ages {age_range}
- Length: approximately {word_count} words (this is important — do not write a shorter story)
- Give the story an engaging, imaginative TITLE on the very first line

{age_profile}

{engagement_devices}

{god_salutation}

{scary_note}**General Rules:**
- Use short paragraphs (3-5 sentences each) for easy reading aloud
- No scary monsters, violence, or adult themes — keep it safe and warm
- If the user specified character names, keep them exactly as given
- Do NOT write any meta-commentary, headers, or explanations — write ONLY the story

{few_shot}

Write the full story now, starting with the title:"""

# -- Self-reflection prompt --------------------------------------------------
REFLECT_SYSTEM = """You are a careful children's book editor reviewing a story draft.
You check the story against a set of rules and fix violations precisely and silently."""

REFLECT_PROMPT = """Review the following children's bedtime story and check it satisfies ALL of these rules:

1. Vocabulary is appropriate for ages {age_range}
2. The story has a clear beginning, middle, and satisfying end
3. The tone is {tone} throughout
4. There is at least one dialogue exchange between characters
5. The story ends with a gentle God/gratitude salutation paragraph
6. No scary themes, violence, or adult content
7. It faithfully addresses the original request: "{request}"

Story to review:
{story}

CRITICAL INSTRUCTIONS:
- Return ONLY the story text. Begin your response with the story title.
- NEVER write phrases like "The story satisfies all the rules", "No changes needed", "Here is the corrected story:", or ANY commentary.
- If all rules are satisfied: return the story EXACTLY as written, word for word.
- If a rule is violated: fix ONLY the specific problem, then return the complete corrected story.
- Your response must start with the story title on the first line."""

# -- Constants ---------------------------------------------------------------
TONE_DESCRIPTIONS = {
    "gentle":       "soft, warm, and reassuring",
    "exciting":     "adventurous and full of wonder — but never scary",
    "funny":        "light-hearted, silly, and fun",
    "heartwarming": "touching and emotionally warm",
    "calm":         "slow-paced, dreamy, and soothing like a lullaby",
}

WORD_COUNTS = {
    "short":  "550-650",
    "medium": "800-950",
    "long":   "1100-1300",
}

AGE_RANGES = {
    "young":  "5-6",
    "middle": "7-8",
    "older":  "9-10",
}


def _build_prompt(user_request: str, arc: str, metadata: dict) -> str:
    """Build the full storyteller prompt. Shared by blocking and streaming variants."""
    tone     = metadata.get("tone", "gentle")
    age_lean = metadata.get("age_lean", "middle")
    length   = metadata.get("length_hint", "medium")

    # Issue 7: inject safety redirect when user mentioned scary elements
    if metadata.get("has_scary_elements"):
        scary_note = (
            "SAFETY REDIRECT: The user's request mentioned scary or frightening elements. "
            "You MUST redirect these into gentle, age-appropriate versions. Examples:\n"
            "  - A 'ghost' becomes a friendly ghost who just wants to play\n"
            "  - A 'monster' becomes a misunderstood creature who loves baking cookies\n"
            "  - 'Darkness' becomes a cosy, starlit night that feels safe and magical\n"
            "Never write content that would genuinely frighten a child. Keep it warm and comforting.\n\n"
        )
    else:
        scary_note = ""

    return PROMPT_TEMPLATE.format(
        request=user_request,
        arc=arc,
        genre=metadata.get("genre", "fantasy"),
        tone=tone,
        tone_description=TONE_DESCRIPTIONS.get(tone, "warm and gentle"),
        age_range=AGE_RANGES.get(age_lean, "7-8"),
        word_count=WORD_COUNTS.get(length, "800-950"),
        age_profile=AGE_PROFILES.get(age_lean, AGE_PROFILES["middle"]),
        engagement_devices=ENGAGEMENT_DEVICES,
        god_salutation=GOD_SALUTATION,
        scary_note=scary_note,
        few_shot=FEW_SHOT_EXAMPLE,
    )


def write_story(user_request: str, arc: str, metadata: dict) -> str:
    """
    Generate the full prose story (blocking, non-streaming).

    Returns:
        The full story as a string (title on first line, then paragraphs).
    """
    prompt = _build_prompt(user_request, arc, metadata)
    story = call_model(prompt, system=SYSTEM, temperature=0.85, max_tokens=2200)
    return story.strip()


def write_story_stream(
    user_request: str, arc: str, metadata: dict
) -> Generator[str, None, None]:
    """
    Streaming variant — yields story text chunks as they arrive from the API.
    Used by the Streamlit UI for word-by-word display via st.write_stream().
    """
    prompt = _build_prompt(user_request, arc, metadata)
    yield from stream_model(prompt, system=SYSTEM, temperature=0.85, max_tokens=2200)


def self_reflect_story(story: str, user_request: str, metadata: dict) -> str:
    """
    Self-reflection pass: the model re-reads its own draft and fixes any
    rule violations before the Judge ever sees it.

    Implements the 'self-reflection' agent design pattern.
    Includes a length-based fallback guard — if the reflected text is less than
    60% of the original, the model returned commentary instead of story text,
    and we silently fall back to the original draft.

    Args:
        story:        Raw story draft from write_story() or write_story_stream().
        user_request: The original user request.
        metadata:     Classification metadata dict.

    Returns:
        The corrected story, or the original draft if reflection failed.
    """
    age_lean = metadata.get("age_lean", "middle")
    tone     = metadata.get("tone", "gentle")

    prompt = REFLECT_PROMPT.format(
        age_range=AGE_RANGES.get(age_lean, "7-8"),
        tone=tone,
        request=user_request,
        story=story,
    )

    reflected = call_model(
        prompt, system=REFLECT_SYSTEM, temperature=0.2, max_tokens=2200
    ).strip()

    # Guard: if model returned commentary instead of story, fall back to original
    if not reflected or len(reflected) < len(story) * 0.6:
        return story

    # Guard: if model started with commentary phrases, strip them and check again
    commentary_starts = (
        "the story satisfies",
        "no changes needed",
        "here is the corrected",
        "here is the story",
        "the story is",
        "all rules are",
    )
    if reflected.lower().startswith(commentary_starts):
        return story

    return reflected


# Minimum word count thresholds per length_hint
MIN_WORDS = {"short": 450, "medium": 650, "long": 900}


EXPAND_SYSTEM = """You are a skilled children's book editor expanding a draft story.
Add rich detail, more scenes, more dialogue, and sensory moments to reach the target length."""


def expand_story_if_short(story: str, user_request: str, metadata: dict) -> str:
    """
    Post-generation length enforcement.

    gpt-3.5-turbo tends to end stories early even when given a long word-count
    target. This function detects under-length stories and asks the model to
    expand the middle section before the story reaches the Judge.

    Args:
        story:        The raw story text (title on first line).
        user_request: The original user request.
        metadata:     Classification metadata dict.

    Returns:
        The story, expanded if it was below the minimum threshold for its
        length category, or unchanged if it was already long enough.
    """
    length_hint = metadata.get("length_hint", "medium")
    min_words   = MIN_WORDS.get(length_hint, 650)
    word_count  = len(story.split())

    if word_count >= min_words:
        return story  # Already long enough

    age_lean   = metadata.get("age_lean", "middle")
    age_range  = AGE_RANGES.get(age_lean, "7-8")
    target     = int(min_words * 1.15)  # Aim a bit above minimum

    expand_prompt = f"""The following children's bedtime story is too short ({word_count} words).
Expand it to approximately {target} words by adding:
- More scenes and mini-adventures
- More dialogue between characters
- Richer sensory details (sounds, smells, textures)
- More inner thoughts and feelings
- Keep the same title, opening, and ending exactly as they are
- Vocabulary must stay appropriate for ages {age_range}
- Do NOT change characters' names
- Keep the God/gratitude salutation at the very end

Current story:
{story}

Rewrite the complete, expanded story now (title on first line):"""

    expanded = call_model(
        expand_prompt, system=EXPAND_SYSTEM, temperature=0.8, max_tokens=2500
    ).strip()

    # Only use expanded version if it's actually longer
    return expanded if expanded and len(expanded.split()) > word_count else story
