"""
agents/classifier.py

Classifies the user's story request into structured metadata used to tailor
all downstream prompts. Uses structured JSON output from the LLM.
"""

import json
from utils.llm import call_model

SYSTEM = """You are a children's story metadata expert.
Your job is to analyze a story request and output structured JSON tags.
Always respond with valid JSON and nothing else."""

PROMPT_TEMPLATE = """Analyze the following bedtime story request and return a JSON object with these fields:

- genre: one of ["fantasy", "adventure", "friendship", "animals", "family", "mystery", "humor", "bedtime_lullaby"]
- tone: one of ["gentle", "exciting", "funny", "heartwarming", "calm"]
- age_lean: one of ["young" (ages 5-6), "middle" (ages 7-8), "older" (ages 9-10)]
- themes: list of up to 3 themes present (e.g. ["courage", "kindness", "teamwork"])
- characters: list of characters mentioned by the user (leave empty [] if none specified)
- has_scary_elements: true if the request mentions scary/frightening things (ghosts, monsters, darkness, fear), false otherwise
- length_hint: one of ["short" (~600 words), "medium" (~850 words), "long" (~1200 words)]
  Choose "short" for simple quick stories, "medium" for most requests, "long" if the user asks for a detailed or long story

Story request: "{request}"

Respond with ONLY valid JSON, no explanation, no markdown code blocks."""


def classify(user_request: str) -> dict:
    """
    Classify a story request into structured metadata.

    Args:
        user_request: The raw user story request string.

    Returns:
        A dict with keys: genre, tone, age_lean, themes, characters,
        has_scary_elements, length_hint.
    """
    prompt = PROMPT_TEMPLATE.format(request=user_request)
    raw = call_model(prompt, system=SYSTEM, temperature=0.1, max_tokens=400)

    # Strip markdown code fences if the model adds them despite instructions
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback defaults if JSON parsing fails
        metadata = {
            "genre": "fantasy",
            "tone": "gentle",
            "age_lean": "middle",
            "themes": ["friendship"],
            "characters": [],
            "has_scary_elements": False,
            "length_hint": "medium",
        }

    return metadata
