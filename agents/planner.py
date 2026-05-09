"""
agents/planner.py

Generates a structured 7-beat story arc before any prose is written.
The 7-beat arc gives the storyteller enough structure to produce longer,
richer stories with proper narrative tension and a warm closing beat.
Uses chain-of-thought prompting to reason through the arc step-by-step.
"""

from utils.llm import call_model

SYSTEM = """You are an expert children's book author and story architect.
You specialize in creating age-appropriate, engaging story outlines for children aged 5-10.
Your arcs always have clear tension, a satisfying resolution, and a warm sleepy ending."""

PROMPT_TEMPLATE = """Create a 7-beat story arc for a children's bedtime story based on the following:

**User's Request:** {request}

**Story Metadata:**
- Genre: {genre}
- Tone: {tone}
- Age Group: {age_lean} (ages {age_range})
- Themes: {themes}
- Characters: {characters}

Think through this step-by-step before writing the arc:

Step 1: Who is the main character? What is their everyday world like and what do they wish for?
Step 2: What unexpected event or invitation kicks the adventure off?
Step 3: What is the first challenge or obstacle they face along the way?
Step 4: What is the most exciting or tense moment — where things look uncertain?
Step 5: How does the character find the courage, help, or idea to turn things around?
Step 6: How is everything resolved? What has the character learned or gained?
Step 7: What is the final warm, sleepy image — and the gentle gratitude or lesson the child carries to sleep?

Now write the 7-beat arc in this exact format (2-3 sentences each):

BEAT 1 - SETUP: [Introduce the character, their cosy world, and what they wish for]
BEAT 2 - INCITING EVENT: [Something unexpected happens that starts the adventure]
BEAT 3 - FIRST CHALLENGE: [An obstacle or difficulty the character must face]
BEAT 4 - DARK MOMENT: [The most tense moment — things look like they won't work out]
BEAT 5 - TURNING POINT: [The character finds courage, help, or a clever idea]
BEAT 6 - RESOLUTION: [Everything resolves happily and the character has grown]
BEAT 7 - WARM CLOSE: [A peaceful, sleepy final image and the gentle lesson or gratitude]"""

AGE_RANGES = {
    "young": "5-6",
    "middle": "7-8",
    "older": "9-10",
}


def plan_story(user_request: str, metadata: dict) -> str:
    """
    Generate a 7-beat story arc from the user request and classification metadata.

    Args:
        user_request: The raw user story request string.
        metadata:     Dict from classifier.classify().

    Returns:
        A string containing the 7-beat story arc.
    """
    characters = metadata.get("characters", [])
    character_str = ", ".join(characters) if characters else "to be invented by the author"

    themes = metadata.get("themes", ["friendship"])
    theme_str = ", ".join(themes)

    age_lean = metadata.get("age_lean", "middle")
    age_range = AGE_RANGES.get(age_lean, "7-8")

    prompt = PROMPT_TEMPLATE.format(
        request=user_request,
        genre=metadata.get("genre", "fantasy"),
        tone=metadata.get("tone", "gentle"),
        age_lean=age_lean,
        age_range=age_range,
        themes=theme_str,
        characters=character_str,
    )

    arc = call_model(prompt, system=SYSTEM, temperature=0.7, max_tokens=600)
    return arc.strip()
