"""
agents/reviser.py

Revises a story based on targeted feedback from the LLM judge or user.
Uses targeted revision prompting — changes only what was flagged while
preserving everything that was working (avoids "revision drift").

For expansion requests (user wants a longer story), the length constraint is
lifted and the reviser is given an explicit word count target to reach.
"""

from utils.llm import call_model

SYSTEM = """You are a skilled children's book editor.
You make targeted, high-quality improvements to children's stories based on specific feedback.
You preserve what works and fix only what needs fixing — but you are not afraid to expand
the story generously when asked to make it longer or more detailed."""

PROMPT_TEMPLATE = """You need to improve a children's bedtime story based on specific editor feedback.

**Original User Request:** {request}

**Current Story:**
{story}

**Editor's Critique:**
- Overall Score: {score}/10
- Strengths to PRESERVE: {strengths}
- Weaknesses to FIX: {weaknesses}
- Specific Revision Instructions: {revision_instructions}

**Your Task:**
Rewrite the story making the improvements specified above.
- Keep everything that was working (the strengths listed above)
- Fix the specific weaknesses identified
- Maintain the same title (or improve it slightly if the title is weak)
- Do not change characters' names
- The result must still be appropriate for children ages 5-10
- End the story with a gentle God/gratitude salutation if not already present
{length_instruction}

Write the improved story now (title on the first line, then the story):"""

EXPANSION_INSTRUCTION = """- TARGET WORD COUNT: approximately {target_words} words
- This story needs to be substantially longer — add more scenes, more dialogue,
  more sensory detail, more adventures. Do not end early."""

STANDARD_INSTRUCTION = "- Length: maintain or improve the richness of the current story"


def revise_story(
    user_request: str,
    story: str,
    critique: dict,
) -> str:
    """
    Revise a story based on the judge's structured critique.

    Detects expansion requests (keywords: longer, more detail, expand, etc.)
    and lifts the length constraint when found, giving the reviser explicit
    permission to add substantial content.

    Args:
        user_request: The original user request string.
        story:        The current story text.
        critique:     The dict returned by judge.judge_story().

    Returns:
        The revised story as a string.
    """
    strengths            = critique.get("strengths", [])
    weaknesses           = critique.get("weaknesses", [])
    revision_instructions = critique.get("revision_instructions", "Improve the story overall.")
    score                = critique.get("total_score", 0)

    # Detect expansion requests in revision instructions
    expansion_keywords = ("longer", "more detail", "expand", "elaborate", "more scenes",
                          "more dialogue", "add more", "make it longer", "lengthen")
    is_expansion = any(kw in revision_instructions.lower() for kw in expansion_keywords)

    current_words = len(story.split())
    if is_expansion:
        target_words = max(current_words * 2, 800)
        length_instruction = EXPANSION_INSTRUCTION.format(target_words=target_words)
        max_tok = 2500
    else:
        length_instruction = STANDARD_INSTRUCTION
        max_tok = 2200

    prompt = PROMPT_TEMPLATE.format(
        request=user_request,
        story=story,
        score=score,
        strengths="; ".join(strengths) if strengths else "Good effort overall",
        weaknesses="; ".join(weaknesses) if weaknesses else "Minor improvements needed",
        revision_instructions=revision_instructions,
        length_instruction=length_instruction,
    )

    revised = call_model(prompt, system=SYSTEM, temperature=0.75, max_tokens=max_tok)
    return revised.strip()
