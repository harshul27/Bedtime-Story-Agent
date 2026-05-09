"""
utils/tts.py

OpenAI Text-to-Speech wrapper.
Converts story text to MP3 audio bytes using the OpenAI TTS API (tts-1 model).
Audio is suitable for playback via Streamlit's st.audio() component.

OpenAI TTS limit: 4096 characters per request.
Longer stories (target: 550-1300 words ≈ 3300-7800 chars) exceed this limit.

Solution: split at sentence boundaries into ≤4000-char chunks, make one API
call per chunk, then concatenate the resulting MP3 bytes. MP3 files support
byte-level concatenation and play sequentially in any audio player.

Reliability features (Issue 5):
  - Hard word-boundary split as final fallback if sentence + comma splitting
    still produces a chunk > max_chars (e.g. very long run-on sentences
    with no punctuation at all).

Available voices and their character:
    Female:
        nova    — warm, friendly, engaging (best for bedtime)
        shimmer — soft, gentle, soothing
    Male:
        fable   — expressive storytelling cadence
        echo    — clear, friendly, easy for kids to follow
"""

import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Hard limit from OpenAI TTS API; we use 4000 to leave a safety margin
_TTS_CHAR_LIMIT = 4000

# Voice catalogue exposed to the UI
VOICES = {
    "Female": {
        "Nova (Warm & Friendly)": "nova",
        "Shimmer (Soft & Soothing)": "shimmer",
    },
    "Male": {
        "Fable (Storytelling)": "fable",
        "Echo (Clear & Friendly)": "echo",
    },
}


def _hard_split(text: str, max_chars: int) -> list[str]:
    """
    Last-resort fallback: split at word boundaries closest to max_chars.
    Used when a piece of text has no sentence-ending punctuation and no commas.
    Guarantees every chunk is within max_chars.

    Args:
        text:      Text to split (may be arbitrarily long).
        max_chars: Hard maximum characters per chunk.

    Returns:
        List of chunks, each <= max_chars.
    """
    chunks = []
    while len(text) > max_chars:
        # Find the last space at or before max_chars
        split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            # No space found — force a hard character split (pathological case)
            split_at = max_chars
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return [c for c in chunks if c.strip()]


def _split_into_chunks(text: str, max_chars: int = _TTS_CHAR_LIMIT) -> list[str]:
    """
    Split text into chunks that are each <= max_chars, in order of preference:
      1. Split at sentence boundaries (. ! ?) — best for natural speech
      2. Split at comma boundaries — for long sentences without sentence-end punctuation
      3. Split at word boundaries — hard fallback for any remaining oversized chunks

    Args:
        text:      The full narration text (title + story body).
        max_chars: Maximum characters per chunk.

    Returns:
        List of non-empty text chunks, each within the character limit.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence

        if len(candidate) <= max_chars:
            current = candidate
        else:
            # Flush current accumulation
            if current:
                chunks.append(current.strip())
                current = ""

            if len(sentence) > max_chars:
                # Try comma-split first
                parts = sentence.split(", ")
                sub = ""
                for part in parts:
                    candidate_sub = (sub + ", " + part).strip(", ") if sub else part
                    if len(candidate_sub) <= max_chars:
                        sub = candidate_sub
                    else:
                        if sub:
                            # sub still might be > max_chars if a single part is long
                            if len(sub) > max_chars:
                                chunks.extend(_hard_split(sub, max_chars))
                            else:
                                chunks.append(sub.strip())
                        sub = part if len(part) <= max_chars else ""
                        if len(part) > max_chars:
                            chunks.extend(_hard_split(part, max_chars))
                current = sub
            else:
                current = sentence

    if current:
        chunks.append(current.strip())

    # Final safety pass: hard-split any chunk that's still over the limit
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            final_chunks.extend(_hard_split(chunk, max_chars))
        else:
            final_chunks.append(chunk)

    return [c for c in final_chunks if c.strip()]


def generate_audio(text: str, voice: str = "nova") -> bytes:
    """
    Convert story text to MP3 audio bytes using OpenAI TTS.

    Automatically handles stories longer than 4096 characters by splitting
    them into sentence-boundary-respecting chunks and concatenating the
    resulting MP3 bytes into a single continuous audio stream.

    Args:
        text:  The narration text (title + story body). May be any length.
        voice: One of the OpenAI TTS voice IDs (nova, shimmer, fable, echo).

    Returns:
        Raw MP3 audio as bytes, ready for st.audio(audio_bytes, format="audio/mp3").
        If the text is chunked, the bytes are the concatenated MP3 segments.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
        openai.APIError: On API-level failures.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Set it in a .env file or as an environment variable."
        )

    client = OpenAI(api_key=api_key)

    def _call_tts(chunk: str) -> bytes:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=chunk,
            response_format="mp3",
        )
        return response.content

    # Fast path: text fits in a single API call
    if len(text) <= _TTS_CHAR_LIMIT:
        return _call_tts(text)

    # Long text: split and concatenate MP3 bytes
    chunks = _split_into_chunks(text)
    return b"".join(_call_tts(chunk) for chunk in chunks)
