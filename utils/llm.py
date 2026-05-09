"""
utils/llm.py

Shared LLM client using the modern openai>=1.0 API.
Keeps the same function signature as the original call_model() in main.py
so the skeleton contract is preserved, while adding system-message support.

Reliability features:
  - Exponential backoff retry for 429 (rate limit), 5xx server errors,
    timeout, and connection errors (Issues 2, 12)
  - Up to 3 attempts with jitter to spread load
  - 4xx client errors (bad request, invalid key) are NOT retried

Known limitation (Issue 12 — documented):
  The OpenAI client is cached as a module-level singleton on first call.
  If the OPENAI_API_KEY environment variable is changed at runtime, the
  cached client will continue using the original key until the Python
  process is restarted. This is acceptable for a single-user app but
  should be addressed with per-request client construction in production.
"""

import os
import time
import random
from typing import Generator

from openai import (
    OpenAI,
    RateLimitError,
    APIStatusError,
    APITimeoutError,
    APIConnectionError,
)
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None

_MAX_RETRIES = 3
_BASE_WAIT_SECONDS = 1.5   # First retry waits ~1.5s, second ~3s, third ~6s


def _get_client() -> OpenAI:
    """
    Return the shared OpenAI client, creating it on first call.

    Note: client is cached as a module-level singleton. If the API key
    changes after first call, restart the process to pick up the new key.
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Set it in a .env file or as an environment variable."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def _retry(fn, *args, **kwargs):
    """
    Execute fn(*args, **kwargs) with exponential backoff retry.

    Retries on transient errors:
      - RateLimitError (429)  — always retry with backoff + jitter
      - APIStatusError 5xx    — server-side failure, retry
      - APITimeoutError       — request timed out, retry
      - APIConnectionError    — network issue, retry

    Does NOT retry on:
      - APIStatusError 4xx (other than 429) — client error, no retry
      - Any other exception — propagate immediately

    Raises the original exception if all retries are exhausted.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_WAIT_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code < 500:
                raise   # 4xx client errors: don't retry
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BASE_WAIT_SECONDS * (2 ** attempt))
        except (APITimeoutError, APIConnectionError):
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BASE_WAIT_SECONDS * (2 ** attempt))


def call_model(
    prompt: str,
    system: str = "You are a helpful assistant.",
    max_tokens: int = 3000,
    temperature: float = 0.7,
) -> str:
    """
    Call gpt-3.5-turbo and return the complete text response (non-streaming).
    Retries automatically on transient API errors.

    Args:
        prompt:      The user message.
        system:      The system message (role / persona).
        max_tokens:  Maximum tokens in the response.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).

    Returns:
        The model's text response as a string (empty string if no content).
    """
    client = _get_client()

    def _call():
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content or ""

    return _retry(_call)


def stream_model(
    prompt: str,
    system: str = "You are a helpful assistant.",
    max_tokens: int = 1200,
    temperature: float = 0.85,
) -> Generator[str, None, None]:
    """
    Streaming variant — yields text chunks as they arrive from the API.
    Used by write_story_stream() to render the story word-by-word in the UI.

    Note: streaming responses cannot be transparently retried mid-stream.
    If a streaming call fails, the exception propagates to the caller
    (caught by app.py's try/except block).

    Args:
        prompt:      The user message.
        system:      The system message.
        max_tokens:  Maximum tokens in the response.
        temperature: Sampling temperature.

    Yields:
        String chunks of the model's response as they stream in.
    """
    client = _get_client()
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content
