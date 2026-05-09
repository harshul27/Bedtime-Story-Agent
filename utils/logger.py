"""
utils/logger.py

Centralised session activity logger for the Bedtime Story Generator.
Writes structured, timestamped entries to agent.log in the project root.

Behaviour:
  - Each new Streamlit browser session clears agent.log and starts fresh.
    This ensures the log always reflects only the most recent session's
    activity, making it easy to review without digging through history.
  - Within a single browser session, all reruns append to the same file.
  - If multiple browser tabs are open, the latest tab's session_id wins
    (acceptable for a single-user local app).

Usage:
    from utils.logger import get_logger
    log = get_logger(session_id, fresh_start=True)   # new browser session
    log = get_logger(session_id)                      # existing session rerun

Log format:
    2026-05-08 16:20:01 | INFO  | [abc12345] Judge pass 1: score=8/12
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "agent.log"

# Module-level cache so we never create duplicate handlers
_loggers: dict[str, logging.Logger] = {}


def _clear_log() -> None:
    """Truncate agent.log and write a clean session boundary header."""
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        f.write(f"{'─' * 72}\n")
        f.write(f"  Bedtime Story Agent — Session Log\n")
        f.write(f"  Started: {now}\n")
        f.write(f"{'─' * 72}\n\n")


def get_logger(session_id: str, fresh_start: bool = False) -> logging.Logger:
    """
    Return a logger namespaced to the given session_id.

    Each log line includes timestamp, level, and a short session prefix
    so entries from concurrent sessions are distinguishable at a glance.

    Writes to:
      - agent.log (cleared on fresh_start, appended otherwise)
      - stdout     (INFO and above only)

    Args:
        session_id:  UUID string identifying the browser or CLI session.
        fresh_start: If True, clear agent.log before creating the handler.
                     Pass True only on the first call for a new browser session.

    Returns:
        A configured logging.Logger instance, cached per session_id.
    """
    short_id = session_id[:8]
    logger_name = f"story_agent.{short_id}"

    if logger_name in _loggers:
        return _loggers[logger_name]

    # New session — optionally clear the log file
    if fresh_start:
        _clear_log()

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent double-logging via root logger

    fmt = logging.Formatter(
        fmt=f"%(asctime)s | %(levelname)-5s | [{short_id}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — append mode (log was already cleared above if fresh_start)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _loggers[logger_name] = logger
    logger.info("Session started")
    return logger
