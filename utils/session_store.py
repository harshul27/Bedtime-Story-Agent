"""
utils/session_store.py

SQLite-backed story library. Provides durable, persistent storage for every
generated story, surviving page refreshes and server restarts.

Reliability features (Issue 9):
  - WAL (Write-Ahead Logging) mode enabled: allows concurrent reads during
    writes, reducing lock contention when multiple users generate stories
    simultaneously. Critical for Streamlit Cloud where multiple sessions
    share the same SQLite file.
  - Connection timeout set to 10s to avoid indefinite blocking.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "story_library.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    # WAL mode: concurrent reads during writes, much better multi-user behaviour
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the stories table if it does not already exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                request     TEXT NOT NULL,
                title       TEXT NOT NULL,
                story_body  TEXT NOT NULL,
                final_score INTEGER NOT NULL DEFAULT 0,
                genre       TEXT,
                tone        TEXT,
                age_lean    TEXT,
                metadata    TEXT,
                arc         TEXT,
                iterations  INTEGER DEFAULT 0,
                timestamp   TEXT NOT NULL
            )
        """)
        conn.commit()


def _extract_fields(result: dict) -> tuple:
    """Extract DB field values from a result dict."""
    meta = result.get("metadata", {})
    history = result.get("judge_history", [])
    final_score = history[-1].get("total_score", 0) if history else 0
    return (
        result.get("request", ""),
        result.get("title", "Untitled"),
        result.get("story", ""),
        final_score,
        meta.get("genre", ""),
        meta.get("tone", ""),
        meta.get("age_lean", ""),
        json.dumps(meta),
        result.get("arc", ""),
        result.get("iterations", 0),
    )


def save_story(session_id: str, result: dict) -> int:
    """
    INSERT a new story record. Returns the new row id.
    Call this only when a brand-new story concept is created.
    For revisions, use update_story() instead.
    """
    fields = _extract_fields(result)
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO stories
                (session_id, request, title, story_body, final_score,
                 genre, tone, age_lean, metadata, arc, iterations, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, *fields, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cursor.lastrowid


def update_story(story_id: int, result: dict) -> None:
    """
    UPDATE an existing story record in-place.
    Used for revisions so only one record exists per story concept
    in the Story Library sidebar.

    Args:
        story_id: The primary key returned by save_story().
        result:   The updated result dict from revise_pipeline().
    """
    fields = _extract_fields(result)
    request, title, story_body, final_score, genre, tone, age_lean, metadata, arc, iterations = fields

    with _connect() as conn:
        conn.execute(
            """
            UPDATE stories SET
                title       = ?,
                story_body  = ?,
                final_score = ?,
                genre       = ?,
                tone        = ?,
                age_lean    = ?,
                metadata    = ?,
                arc         = ?,
                iterations  = ?,
                timestamp   = ?
            WHERE id = ?
            """,
            (
                title, story_body, final_score, genre, tone,
                age_lean, metadata, arc, iterations,
                datetime.now(timezone.utc).isoformat(),
                story_id,
            ),
        )
        conn.commit()


def get_all_stories(limit: int = 100) -> list:
    """Return all stories, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM stories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_story_by_id(story_id: int) -> dict | None:
    """Return a single story by primary key, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM stories WHERE id = ?", (story_id,)
        ).fetchone()
    return dict(row) if row else None
