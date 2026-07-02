"""
PostgreSQL-backed conversation session memory.

Why not ConversationBufferMemory (what the original project used)?
  - Not persisted: reload the app, history is gone
  - Unbounded: long conversations overflow the LLM context window silently
  - No session isolation: all users share the same in-process object

This implementation:
  - Persists to PostgreSQL (survives restarts, scales horizontally)
  - Caps history at last N turns (prevents context overflow)
  - Session-isolated: each UUID is a separate conversation
  - Formats history as a clean string for prompt injection

Schema (created by scripts/init_db.sql):
  chat_sessions(id, session_id UUID, role, content, created_at)
"""

import psycopg2
import psycopg2.extras

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

MAX_HISTORY_TURNS = 6   # last 6 exchanges = 12 rows (6 user + 6 assistant)


class SessionMemory:
    """Read/write conversation history for a given session_id."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._conn_str = settings.database_url

    def _get_conn(self) -> psycopg2.extensions.connection:
        """Get a fresh connection. Short-lived — we don't pool at this layer."""
        return psycopg2.connect(self._conn_str, connect_timeout=3)

    def get_history(self) -> list[dict]:
        """
        Return the last MAX_HISTORY_TURNS turns as a list of {role, content} dicts.
        Ordered oldest → newest (correct order for prompt assembly).
        """
        try:
            with self._get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT role, content FROM (
                            SELECT role, content, created_at
                            FROM chat_sessions
                            WHERE session_id = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        ) sub
                        ORDER BY created_at ASC
                        """,
                        (self._session_id, MAX_HISTORY_TURNS * 2),
                    )
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("session_read_failed", session=self._session_id, error=str(e))
            return []

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Persist one exchange (user + assistant) to the database."""
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO chat_sessions (session_id, role, content)
                        VALUES (%s, %s, %s)
                        """,
                        [
                            (self._session_id, "user", user_message),
                            (self._session_id, "assistant", assistant_message),
                        ],
                    )
                conn.commit()
        except Exception as e:
            # Memory failure is non-fatal — conversation continues without history
            logger.warning("session_write_failed", session=self._session_id, error=str(e))

    def format_for_prompt(self) -> str:
        """
        Format history as a string block for injection into the LLM prompt.
        Returns empty string if no history exists.
        """
        history = self.get_history()
        if not history:
            return ""

        lines = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")

        return "\n".join(lines)
