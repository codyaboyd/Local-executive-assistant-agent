"""SQLite-backed persistent chat session storage."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from exec_agent.config import get_settings
from exec_agent.chat import ChatMessage, ChatSession


@dataclass(frozen=True)
class PersistedChatSession:
    """A chat session record loaded from SQLite."""

    name: str
    messages: list[ChatMessage]
    summary: str
    created_at: str
    updated_at: str


def default_sessions_path() -> Path:
    """Return the default SQLite database path for chat sessions."""

    return get_settings().expanded_data_dir / "chat_sessions.sqlite3"


class ChatSessionStore:
    """Manage named persistent chat sessions in a local SQLite database."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_sessions_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    name TEXT PRIMARY KEY,
                    messages TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at)")

    def get(self, name: str) -> PersistedChatSession | None:
        """Return a named session, if it exists."""

        with self._connect() as connection:
            row = connection.execute("SELECT * FROM chat_sessions WHERE name = ?", (name,)).fetchone()
        return _row_to_session(row) if row is not None else None

    def load_chat_session(self, name: str) -> tuple[ChatSession, str]:
        """Load a ChatSession and summary, creating an empty in-memory session for new names."""

        persisted = self.get(name)
        if persisted is None:
            return ChatSession(), ""
        return ChatSession(messages=list(persisted.messages)), persisted.summary

    def save_chat_session(self, name: str, session: ChatSession, summary: str) -> PersistedChatSession:
        """Insert or update a named chat session."""

        now = _now()
        messages_json = _serialize_messages(session.messages)
        with self._connect() as connection:
            existing = connection.execute("SELECT created_at FROM chat_sessions WHERE name = ?", (name,)).fetchone()
            created_at = str(existing["created_at"]) if existing is not None else now
            connection.execute(
                """
                INSERT INTO chat_sessions (name, messages, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    messages = excluded.messages,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (name, messages_json, summary, created_at, now),
            )
        return self.get(name)  # type: ignore[return-value]

    def list(self) -> list[PersistedChatSession]:
        """Return all sessions ordered by most recently updated first."""

        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM chat_sessions ORDER BY updated_at DESC, name ASC").fetchall()
        return [_row_to_session(row) for row in rows]

    def delete(self, name: str) -> bool:
        """Delete a named session and return whether it existed."""

        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM chat_sessions WHERE name = ?", (name,))
            return cursor.rowcount > 0


def summarize_messages(messages: Iterable[ChatMessage], *, max_chars: int = 1200) -> str:
    """Build a compact continuity summary from a chat transcript."""

    lines = [f"{message.role.title()}: {message.content}" for message in messages]
    summary = "\n".join(lines).strip()
    if len(summary) <= max_chars:
        return summary
    return "…" + summary[-max_chars:]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _serialize_messages(messages: Iterable[ChatMessage]) -> str:
    return json.dumps([{"role": message.role, "content": message.content} for message in messages])


def _deserialize_messages(raw: str) -> list[ChatMessage]:
    payload = json.loads(raw or "[]")
    return [ChatMessage(role=str(item["role"]), content=str(item["content"])) for item in payload]


def _row_to_session(row: sqlite3.Row) -> PersistedChatSession:
    return PersistedChatSession(
        name=str(row["name"]),
        messages=_deserialize_messages(str(row["messages"])),
        summary=str(row["summary"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
