"""SQLite-backed long-term memory storage."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from exec_agent.config import get_settings


@dataclass(frozen=True)
class LongTermMemory:
    """A persisted long-term memory record."""

    id: int
    content: str
    tags: list[str]
    source: str
    created_at: str
    updated_at: str


def default_memory_path() -> Path:
    """Return the default SQLite database path for long-term memories."""

    return get_settings().expanded_data_dir / "long_term_memory.sqlite3"


class LongTermMemoryStore:
    """Manage long-term memories in a local SQLite database."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_memory_path()
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
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'manual',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_long_term_memories_content ON long_term_memories(content)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_long_term_memories_tags ON long_term_memories(tags)")

    def add(self, content: str, tags: Iterable[str] = (), source: str = "manual") -> LongTermMemory:
        """Add a memory and return the persisted record."""

        now = _now()
        tag_text = _serialize_tags(tags)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO long_term_memories (content, tags, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content, tag_text, source, now, now),
            )
            memory_id = int(cursor.lastrowid)
        return self.get(memory_id)  # type: ignore[return-value]

    def list(self) -> list[LongTermMemory]:
        """Return all memories ordered by newest first."""

        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM long_term_memories ORDER BY created_at DESC, id DESC").fetchall()
        return [_row_to_memory(row) for row in rows]

    def search(self, query: str, *, limit: int = 10) -> list[LongTermMemory]:
        """Search memories by content, tag, or source using tokenized LIKE matching."""

        terms = [term.strip(".,?!:;()[]{}\"\'").lower() for term in query.split()]
        terms = [term for term in terms if term]
        if not terms:
            return []

        clauses = []
        params: list[str | int] = []
        for term in terms:
            pattern = f"%{term}%"
            clauses.append("(lower(content) LIKE ? OR lower(tags) LIKE ? OR lower(source) LIKE ?)")
            params.extend([pattern, pattern, pattern])
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM long_term_memories
                WHERE {" OR ".join(clauses)}
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def get(self, memory_id: int) -> LongTermMemory | None:
        """Return a memory by id, if it exists."""

        with self._connect() as connection:
            row = connection.execute("SELECT * FROM long_term_memories WHERE id = ?", (memory_id,)).fetchone()
        return _row_to_memory(row) if row is not None else None

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by id and return whether a row was removed."""

        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM long_term_memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0


def format_memories_for_prompt(memories: Iterable[LongTermMemory]) -> str:
    """Render memories as compact prompt context."""

    lines = []
    for memory in memories:
        tags = f" tags={','.join(memory.tags)}" if memory.tags else ""
        lines.append(f"- [{memory.id}] {memory.content} (source={memory.source}{tags})")
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _serialize_tags(tags: Iterable[str]) -> str:
    return ",".join(tag.strip() for tag in tags if tag.strip())


def _row_to_memory(row: sqlite3.Row) -> LongTermMemory:
    return LongTermMemory(
        id=int(row["id"]),
        content=str(row["content"]),
        tags=[tag for tag in str(row["tags"]).split(",") if tag],
        source=str(row["source"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
