import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from ai_book_assistant.models import (
    Book,
    Explanations,
    HistoryEntry,
    SearchCriteria,
    SearchRecord,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    query        TEXT NOT NULL,
    strategy     TEXT NOT NULL,
    criteria     TEXT NOT NULL,
    books        TEXT NOT NULL,
    explanations TEXT NOT NULL
)
"""

_BOOKS = TypeAdapter(list[Book])


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SqliteSearchHistory:
    """Keeps past searches in SQLite, exactly as the user saw them.

    A snapshot is stored instead of just book ids, so a past search can be
    shown again instantly, without new Open Library or LLM calls.
    """

    def __init__(
        self,
        path: str | Path,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A short-lived connection: committed on success, always closed."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def add(
        self,
        *,
        query: str,
        strategy: str,
        criteria: SearchCriteria,
        books: list[Book],
        explanations: Explanations,
    ) -> SearchRecord:
        created_at = self.clock()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO searches "
                "(created_at, query, strategy, criteria, books, explanations) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    created_at.isoformat(),
                    query,
                    strategy,
                    criteria.model_dump_json(),
                    _BOOKS.dump_json(books).decode(),
                    explanations.model_dump_json(),
                ),
            )
            search_id = cursor.lastrowid
        return SearchRecord(
            id=search_id,
            created_at=created_at,
            query=query,
            strategy=strategy,
            criteria=criteria,
            books=books,
            explanations=explanations,
        )

    def list_recent(self, limit: int = 20, offset: int = 0) -> list[HistoryEntry]:
        """Newest first, without the saved snapshots."""
        if limit < 1 or offset < 0:
            raise ValueError("limit must be at least 1 and offset cannot be negative")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, query, strategy FROM searches "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [HistoryEntry.model_validate(dict(row)) for row in rows]

    def get(self, search_id: int) -> SearchRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM searches WHERE id = ?", (search_id,)
            ).fetchone()
        if row is None:
            return None
        return SearchRecord(
            id=row["id"],
            created_at=row["created_at"],
            query=row["query"],
            strategy=row["strategy"],
            criteria=SearchCriteria.model_validate_json(row["criteria"]),
            books=_BOOKS.validate_json(row["books"]),
            explanations=Explanations.model_validate_json(row["explanations"]),
        )
