from typing import Protocol

from ai_book_assistant.models import (
    Book,
    Explanations,
    HistoryEntry,
    SearchCriteria,
    SearchRecord,
)


class HistoryStore(Protocol):
    def add(
        self,
        *,
        query: str,
        strategy: str,
        criteria: SearchCriteria,
        books: list[Book],
        explanations: Explanations,
    ) -> SearchRecord: ...

    def list_recent(self, limit: int = 20, offset: int = 0) -> list[HistoryEntry]: ...

    def get(self, search_id: int) -> SearchRecord | None: ...
