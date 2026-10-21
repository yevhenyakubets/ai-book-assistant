from dataclasses import dataclass
from typing import Protocol

from ai_book_assistant.llm.extractor import CriteriaExtractor
from ai_book_assistant.models import Book, SearchCriteria
from ai_book_assistant.search.fallback import (
    BookSearcher,
    SearchOutcome,
    search_with_fallback,
)

CLARIFICATION_MESSAGE = (
    "I couldn't tell what kind of book you're looking for. Try mentioning a "
    "genre, a topic or an author, for example 'a short mystery set in a "
    "village' or 'books by Terry Pratchett'."
)


class BookCatalog(BookSearcher, Protocol):
    """Everything the assistant needs from its source of books."""

    def attach_descriptions(self, books: list[Book]) -> list[Book]: ...


@dataclass
class AssistantResult:
    query: str
    criteria: SearchCriteria
    outcome: SearchOutcome

    @property
    def books(self) -> list[Book]:
        return self.outcome.books

    @property
    def needs_clarification(self) -> bool:
        return self.outcome.needs_clarification

    @property
    def strategy(self) -> str:
        return self.outcome.rung

    @property
    def relaxed(self) -> bool:
        return self.outcome.relaxed

    @property
    def message(self) -> str | None:
        return CLARIFICATION_MESSAGE if self.needs_clarification else None


class BookAssistant:
    def __init__(
        self,
        extractor: CriteriaExtractor,
        catalog: BookCatalog,
        *,
        limit: int = 5,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        self.extractor = extractor
        self.catalog = catalog
        self.limit = limit

    def search(self, query: str) -> AssistantResult:
        criteria = self.extractor.extract(query)
        outcome = search_with_fallback(criteria, query, self.catalog, limit=self.limit)
        if outcome.books:
            self.catalog.attach_descriptions(outcome.books)
        return AssistantResult(query, criteria, outcome)
