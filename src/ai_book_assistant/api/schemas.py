from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from ai_book_assistant.assistant import CLARIFICATION_MESSAGE
from ai_book_assistant.models import Book, SearchCriteria, SearchRecord

MAX_QUERY_LENGTH = 500
OPEN_LIBRARY_URL = "https://openlibrary.org"

Query = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH),
]


class SearchRequest(BaseModel):
    query: Query

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"query": "Cozy mystery novels set in a small village"}]
        }
    )


class BookResult(BaseModel):
    """One book as the user sees it, with its explanation alongside."""

    title: str
    authors: list[str]
    first_publish_year: int | None = None
    description: str | None = None
    explanation: str | None = None
    url: str


class SearchResponse(BaseModel):
    """A search as shown to the user, reshaped from the stored snapshot."""

    id: int
    created_at: datetime
    query: str
    strategy: str
    message: str | None = None
    summary: str | None = None
    results: list[BookResult]
    criteria: SearchCriteria


class ErrorResponse(BaseModel):
    detail: str


def _book_result(book: Book, explanations: dict[str, str]) -> BookResult:
    return BookResult(
        title=book.title,
        authors=book.authors,
        first_publish_year=book.first_publish_year,
        description=book.description,
        explanation=explanations.get(book.key),
        url=f"{OPEN_LIBRARY_URL}{book.key}",
    )


def to_response(record: SearchRecord) -> SearchResponse:
    explanations = record.explanations.by_book
    needs_clarification = record.strategy == "needs_clarification"
    return SearchResponse(
        id=record.id,
        created_at=record.created_at,
        query=record.query,
        strategy=record.strategy,
        message=CLARIFICATION_MESSAGE if needs_clarification else None,
        summary=record.explanations.summary,
        results=[_book_result(book, explanations) for book in record.books],
        criteria=record.criteria,
    )
