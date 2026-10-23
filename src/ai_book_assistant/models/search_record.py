from datetime import datetime

from pydantic import BaseModel, Field

from ai_book_assistant.models.book import Book
from ai_book_assistant.models.criteria import SearchCriteria
from ai_book_assistant.models.explanations import Explanations


class SearchRecord(BaseModel):
    """A saved search: exactly what the user was shown at the time."""

    id: int
    created_at: datetime
    query: str
    strategy: str
    criteria: SearchCriteria
    books: list[Book] = Field(default_factory=list)
    explanations: Explanations = Field(default_factory=Explanations)


class HistoryEntry(BaseModel):
    """One line in the history list, without the saved snapshot."""

    id: int
    created_at: datetime
    query: str
    strategy: str
