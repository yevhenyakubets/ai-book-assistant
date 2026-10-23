from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from ai_book_assistant.assistant import CLARIFICATION_MESSAGE
from ai_book_assistant.models import SearchRecord

MAX_QUERY_LENGTH = 500

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


class SearchResponse(SearchRecord):
    """A saved search, plus a message when nothing could be searched."""

    message: str | None = None


class ErrorResponse(BaseModel):
    detail: str


def to_response(record: SearchRecord) -> SearchResponse:
    needs_clarification = record.strategy == "needs_clarification"
    message = CLARIFICATION_MESSAGE if needs_clarification else None
    return SearchResponse(**dict(record), message=message)
