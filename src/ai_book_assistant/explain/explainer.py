import logging

from pydantic import BaseModel

from ai_book_assistant.explain.prompts import build_explanation_prompt
from ai_book_assistant.llm.base import LLMClient
from ai_book_assistant.llm.structured import StructuredGenerator
from ai_book_assistant.models import Book, Explanations, SearchCriteria
from ai_book_assistant.text import collapse_whitespace

logger = logging.getLogger(__name__)


class _Item(BaseModel):
    id: int
    explanation: str


class _Response(BaseModel):
    summary: str | None = None
    explanations: list[_Item]


def _clean(text: str) -> str:
    """Collapse whitespace and replace non-breaking hyphens with plain ones."""
    return collapse_whitespace(text.replace("\u2011", "-"))


def _validate(data: object, count: int) -> tuple[str | None, dict[int, str]]:
    """Keep usable explanations; ignore unknown ids, blanks and duplicates."""
    response = _Response.model_validate(data)
    found: dict[int, str] = {}
    for item in response.explanations:
        text = _clean(item.explanation)
        if 1 <= item.id <= count and text and item.id not in found:
            found[item.id] = text
    if not found:
        raise ValueError("the response contained no usable explanations")
    return _clean(response.summary or "") or None, found


class BookExplainer:
    """Explains, in one LLM call, why the books fit the user's request."""

    def __init__(self, llm_client: LLMClient, max_attempts: int = 2) -> None:
        self.generator = StructuredGenerator(llm_client, max_attempts)

    def explain(
        self,
        query: str,
        criteria: SearchCriteria,
        books: list[Book],
        *,
        relaxed: bool = False,
        length_filter_applied: bool = False,
    ) -> Explanations:
        """Return a summary and per-book texts; empty if nothing usable came back."""
        if not books:
            return Explanations()

        result = self.generator.generate(
            build_explanation_prompt(
                query,
                criteria,
                books,
                relaxed=relaxed,
                length_filter_applied=length_filter_applied,
            ),
            lambda data: _validate(data, len(books)),
            task="Explanation",
        )
        if result is None:
            logger.warning("Explanation failed; returning books without explanations")
            return Explanations()

        summary, by_id = result
        by_book = {books[i - 1].key: text for i, text in by_id.items()}
        return Explanations(summary=summary, by_book=by_book)
