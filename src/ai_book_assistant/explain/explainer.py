import logging

from pydantic import BaseModel

from ai_book_assistant.explain.prompts import build_explanation_prompt
from ai_book_assistant.llm.base import LLMClient
from ai_book_assistant.llm.structured import StructuredGenerator
from ai_book_assistant.models import Book, SearchCriteria

logger = logging.getLogger(__name__)


class _Item(BaseModel):
    id: int
    explanation: str


class _Response(BaseModel):
    explanations: list[_Item]


def _validate(data: object, count: int) -> dict[int, str]:
    """Keep usable explanations; ignore unknown ids, blanks and duplicates."""
    found: dict[int, str] = {}
    for item in _Response.model_validate(data).explanations:
        text = " ".join(item.explanation.split())
        if 1 <= item.id <= count and text and item.id not in found:
            found[item.id] = text
    if not found:
        raise ValueError("the response contained no usable explanations")
    return found


class BookExplainer:
    """Explains, in one LLM call, why each book fits the user's request."""

    def __init__(self, llm_client: LLMClient, max_attempts: int = 2) -> None:
        self.generator = StructuredGenerator(llm_client, max_attempts)

    def explain(
        self,
        query: str,
        criteria: SearchCriteria,
        books: list[Book],
        *,
        relaxed: bool = False,
    ) -> dict[str, str]:
        """Return {book key: explanation}; empty if nothing usable came back."""
        if not books:
            return {}

        by_id = self.generator.generate(
            build_explanation_prompt(query, criteria, books, relaxed=relaxed),
            lambda data: _validate(data, len(books)),
            task="Explanation",
        )
        if by_id is None:
            logger.warning("Explanation failed; returning books without explanations")
            return {}
        return {books[i - 1].key: text for i, text in by_id.items()}
