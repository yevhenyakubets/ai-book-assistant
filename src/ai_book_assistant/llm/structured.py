import json
import logging
from collections.abc import Callable
from typing import TypeVar

from ai_book_assistant.llm.base import LLMClient

logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_json(response_text: object) -> object:
    """Decode a model response, raising ValueError if it isn't usable JSON."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("the response was empty")
    return json.loads(response_text)


class StructuredGenerator:
    """Ask the model for JSON, validate it, and retry with the reason if rejected.

    Provider errors (LLMError) are not caught here: only unusable output is
    retried. Callers decide what to do when every attempt is rejected.
    """

    def __init__(self, llm_client: LLMClient, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.llm_client = llm_client
        self.max_attempts = max_attempts

    def generate(
        self, prompt: str, validate: Callable[[object], T], *, task: str
    ) -> T | None:
        """Return the validated result, or None if every attempt was rejected."""
        current = prompt
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.llm_client.generate(current)
                logger.debug("%s raw response: %r", task, raw)
                return validate(load_json(raw))
            except ValueError as exc:  # bad JSON or failed validation
                logger.warning(
                    "%s attempt %d/%d rejected: %s",
                    task,
                    attempt,
                    self.max_attempts,
                    exc,
                )
                current = (
                    f"{prompt}\n\n"
                    f"Your previous answer was rejected: {_short(exc)}\n"
                    "Return only the corrected JSON object.\n"
                    "JSON:"
                )
        return None


def _short(exc: Exception) -> str:
    return " ".join(str(exc).split())[:200]
