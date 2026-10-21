import json
import logging

from ai_book_assistant.llm.base import LLMClient
from ai_book_assistant.llm.prompts import build_extraction_prompt
from ai_book_assistant.models import SearchCriteria

logger = logging.getLogger(__name__)


class CriteriaExtractor:
    def __init__(self, llm_client: LLMClient, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.llm_client = llm_client
        self.max_attempts = max_attempts

    def extract(self, user_query: str) -> SearchCriteria:
        if not user_query.strip():
            return SearchCriteria()

        base_prompt = self._build_prompt(user_query)
        prompt = base_prompt

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._parse(self.llm_client.generate(prompt))
            except ValueError as exc:  # bad JSON or failed validation
                logger.warning(
                    "Criteria extraction attempt %d/%d rejected: %s",
                    attempt,
                    self.max_attempts,
                    exc,
                )
                prompt = (
                    f"{base_prompt}\n\n"
                    f"Your previous answer was rejected: {self._short(exc)}\n"
                    "Return only the corrected JSON object.\n"
                    "JSON:"
                )

        logger.warning("Extraction failed; using the raw query as a keyword")
        return SearchCriteria(keywords=[user_query])

    @staticmethod
    def _parse(response_text: object) -> SearchCriteria:
        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError("the response was empty")
        return SearchCriteria.model_validate(json.loads(response_text))

    @staticmethod
    def _short(exc: Exception) -> str:
        return " ".join(str(exc).split())[:200]

    def _build_prompt(self, query: str) -> str:
        return build_extraction_prompt(query)
