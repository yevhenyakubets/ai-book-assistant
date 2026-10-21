import logging

from ai_book_assistant.llm.base import LLMClient
from ai_book_assistant.llm.prompts import build_extraction_prompt
from ai_book_assistant.llm.structured import StructuredGenerator
from ai_book_assistant.models import SearchCriteria

logger = logging.getLogger(__name__)


class CriteriaExtractor:
    def __init__(self, llm_client: LLMClient, max_attempts: int = 2) -> None:
        self.generator = StructuredGenerator(llm_client, max_attempts)

    def extract(self, user_query: str) -> SearchCriteria:
        if not user_query.strip():
            return SearchCriteria()

        criteria = self.generator.generate(
            self._build_prompt(user_query),
            SearchCriteria.model_validate,
            task="Criteria extraction",
        )
        if criteria is None:
            logger.warning("Extraction failed; using the raw query as a keyword")
            return SearchCriteria(keywords=[user_query])
        return criteria

    def _build_prompt(self, query: str) -> str:
        return build_extraction_prompt(query)
