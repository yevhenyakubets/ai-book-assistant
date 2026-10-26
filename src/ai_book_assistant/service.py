from typing import Protocol

from ai_book_assistant.assistant import AssistantResult
from ai_book_assistant.models import SearchRecord
from ai_book_assistant.storage import HistoryStore


class Assistant(Protocol):
    def search(self, query: str) -> AssistantResult: ...


class SearchService:
    def __init__(self, assistant: Assistant, history: HistoryStore) -> None:
        self.assistant = assistant
        self.history = history

    def search(self, query: str) -> SearchRecord:
        result = self.assistant.search(query)
        return self.history.add(
            query=query,
            strategy=result.strategy,
            criteria=result.criteria,
            books=result.books,
            explanations=result.explanations,
        )
