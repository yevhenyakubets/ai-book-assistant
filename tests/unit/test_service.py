import pytest

from ai_book_assistant.assistant import AssistantResult
from ai_book_assistant.models import Explanations, SearchCriteria
from ai_book_assistant.search.fallback import SearchOutcome
from ai_book_assistant.service import SearchService
from ai_book_assistant.storage import SqliteSearchHistory


class FakeAssistant:
    def __init__(
        self, result: AssistantResult | None = None, error: Exception | None = None
    ) -> None:
        self.result = result
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str) -> AssistantResult:
        self.queries.append(query)
        if self.error:
            raise self.error
        assert self.result is not None
        return self.result


def found_result(make_book) -> AssistantResult:
    book = make_book(
        title="The Hobbit",
        authors=["J.R.R. Tolkien"],
        key="/works/OL1W",
    )
    return AssistantResult(
        query="ignored",
        criteria=SearchCriteria(subjects=["fantasy"]),
        outcome=SearchOutcome([book], "full", {"q": "fantasy"}),
        explanations=Explanations(
            summary="Overall.", by_book={"/works/OL1W": "It fits."}
        ),
    )


def test_search_saves_the_result_and_returns_the_saved_record(
    tmp_path, make_book
) -> None:
    assistant = FakeAssistant(found_result(make_book))
    history = SqliteSearchHistory(tmp_path / "history.db")

    record = SearchService(assistant, history).search("fantasy")

    assert assistant.queries == ["fantasy"]
    assert record.query == "fantasy"
    assert record.strategy == "full"
    assert record.criteria == SearchCriteria(subjects=["fantasy"])
    assert len(record.books) == 1
    assert record.explanations.summary == "Overall."
    assert history.get(record.id) == record


def test_clarification_results_are_saved(tmp_path) -> None:
    assistant = FakeAssistant(
        AssistantResult(
            query="ignored",
            criteria=SearchCriteria(),
            outcome=SearchOutcome(
                [], "needs_clarification", None, [], needs_clarification=True
            ),
        )
    )
    history = SqliteSearchHistory(tmp_path / "history.db")

    record = SearchService(assistant, history).search("a good book")

    assert record.query == "a good book"
    assert record.strategy == "needs_clarification"
    assert record.books == []
    assert history.get(record.id) == record


def test_assistant_exception_propagates_without_saving(tmp_path) -> None:
    history = SqliteSearchHistory(tmp_path / "history.db")
    assistant = FakeAssistant(error=RuntimeError("search failed"))

    with pytest.raises(RuntimeError, match="search failed"):
        SearchService(assistant, history).search("fantasy")

    assert history.list_recent() == []
