from datetime import UTC, datetime, timedelta

import httpx
import pytest

from ai_book_assistant.assistant import AssistantResult
from ai_book_assistant.models import Book, Explanations, SearchCriteria
from ai_book_assistant.search import OpenLibraryClient
from ai_book_assistant.search.fallback import SearchOutcome
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


class FakeLLM:
    def __init__(self, responses: tuple[object, ...]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class _HistoryClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self.now
        self.now += timedelta(minutes=1)
        return current


@pytest.fixture
def fake_clock() -> _HistoryClock:
    return _HistoryClock()


@pytest.fixture
def make_assistant():
    def factory(
        result: AssistantResult | None = None, error: Exception | None = None
    ) -> FakeAssistant:
        return FakeAssistant(result=result, error=error)

    return factory


@pytest.fixture
def found_result():
    def factory(book: Book) -> AssistantResult:
        return AssistantResult(
            query="ignored",
            criteria=SearchCriteria(subjects=["fantasy"]),
            outcome=SearchOutcome([book], "full", {"q": "fantasy"}),
            explanations=Explanations(
                summary="Overall.", by_book={"/works/OL1W": "It fits."}
            ),
        )

    return factory


@pytest.fixture
def make_llm():
    def factory(*responses: object) -> FakeLLM:
        return FakeLLM(responses)

    return factory


@pytest.fixture
def make_book():
    def factory(**overrides: object) -> Book:
        values = {"title": "A book", "key": "/works/OL1W"}
        return Book(**(values | overrides))

    return factory


@pytest.fixture
def make_criteria():
    def factory(**fields: object) -> SearchCriteria:
        return SearchCriteria.model_validate(fields)

    return factory


@pytest.fixture
def history(tmp_path, fake_clock):
    return SqliteSearchHistory(tmp_path / "history.db", clock=fake_clock)


@pytest.fixture
def sample_search_doc() -> dict:
    return {
        "title": "Foundation",
        "author_name": ["Isaac Asimov"],
        "first_publish_year": 1951,
        "key": "/works/OL46125W",
        "subject": ["Science Fiction"],
        "cover_i": 14612610,
    }


@pytest.fixture
def mock_open_library_transport(sample_search_doc) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/search.json" in request.url.path:
            return httpx.Response(200, json={"docs": [sample_search_doc]})
        if "/works/" in request.url.path:
            return httpx.Response(
                200, json={"description": "An epic sci-fi masterwork."}
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture
def client(mock_open_library_transport) -> OpenLibraryClient:
    return OpenLibraryClient(transport=mock_open_library_transport)
