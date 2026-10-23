import httpx
import pytest
from fastapi.testclient import TestClient

from ai_book_assistant.api import create_app
from ai_book_assistant.assistant import CLARIFICATION_MESSAGE, AssistantResult
from ai_book_assistant.llm.errors import LLMError
from ai_book_assistant.models import Book, Explanations, SearchCriteria
from ai_book_assistant.search.fallback import SearchOutcome
from ai_book_assistant.storage import SearchHistory

BOOK = Book(
    key="/works/OL1W",
    title="The Hobbit",
    authors=["J.R.R. Tolkien"],
    first_publish_year=1937,
)


class FakeAssistant:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.result


def found() -> AssistantResult:
    return AssistantResult(
        query="ignored",
        criteria=SearchCriteria(subjects=["fantasy"]),
        outcome=SearchOutcome([BOOK], "full", {"q": "x"}, [("full", 1)]),
        explanations=Explanations(
            summary="Overall.", by_book={"/works/OL1W": "It fits."}
        ),
    )


def needs_clarification() -> AssistantResult:
    outcome = SearchOutcome(
        [], "needs_clarification", None, [], needs_clarification=True
    )
    return AssistantResult("ignored", SearchCriteria(), outcome)


@pytest.fixture
def history(tmp_path):
    return SearchHistory(tmp_path / "history.db")


def client_for(assistant, history) -> TestClient:
    return TestClient(create_app(assistant, history))


def test_health(history):
    response = client_for(FakeAssistant(), history).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_books_explanations_and_a_history_id(history):
    client = client_for(FakeAssistant(found()), history)

    response = client.post("/search", json={"query": "fantasy"})

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["query"] == "fantasy"
    assert body["strategy"] == "full"
    assert body["message"] is None
    assert body["books"][0]["title"] == "The Hobbit"
    assert body["explanations"]["summary"] == "Overall."
    assert body["explanations"]["by_book"] == {"/works/OL1W": "It fits."}


def test_every_search_is_saved(history):
    client = client_for(FakeAssistant(found()), history)

    client.post("/search", json={"query": "fantasy"})

    saved = history.get(1)
    assert saved.query == "fantasy"
    assert saved.books == [BOOK]


def test_the_query_is_trimmed_before_it_is_used(history):
    assistant = FakeAssistant(found())

    client_for(assistant, history).post("/search", json={"query": "  fantasy  "})

    assert assistant.queries == ["fantasy"]
    assert history.get(1).query == "fantasy"


def test_a_vague_query_returns_the_message_and_is_saved(history):
    client = client_for(FakeAssistant(needs_clarification()), history)

    response = client.post("/search", json={"query": "a good book"})

    body = response.json()
    assert response.status_code == 200
    assert body["strategy"] == "needs_clarification"
    assert body["message"] == CLARIFICATION_MESSAGE
    assert body["books"] == []
    assert history.get(body["id"]) is not None


@pytest.mark.parametrize(
    "payload",
    [{}, {"query": ""}, {"query": "   "}, {"query": "x" * 501}, {"query": 5}],
)
def test_invalid_requests_are_rejected_without_calling_the_assistant(history, payload):
    assistant = FakeAssistant(found())

    response = client_for(assistant, history).post("/search", json=payload)

    assert response.status_code == 422
    assert assistant.queries == []
    assert history.list_recent() == []


def test_a_rate_limited_model_is_reported_as_busy(history):
    assistant = FakeAssistant(error=LLMError("slow down", status_code=429))

    response = client_for(assistant, history).post("/search", json={"query": "x"})

    assert response.status_code == 503
    assert "busy" in response.json()["detail"]
    assert history.list_recent() == []


def test_other_model_failures_are_reported_as_unavailable(history):
    assistant = FakeAssistant(error=LLMError("bad key", status_code=401))

    response = client_for(assistant, history).post("/search", json={"query": "x"})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
    assert history.list_recent() == []


def test_an_unreachable_catalogue_is_a_bad_gateway(history):
    assistant = FakeAssistant(error=httpx.ConnectError("no route"))

    response = client_for(assistant, history).post("/search", json={"query": "x"})

    assert response.status_code == 502
    assert "Open Library" in response.json()["detail"]
    assert history.list_recent() == []
