import httpx
import pytest
from fastapi.testclient import TestClient

from ai_book_assistant.api import create_app
from ai_book_assistant.assistant import CLARIFICATION_MESSAGE, AssistantResult
from ai_book_assistant.llm.errors import LLMError
from ai_book_assistant.models import SearchCriteria
from ai_book_assistant.search.fallback import SearchOutcome
from ai_book_assistant.service import SearchService


@pytest.fixture
def hobbit(make_book):
    return make_book(
        key="/works/OL1W",
        title="The Hobbit",
        authors=["J.R.R. Tolkien"],
        first_publish_year=1937,
    )


def needs_clarification() -> AssistantResult:
    outcome = SearchOutcome(
        [], "needs_clarification", None, [], needs_clarification=True
    )
    return AssistantResult("ignored", SearchCriteria(), outcome)


def client_for(assistant, history) -> TestClient:
    return TestClient(create_app(SearchService(assistant, history), history))


def test_health(history, make_assistant):
    response = client_for(make_assistant(), history).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_results_with_their_explanations(
    history, make_assistant, found_result, hobbit
):
    client = client_for(make_assistant(found_result(hobbit)), history)

    response = client.post("/search", json={"query": "fantasy"})

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["query"] == "fantasy"
    assert body["strategy"] == "full"
    assert body["message"] is None
    assert body["summary"] == "Overall."
    assert body["results"] == [
        {
            "title": "The Hobbit",
            "authors": ["J.R.R. Tolkien"],
            "first_publish_year": 1937,
            "description": None,
            "explanation": "It fits.",
            "url": "https://openlibrary.org/works/OL1W",
        }
    ]


def test_every_search_is_saved(history, make_assistant, found_result, hobbit):
    client = client_for(make_assistant(found_result(hobbit)), history)

    client.post("/search", json={"query": "fantasy"})

    saved = history.get(1)
    assert saved.query == "fantasy"
    assert saved.books == [hobbit]


def test_the_query_is_trimmed_before_it_is_used(
    history, make_assistant, found_result, hobbit
):
    assistant = make_assistant(found_result(hobbit))

    client_for(assistant, history).post("/search", json={"query": "  fantasy  "})

    assert assistant.queries == ["fantasy"]
    assert history.get(1).query == "fantasy"


def test_a_vague_query_returns_the_message_and_is_saved(history, make_assistant):
    client = client_for(make_assistant(needs_clarification()), history)

    response = client.post("/search", json={"query": "a good book"})

    body = response.json()
    assert response.status_code == 200
    assert body["strategy"] == "needs_clarification"
    assert body["message"] == CLARIFICATION_MESSAGE
    assert body["results"] == []
    assert history.get(body["id"]) is not None


@pytest.mark.parametrize(
    "payload",
    [{}, {"query": ""}, {"query": "   "}, {"query": "x" * 501}, {"query": 5}],
)
def test_invalid_requests_are_rejected_without_calling_the_assistant(
    history, payload, make_assistant
):
    assistant = make_assistant()

    response = client_for(assistant, history).post("/search", json=payload)

    assert response.status_code == 422
    assert assistant.queries == []
    assert history.list_recent() == []


def test_a_rate_limited_model_is_reported_as_busy(history, make_assistant):
    assistant = make_assistant(error=LLMError("slow down", status_code=429))

    response = client_for(assistant, history).post("/search", json={"query": "x"})

    assert response.status_code == 503
    assert "busy" in response.json()["detail"]
    assert history.list_recent() == []


def test_other_model_failures_are_reported_as_unavailable(history, make_assistant):
    assistant = make_assistant(error=LLMError("bad key", status_code=401))

    response = client_for(assistant, history).post("/search", json={"query": "x"})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
    assert history.list_recent() == []


def test_an_unreachable_catalogue_is_a_bad_gateway(history, make_assistant):
    assistant = make_assistant(error=httpx.ConnectError("no route"))

    response = client_for(assistant, history).post("/search", json={"query": "x"})

    assert response.status_code == 502
    assert "Open Library" in response.json()["detail"]
    assert history.list_recent() == []


def test_history_starts_empty(history, make_assistant):
    assert client_for(make_assistant(), history).get("/history").json() == []


def test_history_lists_searches_newest_first_without_snapshots(
    history, make_assistant, found_result, hobbit
):
    client = client_for(make_assistant(found_result(hobbit)), history)
    for query in ("one", "two", "three"):
        client.post("/search", json={"query": query})

    entries = client.get("/history").json()

    assert [entry["query"] for entry in entries] == ["three", "two", "one"]
    assert set(entries[0]) == {"id", "created_at", "query", "strategy"}


def test_history_supports_paging(history, make_assistant, found_result, hobbit):
    client = client_for(make_assistant(found_result(hobbit)), history)
    for query in ("one", "two", "three"):
        client.post("/search", json={"query": query})

    entries = client.get("/history?limit=2&offset=1").json()

    assert [entry["query"] for entry in entries] == ["two", "one"]


@pytest.mark.parametrize("params", ["limit=0", "limit=101", "offset=-1", "limit=abc"])
def test_history_rejects_bad_paging(history, params, make_assistant):
    response = client_for(make_assistant(), history).get(f"/history?{params}")

    assert response.status_code == 422


def test_a_saved_search_can_be_opened_again(
    history, make_assistant, found_result, hobbit
):
    client = client_for(make_assistant(found_result(hobbit)), history)
    created = client.post("/search", json={"query": "fantasy"}).json()

    replayed = client.get(f"/history/{created['id']}")

    assert replayed.status_code == 200
    assert replayed.json() == created


def test_a_saved_clarification_keeps_its_message(history, make_assistant):
    client = client_for(make_assistant(needs_clarification()), history)
    created = client.post("/search", json={"query": "a good book"}).json()

    replayed = client.get(f"/history/{created['id']}").json()

    assert replayed["message"] == CLARIFICATION_MESSAGE


def test_an_unknown_search_is_not_found(history, make_assistant):
    response = client_for(make_assistant(), history).get("/history/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Search not found"}


def test_a_non_numeric_search_id_is_rejected(history, make_assistant):
    response = client_for(make_assistant(), history).get("/history/abc")

    assert response.status_code == 422


def test_the_docs_list_every_endpoint(history, make_assistant):
    paths = client_for(make_assistant(), history).get("/openapi.json").json()["paths"]

    assert set(paths) == {"/health", "/search", "/history", "/history/{search_id}"}


def test_results_leave_out_noisy_catalogue_fields(history, make_book, make_assistant):
    noisy = make_book(
        key="/works/OL2W",
        title="Noisy",
        subjects=["Fantasy", "Fiction"],
        covers=[123],
    )
    result = AssistantResult(
        "ignored",
        SearchCriteria(subjects=["fantasy"]),
        SearchOutcome([noisy], "full", None, []),
    )

    body = client_for(make_assistant(result), history).post(
        "/search", json={"query": "x"}
    )

    book = body.json()["results"][0]
    assert "subjects" not in book
    assert "covers" not in book
    assert book["explanation"] is None
