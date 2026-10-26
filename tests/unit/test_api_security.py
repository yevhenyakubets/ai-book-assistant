from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from ai_book_assistant.api import create_app
from ai_book_assistant.models import SearchCriteria, SearchRecord

TOKEN = "s3cret-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
PROTECTED = [
    ("POST", "/search", {"json": {"query": "fantasy"}}),
    ("GET", "/history", {}),
    ("GET", "/history/1", {}),
]


class FakeService:
    def search(self, query):
        return SearchRecord(
            id=1,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            query=query,
            strategy="full",
            criteria=SearchCriteria(subjects=["fantasy"]),
        )


@pytest.fixture
def open_client(history):
    return TestClient(create_app(FakeService(), history))


@pytest.fixture
def locked_client(history):
    return TestClient(create_app(FakeService(), history, access_token=TOKEN))


def test_without_a_configured_token_everything_is_open(open_client):
    assert open_client.post("/search", json={"query": "fantasy"}).status_code == 200
    assert open_client.get("/history").status_code == 200


@pytest.mark.parametrize("method, path, kwargs", PROTECTED)
def test_protected_endpoints_reject_requests_without_a_token(
    locked_client, method, path, kwargs
):
    response = locked_client.request(method, path, **kwargs)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("method, path, kwargs", PROTECTED)
def test_protected_endpoints_reject_a_wrong_token(locked_client, method, path, kwargs):
    headers = {"Authorization": "Bearer wrong"}

    response = locked_client.request(method, path, headers=headers, **kwargs)

    assert response.status_code == 401


def test_the_right_token_is_accepted(locked_client):
    response = locked_client.post("/search", json={"query": "fantasy"}, headers=HEADERS)

    assert response.status_code == 200
    assert locked_client.get("/history", headers=HEADERS).status_code == 200


def test_another_authorization_scheme_is_rejected(locked_client):
    headers = {"Authorization": f"Basic {TOKEN}"}

    assert locked_client.get("/history", headers=headers).status_code == 401


@pytest.mark.parametrize("path", ["/health", "/docs", "/openapi.json"])
def test_health_and_docs_stay_open(locked_client, path):
    assert locked_client.get(path).status_code == 200


def test_the_docs_offer_an_authorize_button(locked_client):
    schema = locked_client.get("/openapi.json").json()

    assert "HTTPBearer" in schema["components"]["securitySchemes"]
