import json

import httpx
import pytest

from ai_book_assistant.search import OpenLibraryClient


def test_search_books(client: OpenLibraryClient) -> None:
    books = client.search_books("Foundation", limit=1)

    assert len(books) == 1
    assert books[0].title == "Foundation"
    assert books[0].key == "/works/OL46125W"


def test_fetch_work_description(client: OpenLibraryClient) -> None:
    desc = client.fetch_work_description("/works/OL46125W")

    assert desc == "An epic sci-fi masterwork."


def test_fetch_work_description_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    desc = client.fetch_work_description("/works/OL9999W")
    assert desc is None


def test_fetch_work_description_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    assert client.fetch_work_description("/works/OL46125W") is None


def test_fetch_work_description_structured_value() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"description": {"type": "/type/text", "value": "  A story.  "}}
        )

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    assert client.fetch_work_description("/works/OL46125W") == "A story."


def test_fetch_work_description_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        client.fetch_work_description("/works/OL46125W")


def test_fetch_work_description_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("request timed out", request=request)

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ReadTimeout):
        client.fetch_work_description("/works/OL46125W")


def test_fetch_work_description_malformed_json_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="work response must be a JSON object"):
        client.fetch_work_description("/works/OL46125W")


def test_fetch_work_description_malformed_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"Not valid JSON")

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    with pytest.raises(json.JSONDecodeError):
        client.fetch_work_description("/works/OL46125W")


def test_fetch_work_description_rejects_invalid_key() -> None:
    client = OpenLibraryClient(transport=httpx.MockTransport(lambda request: None))
    with pytest.raises(ValueError, match="Invalid Open Library work key"):
        client.fetch_work_description("/works/OL1W/../../search.json")


def test_search_books_malformed_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"Not valid JSON")

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    with pytest.raises(json.JSONDecodeError):
        client.search_books("Foundation")


def test_search_books_malformed_response_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"docs": "not a list"})

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="must contain a docs list"):
        client.search_books("Foundation")


def test_search_books_filters_invalid_keys() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "docs": [
                    {"title": "Valid", "key": "/works/OL1W"},
                    {"title": "Invalid Key Type", "key": 123},
                    {"title": "Invalid Key Format", "key": "/books/OL2B"},
                    {"title": "Invalid Path", "key": "/works/OL1W/../OL2W"},
                    "not a document",
                ]
            },
        )

    client = OpenLibraryClient(transport=httpx.MockTransport(handler))
    books = client.search_books("Test")

    assert len(books) == 1
    assert books[0].title == "Valid"


def test_search_books_rejects_non_positive_limit(client: OpenLibraryClient) -> None:
    with pytest.raises(ValueError, match="limit must be greater than zero"):
        client.search_books("Foundation", limit=0)


def test_search_books_with_descriptions(
    client: OpenLibraryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    get_client = client._get_client
    created_clients = []

    def tracked_get_client() -> httpx.Client:
        http_client = get_client()
        created_clients.append(http_client)
        return http_client

    monkeypatch.setattr(client, "_get_client", tracked_get_client)
    books = client.search_books_with_descriptions("Foundation", limit=1)

    assert len(books) == 1
    assert books[0].title == "Foundation"
    assert books[0].description == "An epic sci-fi masterwork."
    assert len(created_clients) == 1
