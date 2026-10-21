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


def test_attach_descriptions_attaches_each_book() -> None:
    docs = [
        {"key": "/works/OL1W", "title": "One"},
        {"key": "/works/OL2W", "title": "Two"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search.json"):
            return httpx.Response(200, json={"docs": docs})
        work_id = request.url.path.split("/")[-1].removesuffix(".json")
        return httpx.Response(200, json={"description": f"Description {work_id}"})

    client = OpenLibraryClient(
        transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    books = client.search_books({"q": "test"})

    assert client.attach_descriptions(books) == books
    assert [book.description for book in books] == [
        "Description OL1W",
        "Description OL2W",
    ]


def test_attach_descriptions_continues_after_work_lookup_fails() -> None:
    docs = [
        {"key": "/works/OL1W", "title": "One"},
        {"key": "/works/OL2W", "title": "Two"},
        {"key": "/works/OL3W", "title": "Three"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search.json"):
            return httpx.Response(200, json={"docs": docs})
        work_id = request.url.path.split("/")[-1].removesuffix(".json")
        if work_id == "OL2W":
            return httpx.Response(500)
        return httpx.Response(200, json={"description": f"Description {work_id}"})

    client = OpenLibraryClient(
        transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    books = client.search_books({"q": "test"})

    client.attach_descriptions(books)

    assert [book.description for book in books] == [
        "Description OL1W",
        None,
        "Description OL3W",
    ]


def test_attach_descriptions_survives_malformed_work_response() -> None:
    docs = [
        {"key": "/works/OL1W", "title": "One"},
        {"key": "/works/OL2W", "title": "Two"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search.json"):
            return httpx.Response(200, json={"docs": docs})
        if request.url.path.endswith("OL1W.json"):
            return httpx.Response(200, content=b"<html>not json</html>")
        return httpx.Response(200, json={"description": "A fine book."})

    client = OpenLibraryClient(
        transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    books = client.search_books({"q": "test"})

    client.attach_descriptions(books)

    assert [book.description for book in books] == [None, "A fine book."]
