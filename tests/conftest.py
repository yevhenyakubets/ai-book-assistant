import httpx
import pytest

from ai_book_assistant.search import OpenLibraryClient


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
