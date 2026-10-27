import json

import httpx
from fastapi.testclient import TestClient

from ai_book_assistant.api import create_app
from ai_book_assistant.assistant import BookAssistant
from ai_book_assistant.explain import BookExplainer
from ai_book_assistant.llm.extractor import CriteriaExtractor
from ai_book_assistant.search import OpenLibraryClient
from ai_book_assistant.service import SearchService
from ai_book_assistant.storage import SqliteSearchHistory


def test_search_api_runs_the_full_book_search_flow(tmp_path, make_llm) -> None:
    query = "A short fantasy novel by Andrzej Sapkowski in English"
    llm = make_llm(
        json.dumps(
            {
                "subjects": ["fantasy"],
                "author": "Andrzej Sapkowski",
                "language": "eng",
                "preferences": {"length": "short"},
            }
        ),
        json.dumps(
            {
                "summary": "A concise fantasy novel from the requested author.",
                "explanations": [
                    {"id": 1, "explanation": "A fantasy novel with a compact length."}
                ],
            }
        ),
    )
    requests: list[httpx.Request] = []

    def open_library_response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/search.json":
            return httpx.Response(
                200,
                json={
                    "docs": [
                        {
                            "title": "The Last Wish",
                            "author_name": ["Andrzej Sapkowski"],
                            "first_publish_year": 1993,
                            "key": "/works/OL123W",
                            "subject": ["Fantasy"],
                            "cover_i": 12345,
                            "number_of_pages_median": 280,
                            "ratings_average": 4.2,
                            "ratings_count": 1500,
                            "want_to_read_count": 3000,
                            "language": ["eng", "pol"],
                            "first_sentence": ["The night was clear."],
                        }
                    ]
                },
            )
        if request.url.path == "/works/OL123W.json":
            return httpx.Response(
                200, json={"description": "A collection of fantasy stories."}
            )
        return httpx.Response(404)

    history = SqliteSearchHistory(tmp_path / "history.db")
    assistant = BookAssistant(
        CriteriaExtractor(llm),
        OpenLibraryClient(transport=httpx.MockTransport(open_library_response)),
        explainer=BookExplainer(llm),
    )
    app = create_app(SearchService(assistant, history), history)

    with TestClient(app) as client:
        search_response = client.post("/search", json={"query": query})
        assert search_response.status_code == 200
        search = search_response.json()

        assert search["query"] == query
        assert search["strategy"] == "full"
        assert search["summary"] == "A concise fantasy novel from the requested author."
        assert search["criteria"]["language"] == "eng"
        assert search["criteria"]["preferences"]["length"] == "short"
        assert search["results"] == [
            {
                "title": "The Last Wish",
                "authors": ["Andrzej Sapkowski"],
                "first_publish_year": 1993,
                "description": "A collection of fantasy stories.",
                "number_of_pages_median": 280,
                "ratings_average": 4.2,
                "ratings_count": 1500,
                "want_to_read_count": 3000,
                "languages": ["eng", "pol"],
                "cover_url": "https://covers.openlibrary.org/b/id/12345-M.jpg",
                "first_sentence": "The night was clear.",
                "explanation": "A fantasy novel with a compact length.",
                "url": "https://openlibrary.org/works/OL123W",
            }
        ]

        search_history = client.get("/history")
        assert search_history.status_code == 200
        assert search_history.json()[0]["id"] == search["id"]
        assert search_history.json()[0]["query"] == query

        saved_search = client.get(f"/history/{search['id']}")
        assert saved_search.status_code == 200
        assert saved_search.json() == search

    search_request = next(
        request for request in requests if request.url.path == "/search.json"
    )
    assert search_request.url.params["author"] == "Andrzej Sapkowski"
    assert 'subject:"fantasy"' in search_request.url.params["q"]
    assert "number_of_pages_median:[1 TO 300]" in search_request.url.params["q"]
    assert "language:eng" in search_request.url.params["q"]
    assert len(llm.prompts) == 2
