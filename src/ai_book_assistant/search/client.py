import re
from collections.abc import Mapping

import httpx

from ai_book_assistant.models import Book


class OpenLibraryClient:
    BASE_URL = "https://openlibrary.org"
    WORK_KEY_PATTERN = re.compile(r"/works/[A-Za-z0-9_-]+")

    def __init__(
        self, timeout: float = 10.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.timeout = timeout
        self.transport = transport
        self.headers = {
            "User-Agent": (
                "AIBookAssistant/0.1 "
                "(https://github.com/yevhenyakubets/ai-book-assistant)"
            )
        }

    def _get_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self.transport)

    def search_books(self, query: str, limit: int = 5) -> list[Book]:
        with self._get_client() as client:
            return self._search_books(client, query, limit)

    def _search_books(self, client: httpx.Client, query: str, limit: int) -> list[Book]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        url = f"{self.BASE_URL}/search.json"
        params = {"q": query, "limit": limit}

        response = client.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        data = self._json_object(response, "search")
        docs = data.get("docs")
        if not isinstance(docs, list):
            raise ValueError("Open Library search response must contain a docs list")

        books = []
        for doc in docs:
            if not isinstance(doc, Mapping):
                continue
            key = doc.get("key")
            if not self._is_valid_work_key(key):
                continue
            books.append(Book.from_search_doc(doc))

        return books

    def fetch_work_description(self, work_key: str) -> str | None:
        with self._get_client() as client:
            return self._fetch_work_description(client, work_key)

    def _fetch_work_description(
        self, client: httpx.Client, work_key: str
    ) -> str | None:
        if not self._is_valid_work_key(work_key):
            raise ValueError(f"Invalid Open Library work key: {work_key!r}")

        url = f"{self.BASE_URL}{work_key}.json"
        response = client.get(url, headers=self.headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = self._json_object(response, "work")

        description = data.get("description")
        if isinstance(description, Mapping):
            description = description.get("value")
        return description.strip() or None if isinstance(description, str) else None

    def search_books_with_descriptions(self, query: str, limit: int = 5) -> list[Book]:
        with self._get_client() as client:
            books = self._search_books(client, query, limit)
            for book in books:
                description = self._fetch_work_description(client, book.key)
                book.update_from_work_data({"description": description})
            return books

    @classmethod
    def _is_valid_work_key(cls, key: object) -> bool:
        return isinstance(key, str) and cls.WORK_KEY_PATTERN.fullmatch(key) is not None

    @staticmethod
    def _json_object(response: httpx.Response, resource: str) -> Mapping[str, object]:
        data = response.json()
        if not isinstance(data, Mapping):
            raise ValueError(f"Open Library {resource} response must be a JSON object")
        return data
