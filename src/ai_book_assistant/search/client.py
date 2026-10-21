import logging
import re
import time
from collections.abc import Mapping

import httpx

from ai_book_assistant.models import Book
from ai_book_assistant.search.parsing import book_from_search_doc, extract_description

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class OpenLibraryClient:
    BASE_URL = "https://openlibrary.org"
    WORK_KEY_PATTERN = re.compile(r"/works/[A-Za-z0-9_-]+")

    def __init__(
        self,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 3,
        backoff: float = 0.5,
        sleep=time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.timeout = timeout
        self.transport = transport
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.sleep = sleep
        self.headers = {
            "User-Agent": (
                "AIBookAssistant/0.1 "
                "(https://github.com/yevhenyakubets/ai-book-assistant)"
            )
        }

    def _get_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self.transport)

    def _get(self, client: httpx.Client, url: str, **kwargs) -> httpx.Response:
        """GET with retries on timeouts, connection errors, 429 and 5xx."""
        for attempt in range(1, self.max_attempts + 1):
            is_last = attempt == self.max_attempts
            try:
                response = client.get(url, headers=self.headers, **kwargs)
            except httpx.TransportError as exc:
                if is_last:
                    raise
                logger.warning(
                    "Open Library request failed (%s); attempt %d/%d",
                    type(exc).__name__,
                    attempt,
                    self.max_attempts,
                )
            else:
                if is_last or response.status_code not in RETRYABLE_STATUS:
                    return response
                logger.warning(
                    "Open Library returned %d; attempt %d/%d",
                    response.status_code,
                    attempt,
                    self.max_attempts,
                )
            self.sleep(self.backoff * 2 ** (attempt - 1))

    def search_books(self, query_or_params: str | dict, limit: int = 5) -> list[Book]:
        with self._get_client() as client:
            return self._search_books(client, query_or_params, limit)

    def _search_books(
        self, client: httpx.Client, query_or_params: str | dict, limit: int
    ) -> list[Book]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        url = f"{self.BASE_URL}/search.json"

        if isinstance(query_or_params, dict):
            params = {**query_or_params, "limit": limit}
        else:
            params = {"q": query_or_params, "limit": limit}

        response = self._get(client, url, params=params)
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
            books.append(book_from_search_doc(doc))

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
        response = self._get(client, url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = self._json_object(response, "work")

        return extract_description(data.get("description"))

    def attach_descriptions(self, books: list[Book]) -> list[Book]:
        with self._get_client() as client:
            for book in books:
                try:
                    description = self._fetch_work_description(client, book.key)
                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning(
                        "Could not fetch description for %s: %s", book.key, exc
                    )
                    description = None
                book.description = description
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
