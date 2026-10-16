import httpx

from ai_book_assistant.models import Book


class OpenLibraryClient:
    BASE_URL = "https://openlibrary.org"

    def __init__(
        self, timeout: float = 10.0, transport: httpx.BaseTransport | None = None
    ):
        self.timeout = timeout
        self.transport = transport
        self.headers = {"User-Agent": "AIBookAssistant/1.0 (dev@example.com)"}

    def _get_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self.transport)

    def search_books(self, query: str, limit: int = 5) -> list[Book]:
        url = f"{self.BASE_URL}/search.json"
        params = {"q": query, "limit": limit}

        with self._get_client() as client:
            response = client.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()

        return [Book.from_search_doc(doc) for doc in data.get("docs", [])]

    def fetch_work_description(self, work_key: str) -> str | None:
        url = f"{self.BASE_URL}{work_key}.json"

        with self._get_client() as client:
            response = client.get(url, headers=self.headers)
            if response.status_code != 200:
                return None
            data = response.json()

        description = data.get("description")
        if isinstance(description, dict):
            return description.get("value")
        return description
