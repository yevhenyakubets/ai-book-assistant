import pytest

from ai_book_assistant.assistant import (
    CLARIFICATION_MESSAGE,
    BookAssistant,
)
from ai_book_assistant.models import Book, SearchCriteria


class FakeExtractor:
    def __init__(self, criteria: SearchCriteria | Exception) -> None:
        self.criteria = criteria

    def extract(self, query: str) -> SearchCriteria:
        if isinstance(self.criteria, Exception):
            raise self.criteria
        return self.criteria


class FakeCatalog:
    def __init__(self, books: list[Book]) -> None:
        self.books = books
        self.calls: list[tuple[dict, int]] = []
        self.described: list[list[Book]] = []

    def search_books(self, params: dict, limit: int = 5) -> list[Book]:
        self.calls.append((params, limit))
        return self.books

    def attach_descriptions(self, books: list[Book]) -> list[Book]:
        self.described.append(books)
        for book in books:
            book.description = f"About {book.title}"
        return books


def test_happy_path_returns_books_and_full_strategy() -> None:
    criteria = SearchCriteria(subjects=["fantasy"])
    books = [Book(title="A book", key="/works/OL1W")]
    catalog = FakeCatalog(books)
    assistant = BookAssistant(FakeExtractor(criteria), catalog)

    result = assistant.search("fantasy")

    assert result.query == "fantasy"
    assert result.criteria == criteria
    assert result.books == books
    assert result.strategy == "full"
    assert result.message is None
    assert not result.needs_clarification


def test_descriptions_are_attached_to_the_results() -> None:
    books = [Book(title="A book", key="/works/OL1W")]
    catalog = FakeCatalog(books)
    assistant = BookAssistant(
        FakeExtractor(SearchCriteria(subjects=["fantasy"])), catalog
    )

    result = assistant.search("fantasy")

    assert [book.description for book in result.books] == ["About A book"]
    assert catalog.described == [books]


def test_empty_criteria_returns_clarification_without_searching() -> None:
    catalog = FakeCatalog([])
    assistant = BookAssistant(FakeExtractor(SearchCriteria()), catalog)

    result = assistant.search("recommend a book")

    assert result.needs_clarification
    assert result.message == CLARIFICATION_MESSAGE
    assert catalog.calls == []
    assert catalog.described == []


def test_no_results_means_no_description_lookups() -> None:
    catalog = FakeCatalog([])
    assistant = BookAssistant(
        FakeExtractor(SearchCriteria(subjects=["fantasy"])), catalog
    )

    result = assistant.search("fantasy")

    assert result.books == []
    assert catalog.described == []


def test_relaxed_result_is_exposed() -> None:
    books = [Book(title="A book", key="/works/OL1W")]

    class RelaxedCatalog(FakeCatalog):
        def search_books(self, params: dict, limit: int = 5) -> list[Book]:
            self.calls.append((params, limit))
            return [] if len(self.calls) == 1 else books

    result = BookAssistant(
        FakeExtractor(SearchCriteria(subjects=["fantasy"], keywords=["magic"])),
        RelaxedCatalog([]),
    ).search("fantasy magic")
    assert result.strategy == "without_keywords"
    assert result.relaxed


def test_extractor_exception_propagates() -> None:
    error = RuntimeError("extraction failed")
    assistant = BookAssistant(FakeExtractor(error), FakeCatalog([]))

    with pytest.raises(RuntimeError, match="extraction failed"):
        assistant.search("fantasy")


def test_limit_is_forwarded_to_the_catalog() -> None:
    catalog = FakeCatalog([Book(title="A book", key="/works/OL1W")])
    assistant = BookAssistant(
        FakeExtractor(SearchCriteria(subjects=["fantasy"])), catalog, limit=12
    )

    assistant.search("fantasy")

    assert catalog.calls[0][1] == 12


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit must be greater than zero"):
        BookAssistant(FakeExtractor(SearchCriteria()), FakeCatalog([]), limit=0)
