import pytest

from ai_book_assistant.assistant import (
    CLARIFICATION_MESSAGE,
    BookAssistant,
)
from ai_book_assistant.llm.errors import LLMError
from ai_book_assistant.models import Book, Explanations, SearchCriteria


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


class RelaxedCatalog(FakeCatalog):
    """Finds nothing on the first query, then returns its books."""

    def search_books(self, params: dict, limit: int = 5) -> list[Book]:
        self.calls.append((params, limit))
        return [] if len(self.calls) == 1 else self.books


class FakeExplainer:
    def __init__(
        self, result: Explanations | None = None, error: Exception | None = None
    ) -> None:
        self.result = Explanations() if result is None else result
        self.error = error
        self.calls: list[tuple] = []

    def explain(self, query, criteria, books, *, relaxed=False):
        self.calls.append((query, criteria, books, relaxed))
        if self.error:
            raise self.error
        return self.result


def _assistant(catalog, explainer=None, criteria=None, **kwargs) -> BookAssistant:
    extractor = FakeExtractor(criteria or SearchCriteria(subjects=["fantasy"]))
    return BookAssistant(extractor, catalog, explainer=explainer, **kwargs)


def test_happy_path_returns_books_and_full_strategy(make_book) -> None:
    criteria = SearchCriteria(subjects=["fantasy"])
    books = [make_book()]
    assistant = _assistant(FakeCatalog(books), criteria=criteria)

    result = assistant.search("fantasy")

    assert result.query == "fantasy"
    assert result.criteria == criteria
    assert result.books == books
    assert result.strategy == "full"
    assert result.message is None
    assert not result.needs_clarification


def test_descriptions_are_attached_to_the_results(make_book) -> None:
    books = [make_book()]
    catalog = FakeCatalog(books)

    result = _assistant(catalog).search("fantasy")

    assert [book.description for book in result.books] == ["About A book"]
    assert catalog.described == [books]


def test_empty_criteria_returns_clarification_without_searching() -> None:
    catalog = FakeCatalog([])
    explainer = FakeExplainer()

    result = _assistant(catalog, explainer, SearchCriteria()).search("a good book")

    assert result.needs_clarification
    assert result.message == CLARIFICATION_MESSAGE
    assert catalog.calls == []
    assert catalog.described == []
    assert explainer.calls == []


def test_no_results_means_no_description_or_explanation_calls() -> None:
    catalog = FakeCatalog([])
    explainer = FakeExplainer()

    result = _assistant(catalog, explainer).search("fantasy")

    assert result.books == []
    assert catalog.described == []
    assert explainer.calls == []


def test_relaxed_result_is_exposed(make_book) -> None:
    criteria = SearchCriteria(subjects=["fantasy"], keywords=["magic"])

    result = _assistant(RelaxedCatalog([make_book()]), criteria=criteria).search("x")

    assert result.strategy == "without_keywords"
    assert result.relaxed


def test_extractor_exception_propagates() -> None:
    error = RuntimeError("extraction failed")
    assistant = BookAssistant(FakeExtractor(error), FakeCatalog([]))

    with pytest.raises(RuntimeError, match="extraction failed"):
        assistant.search("fantasy")


def test_limit_is_forwarded_to_the_catalog(make_book) -> None:
    catalog = FakeCatalog([make_book()])

    _assistant(catalog, limit=12).search("fantasy")

    assert catalog.calls[0][1] == 12


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit must be greater than zero"):
        BookAssistant(FakeExtractor(SearchCriteria()), FakeCatalog([]), limit=0)


def test_explanations_are_returned_with_the_result(make_book) -> None:
    explanations = Explanations(summary="Overall.", by_book={"/works/OL1W": "It fits."})
    explainer = FakeExplainer(explanations)

    result = _assistant(FakeCatalog([make_book()]), explainer).search("fantasy")

    assert result.explanations == explanations


def test_explainer_sees_the_books_with_their_descriptions(make_book) -> None:
    explainer = FakeExplainer()

    _assistant(FakeCatalog([make_book()]), explainer).search("fantasy")

    query, _, books, relaxed = explainer.calls[0]
    assert query == "fantasy"
    assert books[0].description == "About A book"
    assert relaxed is False


def test_relaxed_flag_is_forwarded_to_the_explainer(make_book) -> None:
    explainer = FakeExplainer()
    criteria = SearchCriteria(subjects=["fantasy"], keywords=["magic"])

    _assistant(RelaxedCatalog([make_book()]), explainer, criteria).search("x")

    assert explainer.calls[0][3] is True


def test_without_an_explainer_there_are_no_explanations(make_book) -> None:
    result = _assistant(FakeCatalog([make_book()])).search("fantasy")

    assert result.explanations == Explanations()
    assert len(result.books) == 1


def test_llm_failure_while_explaining_keeps_the_books(make_book) -> None:
    explainer = FakeExplainer(error=LLMError("overloaded", status_code=503))

    result = _assistant(FakeCatalog([make_book()]), explainer).search("fantasy")

    assert len(result.books) == 1
    assert result.explanations == Explanations()


def test_unexpected_explainer_errors_propagate(make_book) -> None:
    explainer = FakeExplainer(error=RuntimeError("bug"))

    with pytest.raises(RuntimeError, match="bug"):
        _assistant(FakeCatalog([make_book()]), explainer).search("fantasy")
