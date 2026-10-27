import pytest

from ai_book_assistant.search.parsing import (
    book_from_search_doc,
    dedupe_books,
    extract_description,
)


def test_book_from_search_doc(sample_search_doc: dict[str, object]) -> None:
    book = book_from_search_doc(sample_search_doc)

    assert book.title == "Foundation"
    assert book.authors == ["Isaac Asimov"]
    assert book.first_publish_year == 1951
    assert book.key == "/works/OL46125W"
    assert book.subjects == ["Science Fiction"]
    assert book.covers == [14612610]


def test_book_from_search_doc_includes_richer_open_library_fields() -> None:
    book = book_from_search_doc(
        {
            "key": "/works/OL1W",
            "title": "A story",
            "number_of_pages_median": 240,
            "ratings_average": 4.25,
            "ratings_count": 120,
            "want_to_read_count": 350,
            "language": ["eng", "fre"],
            "first_sentence": ["  It began at dawn.  ", "Another version."],
            "cover_i": 123,
        }
    )

    assert book.number_of_pages_median == 240
    assert book.ratings_average == 4.25
    assert book.ratings_count == 120
    assert book.want_to_read_count == 350
    assert book.languages == ["eng", "fre"]
    assert book.first_sentence == "It began at dawn."
    assert book.cover_url == "https://covers.openlibrary.org/b/id/123-M.jpg"


def test_book_from_search_doc_messy_types() -> None:
    doc = {
        "title": "Messy Book",
        "author_name": "Single Author String (Invalid Type)",
        "first_publish_year": "ancient",
        "subject": "Fiction (Invalid Type)",
        "cover_i": "not-an-int",
        "number_of_pages_median": True,
        "ratings_average": float("nan"),
        "ratings_count": -1,
        "want_to_read_count": "popular",
        "language": "eng",
        "first_sentence": {"unexpected": "shape"},
        "key": "/works/OL999W",
    }
    book = book_from_search_doc(doc)

    assert book.title == "Messy Book"
    assert book.authors == []
    assert book.first_publish_year is None
    assert book.subjects == []
    assert book.covers == []
    assert book.number_of_pages_median is None
    assert book.ratings_average is None
    assert book.ratings_count is None
    assert book.want_to_read_count is None
    assert book.languages == []
    assert book.first_sentence is None
    assert book.key == "/works/OL999W"


def test_book_from_search_doc_requires_work_key() -> None:
    with pytest.raises(ValueError, match="must contain a work key"):
        book_from_search_doc({"title": "Missing key"})


def test_duplicate_authors_are_removed_in_order() -> None:
    doc = {
        "key": "/works/OL1W",
        "title": "T",
        "author_name": ["Peter Watts", "Peter Watts", "Someone Else"],
    }
    assert book_from_search_doc(doc).authors == ["Peter Watts", "Someone Else"]


def test_author_spelling_variants_are_merged() -> None:
    doc = {
        "key": "/works/OL1W",
        "title": "T",
        "author_name": ["G. A. Aiken", "G.A. Aiken"],
    }
    assert book_from_search_doc(doc).authors == ["G. A. Aiken"]


def test_blank_and_punctuation_only_authors_are_dropped() -> None:
    doc = {
        "key": "/works/OL1W",
        "title": "T",
        "author_name": ["  ", "...", "Real Name"],
    }
    assert book_from_search_doc(doc).authors == ["Real Name"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  A wonderful tale.  ", "A wonderful tale."),
        (
            {"type": "/type/text", "value": " Structured description. "},
            "Structured description.",
        ),
        ({"value": 123}, None),
        ("   ", None),
        (None, None),
    ],
)
def test_extract_description(value: object, expected: str | None) -> None:
    assert extract_description(value) == expected


def test_dedupe_books_keeps_the_first_of_identical_books(make_book) -> None:
    books = [
        make_book(key="/works/OL1W", title="Space travel", authors=["Ben Bova"]),
        make_book(key="/works/OL2W", title="Space Travel!", authors=["Ben  Bova"]),
        make_book(key="/works/OL3W", title="Mind Changer", authors=["James White"]),
        make_book(key="/works/OL4W", title="Space travel", authors=["Ben Bova"]),
    ]

    assert [book.key for book in dedupe_books(books)] == [
        "/works/OL1W",
        "/works/OL3W",
    ]


def test_dedupe_books_keeps_same_title_by_different_authors(make_book) -> None:
    books = [
        make_book(
            key="/works/OL1W",
            title="Artificial intelligence",
            authors=["Stuart Russell"],
        ),
        make_book(
            key="/works/OL2W",
            title="Artificial intelligence",
            authors=["Patrick Winston"],
        ),
    ]

    assert len(dedupe_books(books)) == 2


def test_dedupe_books_treats_missing_authors_as_equal(make_book) -> None:
    books = [
        make_book(key="/works/OL1W", title="Untitled", authors=[]),
        make_book(key="/works/OL2W", title="Untitled", authors=[]),
    ]

    assert [book.key for book in dedupe_books(books)] == ["/works/OL1W"]
