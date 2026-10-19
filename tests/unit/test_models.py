import pytest

from ai_book_assistant.models import Book


def test_book_from_search_doc(sample_search_doc: dict[str, object]) -> None:
    book = Book.from_search_doc(sample_search_doc)

    assert book.title == "Foundation"
    assert book.authors == ["Isaac Asimov"]
    assert book.first_publish_year == 1951
    assert book.key == "/works/OL46125W"
    assert book.subjects == ["Science Fiction"]
    assert book.covers == [14612610]


def test_book_from_search_doc_messy_types() -> None:
    doc = {
        "title": "Messy Book",
        "author_name": "Single Author String (Invalid Type)",
        "first_publish_year": "ancient",
        "subject": "Fiction (Invalid Type)",
        "cover_i": "not-an-int",
        "key": "/works/OL999W",
    }
    book = Book.from_search_doc(doc)

    assert book.title == "Messy Book"
    assert book.authors == []
    assert book.first_publish_year is None
    assert book.subjects == []
    assert book.covers == []
    assert book.key == "/works/OL999W"


def test_book_update_from_work_data_string_description() -> None:
    book = Book(title="Test", key="/works/OL123W")
    work_data = {"description": "A wonderful tale."}

    book.update_from_work_data(work_data)
    assert book.description == "A wonderful tale."


def test_book_update_from_work_data_dict_description() -> None:
    book = Book(title="Test", key="/works/OL123W")
    work_data = {
        "description": {
            "type": "/type/text",
            "value": "Structured description.",
        }
    }

    book.update_from_work_data(work_data)
    assert book.description == "Structured description."


def test_book_from_search_doc_requires_work_key() -> None:
    with pytest.raises(ValueError, match="must contain a work key"):
        Book.from_search_doc({"title": "Missing key"})


def test_book_update_from_work_data_normalizes_invalid_description() -> None:
    book = Book(title="Test", key="/works/OL123W", description="Existing")

    book.update_from_work_data({"description": {"value": 123}})

    assert book.description is None
