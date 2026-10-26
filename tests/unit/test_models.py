import pytest

from ai_book_assistant.models import Book


def test_book_requires_non_empty_key() -> None:
    with pytest.raises(ValueError):
        Book(title="Test", key="")


def test_book_assignment_is_validated() -> None:
    book = Book(title="Test", key="/works/OL123W")

    with pytest.raises(ValueError):
        book.key = ""
