import re
from collections.abc import Mapping

from ai_book_assistant.models import Book
from ai_book_assistant.text import collapse_whitespace


def _fingerprint(text: str) -> str:
    """Lowercase text without spacing or punctuation, for comparing names."""
    return re.sub(r"[\W_]+", "", text).casefold()


def dedupe_names(names: list[str]) -> list[str]:
    """Drop repeated names, ignoring case, spacing and punctuation.

    Open Library often lists the same person twice, for example
    "G. A. Aiken" and "G.A. Aiken". The first spelling is kept.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        cleaned = collapse_whitespace(name)
        fingerprint = _fingerprint(cleaned)
        if cleaned and fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(cleaned)
    return unique


def dedupe_books(books: list[Book]) -> list[Book]:
    """Drop books that repeat an earlier title and first author.

    Open Library often has several works for the same book. The first
    one, which is the most relevant, is kept.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Book] = []
    for book in books:
        first_author = book.authors[0] if book.authors else ""
        fingerprint = (_fingerprint(book.title), _fingerprint(first_author))
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(book)
    return unique


def extract_description(value: object) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    return value.strip() or None if isinstance(value, str) else None


def book_from_search_doc(doc: Mapping[str, object]) -> Book:
    title = doc.get("title")
    if not isinstance(title, str) or not title.strip():
        title = "Unknown Title"

    cover_id = doc.get("cover_i")
    covers = [cover_id] if type(cover_id) is int else []

    authors = doc.get("author_name")
    if isinstance(authors, list):
        authors = dedupe_names(
            [author for author in authors if isinstance(author, str)]
        )
    else:
        authors = []

    subjects = doc.get("subject")
    if isinstance(subjects, list):
        subjects = [subject for subject in subjects if isinstance(subject, str)]
    else:
        subjects = []

    year = doc.get("first_publish_year")
    if type(year) is not int:
        year = None

    key = doc.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Open Library search document must contain a work key")

    return Book(
        title=title,
        authors=authors,
        first_publish_year=year,
        key=key,
        subjects=subjects,
        covers=covers,
    )
