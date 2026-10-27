import math
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


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _rating_average(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        rating = float(value)
        if math.isfinite(rating) and 0 <= rating <= 5:
            return rating
    return None


def _string_values(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            cleaned
            for item in value
            if isinstance(item, str) and (cleaned := collapse_whitespace(item))
        )
    )


def _first_sentence(value: object) -> str | None:
    sentences = [value] if isinstance(value, str) else value
    if not isinstance(sentences, list):
        return None
    return next(
        (
            cleaned
            for sentence in sentences
            if isinstance(sentence, str) and (cleaned := collapse_whitespace(sentence))
        ),
        None,
    )


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

    first_sentence = _first_sentence(doc.get("first_sentence"))

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
        number_of_pages_median=_nonnegative_int(doc.get("number_of_pages_median")),
        ratings_average=_rating_average(doc.get("ratings_average")),
        ratings_count=_nonnegative_int(doc.get("ratings_count")),
        want_to_read_count=_nonnegative_int(doc.get("want_to_read_count")),
        languages=_string_values(doc.get("language")),
        first_sentence=first_sentence,
    )
