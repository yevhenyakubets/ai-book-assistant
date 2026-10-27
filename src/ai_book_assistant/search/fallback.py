from dataclasses import dataclass, field
from typing import Protocol

from ai_book_assistant.models import Book, SearchCriteria
from ai_book_assistant.models.strategy import FULL, NEEDS_CLARIFICATION, NO_RESULTS
from ai_book_assistant.search.query_builder import (
    build_raw_params,
    build_search_params,
)

RAW_QUERY_RUNG = "raw_query"


class BookSearcher(Protocol):
    def search_books(self, query_or_params: dict, limit: int = 5) -> list[Book]: ...


@dataclass(frozen=True)
class Rung:
    name: str
    include_keywords: bool = True
    include_years: bool = True
    include_length: bool = True
    max_subjects: int | None = None


# Strictest first. Page count is relaxed first because Open Library has
# incomplete page metadata; keywords are less reliable than subjects, and
# explicit year constraints are relaxed last.
RUNGS = (
    Rung(FULL),
    Rung("without_length_filter", include_length=False),
    Rung("without_keywords", include_keywords=False, include_length=False),
    Rung(
        "primary_subject",
        include_keywords=False,
        include_length=False,
        max_subjects=1,
    ),
    Rung(
        "ignoring_years",
        include_keywords=False,
        include_years=False,
        include_length=False,
        max_subjects=1,
    ),
)


@dataclass
class SearchOutcome:
    books: list[Book]
    rung: str
    params: dict | None
    attempts: list[tuple[str, int]] = field(default_factory=list)
    needs_clarification: bool = False
    length_filter_applied: bool = False

    @property
    def relaxed(self) -> bool:
        """True when the search had to be loosened from the full criteria."""
        return self.rung not in (FULL, NO_RESULTS, NEEDS_CLARIFICATION)


def _is_searchable(criteria: SearchCriteria, rung: Rung) -> bool:
    # A year range alone would return random books, so require real content.
    return bool(
        criteria.subjects
        or criteria.author
        or (rung.include_keywords and criteria.keywords)
    )


def candidate_queries(
    criteria: SearchCriteria, raw_query: str
) -> list[tuple[str, dict]]:
    """Ordered (rung name, params) pairs, strictest first, duplicates removed."""
    candidates: list[tuple[str, dict]] = []
    seen: set[tuple] = set()

    def add(name: str, params: dict | None) -> None:
        if params is None:
            return
        fingerprint = tuple(sorted(params.items()))
        if fingerprint not in seen:
            seen.add(fingerprint)
            candidates.append((name, params))

    for rung in RUNGS:
        if _is_searchable(criteria, rung):
            add(
                rung.name,
                build_search_params(
                    criteria,
                    include_keywords=rung.include_keywords,
                    include_years=rung.include_years,
                    include_length=rung.include_length,
                    max_subjects=rung.max_subjects,
                ),
            )
    add(RAW_QUERY_RUNG, build_raw_params(raw_query, criteria.language))
    return candidates


def search_with_fallback(
    criteria: SearchCriteria,
    raw_query: str,
    client: BookSearcher,
    limit: int = 5,
    min_results: int = 1,
) -> SearchOutcome:
    attempts: list[tuple[str, int]] = []
    if not criteria.is_searchable:
        return SearchOutcome(
            [], NEEDS_CLARIFICATION, None, [], needs_clarification=True
        )
    for name, params in candidate_queries(criteria, raw_query):
        books = client.search_books(params, limit)
        attempts.append((name, len(books)))
        if len(books) >= min_results:
            return SearchOutcome(
                books,
                name,
                params,
                attempts,
                length_filter_applied=(
                    criteria.preferences.length == "short" and name == FULL
                ),
            )
    return SearchOutcome([], NO_RESULTS, None, attempts)
