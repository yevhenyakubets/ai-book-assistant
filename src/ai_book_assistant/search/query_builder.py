import re

from ai_book_assistant.models import SearchCriteria
from ai_book_assistant.models.criteria import SHORT_MAX_PAGES
from ai_book_assistant.text import collapse_whitespace

SEARCH_FIELDS = (
    "key,title,author_name,first_publish_year,subject,cover_i,"
    "number_of_pages_median,ratings_average,ratings_count,want_to_read_count,"
    "language,first_sentence"
)

# Characters with special meaning in Open Library's (Solr) query syntax.
_UNSAFE_CHARS = re.compile(r'[\\"():\[\]{}^~*?!/|&]')


def _clean(term: str) -> str:
    return collapse_whitespace(_UNSAFE_CHARS.sub(" ", term))


def _keyword_clause(keyword: str) -> str:
    return f'"{keyword}"' if " " in keyword else keyword


def _year_clause(year_min: int | None, year_max: int | None) -> str | None:
    if year_min is None and year_max is None:
        return None
    low = "*" if year_min is None else year_min
    high = "*" if year_max is None else year_max
    return f"first_publish_year:[{low} TO {high}]"


def _length_clause(length: str | None) -> str | None:
    if length == "short":
        return f"number_of_pages_median:[1 TO {SHORT_MAX_PAGES}]"
    return None


def _language_clause(language: str | None) -> str | None:
    return f"language:{language}" if language else None


def build_search_params(
    criteria: SearchCriteria,
    *,
    include_keywords: bool = True,
    include_years: bool = True,
    include_length: bool = True,
    max_subjects: int | None = None,
) -> dict[str, str] | None:
    """Turn criteria into Open Library search parameters.

    Returns None when the criteria contain nothing searchable. The flags
    let the fallback ladder loosen the query one piece at a time.
    """
    clauses: list[str] = []

    subjects = criteria.subjects
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    for subject in subjects:
        cleaned = _clean(subject)
        if cleaned:
            clauses.append(f'subject:"{cleaned}"')

    if include_keywords:
        for keyword in criteria.keywords:
            cleaned = _clean(keyword)
            if cleaned:
                clauses.append(_keyword_clause(cleaned))

    if include_years:
        year_clause = _year_clause(criteria.year_min, criteria.year_max)
        if year_clause:
            clauses.append(year_clause)

    if include_length:
        length_clause = _length_clause(criteria.preferences.length)
        if length_clause:
            clauses.append(length_clause)

    language_clause = _language_clause(criteria.language)
    if language_clause:
        clauses.append(language_clause)

    params: dict[str, str] = {}
    if clauses:
        params["q"] = " AND ".join(clauses)

    if criteria.author:
        author = _clean(criteria.author)
        if author:
            params["author"] = author

    if not params:
        return None

    params["fields"] = SEARCH_FIELDS
    return params


def build_raw_params(query: str, language: str | None = None) -> dict[str, str] | None:
    """Last-resort search: the user's own words as a plain query."""
    cleaned = _clean(query)
    if not cleaned:
        return None
    language_clause = _language_clause(language)
    if language_clause:
        cleaned = f"{cleaned} AND {language_clause}"
    return {"q": cleaned, "fields": SEARCH_FIELDS}
