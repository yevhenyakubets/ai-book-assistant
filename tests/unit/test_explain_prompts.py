import json

from ai_book_assistant.explain.prompts import (
    MAX_DESCRIPTION_CHARS,
    MAX_SUBJECTS,
    build_explanation_prompt,
)
from ai_book_assistant.models import Book, SearchCriteria


def _book(n: int, **kwargs) -> Book:
    return Book(key=f"/works/OL{n}W", title=f"Book {n}", **kwargs)


def _payload(prompt: str) -> dict:
    """The JSON input that follows the last 'Input:' marker."""
    text = prompt.rsplit("Input:\n", 1)[1].removesuffix("\nOutput:")
    return json.loads(text)


def _prompt(books, query="q", criteria=None, **kwargs) -> str:
    return build_explanation_prompt(
        query, criteria or SearchCriteria(), books, **kwargs
    )


def test_books_get_sequential_ids_and_no_open_library_keys():
    prompt = _prompt([_book(7), _book(8)])

    assert [b["id"] for b in _payload(prompt)["books"]] == [1, 2]
    assert "/works/OL7W" not in prompt


def test_long_descriptions_are_shortened():
    book = _book(1, description="word " * 500)

    description = _payload(_prompt([book]))["books"][0]["description"]

    assert len(description) <= MAX_DESCRIPTION_CHARS
    assert description.endswith("...")


def test_missing_description_is_null():
    assert _payload(_prompt([_book(1)]))["books"][0]["description"] is None


def test_subjects_are_capped():
    book = _book(1, subjects=[f"s{i}" for i in range(20)])

    subjects = _payload(_prompt([book]))["books"][0]["subjects"]

    assert len(subjects) == MAX_SUBJECTS


def test_searched_fields_and_preferences_are_kept_apart():
    criteria = SearchCriteria.model_validate(
        {
            "subjects": ["fantasy"],
            "year_min": 2000,
            "preferences": {"tone": ["dark"], "audience": "adults"},
        }
    )

    payload = _payload(_prompt([_book(1)], criteria=criteria, relaxed=True))

    assert payload["search"]["subjects"] == ["fantasy"]
    assert "preferences" not in payload["search"]
    assert payload["preferences"]["tone"] == ["dark"]
    assert payload["search_loosened"] is True


def test_the_query_is_passed_through_unchanged():
    query = 'книги про "AI"\nдля початківців'

    assert _payload(_prompt([_book(1)], query=query))["request"] == query


def test_prompt_ends_waiting_for_the_output():
    assert _prompt([_book(1)]).endswith("Output:")


def test_length_is_reported_as_unverifiable_not_as_a_preference():
    criteria = SearchCriteria.model_validate(
        {"subjects": ["fantasy"], "preferences": {"length": "short"}}
    )

    payload = _payload(_prompt([_book(1)], criteria=criteria))

    assert payload["cannot_be_checked"] == ["length: short"]
    assert "length" not in payload["preferences"]


def test_nothing_is_unverifiable_when_no_length_was_asked():
    assert _payload(_prompt([_book(1)]))["cannot_be_checked"] == []
