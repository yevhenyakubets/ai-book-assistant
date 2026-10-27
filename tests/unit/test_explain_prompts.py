import json

from ai_book_assistant.explain.prompts import (
    MAX_DESCRIPTION_CHARS,
    MAX_SUBJECTS,
    build_explanation_prompt,
)
from ai_book_assistant.models import SearchCriteria


def _payload(prompt: str) -> dict:
    """The JSON input that follows the last 'Input:' marker."""
    text = prompt.rsplit("Input:\n", 1)[1].removesuffix("\nOutput:")
    return json.loads(text)


def _prompt(books, query="q", criteria=None, **kwargs) -> str:
    return build_explanation_prompt(
        query, criteria or SearchCriteria(), books, **kwargs
    )


def test_books_get_sequential_ids_and_no_open_library_keys(make_book):
    prompt = _prompt(
        [
            make_book(key="/works/OL7W", title="Book 7"),
            make_book(key="/works/OL8W", title="Book 8"),
        ]
    )

    assert [b["id"] for b in _payload(prompt)["books"]] == [1, 2]
    assert "/works/OL7W" not in prompt


def test_long_descriptions_are_shortened(make_book):
    book = make_book(description="word " * 500)

    description = _payload(_prompt([book]))["books"][0]["description"]

    assert len(description) <= MAX_DESCRIPTION_CHARS
    assert description.endswith("...")


def test_missing_description_is_null(make_book):
    assert _payload(_prompt([make_book()]))["books"][0]["description"] is None


def test_richer_open_library_metadata_is_in_the_explanation_prompt(make_book):
    book = make_book(
        number_of_pages_median=240,
        ratings_average=4.25,
        ratings_count=120,
        want_to_read_count=350,
        languages=["eng", "fre"],
        covers=[123],
        first_sentence="It began at dawn.",
    )

    data = _payload(_prompt([book]))["books"][0]

    assert data["number_of_pages_median"] == 240
    assert data["ratings_average"] == 4.25
    assert data["ratings_count"] == 120
    assert data["want_to_read_count"] == 350
    assert data["languages"] == ["eng", "fre"]
    assert data["cover_url"] == "https://covers.openlibrary.org/b/id/123-M.jpg"
    assert data["first_sentence"] == "It began at dawn."


def test_subjects_are_capped(make_book):
    book = make_book(subjects=[f"s{i}" for i in range(20)])

    subjects = _payload(_prompt([book]))["books"][0]["subjects"]

    assert len(subjects) == MAX_SUBJECTS


def test_searched_fields_and_preferences_are_kept_apart(make_book):
    criteria = SearchCriteria.model_validate(
        {
            "subjects": ["fantasy"],
            "year_min": 2000,
            "preferences": {"tone": ["dark"], "audience": "adults"},
        }
    )

    payload = _payload(_prompt([make_book()], criteria=criteria, relaxed=True))

    assert payload["search"]["subjects"] == ["fantasy"]
    assert "preferences" not in payload["search"]
    assert payload["preferences"]["tone"] == ["dark"]
    assert payload["search_loosened"] is True


def test_the_query_is_passed_through_unchanged(make_book):
    query = 'книги про "AI"\nдля початківців'

    assert _payload(_prompt([make_book()], query=query))["request"] == query


def test_prompt_ends_waiting_for_the_output(make_book):
    assert _prompt([make_book()]).endswith("Output:")


def test_length_is_reported_as_unverifiable_not_as_a_preference(make_book):
    criteria = SearchCriteria.model_validate(
        {"subjects": ["fantasy"], "preferences": {"length": "short"}}
    )

    payload = _payload(_prompt([make_book()], criteria=criteria))

    assert payload["cannot_be_checked"] == ["length: short"]
    assert "length" not in payload["preferences"]


def test_nothing_is_unverifiable_when_no_length_was_asked(make_book):
    assert _payload(_prompt([make_book()]))["cannot_be_checked"] == []
