from ai_book_assistant.models import SearchCriteria


def test_normalizes_and_deduplicates_terms():
    c = SearchCriteria.model_validate(
        {"subjects": ["  Dark Fantasy ", "dark fantasy", "Magic"]}
    )
    assert c.subjects == ["dark fantasy", "magic"]


def test_caps_list_sizes():
    c = SearchCriteria.model_validate(
        {"subjects": ["a", "b", "c", "d"], "keywords": ["w", "x", "y", "z"]}
    )
    assert len(c.subjects) == 3
    assert len(c.keywords) == 3


def test_null_values_become_empty_defaults():
    c = SearchCriteria.model_validate(
        {"subjects": None, "keywords": None, "author": None, "preferences": None}
    )
    assert c.subjects == []
    assert c.keywords == []
    assert c.preferences.tone == []


def test_reversed_years_are_swapped():
    c = SearchCriteria.model_validate({"year_min": 2010, "year_max": 1990})
    assert (c.year_min, c.year_max) == (1990, 2010)


def test_implausible_year_is_dropped():
    c = SearchCriteria.model_validate({"year_min": 42, "year_max": "not a year"})
    assert c.year_min is None
    assert c.year_max is None


def test_old_year_key_is_ignored():
    c = SearchCriteria.model_validate({"year": 2010})
    assert c.year_min is None and c.year_max is None
