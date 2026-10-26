from ai_book_assistant.search.query_builder import SEARCH_FIELDS, build_search_params


def test_subject_and_year_range(make_criteria):
    params = build_search_params(make_criteria(subjects=["fantasy"], year_max=1999))
    assert params == {
        "q": 'subject:"fantasy" AND first_publish_year:[* TO 1999]',
        "fields": SEARCH_FIELDS,
    }


def test_open_ended_lower_bound(make_criteria):
    params = build_search_params(
        make_criteria(subjects=["science fiction"], year_min=2015)
    )
    assert params["q"] == (
        'subject:"science fiction" AND first_publish_year:[2015 TO *]'
    )


def test_multiple_subjects_are_anded(make_criteria):
    params = build_search_params(
        make_criteria(subjects=["science fiction", "artificial intelligence"])
    )
    assert params["q"] == (
        'subject:"science fiction" AND subject:"artificial intelligence"'
    )


def test_keywords_quote_phrases_only(make_criteria):
    params = build_search_params(make_criteria(keywords=["climate change", "wind"]))
    assert params["q"] == '"climate change" AND wind'


def test_author_only_uses_separate_param(make_criteria):
    params = build_search_params(make_criteria(author="Terry Pratchett"))
    assert params == {"author": "Terry Pratchett", "fields": SEARCH_FIELDS}


def test_empty_criteria_returns_none(make_criteria):
    assert build_search_params(make_criteria()) is None


def test_flags_loosen_the_query(make_criteria):
    criteria = make_criteria(
        subjects=["fantasy", "romance"], keywords=["dragons"], year_min=2000
    )
    params = build_search_params(
        criteria, include_keywords=False, include_years=False, max_subjects=1
    )
    assert params["q"] == 'subject:"fantasy"'


def test_special_characters_are_stripped(make_criteria):
    params = build_search_params(make_criteria(subjects=['sci"fi: (x)']))
    assert params["q"] == 'subject:"sci fi x"'
