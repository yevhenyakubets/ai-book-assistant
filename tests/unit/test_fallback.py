from ai_book_assistant.models import Book
from ai_book_assistant.search.fallback import (
    candidate_queries,
    search_with_fallback,
)


class FakeClient:
    def __init__(self, counts):
        self.counts = list(counts)
        self.calls = []

    def search_books(self, params, limit=5):
        self.calls.append(params)
        count = self.counts.pop(0) if self.counts else 0
        return [Book(title=f"B{i}", key=f"/works/OL{i}W") for i in range(count)]


def names(criteria, raw="query"):
    return [name for name, _ in candidate_queries(criteria, raw)]


def test_full_query_stops_the_ladder_early(make_criteria):
    client = FakeClient([3])
    outcome = search_with_fallback(make_criteria(subjects=["fantasy"]), "q", client)
    assert outcome.rung == "full"
    assert not outcome.relaxed
    assert len(client.calls) == 1


def test_dune_style_query_falls_back_to_primary_subject(make_criteria):
    criteria = make_criteria(
        subjects=["science fiction", "space opera"], keywords=["desert", "politics"]
    )
    client = FakeClient([0, 0, 2])
    outcome = search_with_fallback(criteria, "like dune", client)
    assert outcome.rung == "primary_subject"
    assert outcome.relaxed
    assert outcome.attempts == [
        ("full", 0),
        ("without_keywords", 0),
        ("primary_subject", 2),
    ]


def test_duplicate_rungs_are_skipped(make_criteria):
    assert names(make_criteria(subjects=["fantasy"]), raw="") == ["full"]


def test_years_are_dropped_last(make_criteria):
    criteria = make_criteria(subjects=["fantasy"], year_min=2011)
    assert names(criteria, raw="") == ["full", "ignoring_years"]


def test_page_filter_is_dropped_before_other_search_constraints(make_criteria):
    criteria = make_criteria(
        subjects=["fantasy"],
        keywords=["dragons"],
        preferences={"length": "short"},
    )

    assert names(criteria, raw="") == [
        "full",
        "without_length_filter",
        "without_keywords",
    ]


def test_short_filter_falls_back_when_page_metadata_is_missing(make_criteria):
    criteria = make_criteria(subjects=["fantasy"], preferences={"length": "short"})
    client = FakeClient([0, 2])

    outcome = search_with_fallback(criteria, "fantasy", client)

    assert outcome.rung == "without_length_filter"
    assert outcome.relaxed
    assert outcome.length_filter_applied is False
    assert "number_of_pages_median" in client.calls[0]["q"]
    assert "number_of_pages_median" not in client.calls[1]["q"]


def test_full_short_query_records_that_the_filter_was_applied(make_criteria):
    criteria = make_criteria(subjects=["fantasy"], preferences={"length": "short"})

    outcome = search_with_fallback(criteria, "fantasy", FakeClient([2]))

    assert outcome.rung == "full"
    assert outcome.length_filter_applied is True


def test_keywords_only_never_runs_a_years_only_query(make_criteria):
    criteria = make_criteria(keywords=["x"], year_min=2000)
    assert names(criteria) == ["full", "raw_query"]


def test_empty_criteria_asks_for_clarification_without_searching(make_criteria):
    client = FakeClient([4])
    outcome = search_with_fallback(make_criteria(), "give me a good book", client)
    assert outcome.needs_clarification
    assert outcome.books == []
    assert client.calls == []


def test_years_only_criteria_asks_for_clarification(make_criteria):
    client = FakeClient([4])
    criteria = make_criteria(year_min=1980, year_max=1989)
    outcome = search_with_fallback(criteria, "from the eighties", client)
    assert outcome.needs_clarification
    assert client.calls == []


def test_nothing_found_anywhere(make_criteria):
    outcome = search_with_fallback(make_criteria(subjects=["x"]), "x", FakeClient([]))
    assert outcome.books == []
    assert outcome.rung == "none"
    assert outcome.params is None
