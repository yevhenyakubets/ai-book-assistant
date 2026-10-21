from ai_book_assistant.models import Book, SearchCriteria
from ai_book_assistant.search.fallback import (
    candidate_queries,
    search_with_fallback,
)


def make(**kwargs) -> SearchCriteria:
    return SearchCriteria.model_validate(kwargs)


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


def test_full_query_stops_the_ladder_early():
    client = FakeClient([3])
    outcome = search_with_fallback(make(subjects=["fantasy"]), "q", client)
    assert outcome.rung == "full"
    assert not outcome.relaxed
    assert len(client.calls) == 1


def test_dune_style_query_falls_back_to_primary_subject():
    criteria = make(
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


def test_duplicate_rungs_are_skipped():
    assert names(make(subjects=["fantasy"]), raw="") == ["full"]


def test_years_are_dropped_last():
    criteria = make(subjects=["fantasy"], year_min=2011)
    assert names(criteria, raw="") == ["full", "ignoring_years"]


def test_keywords_only_never_runs_a_years_only_query():
    criteria = make(keywords=["x"], year_min=2000)
    assert names(criteria) == ["full", "raw_query"]


def test_empty_criteria_asks_for_clarification_without_searching():
    client = FakeClient([4])
    outcome = search_with_fallback(make(), "give me a good book", client)
    assert outcome.needs_clarification
    assert outcome.books == []
    assert client.calls == []


def test_years_only_criteria_asks_for_clarification():
    client = FakeClient([4])
    criteria = make(year_min=1980, year_max=1989)
    outcome = search_with_fallback(criteria, "from the eighties", client)
    assert outcome.needs_clarification
    assert client.calls == []


def test_nothing_found_anywhere():
    outcome = search_with_fallback(make(subjects=["x"]), "x", FakeClient([]))
    assert outcome.books == []
    assert outcome.rung == "none"
    assert outcome.params is None
