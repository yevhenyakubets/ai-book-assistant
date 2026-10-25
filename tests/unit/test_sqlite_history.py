from datetime import UTC, datetime

import pytest

from ai_book_assistant.models import Explanations, SearchCriteria
from ai_book_assistant.storage import SqliteSearchHistory

START = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
EXPLANATIONS = Explanations(summary="Overall.", by_book={"/works/OL1W": "It fits."})


@pytest.fixture
def hobbit(make_book):
    return make_book(
        key="/works/OL1W",
        title="The Hobbit",
        authors=["J.R.R. Tolkien"],
        first_publish_year=1937,
        subjects=["Fantasy"],
        description="A hobbit goes on an adventure.",
    )


def _save(
    history,
    hobbit,
    query="fantasy",
    strategy="full",
    books=None,
    explanations=EXPLANATIONS,
):
    if books is None:
        books = (hobbit,)
    return history.add(
        query=query,
        strategy=strategy,
        criteria=SearchCriteria(subjects=["fantasy"]),
        books=list(books),
        explanations=explanations,
    )


def test_add_returns_the_saved_record(history, hobbit):
    record = _save(history, hobbit)

    assert record.id == 1
    assert record.created_at == START
    assert record.query == "fantasy"
    assert record.books == [hobbit]


def test_get_returns_exactly_what_was_saved(history, hobbit):
    saved = _save(history, hobbit)

    assert history.get(saved.id) == saved


def test_non_ascii_text_survives_a_round_trip(history, hobbit):
    saved = _save(history, hobbit, query="Книги про штучний інтелект")

    assert history.get(saved.id).query == "Книги про штучний інтелект"


def test_unknown_id_returns_none(history):
    assert history.get(999) is None


def test_list_recent_is_newest_first_and_omits_the_snapshot(history, hobbit):
    for query in ("one", "two", "three"):
        _save(history, hobbit, query=query)

    entries = history.list_recent()

    assert [entry.query for entry in entries] == ["three", "two", "one"]
    assert not hasattr(entries[0], "books")


def test_list_recent_supports_paging(history, hobbit):
    for number in range(5):
        _save(history, hobbit, query=f"query {number}")

    entries = history.list_recent(limit=2, offset=2)

    assert [entry.query for entry in entries] == ["query 2", "query 1"]


@pytest.mark.parametrize("limit, offset", [(0, 0), (-1, 0), (5, -1)])
def test_list_recent_rejects_bad_paging(history, limit, offset):
    with pytest.raises(ValueError):
        history.list_recent(limit=limit, offset=offset)


def test_searches_without_results_are_saved_too(history, hobbit):
    saved = _save(
        history,
        hobbit,
        strategy="needs_clarification",
        books=[],
        explanations=Explanations(),
    )

    loaded = history.get(saved.id)

    assert loaded.strategy == "needs_clarification"
    assert loaded.books == []
    assert loaded.explanations == Explanations()


def test_history_survives_reopening_the_database(tmp_path, hobbit, fake_clock):
    path = tmp_path / "history.db"
    saved = _save(SqliteSearchHistory(path, clock=fake_clock), hobbit)

    assert SqliteSearchHistory(path).get(saved.id) == saved


def test_missing_parent_directories_are_created(tmp_path, hobbit):
    history = SqliteSearchHistory(tmp_path / "data" / "nested" / "history.db")

    assert _save(history, hobbit).id == 1


def test_quotes_and_sql_in_the_query_are_stored_as_plain_text(history, hobbit):
    nasty = "'; DROP TABLE searches; --"

    saved = _save(history, hobbit, query=nasty)

    assert history.get(saved.id).query == nasty
    assert len(history.list_recent()) == 1
