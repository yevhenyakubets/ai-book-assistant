import json

import pytest

from ai_book_assistant.explain import BookExplainer
from ai_book_assistant.models import Book, Explanations, SearchCriteria

BOOKS = [
    Book(key="/works/OL1W", title="One"),
    Book(key="/works/OL2W", title="Two"),
]
CRITERIA = SearchCriteria(subjects=["fantasy"])


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def reply(*items, summary=None):
    payload = {"explanations": [{"id": i, "explanation": t} for i, t in items]}
    if summary is not None:
        payload["summary"] = summary
    return json.dumps(payload)


def explain(llm, books=BOOKS, **kwargs):
    return BookExplainer(llm).explain("a query", CRITERIA, books, **kwargs)


def test_explanations_are_mapped_to_book_keys():
    llm = FakeLLM([reply((1, "First."), (2, "Second."))])

    result = explain(llm)

    assert result.by_book == {"/works/OL1W": "First.", "/works/OL2W": "Second."}
    assert len(llm.prompts) == 1


def test_the_summary_is_returned_with_whitespace_normalized():
    llm = FakeLLM([reply((1, "First."), summary="  Overall   note. ")])

    assert explain(llm).summary == "Overall note."


def test_a_missing_or_blank_summary_becomes_none():
    assert explain(FakeLLM([reply((1, "First."))])).summary is None
    assert explain(FakeLLM([reply((1, "First."), summary="  ")])).summary is None


def test_non_breaking_hyphens_are_replaced_with_plain_ones():
    answer = reply((1, "A short\u2011story book."), summary="Sci\u2011fi picks.")

    result = explain(FakeLLM([answer]))

    assert result.by_book == {"/works/OL1W": "A short-story book."}
    assert result.summary == "Sci-fi picks."


def test_a_partial_answer_is_accepted():
    llm = FakeLLM([reply((2, "Only the second."))])

    assert explain(llm).by_book == {"/works/OL2W": "Only the second."}


def test_unknown_ids_blanks_and_duplicates_are_ignored():
    answer = reply(
        (1, "  Spaced   out. "),
        (1, "Duplicate."),
        (2, "   "),
        (9, "Unknown."),
    )

    assert explain(FakeLLM([answer])).by_book == {"/works/OL1W": "Spaced out."}


def test_string_ids_are_accepted():
    answer = json.dumps({"explanations": [{"id": "1", "explanation": "First."}]})

    assert explain(FakeLLM([answer])).by_book == {"/works/OL1W": "First."}


def test_bad_output_is_retried():
    llm = FakeLLM(["not json", reply((1, "First."))])

    assert explain(llm).by_book == {"/works/OL1W": "First."}
    assert "rejected" in llm.prompts[1]


def test_an_answer_without_usable_explanations_is_retried():
    llm = FakeLLM([reply((7, "Wrong id.")), reply((1, "First."))])

    assert explain(llm).by_book == {"/works/OL1W": "First."}


def test_gives_back_nothing_when_every_attempt_fails():
    llm = FakeLLM(["garbage", "still garbage"])

    assert explain(llm) == Explanations()
    assert len(llm.prompts) == 2


def test_no_books_means_no_llm_call():
    llm = FakeLLM([])

    assert explain(llm, books=[]) == Explanations()
    assert llm.prompts == []


def test_provider_errors_are_not_swallowed():
    class BrokenLLM:
        def generate(self, prompt):
            raise RuntimeError("503")

    with pytest.raises(RuntimeError):
        explain(BrokenLLM())


def test_relaxed_flag_reaches_the_prompt():
    llm = FakeLLM([reply((1, "First."))])

    explain(llm, relaxed=True)

    assert '"search_loosened": true' in llm.prompts[0]
