import json

import pytest

from ai_book_assistant.explain import BookExplainer
from ai_book_assistant.models import Book, SearchCriteria

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


def reply(*items):
    entries = [{"id": i, "explanation": text} for i, text in items]
    return json.dumps({"explanations": entries})


def explain(llm, books=BOOKS, **kwargs):
    return BookExplainer(llm).explain("a query", CRITERIA, books, **kwargs)


def test_explanations_are_mapped_to_book_keys():
    llm = FakeLLM([reply((1, "First."), (2, "Second."))])

    assert explain(llm) == {"/works/OL1W": "First.", "/works/OL2W": "Second."}
    assert len(llm.prompts) == 1


def test_a_partial_answer_is_accepted():
    llm = FakeLLM([reply((2, "Only the second."))])

    assert explain(llm) == {"/works/OL2W": "Only the second."}


def test_unknown_ids_blanks_and_duplicates_are_ignored():
    answer = reply(
        (1, "  Spaced   out. "),
        (1, "Duplicate."),
        (2, "   "),
        (9, "Unknown."),
    )

    assert explain(FakeLLM([answer])) == {"/works/OL1W": "Spaced out."}


def test_string_ids_are_accepted():
    answer = json.dumps({"explanations": [{"id": "1", "explanation": "First."}]})

    assert explain(FakeLLM([answer])) == {"/works/OL1W": "First."}


def test_bad_output_is_retried():
    llm = FakeLLM(["not json", reply((1, "First."))])

    assert explain(llm) == {"/works/OL1W": "First."}
    assert "rejected" in llm.prompts[1]


def test_an_answer_without_usable_explanations_is_retried():
    llm = FakeLLM([reply((7, "Wrong id.")), reply((1, "First."))])

    assert explain(llm) == {"/works/OL1W": "First."}


def test_gives_back_nothing_when_every_attempt_fails():
    llm = FakeLLM(["garbage", "still garbage"])

    assert explain(llm) == {}
    assert len(llm.prompts) == 2


def test_no_books_means_no_llm_call():
    llm = FakeLLM([])

    assert explain(llm, books=[]) == {}
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
