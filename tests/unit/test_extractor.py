import json

import pytest

from ai_book_assistant.llm.extractor import CriteriaExtractor

GOOD = json.dumps({"subjects": ["fantasy"], "year_max": 1999})


class BrokenLLM:
    def generate(self, prompt):
        raise RuntimeError("503 service unavailable")


def test_valid_response_needs_one_call(make_llm):
    llm = make_llm(GOOD)
    criteria = CriteriaExtractor(llm).extract("fantasy before 2000")
    assert criteria.subjects == ["fantasy"]
    assert criteria.year_max == 1999
    assert len(llm.prompts) == 1


def test_invalid_json_is_retried_with_the_reason(make_llm):
    llm = make_llm("not json at all", GOOD)
    criteria = CriteriaExtractor(llm).extract("fantasy before 2000")
    assert criteria.subjects == ["fantasy"]
    assert len(llm.prompts) == 2
    assert "rejected" in llm.prompts[1]


def test_non_object_json_is_retried(make_llm):
    llm = make_llm("[1, 2, 3]", GOOD)
    criteria = CriteriaExtractor(llm).extract("fantasy before 2000")
    assert criteria.subjects == ["fantasy"]
    assert len(llm.prompts) == 2


def test_empty_response_is_retried(make_llm):
    llm = make_llm(None, GOOD)
    criteria = CriteriaExtractor(llm).extract("fantasy before 2000")
    assert criteria.subjects == ["fantasy"]


def test_two_failures_fall_back_to_the_raw_query(make_llm):
    llm = make_llm("garbage", "still garbage")
    criteria = CriteriaExtractor(llm).extract("tell me something fun")
    assert criteria.keywords == ["tell me something fun"]
    assert criteria.subjects == []
    assert len(llm.prompts) == 2


def test_blank_query_skips_the_llm(make_llm):
    llm = make_llm()
    criteria = CriteriaExtractor(llm).extract("   ")
    assert not criteria.is_searchable
    assert llm.prompts == []


def test_provider_errors_are_not_swallowed():
    with pytest.raises(RuntimeError):
        CriteriaExtractor(BrokenLLM()).extract("fantasy")


def test_max_attempts_must_be_positive(make_llm):
    with pytest.raises(ValueError):
        CriteriaExtractor(make_llm(), max_attempts=0)
