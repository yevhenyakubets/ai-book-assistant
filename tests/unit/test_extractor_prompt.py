from ai_book_assistant.llm.extractor import CriteriaExtractor
from ai_book_assistant.llm.prompts import EXAMPLES
from ai_book_assistant.models import SearchCriteria


def build(query: str) -> str:
    return CriteriaExtractor(llm_client=None)._build_prompt(query)


def test_prompt_contains_the_user_query():
    assert "Books about whales" in build("Books about whales")


def test_prompt_escapes_quotes_in_the_query():
    prompt = build('a "tricky" {query}')
    assert '\\"tricky\\"' in prompt


def test_prompt_ends_waiting_for_json():
    assert build("anything").rstrip().endswith("JSON:")


def test_examples_are_valid_criteria():
    for _, fields in EXAMPLES:
        SearchCriteria.model_validate(fields)


def test_prompt_maps_explicit_300_page_limit_to_short_length():
    assert "no more than 300 pages" in build("A fantasy novel no more than 300 pages")
