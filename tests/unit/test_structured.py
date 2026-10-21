import pytest

from ai_book_assistant.llm.structured import StructuredGenerator, load_json


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def must_have_name(data: object) -> str:
    if not isinstance(data, dict) or "name" not in data:
        raise ValueError("missing name")
    return data["name"]


def run(llm, **kwargs):
    generator = StructuredGenerator(llm, **kwargs)
    return generator.generate("PROMPT", must_have_name, task="Test task")


def test_valid_response_needs_one_call():
    llm = FakeLLM(['{"name": "x"}'])

    assert run(llm) == "x"
    assert llm.prompts == ["PROMPT"]


def test_invalid_json_is_retried_with_the_reason():
    llm = FakeLLM(["not json", '{"name": "x"}'])

    assert run(llm) == "x"
    assert len(llm.prompts) == 2
    assert llm.prompts[1].startswith("PROMPT")
    assert "rejected" in llm.prompts[1]


def test_validation_errors_are_retried_with_their_message():
    llm = FakeLLM(['{"other": 1}', '{"name": "x"}'])

    assert run(llm) == "x"
    assert "missing name" in llm.prompts[1]


def test_empty_response_is_retried():
    assert run(FakeLLM([None, '{"name": "x"}'])) == "x"


def test_returns_none_when_every_attempt_is_rejected():
    llm = FakeLLM(["garbage", "still garbage"])

    assert run(llm) is None
    assert len(llm.prompts) == 2


def test_max_attempts_is_respected():
    llm = FakeLLM(["garbage", "garbage", '{"name": "x"}'])

    assert run(llm, max_attempts=1) is None
    assert len(llm.prompts) == 1


def test_provider_errors_are_not_swallowed():
    class BrokenLLM:
        def generate(self, prompt):
            raise RuntimeError("503")

    with pytest.raises(RuntimeError):
        run(BrokenLLM())


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        StructuredGenerator(FakeLLM([]), max_attempts=0)


@pytest.mark.parametrize("bad", [None, "", "   ", 42])
def test_load_json_rejects_blank_and_non_text(bad):
    with pytest.raises(ValueError):
        load_json(bad)
