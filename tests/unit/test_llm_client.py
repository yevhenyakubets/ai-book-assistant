import logging
from types import SimpleNamespace

import httpx
import openai
import pytest

from ai_book_assistant.llm.client import (
    DEFAULT_MODEL,
    MAX_OUTPUT_TOKENS,
    OpenAILLMClient,
)
from ai_book_assistant.llm.errors import LLMError

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def fake_openai(result):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, calls


def reply(text, finish_reason="stop", usage=None):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason=finish_reason
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def status_error(cls, status, body=None):
    response = httpx.Response(status, request=REQUEST)
    return cls("boom", response=response, body=body)


def test_returns_the_model_text_with_json_mode_and_an_output_cap():
    client, calls = fake_openai(reply('{"subjects": []}'))

    assert OpenAILLMClient(client=client).generate("hi") == '{"subjects": []}'

    sent = calls[0]
    assert sent["model"] == DEFAULT_MODEL
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["max_completion_tokens"] == MAX_OUTPUT_TOKENS
    assert "temperature" not in sent


def test_rate_limit_becomes_llm_error_with_status():
    client, _ = fake_openai(status_error(openai.RateLimitError, 429))

    with pytest.raises(LLMError) as info:
        OpenAILLMClient(client=client).generate("hi")

    assert info.value.status_code == 429


def test_an_empty_balance_is_not_reported_as_a_rate_limit():
    error = status_error(
        openai.RateLimitError, 429, body={"code": "insufficient_quota"}
    )
    client, _ = fake_openai(error)

    with pytest.raises(LLMError) as info:
        OpenAILLMClient(client=client).generate("hi")

    assert info.value.status_code == 402


def test_bad_key_becomes_llm_error_with_status():
    client, _ = fake_openai(status_error(openai.AuthenticationError, 401))

    with pytest.raises(LLMError) as info:
        OpenAILLMClient(client=client).generate("hi")

    assert info.value.status_code == 401


def test_connection_problem_becomes_llm_error_without_status():
    client, _ = fake_openai(openai.APIConnectionError(request=REQUEST))

    with pytest.raises(LLMError) as info:
        OpenAILLMClient(client=client).generate("hi")

    assert info.value.status_code is None


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAILLMClient()


def test_model_comes_from_the_argument_then_the_environment(monkeypatch):
    client, _ = fake_openai(reply("{}"))

    monkeypatch.setenv("OPENAI_MODEL", "from-env")
    assert OpenAILLMClient(client=client).model == "from-env"
    assert OpenAILLMClient("explicit", client=client).model == "explicit"

    monkeypatch.delenv("OPENAI_MODEL")
    assert OpenAILLMClient(client=client).model == DEFAULT_MODEL


def test_token_usage_is_logged(caplog):
    caplog.set_level(logging.INFO, logger="ai_book_assistant.llm.client")
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
    )
    client, _ = fake_openai(reply("{}", usage=usage))

    OpenAILLMClient(client=client).generate("hi")

    assert "100 in, 20 out, 5 reasoning" in caplog.text


def test_hitting_the_output_limit_is_warned_about(caplog):
    caplog.set_level(logging.WARNING, logger="ai_book_assistant.llm.client")
    client, _ = fake_openai(reply("{", finish_reason="length"))

    OpenAILLMClient(client=client).generate("hi")

    assert "output limit" in caplog.text
