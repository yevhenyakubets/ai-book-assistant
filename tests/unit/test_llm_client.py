from types import SimpleNamespace

import groq
import httpx
import pytest

from ai_book_assistant.llm.client import GroqLLMClient
from ai_book_assistant.llm.errors import LLMError

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def fake_groq(result):
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


def reply(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def status_error(cls, status):
    response = httpx.Response(status, request=REQUEST)
    return cls("boom", response=response, body=None)


def test_returns_the_model_text_and_requests_json_mode():
    client, calls = fake_groq(reply('{"subjects": []}'))
    assert GroqLLMClient(client=client).generate("hi") == '{"subjects": []}'
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["temperature"] == 0


def test_rate_limit_becomes_llm_error_with_status():
    client, _ = fake_groq(status_error(groq.RateLimitError, 429))
    with pytest.raises(LLMError) as info:
        GroqLLMClient(client=client).generate("hi")
    assert info.value.status_code == 429


def test_bad_key_becomes_llm_error_with_status():
    client, _ = fake_groq(status_error(groq.AuthenticationError, 401))
    with pytest.raises(LLMError) as info:
        GroqLLMClient(client=client).generate("hi")
    assert info.value.status_code == 401


def test_connection_problem_becomes_llm_error_without_status():
    client, _ = fake_groq(groq.APIConnectionError(request=REQUEST))
    with pytest.raises(LLMError) as info:
        GroqLLMClient(client=client).generate("hi")
    assert info.value.status_code is None


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GroqLLMClient()
