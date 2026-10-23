import pytest

from ai_book_assistant.assistant import BookAssistant
from ai_book_assistant.bootstrap import build_assistant
from ai_book_assistant.explain import BookExplainer
from ai_book_assistant.search.client import OpenLibraryClient


def test_build_assistant_wires_extraction_search_and_explanation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    assistant = build_assistant()

    assert isinstance(assistant, BookAssistant)
    assert isinstance(assistant.catalog, OpenLibraryClient)
    assert isinstance(assistant.explainer, BookExplainer)


def test_build_assistant_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_assistant()
