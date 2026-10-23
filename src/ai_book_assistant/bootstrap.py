from ai_book_assistant.assistant import BookAssistant
from ai_book_assistant.explain import BookExplainer
from ai_book_assistant.llm.client import GroqLLMClient
from ai_book_assistant.llm.extractor import CriteriaExtractor
from ai_book_assistant.search.client import OpenLibraryClient


def build_assistant() -> BookAssistant:
    """The real assistant: Groq for the language model, Open Library for books."""
    llm = GroqLLMClient()
    return BookAssistant(
        CriteriaExtractor(llm),
        OpenLibraryClient(),
        explainer=BookExplainer(llm),
    )
