from ai_book_assistant.assistant import BookAssistant
from ai_book_assistant.explain import BookExplainer
from ai_book_assistant.llm.client import OpenAILLMClient
from ai_book_assistant.llm.extractor import CriteriaExtractor
from ai_book_assistant.search.client import OpenLibraryClient


def build_assistant() -> BookAssistant:
    """The real assistant: OpenAI for the language model, Open Library for books."""
    llm = OpenAILLMClient()
    return BookAssistant(
        CriteriaExtractor(llm),
        OpenLibraryClient(),
        explainer=BookExplainer(llm),
    )
