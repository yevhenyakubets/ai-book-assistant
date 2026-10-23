from fastapi import FastAPI

from ai_book_assistant.api.errors import register_error_handlers
from ai_book_assistant.api.routes import router
from ai_book_assistant.assistant import BookAssistant
from ai_book_assistant.storage import SearchHistory

DESCRIPTION = (
    "Describe the kind of book you want in plain language. An LLM turns the "
    "request into search criteria, Open Library is searched, and each result "
    "comes back with a short explanation of why it fits. Every search is "
    "saved and can be opened again from the history endpoints."
)


def create_app(assistant: BookAssistant, history: SearchHistory) -> FastAPI:
    app = FastAPI(title="AI Book Assistant", description=DESCRIPTION)
    app.state.assistant = assistant
    app.state.history = history
    app.include_router(router)
    register_error_handlers(app)
    return app
