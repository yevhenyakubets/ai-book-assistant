from fastapi import FastAPI

from ai_book_assistant.api.errors import register_error_handlers
from ai_book_assistant.api.routes import router
from ai_book_assistant.service import SearchService
from ai_book_assistant.storage import HistoryStore

DESCRIPTION = (
    "Describe the kind of book you want in plain language. An LLM turns the "
    "request into search criteria, Open Library is searched, and each result "
    "comes back with a short explanation of why it fits. Every search is "
    "saved and can be opened again from the history endpoints."
)


def create_app(service: SearchService, history: HistoryStore) -> FastAPI:
    app = FastAPI(title="AI Book Assistant", description=DESCRIPTION)
    app.state.service = service
    app.state.history = history
    app.include_router(router)
    register_error_handlers(app)
    return app
