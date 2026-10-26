import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from ai_book_assistant.api.app import create_app
from ai_book_assistant.bootstrap import build_assistant
from ai_book_assistant.service import SearchService
from ai_book_assistant.storage import SqliteSearchHistory

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_PATH = "data/history.db"


def create_default_app() -> FastAPI:
    """Build the real application: OpenAI, Open Library and a SQLite history file.

    Run it with:
        uvicorn --factory ai_book_assistant.api.main:create_default_app
    """
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    history = SqliteSearchHistory(os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH))
    access_token = os.getenv("APP_ACCESS_TOKEN") or None
    if access_token is None:
        logger.warning("APP_ACCESS_TOKEN is not set: the API is open to everyone")
    service = SearchService(build_assistant(), history)
    return create_app(service, history, access_token)


def run() -> None:
    uvicorn.run(
        "ai_book_assistant.api.main:create_default_app",
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
