import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI

from ai_book_assistant.api.app import create_app
from ai_book_assistant.bootstrap import build_assistant
from ai_book_assistant.storage import SearchHistory

DEFAULT_DATABASE_PATH = "data/history.db"


def create_default_app() -> FastAPI:
    """Build the real application: Groq, Open Library and a SQLite history file.

    Run it with:
        uvicorn --factory ai_book_assistant.api.main:create_default_app
    """
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    history = SearchHistory(os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH))
    return create_app(build_assistant(), history)
