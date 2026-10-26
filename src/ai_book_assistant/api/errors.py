import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_book_assistant.llm.errors import LLMError

logger = logging.getLogger(__name__)


async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
    if exc.status_code == 429:
        detail = "The language model is busy right now. Please try again in a minute."
    else:
        detail = "The language model is unavailable. Please try again later."
    return JSONResponse(status_code=503, content={"detail": detail})


async def handle_catalogue_error(
    request: Request, exc: httpx.HTTPError
) -> JSONResponse:
    logger.warning("Open Library request failed: %s", exc)
    detail = "The book catalogue (Open Library) is unavailable. Try again later."
    return JSONResponse(status_code=502, content={"detail": detail})


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(LLMError, handle_llm_error)
    app.add_exception_handler(httpx.HTTPError, handle_catalogue_error)
