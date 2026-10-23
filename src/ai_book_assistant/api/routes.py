from typing import Annotated

from fastapi import APIRouter, Depends, Request

from ai_book_assistant.api.schemas import (
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    to_response,
)
from ai_book_assistant.assistant import BookAssistant
from ai_book_assistant.storage import SearchHistory

router = APIRouter()

SERVICE_ERRORS = {
    502: {
        "model": ErrorResponse,
        "description": "The book catalogue (Open Library) is unavailable",
    },
    503: {
        "model": ErrorResponse,
        "description": "The language model is busy or unavailable",
    },
}


def get_assistant(request: Request) -> BookAssistant:
    return request.app.state.assistant


def get_history(request: Request) -> SearchHistory:
    return request.app.state.history


AssistantDep = Annotated[BookAssistant, Depends(get_assistant)]
HistoryDep = Annotated[SearchHistory, Depends(get_history)]


@router.get("/health", summary="Check that the service is running")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/search",
    response_model=SearchResponse,
    responses=SERVICE_ERRORS,
    summary="Find books for a request and explain why they fit",
)
def search(
    body: SearchRequest, assistant: AssistantDep, history: HistoryDep
) -> SearchResponse:
    """Turn a free-text request into up to five books, each with an explanation.

    Every search is saved. Use the returned `id` with `GET /history/{id}`.
    """
    result = assistant.search(body.query)
    record = history.add(
        query=body.query,
        strategy=result.strategy,
        criteria=result.criteria,
        books=result.books,
        explanations=result.explanations,
    )
    return to_response(record)
