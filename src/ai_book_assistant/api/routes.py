from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ai_book_assistant.api.schemas import (
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    to_response,
)
from ai_book_assistant.assistant import BookAssistant
from ai_book_assistant.models import HistoryEntry
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


@router.get(
    "/history",
    response_model=list[HistoryEntry],
    summary="List past searches, newest first",
)
def list_history(
    history: HistoryDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[HistoryEntry]:
    return history.list_recent(limit=limit, offset=offset)


@router.get(
    "/history/{search_id}",
    response_model=SearchResponse,
    responses={404: {"model": ErrorResponse, "description": "No such search"}},
    summary="Show a past search exactly as it was returned",
)
def get_search(search_id: int, history: HistoryDep) -> SearchResponse:
    record = history.get(search_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Search not found")
    return to_response(record)
