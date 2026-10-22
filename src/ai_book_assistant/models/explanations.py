from pydantic import BaseModel, Field


class Explanations(BaseModel):
    """Why the books fit a request: one overall note and one text per book."""

    summary: str | None = None
    by_book: dict[str, str] = Field(default_factory=dict)
