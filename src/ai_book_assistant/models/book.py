from pydantic import BaseModel, ConfigDict, Field


class Book(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    title: str
    authors: list[str] = Field(default_factory=list)
    first_publish_year: int | None = None
    key: str = Field(min_length=1)
    description: str | None = None
    subjects: list[str] = Field(default_factory=list)
    covers: list[int] = Field(default_factory=list)
