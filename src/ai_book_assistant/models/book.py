from pydantic import BaseModel, ConfigDict, Field

COVER_IMAGE_URL = "https://covers.openlibrary.org/b/id"


class Book(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    title: str
    authors: list[str] = Field(default_factory=list)
    first_publish_year: int | None = None
    key: str = Field(min_length=1)
    description: str | None = None
    subjects: list[str] = Field(default_factory=list)
    covers: list[int] = Field(default_factory=list)
    number_of_pages_median: int | None = None
    ratings_average: float | None = None
    ratings_count: int | None = None
    want_to_read_count: int | None = None
    languages: list[str] = Field(default_factory=list)
    first_sentence: str | None = None

    @property
    def cover_url(self) -> str | None:
        if not self.covers or self.covers[0] < 1:
            return None
        return f"{COVER_IMAGE_URL}/{self.covers[0]}-M.jpg"
