from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from ai_book_assistant.text import collapse_whitespace

MAX_SUBJECTS = 3
MAX_KEYWORDS = 3
MIN_YEAR = 1000
MAX_YEAR = 2100


def _normalize_terms(value: object) -> list[str]:
    """Turn whatever the LLM returned into a clean, de-duplicated list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    terms: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        term = collapse_whitespace(item.replace("-", " ")).lower()
        if term and term not in terms:
            terms.append(term)
    return terms


class Preferences(BaseModel):
    """Soft preferences Open Library can't search for.

    These are never sent to the search API. They are kept so the
    explanation step can judge and describe how well a book fits.
    """

    tone: list[str] = Field(
        default_factory=list,
        description="Mood or tone words, e.g. 'dark', 'cozy', 'serious'",
    )
    audience: str | None = Field(
        default=None, description="Intended reader, e.g. 'beginners', 'children'"
    )
    length: str | None = Field(
        default=None, description="Length hint, e.g. 'short', 'long'"
    )

    @field_validator("tone", mode="before")
    @classmethod
    def _clean_tone(cls, value: object) -> list[str]:
        return _normalize_terms(value)

    @field_validator("audience", "length", mode="before")
    @classmethod
    def _clean_text(cls, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        return collapse_whitespace(value).lower() or None


class SearchCriteria(BaseModel):
    subjects: list[str] = Field(
        default_factory=list,
        description="1-3 short genre/topic tags, one concept each",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Up to 3 topic words that are not genres or tone",
    )
    author: str | None = Field(default=None, description="Author, only if named")
    year_min: int | None = Field(default=None, description="Earliest publication year")
    year_max: int | None = Field(default=None, description="Latest publication year")
    preferences: Preferences = Field(default_factory=Preferences)

    @field_validator("subjects", "keywords", mode="before")
    @classmethod
    def _clean_terms(cls, value: object, info: ValidationInfo) -> list[str]:
        limit = MAX_SUBJECTS if info.field_name == "subjects" else MAX_KEYWORDS
        return _normalize_terms(value)[:limit]

    @field_validator("author", mode="before")
    @classmethod
    def _clean_author(cls, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        return collapse_whitespace(value) or None

    @field_validator("year_min", "year_max", mode="before")
    @classmethod
    def _clean_year(cls, value: object) -> int | None:
        try:
            year = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return year if MIN_YEAR <= year <= MAX_YEAR else None

    @field_validator("preferences", mode="before")
    @classmethod
    def _clean_preferences(cls, value: object) -> object:
        return {} if value is None else value

    @model_validator(mode="after")
    def _order_years(self) -> "SearchCriteria":
        if (
            self.year_min is not None
            and self.year_max is not None
            and self.year_min > self.year_max
        ):
            self.year_min, self.year_max = self.year_max, self.year_min
        return self

    @property
    def is_searchable(self) -> bool:
        """True when there is something Open Library can actually search for.

        Years, tone, audience and length alone are not enough: a year range by
        itself would return arbitrary books.
        """
        return bool(self.subjects or self.keywords or self.author)
