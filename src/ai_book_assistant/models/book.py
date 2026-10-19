from collections.abc import Mapping

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

    @classmethod
    def from_search_doc(cls, doc: Mapping[str, object]) -> "Book":
        title = doc.get("title")
        if not isinstance(title, str) or not title.strip():
            title = "Unknown Title"

        cover_id = doc.get("cover_i")
        covers = [cover_id] if type(cover_id) is int else []

        authors = doc.get("author_name")
        if isinstance(authors, list):
            authors = [author for author in authors if isinstance(author, str)]
        else:
            authors = []

        subjects = doc.get("subject")
        if isinstance(subjects, list):
            subjects = [subject for subject in subjects if isinstance(subject, str)]
        else:
            subjects = []

        year = doc.get("first_publish_year")
        if type(year) is not int:
            year = None

        key = doc.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Open Library search document must contain a work key")

        return cls(
            title=title,
            authors=authors,
            first_publish_year=year,
            key=key,
            subjects=subjects,
            covers=covers,
        )

    def update_from_work_data(self, work_data: Mapping[str, object]) -> None:
        desc = work_data.get("description")
        if isinstance(desc, Mapping):
            desc = desc.get("value")
        self.description = desc.strip() or None if isinstance(desc, str) else None
