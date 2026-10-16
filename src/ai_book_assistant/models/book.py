from pydantic import BaseModel, Field


class Book(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    first_publish_year: int | None = None
    key: str  # e.g., "/works/OL46125W"
    description: str | None = None
    subjects: list[str] = Field(default_factory=list)
    covers: list[int] = Field(default_factory=list)

    @classmethod
    def from_search_doc(cls, doc: dict) -> "Book":
        cover_id = doc.get("cover_i")
        covers = [cover_id] if cover_id else []
        return cls(
            title=doc.get("title", "Unknown Title"),
            authors=doc.get("author_name", []),
            first_publish_year=doc.get("first_publish_year"),
            key=doc.get("key", ""),
            subjects=doc.get("subject", []),
            covers=covers,
        )

    def update_from_work_data(self, work_data: dict) -> None:
        desc = work_data.get("description")
        if isinstance(desc, dict):
            self.description = desc.get("value")
        else:
            self.description = desc
