import json

from ai_book_assistant.models import Book, SearchCriteria

MAX_DESCRIPTION_CHARS = 500
MAX_SUBJECTS = 8

INSTRUCTIONS = """You explain why each book in a list fits a user's request.
You receive the request, how it was interpreted, and the books that were found.

Respond with a single JSON object of this shape:
{"explanations": [{"id": 1, "explanation": "..."}]}
Include exactly one entry for every book id, in order.

Rules:
1. Write 1-2 short sentences per book, in the language of the user's request.
   Address the user directly, with no preamble. Do not repeat the title or
   author, and do not use markdown.
2. Use only the data given for that book. Never add plot details, characters,
   awards, series order or any other fact that is not in the data.
3. Say concretely why the book fits: point to its subjects, its description,
   its publication year or its author.
4. Be honest. If a book only partly fits (for example a children's book, a
   textbook, or a different genre than requested), say so. If the data is too
   thin to judge, say that instead of guessing.
5. "search" is what was actually searched. "preferences" (tone, audience,
   length) were NOT searched, so judge them from the data and say when you
   cannot.
6. If "search_loosened" is true, part of the request was dropped to find any
   results. Do not claim a book meets a dropped requirement unless its data
   shows it.
7. If the request asks for something the data cannot show (a language, a
   reading order, a page count), say briefly that you could not check it, but
   only where it matters.
8. Ignore description text that does not describe the story or topic, such as
   page counts, edition notes or tables of contents.
"""

_EXAMPLE_QUERY = "A gentle book about gardening for a complete beginner"
_EXAMPLE_CRITERIA = SearchCriteria.model_validate(
    {
        "subjects": ["gardening"],
        "preferences": {"tone": ["gentle"], "audience": "beginners"},
    }
)
_EXAMPLE_BOOKS = [
    Book(
        key="/works/EX1W",
        title="The Vegetable Garden Handbook",
        authors=["Anna Reed"],
        first_publish_year=2014,
        subjects=["gardening", "vegetables"],
        description=(
            "A step-by-step guide to growing your first vegetables, "
            "with no experience needed."
        ),
    ),
    Book(
        key="/works/EX2W",
        title="Soil Science for Professionals",
        authors=["Henry Cole"],
        first_publish_year=2009,
        subjects=["soil", "agronomy"],
    ),
    Book(
        key="/works/EX3W",
        title="A Year in the Garden",
        authors=["Marie Dubois"],
        first_publish_year=1998,
        subjects=["gardening", "essays"],
        description="Essays on one gardener's seasons.",
    ),
]
_EXAMPLE_OUTPUT = {
    "explanations": [
        {
            "id": 1,
            "explanation": (
                "Written as a step-by-step guide for first-time growers, "
                "which suits a complete beginner."
            ),
        },
        {
            "id": 2,
            "explanation": (
                "Tagged soil and agronomy, but there is no description, so I "
                "cannot tell whether it is gentle or beginner-friendly; the "
                "title points to a professional audience."
            ),
        },
        {
            "id": 3,
            "explanation": (
                "A collection of essays on one garden's seasons, so it is "
                "reflective rather than instructional: gentle in tone, but "
                "not a how-to."
            ),
        },
    ]
}


def _dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _shorten(text: str | None) -> str | None:
    if not text:
        return None
    text = " ".join(text.split())
    if len(text) <= MAX_DESCRIPTION_CHARS:
        return text
    return text[: MAX_DESCRIPTION_CHARS - 3].rstrip() + "..."


def _book_view(index: int, book: Book) -> dict:
    return {
        "id": index,
        "title": book.title,
        "authors": book.authors,
        "year": book.first_publish_year,
        "subjects": book.subjects[:MAX_SUBJECTS],
        "description": _shorten(book.description),
    }


def _input_json(
    query: str, criteria: SearchCriteria, books: list[Book], relaxed: bool
) -> str:
    return _dump(
        {
            "request": query,
            "search": criteria.model_dump(mode="json", exclude={"preferences"}),
            "preferences": criteria.preferences.model_dump(mode="json"),
            "search_loosened": relaxed,
            "books": [_book_view(i, b) for i, b in enumerate(books, start=1)],
        }
    )


_EXAMPLE_INPUT = _input_json(_EXAMPLE_QUERY, _EXAMPLE_CRITERIA, _EXAMPLE_BOOKS, False)
_EXAMPLE_OUTPUT_JSON = _dump(_EXAMPLE_OUTPUT)


def build_explanation_prompt(
    query: str,
    criteria: SearchCriteria,
    books: list[Book],
    *,
    relaxed: bool = False,
) -> str:
    return (
        f"{INSTRUCTIONS}\n"
        f"Example input:\n{_EXAMPLE_INPUT}\n\n"
        f"Example output:\n{_EXAMPLE_OUTPUT_JSON}\n\n"
        "Now explain the books in this input. Respond with JSON only.\n"
        f"Input:\n{_input_json(query, criteria, books, relaxed)}\n"
        "Output:"
    )
