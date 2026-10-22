import json

from ai_book_assistant.models import Book, SearchCriteria

MAX_DESCRIPTION_CHARS = 500
MAX_SUBJECTS = 8

INSTRUCTIONS = """You help a user judge a list of books found for their request.
You receive the request, how it was interpreted, and the books that were found.

Respond with a single JSON object of this shape:
{"summary": "...", "explanations": [{"id": 1, "explanation": "..."}]}
Include exactly one explanation for every book id, in order.

Summary: 1-2 sentences about the list as a whole. Say what kind of books these
are and, once, anything in the request that could not be searched or checked
(a page count, a language, a reading order, an unrelated extra wish). If
"search_loosened" is true, say which part of the request was dropped.

Explanations:
1. Write 1-2 short sentences per book, in the language of the user's request.
   Start with what the book is or is about, then how it suits the request.
   Be warm and matter-of-fact, like a bookseller, with no preamble.
2. Do not repeat the request or its filters (the year, the author, the genre
   the user asked for), and do not repeat the title or author. Do not use
   markdown.
3. Use only the information given for that book. Never add plot details,
   characters, awards, series order or any other fact that is not there.
4. Be honest, but proportionate. Mention a limitation only when it changes
   whether the book suits the request, such as a children's book, a textbook,
   a different genre, or too little information to judge. Do not list every
   thing you could not check; that belongs in the summary.
5. Every book was found by searching for the request's terms, so it may match
   through its title or catalogue tags even when its description does not
   mention them. Never claim a book lacks something only because its
   description is silent about it.
6. "search" is what was actually searched. "preferences" (tone, audience,
   length) were NOT searched, so judge them from the information given and
   say when you cannot.
7. Never talk about the information itself ("the data", "the record", "the
   field", "the subjects list"). Say "I couldn't tell" instead.
8. If the request names a language or edition and a book's own title or
   description is in a different language, say so plainly. A subject such as
   "Translations into English" only shows that a translation exists
   somewhere; it does not show that this edition is in English.
9. Ignore description text that does not describe the story or topic, such as
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
    "summary": (
        "Only the first book is clearly written for beginners. Whether the "
        "others are gentle is mostly a guess, since there is little to go on."
    ),
    "explanations": [
        {
            "id": 1,
            "explanation": (
                "A step-by-step guide to growing your first vegetables, "
                "written for people with no experience."
            ),
        },
        {
            "id": 2,
            "explanation": (
                "A professional soil-science text, so probably too technical "
                "for a beginner; there is nothing more to say about it."
            ),
        },
        {
            "id": 3,
            "explanation": (
                "Essays that follow one gardener through the seasons: "
                "reflective and gentle, but not a how-to."
            ),
        },
    ],
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
