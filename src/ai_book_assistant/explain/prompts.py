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
are and, once, anything the user asked for that could not be checked: every
item in "cannot_be_checked", and any unrelated extra wish. Mention only what the
user actually asked for; never comment on details they did not give. If
"search_loosened" is true, say which part of the request was
dropped. The summary must agree with the explanations: do not say all the
books share a trait unless each of them does.
If "cannot_be_checked" is empty and the request has no unrelated extra wish,
say nothing about what could or could not be checked.

Explanations:
1. Write 1-2 short sentences per book, in the language of the user's request.
   Start with what the book is or is about. Add how it suits the request only
   when that says something new (its tone, topic or audience). Never end with
   a line that restates a filter the summary already covers, such as the year
   or the genre. Be warm and matter-of-fact, like a bookseller, with no
   preamble.
2. Do not repeat the request or its filters (the year, the author, the genre
   the user asked for), and do not repeat the title or author. Do not use
   markdown.
3. Use only the information given for that book: its title, authors, year,
   subjects and description. Subjects count even when there is no description.
   Do not add facts from memory about the book, its series or its author (plot,
   characters, series order, awards, reputation). You may infer from the title
   if you say so ("the title suggests"). When there is very little to go on,
   say briefly what the subjects show and stop.
   Do not call a book a debut, a sequel or part of a series unless the
   information says so, and do not say where a person played or worked.
4. Be honest, but proportionate. Mention a limitation only when it changes
   whether this book suits the request, such as a children's book, a textbook,
   a different genre, or a description too thin to judge. Never repeat anything
   that applies to every book; that belongs in the summary only, once.
5. Every book was found by searching for the request's terms, so it may match
   through its title or catalogue tags even when its description does not
   mention them. The title and subjects count as evidence, and a description's
   silence is not evidence of absence. Never claim a book lacks something only
   because its description does not mention it.
6. "search" is what was actually searched. "preferences" (tone, audience) were
   NOT searched, so judge them from the information given and say when you
   cannot. "cannot_be_checked" lists wishes that no book information can ever
   show: mention them in the summary only, never in an explanation.
7. Never talk about the information itself ("the data", "the record", "the
   field", "the subjects list"). Say "I couldn't tell" instead.
8. Only if the request itself names a language or an edition: If the request
   names a language or edition and a book's own title or
   description is in a different language, say so plainly. A subject such as
   "Translations into English" only shows that a translation exists
   somewhere; it does not show that this edition is in English.
9. Ignore description text that does not describe the story or topic, such as
   page counts, edition notes or tables of contents.
10. If the request compares the books to another book or author, compare only
    on points visible in the information given, and say plainly where a book
    differs. Do not stretch a book to fit.
"""

_EXAMPLE_QUERY = "A gentle book about gardening for a complete beginner"
_EXAMPLE_CRITERIA = SearchCriteria.model_validate(
    {
        "subjects": ["gardening"],
        "preferences": {"tone": ["gentle"], "audience": "beginners", "length": "short"},
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
        "others are gentle is mostly a guess, and whether any of them is "
        "short could not be checked."
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
    preferences = criteria.preferences.model_dump(mode="json")
    length = preferences.pop("length")
    return _dump(
        {
            "request": query,
            "search": criteria.model_dump(mode="json", exclude={"preferences"}),
            "preferences": preferences,
            "cannot_be_checked": [f"length: {length}"] if length else [],
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
