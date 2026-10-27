import json

from ai_book_assistant.models import SearchCriteria

RECENT_YEAR = 2015  # what "recent" or "newer" means when the user doesn't say

INSTRUCTIONS = f"""You extract book search criteria from a user's request.
Respond with a single JSON object with exactly these keys:
- "subjects": list of 0-3 short lowercase English genre or topic tags, one concept each
- "keywords": list of 0-3 specific English words or phrases that are not genres
- "author": string or null
- "year_min": integer or null
- "year_max": integer or null
- "preferences": object with "tone" (list of mood words), "audience" (string or null)
  and "length" (string or null)

Rules:
1. Extract only what the user actually asked for. Never add related topics or
   details the user did not mention. If something is not stated, leave it empty or null.
2. Subjects are broad tags such as "fantasy", "mystery", "biography". Use
   "autobiography" when the user says autobiography or memoir, and "biography" only
   when they ask for a biography. If the user asks for "novels" or "fiction" without
   naming a genre, use "fiction".
3. Keywords are for specific topics or settings. Keep fixed terms together
   ("climate change", "artificial intelligence"), but for a descriptive setting use
   the single most distinctive word ("village", not "small village").
   Never use filler such as "book", "novel", "famous", "good" or "best".
4. Never put mood, audience or length in subjects or keywords. They go in
   preferences only: tone ("dark", "cozy"), audience ("beginners", "children"),
   length ("short", "long"). Treat "short", "brief", "not too long", and a
   request for no more than 300 pages as length "short".
5. Years: "after 2010" means year_min 2011; "before 2000" means year_max 1999;
   "in 1998" means both are 1998; "the 1980s" means 1980 to 1989; "recent" or
   "newer" means year_min {RECENT_YEAR}.
6. Set author only when a person is named as the author. If the user names a book to
   compare with, you may infer its genre, but never guess its author or plot details.
7. If the request contains several unrelated wishes, extract only the first one.
8. The request may be in any language. Always answer in English.
9. If the request gives nothing searchable, return empty lists and nulls.
"""

# (user query, fields to extract). Deliberately different from scripts/try_queries.py
# so test runs measure the rules instead of memorized answers.
EXAMPLES = [
    (
        "Memoirs of famous chess players published after 2005",
        {"subjects": ["autobiography", "chess"], "year_min": 2006},
    ),
    (
        "Books about volcanoes published in 1998",
        {"subjects": ["volcanoes"], "year_min": 1998, "year_max": 1998},
    ),
    (
        "Something like The Hobbit but newer",
        {"subjects": ["fantasy"], "year_min": RECENT_YEAR},
    ),
    (
        "A gentle romance set in a bakery, not too long",
        {
            "subjects": ["romance"],
            "keywords": ["bakery"],
            "preferences": {"tone": ["gentle"], "length": "short"},
        },
    ),
    (
        "A fantasy novel no more than 300 pages",
        {
            "subjects": ["fantasy"],
            "preferences": {"length": "short"},
        },
    ),
    (
        "Anything by Ursula K. Le Guin",
        {"author": "Ursula K. Le Guin"},
    ),
    (
        "Livres sur la photographie pour débutants",
        {
            "subjects": ["photography"],
            "preferences": {"audience": "beginners"},
        },
    ),
    (
        "I'd like a detective story from the 70s, and my sister wants gardening books",
        {"subjects": ["detective"], "year_min": 1970, "year_max": 1979},
    ),
    (
        "Recommend me something to read",
        {},
    ),
    (
        "Novels from the 1930s",
        {"subjects": ["fiction"], "year_min": 1930, "year_max": 1939},
    ),
]


def _format_examples() -> str:
    blocks = []
    for query, fields in EXAMPLES:
        criteria = SearchCriteria.model_validate(fields)
        blocks.append(
            f"Query: {json.dumps(query, ensure_ascii=False)}\n"
            f"JSON: {criteria.model_dump_json()}"
        )
    return "\n\n".join(blocks)


EXAMPLES_TEXT = _format_examples()


def build_extraction_prompt(query: str) -> str:
    return (
        f"{INSTRUCTIONS}\n"
        f"Examples:\n\n{EXAMPLES_TEXT}\n\n"
        "Now extract the criteria for this request. Respond with JSON only.\n"
        f"Query: {json.dumps(query, ensure_ascii=False)}\n"
        "JSON:"
    )
