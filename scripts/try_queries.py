import argparse
import logging
import sys
import time

from dotenv import load_dotenv

from ai_book_assistant.bootstrap import build_assistant
from ai_book_assistant.llm.errors import LLMError

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")
logging.getLogger("ai_book_assistant.llm.client").setLevel(logging.INFO)


TEST_QUERIES = [
    # Basics
    "A short science-fiction book about artificial intelligence suitable for beginners",
    "Cozy mystery novels set in a small village",
    "Fantasy romance with dragons, no more than 300 pages",
    # Years
    "Fantasy books published before 2000",
    "Novels from the 1980s",
    "Books published in 2015 about climate change",
    "Give me autobiographies of famous football players published after 2010",
    # Authors
    "Books by Terry Pratchett",
    "Something like Dune but more recent",
    # Edge cases
    "Give me a good book",
    "Книги про штучний інтелект для початківців",
    "I want something with space travel, maybe from the 90s, not too long, "
    "and my brother also likes cookbooks, so anything like that too",
]

PAUSE_BETWEEN_QUERIES = 1.0  # seconds; keeps free-tier rate limits happy


def compact_criteria(criteria) -> dict:
    """Criteria without empty fields, so the output stays readable."""
    data = criteria.model_dump()
    prefs = {k: v for k, v in data.pop("preferences").items() if v}
    data = {k: v for k, v in data.items() if v}
    if prefs:
        data["preferences"] = prefs
    return data


def format_book(book) -> str:
    authors = ", ".join(book.authors[:2]) or "Unknown author"
    year = book.first_publish_year or "?"
    return f"{book.title} - {authors} ({year})"


def format_description(book, width: int = 100) -> str:
    if not book.description:
        return "(no description)"
    text = " ".join(book.description.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def run_query(query, assistant) -> dict:
    """Run one query end to end and return a summary row."""
    print("=" * 78)
    print(f"QUERY: {query!r}")

    row = {
        "query": query,
        "rung": "-",
        "count": 0,
        "described": 0,
        "explained": 0,
        "status": "ok",
    }

    try:
        result = assistant.search(query)
    except LLMError as exc:
        print(f"  EXTRACTION FAILED: {type(exc).__name__}: {exc}")
        row["status"] = "extraction failed"
        return row

    except Exception as exc:
        print(f"  SEARCH FAILED: {type(exc).__name__}: {exc}")
        row["status"] = "search failed"
        return row

    print(f"  Criteria: {compact_criteria(result.criteria)}")

    outcome = result.outcome
    if result.needs_clarification:
        row["status"] = "needs clarification"
        print(f"  Needs clarification: {result.message}")
        return row

    row["rung"] = result.strategy
    row["count"] = len(result.books)
    row["described"] = sum(1 for book in result.books if book.description)
    row["explained"] = len(result.explanations.by_book)
    marker = " (relaxed)" if result.relaxed else ""
    print(f"  Strategy: {result.strategy}{marker}")
    print(f"  Attempts: {outcome.attempts}")
    print(f"  Params:   {outcome.params}")

    if not result.books:
        print("  Found 0 books.")
    else:
        print(f"  Found {len(result.books)} books:")
        if result.explanations.summary:
            print(f"  Summary: {result.explanations.summary}")
        for book in result.books:
            why = result.explanations.by_book.get(book.key, "(no explanation)")
            print(f"    - {format_book(book)}")
            print(f"        Subjects: {', '.join(book.subjects[:8]) or '(none)'}")
            print(f"        {format_description(book)}")
            print(f"        Why: {why}")
    return row


def print_summary(rows) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'#':>2}  {'strategy':<17} {'books':>5}  {'status':<18} query")
    for i, row in enumerate(rows, start=1):
        query = row["query"]
        if len(query) > 45:
            query = query[:42] + "..."
        print(
            f"{i:>2}  {row['rung']:<17} {row['count']:>5}  "
            f"{row['status']:<18} {query}"
        )

    relaxed = sum(1 for r in rows if r["rung"] not in ("full", "-"))
    failed = sum(
        1 for r in rows if r["status"] in ("extraction failed", "search failed")
    )
    clarify = sum(1 for r in rows if r["status"] == "needs clarification")
    empty = sum(1 for r in rows if r["status"] == "ok" and r["count"] == 0)
    print(
        f"\n{len(rows)} queries | {relaxed} relaxed | {empty} with no results | "
        f"{clarify} need clarification | {failed} failed"
    )
    books_total = sum(r["count"] for r in rows)
    described_total = sum(r["described"] for r in rows)
    print(f"{described_total}/{books_total} books have a description")
    explained_total = sum(r["explained"] for r in rows)
    print(f"{explained_total}/{books_total} books have an explanation")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run test queries end to end.")
    parser.add_argument(
        "queries",
        nargs="*",
        help="Optional queries to run instead of the built-in test list.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Also print the raw model responses.",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger("ai_book_assistant").setLevel(logging.DEBUG)
    queries = args.queries or TEST_QUERIES

    assistant = build_assistant()

    rows = []
    for i, query in enumerate(queries):
        if i:
            time.sleep(PAUSE_BETWEEN_QUERIES)
        rows.append(run_query(query, assistant))

    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
