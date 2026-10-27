# AI Book Assistant

AI Book Assistant turns a plain-language book request into a small list of
Open Library books, with metadata and an explanation of why each result may
fit. It exposes a REST API built with FastAPI and saves searches in a local
SQLite database so they can be revisited later.

## How it works

The language model is used at two edges of the search flow: to extract
structured criteria from the request and to explain the returned books. The
application validates those criteria, builds Open Library Search API queries,
and applies a deterministic fallback ladder if a strict search has no results.
Book results are enriched with work descriptions and available catalogue
metadata before they are explained and saved.

```text
Plain-language query
        |
        v
LLM criteria extraction --> Pydantic validation
        |
        v
Open Library search --> fallback strategies --> descriptions and metadata
        |                                           |
        +-------------------- LLM explanations <----+
                                |
                                v
                      API response + SQLite history
```

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for live searches

## Run locally

Clone the repository and install its dependencies:

```bash
git clone https://github.com/yevhenyakubets/ai-book-assistant.git
cd ai-book-assistant
uv sync --dev
```

Create a local environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the API:

```bash
uv run ai-book-assistant
```

The server listens on `http://127.0.0.1:8000` by default. Interactive API
documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

To try a request, submit a natural-language query:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Cozy mystery novels set in a small village"}'
```

An abbreviated response looks like this; available metadata and actual values
vary by search:

```json
{
  "id": 1,
  "created_at": "2026-10-27T12:00:00Z",
  "query": "Cozy mystery novels set in a small village",
  "strategy": "full",
  "message": null,
  "summary": "...",
  "results": [
    {
      "title": "...",
      "authors": ["..."],
      "first_publish_year": 1990,
      "description": "...",
      "number_of_pages_median": null,
      "ratings_average": null,
      "ratings_count": null,
      "want_to_read_count": null,
      "languages": [],
      "cover_url": null,
      "first_sentence": null,
      "explanation": "...",
      "url": "https://openlibrary.org/works/..."
    }
  ],
  "criteria": {
    "subjects": ["mystery"],
    "keywords": ["village"],
    "author": null,
    "language": null,
    "year_min": null,
    "year_max": null,
    "preferences": {
      "tone": ["cozy"],
      "audience": null,
      "length": null
    }
  }
}
```

The response contains the original query, extracted criteria, a summary, and
up to five results. Each result can include the title, author, publication
year, description, page-count median, ratings, popularity, language codes,
cover URL, first sentence, and an explanation. For example, a request such as
`Fantasy novels by Andrzej Sapkowski available in English` asks Open Library
for works with English-language editions; it does not select a particular
edition.

## Configuration

Configuration is read from environment variables. When running locally,
`python-dotenv` loads values from `.env`. The same variables can be provided
directly to the process or through Docker Compose.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Yes | — | API key used for criteria extraction and explanations. |
| `OPENAI_MODEL` | No | `gpt-5.4-mini` | OpenAI model used by the assistant. |
| `APP_ACCESS_TOKEN` | No | Unset | When set, requires a Bearer token for `/search` and `/history` endpoints. `/health` remains public. |
| `DATABASE_PATH` | No | `data/history.db` | Path to the SQLite search-history database. |
| `HOST` | No | `127.0.0.1` | Address used by the local application entry point. |
| `PORT` | No | `8000` | Port used by the local application entry point. |
| `OPEN_LIBRARY_CONTACT_EMAIL` | No | Unset | Optional contact email included in Open Library API requests. |

For example, set `APP_ACCESS_TOKEN` in `.env` to protect search and history
endpoints. Send the same value as an Authorization header:

```bash
curl http://127.0.0.1:8000/history \
  -H "Authorization: Bearer <your-access-token>"
```

If `APP_ACCESS_TOKEN` is unset, these endpoints are open without
authentication. Set a token before exposing the service beyond a trusted
local environment.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Public health check. |
| `POST` | `/search` | Search for books from a JSON body such as `{"query":"Books by Terry Pratchett"}`. Saves the result and returns its `id`. |
| `GET` | `/history` | List recent searches. Supports `limit` (1–100, default 20) and `offset` (default 0). |
| `GET` | `/history/{search_id}` | Retrieve a saved search and its results. |

The `/search` query must contain 1–500 non-whitespace characters. Search and
history endpoints return `401` when access-token authentication is enabled
and the token is missing or invalid. A missing search-history entry returns
`404`; upstream Open Library and OpenAI errors are reported as `502` and `503`
respectively.

## Try example queries

The repository includes an end-to-end script that prints extracted criteria,
search strategy, returned metadata, descriptions, and explanations. It calls
the live OpenAI and Open Library services:

```bash
uv run python -m scripts.try_queries \
  "Fantasy novels by Andrzej Sapkowski available in English"
```

Other useful examples:

```bash
uv run python -m scripts.try_queries \
  "A short science-fiction book about artificial intelligence suitable for beginners"
uv run python -m scripts.try_queries \
  "Fantasy books published before 2000"
uv run python -m scripts.try_queries \
  "Books by Terry Pratchett"
```

The script also has a built-in set of example queries if invoked without
arguments. Use `--verbose` to include raw model responses:

```bash
uv run python -m scripts.try_queries --verbose "Cozy mystery novels set in a small village"
```

## Run tests and checks

The default test command discovers and runs the unit tests. They use mocked
LLM and Open Library clients and do not make live service calls:

```bash
uv run pytest
```

The full-pipeline integration test uses a fake LLM, a mocked Open Library HTTP
transport, and a temporary SQLite database, so it also requires no API key or
network access. CI runs it separately from the unit tests on pull requests
targeting `main` or `develop` and on pushes to those branches. Run it manually
with:

```bash
uv run pytest tests/integration
```

The project also configures Flake8 and Black:

```bash
uv run flake8
uv run black --check .
```

## Run with Docker Compose

After creating `.env` and setting `OPENAI_API_KEY`, build and start the
service:

```bash
docker compose up --build
```

The API is available at `http://127.0.0.1:8000`. Compose binds it to localhost
and keeps SQLite history in a named Docker volume. Stop the service with
`Ctrl+C`, or run `docker compose down` in another terminal. If
`APP_ACCESS_TOKEN` is set in `.env`, send it as a Bearer token for search and
history requests.

## Project structure

```text
src/ai_book_assistant/
├── api/        FastAPI app, routes, schemas, errors, and access-token auth
├── explain/    LLM-based explanations for search results
├── llm/        OpenAI client, criteria extraction, prompts, and validation
├── models/     Search criteria, book, result, and history models
├── search/     Open Library client, query building, parsing, and fallbacks
├── storage/    Search-history interface and SQLite implementation
├── assistant.py
├── bootstrap.py
└── service.py
scripts/
└── try_queries.py   Run one or more live example queries
tests/
└── unit/            Unit tests for the application layers
```

## Search behavior and limitations

- Open Library catalogue data is incomplete and inconsistent. Some works
  have no description, rating, cover, page count, or first sentence, and
  metadata may describe different editions of the same work.
- The language filter finds works that have at least one edition in the
  requested language. It does not choose an edition or guarantee that the
  work's displayed title and description are in that language.
- “Short” is searched using Open Library's median page count, capped at 300
  pages. Since many works have no page-count metadata, the fallback search
  can drop that constraint; results from that fallback are not verified to
  be short.
- Audience and tone requests such as “for beginners” or “cozy” are not
  strict catalogue filters. Search relevance depends on the criteria that
  can be represented in Open Library, and fallback strategies may relax
  constraints to return results.
- Explanations are generated from the available catalogue data and should be
  treated as helpful summaries, not authoritative bibliographic facts.
- Live searches require network access to OpenAI and Open Library. The test
  suite avoids those services by using mocks.
