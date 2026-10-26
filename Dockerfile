FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Run as a non-root user; only the data directory is writable.
RUN useradd --system --uid 10001 app \
    && mkdir -p /app/data \
    && chown app:app /app/data
USER app

ENV HOST=0.0.0.0 \
    PORT=8000 \
    DATABASE_PATH=/app/data/history.db
EXPOSE 8000

# The console script defined in pyproject.toml (uvicorn with the app factory).
CMD ["ai-book-assistant"]
