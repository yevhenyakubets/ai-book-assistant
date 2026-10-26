FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation and disable uv download prompts
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first (leveraging Docker layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application code
ADD . /app

# Run synchronization for the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Ensure the data directory exists for SQLite
RUN mkdir -p /app/data

EXPOSE 8000

# Run the FastAPI application using uvicorn through uv
CMD ["uv", "run", "uvicorn", "--factory", "ai_book_assistant.api.main:create_default_app", "--host", "0.0.0.0", "--port", "8000"]
