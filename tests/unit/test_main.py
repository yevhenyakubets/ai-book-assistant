import pytest
from fastapi.testclient import TestClient

from ai_book_assistant.api.main import create_default_app, run


@pytest.fixture(autouse=True)
def ignore_dotenv_file(monkeypatch):
    monkeypatch.setattr("ai_book_assistant.api.main.load_dotenv", lambda: None)


def test_default_app_uses_the_configured_database(monkeypatch, tmp_path):
    database = tmp_path / "data" / "history.db"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_PATH", str(database))

    client = TestClient(create_default_app())

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/history").json() == []
    assert database.exists()


def test_default_app_refuses_to_start_without_an_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "history.db"))

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_default_app()


def test_run_uses_default_host_and_port(monkeypatch):
    calls = []
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setattr(
        "ai_book_assistant.api.main.uvicorn.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    run()

    assert calls == [
        (
            ("ai_book_assistant.api.main:create_default_app",),
            {"factory": True, "host": "127.0.0.1", "port": 8000},
        )
    ]


def test_run_uses_configured_host_and_port(monkeypatch):
    calls = []
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9001")
    monkeypatch.setattr(
        "ai_book_assistant.api.main.uvicorn.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    run()

    assert calls == [
        (
            ("ai_book_assistant.api.main:create_default_app",),
            {"factory": True, "host": "0.0.0.0", "port": 9001},
        )
    ]
