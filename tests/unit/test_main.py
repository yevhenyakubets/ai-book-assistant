import logging

import pytest
from fastapi.testclient import TestClient

from ai_book_assistant.api.main import create_default_app, run


@pytest.fixture(autouse=True)
def ignore_dotenv_file(monkeypatch):
    monkeypatch.setattr("ai_book_assistant.api.main.load_dotenv", lambda: None)
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)


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


def test_default_app_enforces_the_configured_token(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "history.db"))
    monkeypatch.setenv("APP_ACCESS_TOKEN", "s3cret")
    client = TestClient(create_default_app())

    assert client.get("/history").status_code == 401
    ok = client.get("/history", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200


def test_default_app_warns_when_no_token_is_configured(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "history.db"))
    caplog.set_level(logging.WARNING, logger="ai_book_assistant.api.main")

    create_default_app()

    assert "APP_ACCESS_TOKEN is not set" in caplog.text
