import pytest
from fastapi.testclient import TestClient

from ai_book_assistant.api.main import create_default_app


@pytest.fixture(autouse=True)
def ignore_dotenv_file(monkeypatch):
    monkeypatch.setattr("ai_book_assistant.api.main.load_dotenv", lambda: None)


def test_default_app_uses_the_configured_database(monkeypatch, tmp_path):
    database = tmp_path / "data" / "history.db"
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_PATH", str(database))

    client = TestClient(create_default_app())

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/history").json() == []
    assert database.exists()


def test_default_app_refuses_to_start_without_an_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "history.db"))

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        create_default_app()
