import pytest

from ai_book_assistant.text import collapse_whitespace


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("already compact", "already compact"),
        ("  several   words  ", "several words"),
        ("one\ttwo\nthree", "one two three"),
    ],
)
def test_collapse_whitespace(text: str, expected: str) -> None:
    assert collapse_whitespace(text) == expected
