import httpx
import pytest

from ai_book_assistant.search.client import OpenLibraryClient

OK_BODY = {"docs": []}


def make_client(handler, sleeps=None, **kwargs) -> OpenLibraryClient:
    record = sleeps if sleeps is not None else []
    return OpenLibraryClient(
        transport=httpx.MockTransport(handler),
        sleep=record.append,
        **kwargs,
    )


def test_retries_a_503_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=OK_BODY)

    assert make_client(handler).search_books({"q": "x"}) == []
    assert len(calls) == 2


def test_retries_a_timeout_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, json=OK_BODY)

    assert make_client(handler).search_books({"q": "x"}) == []
    assert len(calls) == 2


def test_gives_up_after_max_attempts_and_backs_off():
    calls = []
    sleeps = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    client = make_client(handler, sleeps, max_attempts=3, backoff=0.5)
    with pytest.raises(httpx.HTTPStatusError):
        client.search_books({"q": "x"})
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]


def test_persistent_timeout_is_raised():
    def handler(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    with pytest.raises(httpx.ConnectTimeout):
        make_client(handler).search_books({"q": "x"})


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400)

    with pytest.raises(httpx.HTTPStatusError):
        make_client(handler).search_books({"q": "x"})
    assert len(calls) == 1


def test_missing_work_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    assert make_client(handler).fetch_work_description("/works/OL1W") is None
    assert len(calls) == 1


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        OpenLibraryClient(max_attempts=0)
