"""Provider behaviour with a fake HTTP session (no network)."""

import json

import pytest
import requests

from src.generation.llm_provider import (
    ChatMessage,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from src.generation.ollama_provider import OllamaProvider
from src.generation.openai_compatible_provider import OpenAICompatibleProvider

MSGS = [ChatMessage("system", "s"), ChatMessage("user", "u")]


class FakeResponse:
    def __init__(self, status=200, payload=None, lines=None, text=""):
        self.status_code = status
        self._payload = payload
        self._lines = lines or []
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_lines(self):
        yield from self._lines

    def close(self):
        pass


class FakeSession:
    def __init__(self, get=None, post=None):
        self._get, self._post = get, post
        self.last_post = None

    def get(self, url, timeout=None):
        if isinstance(self._get, Exception):
            raise self._get
        return self._get

    def post(self, url, **kw):
        self.last_post = (url, kw)
        if isinstance(self._post, Exception):
            raise self._post
        return self._post


TAGS = FakeResponse(payload={"models": [{"name": "llama3.2:latest"}]})


def test_ollama_status_ready():
    p = OllamaProvider("llama3.2:latest", session=FakeSession(get=TAGS))
    st = p.status()
    assert st.available and st.models == ["llama3.2:latest"]


def test_ollama_status_when_server_down():
    p = OllamaProvider("llama3.2", session=FakeSession(get=requests.ConnectionError("refused")))
    st = p.status()
    assert not st.available and "ollama serve" in st.message


def test_ollama_status_when_model_missing():
    p = OllamaProvider("mistral", session=FakeSession(get=TAGS))
    st = p.status()
    assert not st.available and "ollama pull mistral" in st.message


def test_ollama_generate_parses_usage():
    payload = {"model": "llama3.2:latest", "message": {"content": "Hi [S1]"}, "prompt_eval_count": 50, "eval_count": 5}
    p = OllamaProvider("llama3.2:latest", session=FakeSession(post=FakeResponse(payload=payload)))
    r = p.generate(MSGS)
    assert r.text == "Hi [S1]" and r.usage.prompt_tokens == 50 and r.usage.total_tokens == 55


def test_ollama_timeout_raises_timeout_error():
    p = OllamaProvider("m", session=FakeSession(post=requests.Timeout()))
    with pytest.raises(ProviderTimeoutError):
        p.generate(MSGS)


def test_ollama_connection_error_raises_unavailable():
    p = OllamaProvider("m", session=FakeSession(post=requests.ConnectionError()))
    with pytest.raises(ProviderUnavailableError):
        p.generate(MSGS)


def test_ollama_http_500_raises_response_error():
    p = OllamaProvider("m", session=FakeSession(post=FakeResponse(status=500, text="boom")))
    with pytest.raises(ProviderResponseError):
        p.generate(MSGS)


def test_ollama_bad_payload_raises_response_error():
    p = OllamaProvider("m", session=FakeSession(post=FakeResponse(payload={"unexpected": True})))
    with pytest.raises(ProviderResponseError):
        p.generate(MSGS)


def test_ollama_stream_yields_deltas_and_records_usage():
    lines = [
        json.dumps({"message": {"content": "Hello "}}).encode(),
        b"",
        json.dumps({"message": {"content": "world"}}).encode(),
        json.dumps({"done": True, "model": "m", "prompt_eval_count": 10, "eval_count": 2}).encode(),
    ]
    p = OllamaProvider("m", session=FakeSession(post=FakeResponse(lines=lines)))
    assert "".join(p.stream(MSGS)) == "Hello world"
    assert p.last_response.usage.completion_tokens == 2
    assert p.last_response.first_token_s is not None


def test_ollama_stream_error_event():
    lines = [json.dumps({"error": "model crashed"}).encode()]
    p = OllamaProvider("m", session=FakeSession(post=FakeResponse(lines=lines)))
    with pytest.raises(ProviderResponseError):
        list(p.stream(MSGS))


def test_openai_compatible_requires_key():
    p = OpenAICompatibleProvider("gpt", "https://api.example.com/v1", api_key="")
    assert not p.status().available
    with pytest.raises(ProviderUnavailableError):
        p.generate(MSGS)


def test_openai_compatible_refuses_plain_http_remote():
    p = OpenAICompatibleProvider("gpt", "http://api.example.com/v1", api_key="sk-test")
    assert not p.status().available


def test_openai_compatible_generate_and_auth_header():
    payload = {"model": "gpt", "choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    session = FakeSession(post=FakeResponse(payload=payload))
    p = OpenAICompatibleProvider("gpt", "https://api.example.com/v1", api_key="sk-secret-value", session=session)
    r = p.generate(MSGS)
    assert r.text == "ok" and r.usage.total_tokens == 4
    url, kw = session.last_post
    assert url.endswith("/chat/completions") and kw["headers"]["Authorization"] == "Bearer sk-secret-value"
    assert "sk-secret-value" not in repr(p)


def test_openai_compatible_auth_failure():
    p = OpenAICompatibleProvider("gpt", "https://api.example.com/v1", "k", session=FakeSession(post=FakeResponse(status=401)))
    with pytest.raises(ProviderUnavailableError, match="Authentication"):
        p.generate(MSGS)


def test_openai_compatible_stream():
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}',
        b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
        b"data: [DONE]",
    ]
    p = OpenAICompatibleProvider("gpt", "https://api.example.com/v1", "k", session=FakeSession(post=FakeResponse(lines=lines)))
    assert "".join(p.stream(MSGS)) == "Hello"
    assert p.last_response.usage.prompt_tokens == 5


def test_openai_compatible_timeout():
    p = OpenAICompatibleProvider("gpt", "https://api.example.com/v1", "k", session=FakeSession(post=requests.Timeout()))
    with pytest.raises(ProviderTimeoutError):
        p.generate(MSGS)
