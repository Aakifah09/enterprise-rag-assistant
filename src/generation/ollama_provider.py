"""Local inference through the Ollama HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

from __future__ import annotations

import json
import logging
import time
from typing import Iterator

import requests

from src.generation.llm_provider import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderResponseError,
    ProviderStatus,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout_s: float = 180.0, session=None):
        super().__init__(model, timeout_s)
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def list_models(self) -> list[str]:
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
            return sorted(m["name"] for m in resp.json().get("models", []))
        except requests.Timeout as exc:
            raise ProviderTimeoutError("Ollama did not respond within 3 s.") from exc
        except (requests.ConnectionError, OSError) as exc:
            raise ProviderUnavailableError(
                f"Ollama is not reachable at {self.base_url}. Install it from https://ollama.com and run `ollama serve`."
            ) from exc
        except (requests.HTTPError, ValueError, KeyError) as exc:
            raise ProviderResponseError(f"Unexpected response from Ollama: {exc}") from exc

    def status(self) -> ProviderStatus:
        try:
            models = self.list_models()
        except (ProviderUnavailableError, ProviderTimeoutError, ProviderResponseError) as exc:
            return ProviderStatus(False, str(exc))
        if not models:
            return ProviderStatus(False, "Ollama is running but no models are installed. Run `ollama pull llama3.2`.", [])
        if self.model not in models and f"{self.model}:latest" not in models:
            return ProviderStatus(
                False, f"Model '{self.model}' is not installed. Run `ollama pull {self.model}`.", models
            )
        return ProviderStatus(True, f"Ollama ready · {self.model}", models)

    def _payload(self, messages: list[ChatMessage], temperature: float, max_tokens: int, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 7},
        }

    def _post(self, payload: dict, stream: bool) -> requests.Response:
        try:
            resp = self.session.post(
                f"{self.base_url}/api/chat", json=payload, timeout=(5, self.timeout_s), stream=stream
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(f"Ollama did not answer within {self.timeout_s:.0f} s.") from exc
        except (requests.ConnectionError, OSError) as exc:
            raise ProviderUnavailableError(f"Ollama is not reachable at {self.base_url}.") from exc
        if resp.status_code == 404:
            raise ProviderUnavailableError(f"Model '{self.model}' is not available in Ollama.")
        if resp.status_code >= 400:
            raise ProviderResponseError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp

    def generate(self, messages: list[ChatMessage], temperature: float = 0.1, max_tokens: int = 512) -> LLMResponse:
        start = time.perf_counter()
        resp = self._post(self._payload(messages, temperature, max_tokens, stream=False), stream=False)
        try:
            data = resp.json()
            text = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderResponseError("Ollama returned an unexpected payload.") from exc
        response = LLMResponse(
            text=text,
            model=data.get("model", self.model),
            provider=self.name,
            latency_s=time.perf_counter() - start,
            usage=LLMUsage(data.get("prompt_eval_count"), data.get("eval_count")),
        )
        self.last_response = response
        return response

    def stream(self, messages: list[ChatMessage], temperature: float = 0.1, max_tokens: int = 512) -> Iterator[str]:
        start = time.perf_counter()
        resp = self._post(self._payload(messages, temperature, max_tokens, stream=True), stream=True)
        parts: list[str] = []
        first: float | None = None
        final: dict = {}
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise ProviderResponseError(f"Ollama error: {event['error']}")
                delta = event.get("message", {}).get("content", "")
                if delta:
                    if first is None:
                        first = time.perf_counter() - start
                    parts.append(delta)
                    yield delta
                if event.get("done"):
                    final = event
                    break
        except requests.Timeout as exc:
            raise ProviderTimeoutError(f"Ollama stream stalled for more than {self.timeout_s:.0f} s.") from exc
        except (requests.ConnectionError, OSError) as exc:
            raise ProviderUnavailableError("Connection to Ollama was lost during streaming.") from exc
        except ValueError as exc:
            raise ProviderResponseError("Ollama sent malformed streaming data.") from exc
        finally:
            resp.close()
        self.last_response = LLMResponse(
            text="".join(parts),
            model=final.get("model", self.model),
            provider=self.name,
            latency_s=time.perf_counter() - start,
            usage=LLMUsage(final.get("prompt_eval_count"), final.get("eval_count")),
            first_token_s=first,
        )
