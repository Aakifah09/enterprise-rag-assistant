"""Any OpenAI-compatible Chat Completions endpoint.

Works with OpenAI and other services that expose ``POST {base_url}/chat/completions``
-- for example BytePlus ModelArk (base URL
``https://ark.ap-southeast.bytepluses.com/api/v3``, model = your endpoint/model id),
vLLM, LM Studio or Ollama's own ``/v1`` endpoint. Implemented with ``requests``
so no vendor SDK is needed. The API key is read from configuration and never logged.
"""

from __future__ import annotations

import json
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


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key: str, timeout_s: float = 120.0, session=None) -> None:
        super().__init__(model, timeout_s)
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.session = session or requests.Session()

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatibleProvider(model={self.model!r}, base_url={self.base_url!r})"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def status(self) -> ProviderStatus:
        if not self._api_key:
            return ProviderStatus(False, "No API key configured (set OPENAI_COMPATIBLE_API_KEY in .env).")
        if not self.base_url.startswith("https://") and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            return ProviderStatus(False, "Refusing to send an API key over plain HTTP to a remote host.")
        return ProviderStatus(True, f"Configured · {self.model} @ {self.base_url} (not contacted until first query)")

    def _post(self, payload: dict, stream: bool) -> requests.Response:
        st = self.status()
        if not st.available:
            raise ProviderUnavailableError(st.message)
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=(10, self.timeout_s),
                stream=stream,
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(f"The API did not answer within {self.timeout_s:.0f} s.") from exc
        except (requests.ConnectionError, OSError) as exc:
            raise ProviderUnavailableError(f"Could not reach {self.base_url}.") from exc
        if resp.status_code in (401, 403):
            raise ProviderUnavailableError(f"Authentication failed (HTTP {resp.status_code}). Check the API key.")
        if resp.status_code >= 400:
            raise ProviderResponseError(f"API returned HTTP {resp.status_code}.")
        return resp

    def _payload(self, messages: list[ChatMessage], temperature: float, max_tokens: int, stream: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def generate(self, messages: list[ChatMessage], temperature: float = 0.1, max_tokens: int = 512) -> LLMResponse:
        start = time.perf_counter()
        resp = self._post(self._payload(messages, temperature, max_tokens, False), stream=False)
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError("The API returned an unexpected payload.") from exc
        usage = data.get("usage") or {}
        response = LLMResponse(
            text=text,
            model=data.get("model", self.model),
            provider=self.name,
            latency_s=time.perf_counter() - start,
            usage=LLMUsage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
        )
        self.last_response = response
        return response

    def stream(self, messages: list[ChatMessage], temperature: float = 0.1, max_tokens: int = 512) -> Iterator[str]:
        start = time.perf_counter()
        resp = self._post(self._payload(messages, temperature, max_tokens, True), stream=True)
        parts: list[str] = []
        usage: dict = {}
        model = self.model
        first: float | None = None
        try:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                event = json.loads(body)
                model = event.get("model", model)
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content") or ""
                    if delta:
                        if first is None:
                            first = time.perf_counter() - start
                        parts.append(delta)
                        yield delta
        except requests.Timeout as exc:
            raise ProviderTimeoutError("The API stream stalled.") from exc
        except ValueError as exc:
            raise ProviderResponseError("Malformed streaming data from the API.") from exc
        finally:
            resp.close()
        self.last_response = LLMResponse(
            text="".join(parts),
            model=model,
            provider=self.name,
            latency_s=time.perf_counter() - start,
            usage=LLMUsage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
            first_token_s=first,
        )
