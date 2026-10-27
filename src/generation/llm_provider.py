"""Provider-agnostic LLM interface.

The RAG pipeline depends only on :class:`LLMProvider`. Adding a provider means
implementing ``status``, ``generate`` and optionally ``stream``; the pipeline,
prompts, citation checks and evaluation stay unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


class LLMError(RuntimeError):
    """Base class for provider failures."""


class ProviderUnavailableError(LLMError):
    """The provider cannot be reached or the model is not installed."""


class ProviderTimeoutError(LLMError):
    """The provider did not respond within the configured timeout."""


class ProviderResponseError(LLMError):
    """The provider answered with an error or an unparseable payload."""


@dataclass
class ChatMessage:
    role: str  # system | user | assistant
    content: str


@dataclass
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    latency_s: float
    usage: LLMUsage = field(default_factory=LLMUsage)
    first_token_s: float | None = None


@dataclass
class ProviderStatus:
    available: bool
    message: str
    models: list[str] = field(default_factory=list)


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, timeout_s: float = 120.0) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.last_response: LLMResponse | None = None

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Check reachability and whether ``self.model`` can be used."""

    @abstractmethod
    def generate(self, messages: list[ChatMessage], temperature: float = 0.1, max_tokens: int = 512) -> LLMResponse:
        """Return the full completion. Raises :class:`LLMError` subclasses."""

    def stream(self, messages: list[ChatMessage], temperature: float = 0.1, max_tokens: int = 512) -> Iterator[str]:
        """Yield text deltas. Default: a single chunk from :meth:`generate`.

        After the iterator is exhausted ``self.last_response`` holds the full
        response including usage and latency.
        """
        response = self.generate(messages, temperature, max_tokens)
        self.last_response = response
        yield response.text


class UnavailableProvider(LLMProvider):
    """Explicit "no LLM configured" provider. It never produces text."""

    name = "none"

    def __init__(self, reason: str = "No LLM provider is configured.") -> None:
        super().__init__(model="none")
        self.reason = reason

    def status(self) -> ProviderStatus:
        return ProviderStatus(False, self.reason)

    def generate(self, messages: list[ChatMessage], temperature: float = 0.1, max_tokens: int = 512) -> LLMResponse:
        raise ProviderUnavailableError(self.reason)


def estimate_cost(usage: LLMUsage, price_in_per_1k: float | None, price_out_per_1k: float | None) -> float | None:
    """Cost in the configured currency, or ``None`` if tokens or prices are unknown."""
    if price_in_per_1k is None or price_out_per_1k is None:
        return None
    if usage.prompt_tokens is None or usage.completion_tokens is None:
        return None
    return usage.prompt_tokens / 1000 * price_in_per_1k + usage.completion_tokens / 1000 * price_out_per_1k
