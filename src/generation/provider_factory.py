"""Build the configured :class:`LLMProvider`."""

from __future__ import annotations

from src.config import AppConfig
from src.generation.llm_provider import LLMProvider, UnavailableProvider
from src.generation.ollama_provider import OllamaProvider
from src.generation.openai_compatible_provider import OpenAICompatibleProvider


def build_provider(cfg: AppConfig) -> LLMProvider:
    if cfg.llm_provider == "ollama":
        return OllamaProvider(cfg.ollama_model, cfg.ollama_base_url, cfg.llm_timeout_s)
    if cfg.llm_provider == "openai_compatible":
        return OpenAICompatibleProvider(cfg.openai_model, cfg.openai_base_url, cfg.openai_api_key, cfg.llm_timeout_s)
    return UnavailableProvider("LLM generation is disabled (RAG_LLM_PROVIDER=none). Retrieval still works.")
