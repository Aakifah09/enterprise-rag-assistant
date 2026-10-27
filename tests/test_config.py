import logging

import pytest

from src.config import AppConfig, ConfigError
from src.generation.llm_provider import UnavailableProvider
from src.generation.ollama_provider import OllamaProvider
from src.generation.provider_factory import build_provider
from src.utils.logging_utils import RedactSecretsFilter


def test_defaults_are_valid():
    AppConfig().validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": 50},
        {"chunk_overlap": 600, "chunk_size": 800},
        {"chunk_overlap": -1},
        {"top_k": 0},
        {"retrieval_strategy": "magic"},
        {"hybrid_alpha": 1.5},
        {"min_relevance": 2.0},
        {"llm_provider": "skynet"},
        {"temperature": 3.0},
        {"prompt_variant": "unknown"},
        {"device": "tpu"},
        {"max_upload_mb": 0},
    ],
)
def test_invalid_values_rejected(overrides):
    with pytest.raises(ConfigError):
        AppConfig().with_overrides(**overrides)


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_RETRIEVAL_STRATEGY", "dense")
    monkeypatch.setenv("RAG_MIN_RELEVANCE", "0.42")
    cfg = AppConfig.from_env()
    assert cfg.top_k == 7 and cfg.retrieval_strategy == "dense" and cfg.effective_min_relevance == 0.42


def test_env_non_integer_rejected(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "five")
    with pytest.raises(ConfigError):
        AppConfig.from_env()


def test_model_specific_relevance_floor():
    assert AppConfig(embedding_model="BAAI/bge-small-en-v1.5").effective_min_relevance == 0.60
    assert AppConfig(embedding_model="unknown/model").effective_min_relevance == 0.35


def test_public_dict_hides_secrets_and_paths():
    d = AppConfig(openai_api_key="sk-very-secret").public_dict()
    assert "openai_api_key" not in d and "data_dir" not in d
    assert "sk-very-secret" not in repr(AppConfig(openai_api_key="sk-very-secret"))


def test_provider_factory():
    assert isinstance(build_provider(AppConfig(llm_provider="ollama")), OllamaProvider)
    none = build_provider(AppConfig(llm_provider="none"))
    assert isinstance(none, UnavailableProvider) and not none.status().available


def test_log_filter_redacts_keys():
    record = logging.LogRecord("x", logging.INFO, "f", 1, "calling with Bearer abcdefghijklmnop and sk-1234567890abcdef", (), None)
    RedactSecretsFilter().filter(record)
    assert "abcdefghijklmnop" not in record.getMessage() and "sk-1234567890" not in record.getMessage()
