"""Central, validated configuration for the Enterprise Knowledge Assistant.

Values come from (in increasing priority): built-in defaults, a `.env` file in
the project root, and real environment variables. The UI can override the
retrieval settings per session via :meth:`AppConfig.with_overrides`; the
result is validated again so invalid combinations never reach the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

try:  # python-dotenv is optional; the app works with plain env vars.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

RETRIEVAL_STRATEGIES = ("dense", "keyword", "hybrid", "rrf")
LLM_PROVIDERS = ("ollama", "openai_compatible", "none")
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".markdown")


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Known embedding models and the settings that matter for retrieval.

    ``default_min_relevance`` is the cosine-similarity floor below which the
    evidence-sufficiency gate refuses to answer. Cosine scales differ between
    models, so the floor is model specific. The values were chosen by
    inspecting answerable vs. unanswerable questions in the sample benchmark
    (see docs/evaluation.md) and are a starting point, not a universal truth.
    """

    name: str
    query_prefix: str = ""
    default_min_relevance: float = 0.35
    description: str = ""


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    "sentence-transformers/all-MiniLM-L6-v2": EmbeddingModelSpec(
        name="sentence-transformers/all-MiniLM-L6-v2",
        default_min_relevance=0.35,
        description="384-d, fast general-purpose model (22M params).",
    ),
    "BAAI/bge-small-en-v1.5": EmbeddingModelSpec(
        name="BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        default_min_relevance=0.60,
        description="384-d retrieval-tuned model (33M params); uses a query instruction.",
    ),
}


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""

    # Storage
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    # Ingestion
    max_upload_mb: int = 20
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Embeddings & retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_strategy: str = "hybrid"
    top_k: int = 5
    hybrid_alpha: float = 0.6  # weight of the dense score in weighted fusion
    candidate_pool: int = 30  # candidates per retriever before fusion
    min_relevance: float | None = None  # None -> model default
    device: str = "auto"  # auto | cpu | cuda

    # Generation
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = field(default="", repr=False)
    llm_timeout_s: int = 180
    temperature: float = 0.1
    max_answer_tokens: int = 512
    prompt_variant: str = "strict"
    price_per_1k_input: float | None = None
    price_per_1k_output: float | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "knowledge_assistant.db"

    @property
    def effective_min_relevance(self) -> float:
        if self.min_relevance is not None:
            return self.min_relevance
        spec = EMBEDDING_MODELS.get(self.embedding_model)
        return spec.default_min_relevance if spec else 0.35

    @property
    def chunking_key(self) -> str:
        return chunking_key(self.chunk_size, self.chunk_overlap)

    @property
    def llm_model(self) -> str:
        if self.llm_provider == "ollama":
            return self.ollama_model
        if self.llm_provider == "openai_compatible":
            return self.openai_model
        return "none"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build a configuration from environment variables and validate it."""
        data_dir = Path(_env("RAG_DATA_DIR", str(PROJECT_ROOT / "data")))
        outputs_dir = Path(_env("RAG_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs")))
        min_rel = _env_float("RAG_MIN_RELEVANCE", None)
        cfg = cls(
            data_dir=data_dir,
            outputs_dir=outputs_dir,
            max_upload_mb=_env_int("RAG_MAX_UPLOAD_MB", 20),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 800),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 120),
            embedding_model=_env("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            retrieval_strategy=_env("RAG_RETRIEVAL_STRATEGY", "hybrid"),
            top_k=_env_int("RAG_TOP_K", 5),
            hybrid_alpha=float(_env("RAG_HYBRID_ALPHA", "0.6")),
            min_relevance=min_rel,
            device=_env("RAG_DEVICE", "auto"),
            llm_provider=_env("RAG_LLM_PROVIDER", "ollama"),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=_env("OLLAMA_MODEL", "llama3.2:latest"),
            openai_base_url=_env("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1"),
            openai_model=_env("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini"),
            openai_api_key=os.environ.get("OPENAI_COMPATIBLE_API_KEY", ""),
            llm_timeout_s=_env_int("RAG_LLM_TIMEOUT_S", 180),
            temperature=float(_env("RAG_TEMPERATURE", "0.1")),
            max_answer_tokens=_env_int("RAG_MAX_ANSWER_TOKENS", 512),
            prompt_variant=_env("RAG_PROMPT_VARIANT", "strict"),
            price_per_1k_input=_env_float("RAG_PRICE_PER_1K_INPUT", None),
            price_per_1k_output=_env_float("RAG_PRICE_PER_1K_OUTPUT", None),
        )
        cfg.validate()
        return cfg

    def with_overrides(self, **overrides: Any) -> "AppConfig":
        """Return a validated copy with some fields replaced."""
        clean = {k: v for k, v in overrides.items() if v is not None or k == "min_relevance"}
        cfg = replace(self, **clean)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise :class:`ConfigError` if any value is out of range."""
        from src.generation.prompt_templates import PROMPT_VARIANTS  # avoid cycle

        if not 100 <= self.chunk_size <= 8000:
            raise ConfigError("chunk_size must be between 100 and 8000 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size // 2 + 1:
            raise ConfigError("chunk_overlap must be >= 0 and at most half of chunk_size")
        if not 1 <= self.top_k <= 50:
            raise ConfigError("top_k must be between 1 and 50")
        if self.retrieval_strategy not in RETRIEVAL_STRATEGIES:
            raise ConfigError(f"retrieval_strategy must be one of {RETRIEVAL_STRATEGIES}")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ConfigError("hybrid_alpha must be between 0 and 1")
        if self.candidate_pool < self.top_k:
            raise ConfigError("candidate_pool must be >= top_k")
        if self.min_relevance is not None and not -1.0 <= self.min_relevance <= 1.0:
            raise ConfigError("min_relevance must be a cosine similarity in [-1, 1]")
        if self.llm_provider not in LLM_PROVIDERS:
            raise ConfigError(f"llm_provider must be one of {LLM_PROVIDERS}")
        if not 1 <= self.max_upload_mb <= 200:
            raise ConfigError("max_upload_mb must be between 1 and 200")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("temperature must be between 0 and 2")
        if self.llm_timeout_s <= 0:
            raise ConfigError("llm_timeout_s must be positive")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigError("device must be auto, cpu or cuda")
        if self.prompt_variant not in PROMPT_VARIANTS:
            raise ConfigError(f"prompt_variant must be one of {tuple(PROMPT_VARIANTS)}")
        if not self.embedding_model.strip():
            raise ConfigError("embedding_model must not be empty")

    def public_dict(self) -> dict[str, Any]:
        """Configuration as a JSON-safe dict with secrets and local paths removed."""
        d = asdict(self)
        d.pop("openai_api_key", None)
        d.pop("data_dir", None)
        d.pop("outputs_dir", None)
        d["effective_min_relevance"] = self.effective_min_relevance
        return d


def chunking_key(chunk_size: int, chunk_overlap: int) -> str:
    """Stable identifier for a chunking configuration."""
    return f"cs{chunk_size}_ov{chunk_overlap}"


def query_prefix_for(model_name: str) -> str:
    spec = EMBEDDING_MODELS.get(model_name)
    return spec.query_prefix if spec else ""
