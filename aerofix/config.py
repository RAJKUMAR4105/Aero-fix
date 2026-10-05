"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with safe local defaults."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AEROFIX_", extra="ignore")

    data_dir: Path = Field(default=Path("data"))
    chroma_dir: Path = Field(default=Path(".aerofix/chroma"))
    sqlite_path: Path = Field(default=Path(".aerofix/aerofix.sqlite"))
    checkpointer_path: Path = Field(default=Path(".aerofix/checkpoints.sqlite"))
    collection_name: str = "aerofix_manuals"
    retrieval_top_k: int = 6
    # Calibrated against the query-aware blend in retrieval_confidence():
    # grounded in-corpus questions score ~0.50-0.76, out-of-corpus and
    # underspecified questions score below 0.45.
    confidence_threshold: float = 0.45
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    # Local embeddings by default: no OpenAI key is required to index or query.
    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Hybrid retrieval: candidates fetched from the vector index before the
    # keyword (BM25) side is fused in.
    retrieval_candidate_k: int = 30
    hybrid: bool = True
    # Worker processes used when batch-embedding (0/1 = embed in-process).
    # Measured on this machine: in-process torch is faster than the pool.
    embedding_workers: int = 1

    def ensure_runtime_directories(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpointer_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    # Load plain provider credentials (OPENAI_API_KEY / ANTHROPIC_API_KEY) from
    # .env into the process environment; pydantic only maps the AEROFIX_ prefix.
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:  # pragma: no cover - dotenv is optional
        pass
    return Settings()
