"""
Application configuration.

All settings loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # App
    app_name: str = "Gajashastra Sanskrit Intelligence Platform"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/gajashastra",
        description="Async PostgreSQL connection string",
    )
    database_url_sync: str = Field(
        default="postgresql://localhost:5432/gajashastra",
        description="Sync PostgreSQL connection string (for migrations)",
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30

    # Embeddings (Vertex AI via Application Default Credentials)
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 3072
    embedding_task_type: str = "RETRIEVAL_DOCUMENT"
    embedding_query_task_type: str = "RETRIEVAL_QUERY"
    embedding_max_chars: int = 8000
    embedding_batch_size: int = 100
    use_local_embeddings: bool = False  # Never use on Vercel - no torch
    local_embedding_model: str = "all-MiniLM-L6-v2"
    local_embedding_dimension: int = 384
    gcp_project: str = "ferrous-purpose-480705-j1"
    gcp_location: str = "us-central1"

    # LLM (extraction, assistant) — key from gcloud secrets
    anthropic_api_key: str = ""
    extraction_model: str = "claude-sonnet-4-20250514"
    assistant_model: str = "claude-sonnet-4-20250514"
    vision_model: str = "claude-haiku-4-5-20251001"

    # Search
    rrf_k: int = 60
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    mmr_lambda: float = 0.7
    default_search_limit: int = 10
    dedup_cosine_threshold: float = 0.85

    # Chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chars_per_token: int = 4
    max_chunks_per_document: int = 500

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Corpus paths
    corpus_dir: str = ""
    pages_dir: str = ""

    model_config = {"env_prefix": "GAJA_", "env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
