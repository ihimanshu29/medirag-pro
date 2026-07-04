"""
Central configuration using pydantic-settings.
All settings read from environment variables / .env file.
Never instantiate directly — use the `settings` singleton.

Supports three deployment targets via environment variables alone:
  Local:  QDRANT_HOST=localhost, POSTGRES_HOST=localhost
  Free:   QDRANT_URL=https://xyz.qdrant.io, QDRANT_API_KEY=..., DATABASE_URL=postgresql+pg8000://...
  VPS:    QDRANT_HOST=qdrant (Docker service), POSTGRES_HOST=postgres (Docker service)
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────
    groq_api_key: str | None = Field(default=None, description="Groq API key")
    groq_model_name: str = Field(default="llama-3.3-70b-versatile")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=1024)

    # ── Observability ─────────────────────────────────────────────
    langchain_api_key: str | None = Field(default=None, description="LangSmith API key")
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="medirag-pro")

    # ── Vector Store ──────────────────────────────────────────────
    # Free cloud: set QDRANT_URL + QDRANT_API_KEY (Qdrant Cloud)
    # Local/VPS:  set QDRANT_HOST + QDRANT_PORT (self-hosted Docker)
    qdrant_url: str | None = Field(default=None, description="Qdrant Cloud URL (overrides host+port)")
    qdrant_api_key: str = Field(default="", description="Qdrant Cloud API key")
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_name: str = Field(default="medical_docs")

    # ── Database ──────────────────────────────────────────────────
    # Free cloud: set DATABASE_URL directly (Neon gives a full connection string)
    # Local/VPS:  set individual POSTGRES_* vars (assembled into database_url)
    database_url_override: str | None = Field(
        default=None,
        alias="DATABASE_URL",
        description="Full database URL — overrides individual POSTGRES_* vars. Use for Neon/Supabase.",
    )
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="medirag")
    postgres_user: str = Field(default="medirag")
    postgres_password: str = Field(default="medirag_secret")

    # ── Embeddings ────────────────────────────────────────────────
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dimension: int = Field(default=384)
    reranker_model: str = Field(default="BAAI/bge-reranker-base")

    # ── Chunking ──────────────────────────────────────────────────
    parent_chunk_size: int = Field(default=1024)
    child_chunk_size: int = Field(default=256)
    chunk_overlap: int = Field(default=32)

    # ── Retrieval ─────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=20)
    rerank_top_n: int = Field(default=5)
    bm25_index_path: str = Field(default="./data/bm25_index.pkl")
    parent_chunks_path: str = Field(default="./data/parent_chunks.pkl")

    # ── Cache ─────────────────────────────────────────────────────
    cache_dir: str = Field(default="./data/semantic_cache")
    cache_similarity_threshold: float = Field(default=0.92)
    cache_max_size_gb: float = Field(default=1.0)

    # ── App ───────────────────────────────────────────────────────
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    app_version: str = Field(default="0.1.0")

    # ── Frontend ──────────────────────────────────────────────────
    # Set this to the public FastAPI URL in cloud/VPS deployments.
    # Local default: http://localhost:8000
    # Free cloud: https://your-hf-space.hf.space
    # VPS: https://api.yourdomain.com or http://YOUR_IP:8000
    api_backend_url: str = Field(
        default="http://localhost:8000",
        description="Public URL of the FastAPI backend. Read by Streamlit frontend.",
    )

    # ── CORS ──────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    # Local/free: "*"  — no restriction needed
    # VPS with domain: "https://yourdomain.com,https://www.yourdomain.com"
    cors_origins: str = Field(
        default="*",
        description="Comma-separated CORS origins. Use * for open access.",
    )

    # ── Computed ──────────────────────────────────────────────────

    @computed_field
    @property
    def database_url(self) -> str:
        """
        Returns the database connection URL.
        Priority: DATABASE_URL env var (Neon/Supabase) → assembled from POSTGRES_* vars.
        """
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def uses_qdrant_cloud(self) -> bool:
        """True when QDRANT_URL is set — means we're using Qdrant Cloud."""
        return bool(self.qdrant_url)

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @computed_field
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS_ORIGINS into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @computed_field
    @property
    def data_dir(self) -> Path:
        path = Path("./data")
        path.mkdir(exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()


# Module-level singleton for easy import
settings = get_settings()
