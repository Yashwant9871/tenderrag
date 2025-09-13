# app/config.py
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

class Settings(BaseSettings):
    # Vector DB
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    collection: str = Field("tenders", env="COLLECTION")

    # DeepInfra
    deepinfra_base: str = Field(..., env="DEEPINFRA_BASE")
    deepinfra_token: str = Field(..., env="DEEPINFRA_TOKEN")
    tokenizer_name: str = Field("gpt2", env="TOKENIZER_NAME")  # model-specific tokenizer
    # NEW: explicit vector size for embeddings (required)
    deepinfra_vector_size: int = Field(..., env="DEEPINFRA_VECTOR_SIZE")

    # Celery / Redis
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    celery_queue: str = Field("ingest_queue", env="CELERY_QUEUE")

    # Uploads
    upload_dir: str = Field(".data", env="UPLOAD_DIR")
    max_upload_size: int = Field(50 * 1024 * 1024, env="MAX_UPLOAD_SIZE")
    allowed_extensions: List[str] = Field([".pdf", ".docx"], env="ALLOWED_EXTENSIONS")

    # batching
    embed_batch: int = Field(64, env="EMBED_BATCH")
    upsert_batch: int = Field(128, env="UPsert_BATCH")

    # CORS
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")

    # MinIO
    minio_endpoint: Optional[str] = Field(None, env="MINIO_ENDPOINT")
    minio_access_key: Optional[str] = Field(None, env="MINIO_ACCESS_KEY")
    minio_secret_key: Optional[str] = Field(None, env="MINIO_SECRET_KEY")

    secret_key: str = Field("change_me", env="SECRET_KEY")
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")

    # DB
    database_url: str = Field("postgresql+asyncpg://localhost:5432/tenderdb", env="DATABASE_URL")

    # Prometheus
    prometheus_enabled: bool = Field(True, env="PROMETHEUS_ENABLED")

    # Pydantic v2 config
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("tokenizer_name", mode="before")
    def _normalize_tokenizer_name(cls, v):
        if v is None:
            return "gpt2"
        return str(v).strip()
    # ---- field validators (pydantic v2 style) ----
    @field_validator("allowed_extensions", mode="before")
    def _split_allowed_extensions(cls, v):
        """
        Allows ALLOWED_EXTENSIONS as comma-separated string in env, or as a list.
        Example: '.pdf,.docx'
        """
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("cors_origins", mode="before")
    def _split_cors_origins(cls, v):
        """
        Allows CORS_ORIGINS as comma-separated string in env, or as a list.
        """
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("deepinfra_vector_size", mode="before")
    def _validate_vector_size(cls, v):
        """
        Accepts the env value as string or int and ensures it's a positive int.
        """
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        if not isinstance(v, int) or v <= 0:
            raise ValueError("DEEPINFRA_VECTOR_SIZE must be a positive integer")
        return v

settings = Settings()
