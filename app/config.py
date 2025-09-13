# app/config.py
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Vector DB
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    collection: str = Field("tenders", env="COLLECTION")

    # DeepInfra
    deepinfra_base: str = Field("https://api.deepinfra.com/v1/openai", env="DEEPINFRA_BASE")
    deepinfra_token: str = Field(..., env="DEEPINFRA_TOKEN")

    # Celery / Redis
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    celery_queue: str = Field("ingest_queue", env="CELERY_QUEUE")
    deepinfra_vector_size: int = Field(1024, env="DEEPINFRA_VECTOR_SIZE")

    # Uploads
    upload_dir: str = Field(".data", env="UPLOAD_DIR")
    max_upload_size: int = Field(50 * 1024 * 1024, env="MAX_UPLOAD_SIZE")
    allowed_extensions: List[str] = Field([".pdf", ".docx"], env="ALLOWED_EXTENSIONS")

    # batching
    embed_batch: int = Field(64, env="EMBED_BATCH")
    upsert_batch: int = Field(128, env="UPsert_BATCH")
    database_url: str = Field(..., env="DATABASE_URL")
    # CORS
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")

    # --- New fields for your .env ---
    minio_endpoint: Optional[str] = Field(None, env="MINIO_ENDPOINT")
    minio_access_key: Optional[str] = Field(None, env="MINIO_ACCESS_KEY")
    minio_secret_key: Optional[str] = Field(None, env="MINIO_SECRET_KEY")

    secret_key: str = Field("change_me", env="SECRET_KEY")
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # still ignore truly random envs
    }

settings = Settings()
