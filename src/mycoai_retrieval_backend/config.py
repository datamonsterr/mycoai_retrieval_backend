from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MYCOAI_BACKEND_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "MycoAI Retrieval Backend"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api"
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "myco_fungi_features_full"
    retrieval_default_k: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
