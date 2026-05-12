from functools import lru_cache

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

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "myco_fungi_features_full"
    qdrant_api_key: str | None = None

    feature_extractor: str = "EfficientNetV2B0"
    default_k: int = 5
    default_aggregation: str = "weighted"
    result_cache_ttl_seconds: int = 3600

    monorepo_root: str = ""
    species_weights_path: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
