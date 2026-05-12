from datetime import timedelta
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
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite+aiosqlite:///./mycoai.db"
    jwt_secret_key: str = Field(default="change-me-in-production", min_length=16)
    jwt_algorithm: str = "HS256"
    access_token_lifetime: timedelta = timedelta(hours=1)
    refresh_token_lifetime: timedelta = timedelta(days=30)
    password_min_length: int = 8
    bcrypt_rounds: int = 12
    cors_origins: list[str] = ["http://localhost:5173"]
    rate_limit: str = "5/minute"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
