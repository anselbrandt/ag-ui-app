import os

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings using Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )
    database_url: str = Field(
        default="postgresql+asyncpg://postgres@localhost:5432/database",
        description="Database connection URL",
    )
    db_echo: bool = Field(
        default=False,
        description="Echo SQL queries to logs",
    )
    geo_api_key: SecretStr = Field(default=SecretStr(""), description="Mapbox API Key")
    openai_api_key: SecretStr = Field(
        default=SecretStr(""), description="OpenAI API Key"
    )
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )
    tavily_api_key: SecretStr = Field(
        default=SecretStr(""), description="Tavily API Key"
    )
    unipile_account_id: str = Field(default="", description="Unipile account ID")
    unipile_api_key: SecretStr = Field(
        default=SecretStr(""), description="Unipile API Key"
    )
    unipile_dsn: str = Field(default="", description="Unipile DSN")
    weather_api_key: SecretStr = Field(
        default=SecretStr(""), description="Tomorrow.io API Key"
    )


settings = Settings()
