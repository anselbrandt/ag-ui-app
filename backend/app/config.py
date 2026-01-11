import json
import os
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_allowed_users() -> List[str]:
    with open("app/data/allowed_users.json", "r") as file:
        data = json.load(file)
        return data["allowed_users"]


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
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    tavily_api_key: str = Field(default="", description="Tavily API Key")
    geo_api_key: str = Field(default="", description="Mapbox API Key")
    weather_api_key: str = Field(default="", description="Tomorrow.io API Key")


settings = Settings()
