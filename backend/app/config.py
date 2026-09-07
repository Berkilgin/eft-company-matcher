from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "EFT Company Matcher"
    app_version: str = "0.6.0"

    app_data_source: str = "postgres"
    allow_csv_fallback: bool = True

    # Read DATABASE_URL from the environment or the project-root .env file.
    database_url: str = Field(
        ...,
        min_length=1,
        repr=False,
    )

    project_root: Path = PROJECT_ROOT

    companies_csv_path: Path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "companies.csv"
    )

    aliases_csv_path: Path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "aliases.csv"
    )

    embedding_model_path: Path = (
        PROJECT_ROOT
        / "models"
        / "qwen3-embedding-0.6b"
    )

    reranker_model_path: Path = (
        PROJECT_ROOT
        / "models"
        / "qwen3-reranker-0.6b"
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        hide_input_in_errors=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
