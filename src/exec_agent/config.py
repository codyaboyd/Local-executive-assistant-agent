"""Application configuration loading."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env files."""

    app_name: str = "Local Executive Assistant"
    environment: str = Field(default="development", validation_alias="EXEC_AGENT_ENV")
    log_level: str = Field(default="INFO", validation_alias="EXEC_AGENT_LOG_LEVEL")
    data_dir: Path = Field(default=Path("~/.local/share/exec-agent"), validation_alias="EXEC_AGENT_DATA_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def expanded_data_dir(self) -> Path:
        """Return the configured data directory with user home expanded."""

        return self.data_dir.expanduser()


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
