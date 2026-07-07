"""Application configuration loading."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env files."""

    app_name: str = "Local Executive Assistant"
    environment: str = Field(default="development", validation_alias="EXEC_AGENT_ENV")
    log_level: str = Field(default="INFO", validation_alias="EXEC_AGENT_LOG_LEVEL")
    data_dir: Path = Field(default=Path("~/.local/share/exec-agent"), validation_alias="EXEC_AGENT_DATA_DIR")
    model_id: str = Field(default="sshleifer/tiny-gpt2", validation_alias="EXEC_AGENT_MODEL_ID")
    image_caption_model_id: str = Field(default="Salesforce/blip-image-captioning-base", validation_alias="EXEC_AGENT_IMAGE_CAPTION_MODEL_ID")
    image_qa_model_id: str = Field(default="dandelin/vilt-b32-finetuned-vqa", validation_alias="EXEC_AGENT_IMAGE_QA_MODEL_ID")
    device: Literal["cpu", "cuda", "auto"] = Field(default="auto", validation_alias="EXEC_AGENT_DEVICE")
    max_tokens: int = Field(default=64, ge=1, validation_alias="EXEC_AGENT_MAX_TOKENS")
    temperature: float = Field(default=0.7, ge=0.0, validation_alias="EXEC_AGENT_TEMPERATURE")
    hitl: bool = Field(default=False, validation_alias="EXEC_AGENT_HITL")

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
