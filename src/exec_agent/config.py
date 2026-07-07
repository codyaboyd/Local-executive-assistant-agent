"""Application configuration loading."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

RuntimeProfile = Literal["private-offline", "research-online", "test-hitl"]

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
    runtime_profile: RuntimeProfile = Field(default="private-offline", validation_alias="EXEC_AGENT_RUNTIME_PROFILE")
    hitl: bool = Field(default=False, validation_alias="EXEC_AGENT_HITL")
    web_enabled: bool = Field(default=False, validation_alias="EXEC_AGENT_WEB_ENABLED")
    fastcrw_enabled: bool = Field(default=False, validation_alias="FASTCRW_ENABLED")
    fastcrw_crawl_requires_approval: bool = Field(default=False, validation_alias="FASTCRW_CRAWL_REQUIRES_APPROVAL")
    fastcrw_base_url: str = Field(default="http://localhost:3002", validation_alias="FASTCRW_BASE_URL")
    fastcrw_api_prefix: str = Field(default="/v1", validation_alias="FASTCRW_API_PREFIX")
    fastcrw_api_key: str | None = Field(default=None, validation_alias="FASTCRW_API_KEY")
    fastcrw_timeout_seconds: int = Field(default=30, ge=1, validation_alias="FASTCRW_TIMEOUT_SECONDS")
    fastcrw_max_results: int = Field(default=5, ge=1, validation_alias="FASTCRW_MAX_RESULTS")
    fastcrw_enable_scrape: bool = Field(default=True, validation_alias="FASTCRW_ENABLE_SCRAPE")
    fastcrw_enable_crawl: bool = Field(default=False, validation_alias="FASTCRW_ENABLE_CRAWL")

    def model_post_init(self, __context: object) -> None:
        """Apply named runtime profile defaults after environment loading."""

        if self.runtime_profile == "private-offline":
            self.web_enabled = False
            self.fastcrw_enabled = False
            self.fastcrw_crawl_requires_approval = False
        elif self.runtime_profile == "research-online":
            self.web_enabled = True
            self.fastcrw_enabled = True
            self.fastcrw_crawl_requires_approval = False
        elif self.runtime_profile == "test-hitl":
            self.web_enabled = True
            self.fastcrw_enabled = True
            self.hitl = True
            self.fastcrw_crawl_requires_approval = True

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
