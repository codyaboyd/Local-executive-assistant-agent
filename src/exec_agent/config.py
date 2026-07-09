"""Application configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RuntimeProfile = Literal["cpu-safe", "gpu-fast", "private-offline", "research-online", "test-hitl"]
DeviceMode = Literal["cpu", "cuda", "auto"]
ModelPreset = Literal["default", "low_vram", "cpu_only", "quality", "coding", "research"]
AutonomyLevel = Literal["off", "suggest_only", "human_approved", "autonomous_limited", "autonomous_full"]


@dataclass(frozen=True)
class RuntimeProfileDefaults:
    """Defaults applied by a named runtime profile."""

    name: RuntimeProfile
    description: str
    model_id: str
    device: DeviceMode
    web_enabled: bool
    fastcrw_enabled: bool
    hitl: bool
    vector_db_subdir: str
    log_level: str
    fastcrw_crawl_requires_approval: bool = False


RUNTIME_PROFILES: dict[RuntimeProfile, RuntimeProfileDefaults] = {
    "cpu-safe": RuntimeProfileDefaults(
        name="cpu-safe",
        description="Conservative local CPU execution with web disabled and HITL enabled.",
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
        web_enabled=False,
        fastcrw_enabled=False,
        hitl=False,
        vector_db_subdir="profiles/cpu-safe/chroma",
        log_level="INFO",
    ),
    "gpu-fast": RuntimeProfileDefaults(
        name="gpu-fast",
        description="Local GPU execution for faster inference with web disabled and HITL enabled.",
        model_id="distilgpt2",
        device="cuda",
        web_enabled=False,
        fastcrw_enabled=False,
        hitl=False,
        vector_db_subdir="profiles/gpu-fast/chroma",
        log_level="WARNING",
    ),
    "private-offline": RuntimeProfileDefaults(
        name="private-offline",
        description="Private local-only mode; disables web access and FastCRW with HITL enabled.",
        model_id="sshleifer/tiny-gpt2",
        device="auto",
        web_enabled=False,
        fastcrw_enabled=False,
        hitl=False,
        vector_db_subdir="profiles/private-offline/chroma",
        log_level="INFO",
    ),
    "research-online": RuntimeProfileDefaults(
        name="research-online",
        description="Online research mode using the configured self-hosted FastCRW endpoint.",
        model_id="sshleifer/tiny-gpt2",
        device="auto",
        web_enabled=True,
        fastcrw_enabled=True,
        hitl=False,
        vector_db_subdir="profiles/research-online/chroma",
        log_level="INFO",
    ),
    "test-hitl": RuntimeProfileDefaults(
        name="test-hitl",
        description="Integration-test mode with online research and human approval gates enabled.",
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
        web_enabled=True,
        fastcrw_enabled=True,
        hitl=True,
        vector_db_subdir="profiles/test-hitl/chroma",
        log_level="DEBUG",
        fastcrw_crawl_requires_approval=True,
    ),
}


def get_runtime_profile_defaults(profile: RuntimeProfile) -> RuntimeProfileDefaults:
    """Return defaults for a runtime profile."""

    return RUNTIME_PROFILES[profile]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env files."""

    app_name: str = "Local Executive Assistant"
    environment: str = Field(default="development", validation_alias="EXEC_AGENT_ENV")
    log_level: str = Field(default="INFO", validation_alias="EXEC_AGENT_LOG_LEVEL")
    data_dir: Path = Field(default=Path("~/.local/share/exec-agent"), validation_alias="EXEC_AGENT_DATA_DIR")
    model_id: str = Field(default="sshleifer/tiny-gpt2", validation_alias="EXEC_AGENT_MODEL_ID")
    model_preset: ModelPreset = Field(default="default", validation_alias="EXEC_AGENT_MODEL_PRESET")
    model_auto_pull: bool = Field(default=False, validation_alias="EXEC_AGENT_MODEL_AUTO_PULL")
    max_vram_gb: int = Field(default=16, ge=1, validation_alias="EXEC_AGENT_MAX_VRAM_GB")
    general_model_id: str = Field(default="auto", validation_alias="EXEC_AGENT_GENERAL_MODEL_ID")
    coding_model_id: str = Field(default="auto", validation_alias="EXEC_AGENT_CODING_MODEL_ID")
    summary_model_id: str = Field(default="auto", validation_alias="EXEC_AGENT_SUMMARY_MODEL_ID")
    docqa_model_id: str = Field(default="auto", validation_alias="EXEC_AGENT_DOCQA_MODEL_ID")
    research_model_id: str = Field(default="auto", validation_alias="EXEC_AGENT_RESEARCH_MODEL_ID")
    tool_model_id: str = Field(default="auto", validation_alias="EXEC_AGENT_TOOL_MODEL_ID")
    embedding_model_id: str = Field(default="auto", validation_alias="EXEC_AGENT_EMBEDDING_MODEL_ID")
    vision_model_id: str = Field(default="auto", validation_alias="EXEC_AGENT_VISION_MODEL_ID")
    image_caption_model_id: str = Field(default="Salesforce/blip-image-captioning-base", validation_alias="EXEC_AGENT_IMAGE_CAPTION_MODEL_ID")
    image_qa_model_id: str = Field(default="dandelin/vilt-b32-finetuned-vqa", validation_alias="EXEC_AGENT_IMAGE_QA_MODEL_ID")
    device: DeviceMode = Field(default="auto", validation_alias="EXEC_AGENT_DEVICE")
    max_tokens: int = Field(default=64, ge=1, validation_alias="EXEC_AGENT_MAX_TOKENS")
    temperature: float = Field(default=0.7, ge=0.0, validation_alias="EXEC_AGENT_TEMPERATURE")
    runtime_profile: RuntimeProfile = Field(default="private-offline", validation_alias="EXEC_AGENT_RUNTIME_PROFILE")
    hitl: bool = Field(default=False, validation_alias="EXEC_AGENT_HITL")
    actions_hitl: bool = Field(default=True, validation_alias="EXEC_AGENT_ACTIONS_HITL")
    autonomy_level: AutonomyLevel = Field(default="human_approved", validation_alias="EXEC_AGENT_AUTONOMY_LEVEL")
    max_autonomous_steps: int = Field(default=25, ge=1, validation_alias="EXEC_AGENT_MAX_AUTONOMOUS_STEPS")
    require_approval_for_dangerous_commands: bool = Field(default=True, validation_alias="EXEC_AGENT_REQUIRE_APPROVAL_FOR_DANGEROUS_COMMANDS")
    task_timeout_seconds: int = Field(default=1800, ge=1, validation_alias="EXEC_AGENT_TASK_TIMEOUT_SECONDS")
    local_only: bool = Field(default=False, validation_alias="EXEC_AGENT_LOCAL_ONLY")
    web_enabled: bool = Field(default=False, validation_alias="EXEC_AGENT_WEB_ENABLED")
    vector_db_path: Path | None = Field(default=None, validation_alias="EXEC_AGENT_VECTOR_DB_PATH")
    fastcrw_enabled: bool = Field(default=False, validation_alias="FASTCRW_ENABLED")
    fastcrw_crawl_requires_approval: bool = Field(default=False, validation_alias="FASTCRW_CRAWL_REQUIRES_APPROVAL")
    fastcrw_base_url: str = Field(default="http://localhost:3002", validation_alias="FASTCRW_BASE_URL")
    fastcrw_api_prefix: str = Field(default="/v1", validation_alias="FASTCRW_API_PREFIX")
    fastcrw_api_key: str | None = Field(default=None, validation_alias="FASTCRW_API_KEY")
    fastcrw_timeout_seconds: int = Field(default=30, ge=1, validation_alias="FASTCRW_TIMEOUT_SECONDS")
    model_timeout_seconds: int = Field(default=120, ge=1, validation_alias="EXEC_AGENT_MODEL_TIMEOUT_SECONDS")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1, validation_alias="EXEC_AGENT_MAX_UPLOAD_BYTES")
    allowed_upload_extensions: str = Field(default=".pdf,.docx,.png,.jpg,.jpeg,.webp,.txt,.md", validation_alias="EXEC_AGENT_ALLOWED_UPLOAD_EXTENSIONS")
    structured_logging: bool = Field(default=True, validation_alias="EXEC_AGENT_STRUCTURED_LOGGING")
    fastcrw_max_results: int = Field(default=5, ge=1, validation_alias="FASTCRW_MAX_RESULTS")
    fastcrw_enable_scrape: bool = Field(default=True, validation_alias="FASTCRW_ENABLE_SCRAPE")
    fastcrw_enable_crawl: bool = Field(default=False, validation_alias="FASTCRW_ENABLE_CRAWL")
    allowed_dirs: str = Field(default="./workspace,./data,./uploads", validation_alias="EXEC_AGENT_ALLOWED_DIRS")
    readonly_dirs: str = Field(default="", validation_alias="EXEC_AGENT_READONLY_DIRS")
    blocked_paths: str = Field(default="/etc,/root,/home/*/.ssh,/home/*/.gnupg", validation_alias="EXEC_AGENT_BLOCKED_PATHS")
    max_file_size_mb: int = Field(default=25, ge=1, validation_alias="EXEC_AGENT_MAX_FILE_SIZE_MB")
    shell_enabled: bool = Field(default=True, validation_alias="EXEC_AGENT_SHELL_ENABLED")
    shell_workdir: Path = Field(default=Path("./workspace"), validation_alias="EXEC_AGENT_SHELL_WORKDIR")
    shell_timeout_seconds: int = Field(default=120, ge=1, validation_alias="EXEC_AGENT_SHELL_TIMEOUT_SECONDS")
    shell_max_output_chars: int = Field(default=20000, ge=1, validation_alias="EXEC_AGENT_SHELL_MAX_OUTPUT_CHARS")
    shell_allowlist: str = Field(
        default="python,python3,pip,uv,pytest,node,npm,git,grep,find,ls,cat,pwd,mkdir,cp,mv,rm,touch,sed,awk,curl",
        validation_alias="EXEC_AGENT_SHELL_ALLOWLIST",
    )
    shell_denylist: str = Field(
        default="sudo,su,chmod,chown,dd,mkfs,mount,umount,ssh,scp,rsync,shutdown,reboot,systemctl,service,docker",
        validation_alias="EXEC_AGENT_SHELL_DENYLIST",
    )

    def model_post_init(self, __context: object) -> None:
        """Apply named runtime profile controls after environment loading."""

        profile = get_runtime_profile_defaults(self.runtime_profile)
        profile_was_selected = "runtime_profile" in self.model_fields_set

        if profile_was_selected or "model_id" not in self.model_fields_set:
            self.model_id = profile.model_id
        if profile_was_selected or "device" not in self.model_fields_set:
            self.device = profile.device
        if profile_was_selected or "web_enabled" not in self.model_fields_set:
            self.web_enabled = profile.web_enabled
        if profile_was_selected or "fastcrw_enabled" not in self.model_fields_set:
            self.fastcrw_enabled = profile.fastcrw_enabled
        if profile_was_selected or "hitl" not in self.model_fields_set:
            self.hitl = profile.hitl
        if profile_was_selected or "log_level" not in self.model_fields_set:
            self.log_level = profile.log_level
        if profile_was_selected or "fastcrw_crawl_requires_approval" not in self.model_fields_set:
            self.fastcrw_crawl_requires_approval = profile.fastcrw_crawl_requires_approval
        if self.model_preset == "cpu_only":
            self.device = "cpu"
        if self.local_only:
            self.web_enabled = False
            self.fastcrw_enabled = False
            self.fastcrw_enable_scrape = False
            self.fastcrw_enable_crawl = False
        if self.vector_db_path is None:
            self.vector_db_path = self.expanded_data_dir / profile.vector_db_subdir

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def expanded_data_dir(self) -> Path:
        """Return the configured data directory with user home expanded."""

        return self.data_dir.expanduser()

    @property
    def expanded_vector_db_path(self) -> Path:
        """Return the configured vector database directory with user home expanded."""

        if self.vector_db_path is None:
            return self.expanded_data_dir / get_runtime_profile_defaults(self.runtime_profile).vector_db_subdir
        return self.vector_db_path.expanduser()


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
