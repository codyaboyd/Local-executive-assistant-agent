from pathlib import Path

from exec_agent.config import Settings


def test_settings_expands_data_dir() -> None:
    settings = Settings(EXEC_AGENT_DATA_DIR="~/example-exec-agent")

    assert settings.expanded_data_dir == Path("~/example-exec-agent").expanduser()


def test_settings_reads_hitl_flag() -> None:
    settings = Settings(EXEC_AGENT_HITL="true")

    assert settings.hitl is True


def test_private_offline_profile_disables_web_and_fastcrw() -> None:
    settings = Settings(EXEC_AGENT_RUNTIME_PROFILE="private-offline", FASTCRW_ENABLED="true", EXEC_AGENT_WEB_ENABLED="true")

    assert settings.web_enabled is False
    assert settings.fastcrw_enabled is False
    assert settings.hitl is False


def test_research_online_profile_enables_self_hosted_fastcrw_base_url() -> None:
    settings = Settings(EXEC_AGENT_RUNTIME_PROFILE="research-online", FASTCRW_BASE_URL="http://fastcrw.local:3002")

    assert settings.web_enabled is True
    assert settings.fastcrw_enabled is True
    assert settings.fastcrw_base_url == "http://fastcrw.local:3002"
    assert settings.fastcrw_crawl_requires_approval is False


def test_test_hitl_profile_requires_crawl_approval() -> None:
    settings = Settings(EXEC_AGENT_RUNTIME_PROFILE="test-hitl")

    assert settings.web_enabled is True
    assert settings.fastcrw_enabled is True
    assert settings.hitl is True
    assert settings.fastcrw_crawl_requires_approval is True


def test_cpu_safe_profile_controls_runtime_defaults() -> None:
    settings = Settings(EXEC_AGENT_RUNTIME_PROFILE="cpu-safe", EXEC_AGENT_MODEL_ID="ignored", EXEC_AGENT_DEVICE="cuda")

    assert settings.model_id == "sshleifer/tiny-gpt2"
    assert settings.device == "cpu"
    assert settings.web_enabled is False
    assert settings.hitl is False
    assert str(settings.expanded_vector_db_path).endswith("profiles/cpu-safe/chroma")
    assert settings.log_level == "INFO"


def test_gpu_fast_profile_controls_runtime_defaults() -> None:
    settings = Settings(EXEC_AGENT_RUNTIME_PROFILE="gpu-fast")

    assert settings.model_id == "distilgpt2"
    assert settings.device == "cuda"
    assert settings.web_enabled is False
    assert settings.hitl is False
    assert str(settings.expanded_vector_db_path).endswith("profiles/gpu-fast/chroma")
    assert settings.log_level == "WARNING"


def test_explicit_vector_db_path_overrides_profile_default() -> None:
    settings = Settings(EXEC_AGENT_RUNTIME_PROFILE="research-online", EXEC_AGENT_VECTOR_DB_PATH="~/custom-vector")

    assert settings.expanded_vector_db_path == Path("~/custom-vector").expanduser()
