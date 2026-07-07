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
