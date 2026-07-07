from pathlib import Path

from exec_agent.config import Settings


def test_settings_expands_data_dir() -> None:
    settings = Settings(EXEC_AGENT_DATA_DIR="~/example-exec-agent")

    assert settings.expanded_data_dir == Path("~/example-exec-agent").expanduser()


def test_settings_reads_hitl_flag() -> None:
    settings = Settings(EXEC_AGENT_HITL="true")

    assert settings.hitl is True
