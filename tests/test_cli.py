from typer.testing import CliRunner

from exec_agent.cli import app

runner = CliRunner()


def test_chat_command_runs() -> None:
    result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    assert "Executive assistant chat" in result.output


def test_config_command_runs() -> None:
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "Executive Assistant Configuration" in result.output


def test_config_command_shows_hitl() -> None:
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "hitl" in result.output


def test_chat_command_accepts_hitl_flag() -> None:
    result = runner.invoke(app, ["chat", "--hitl"])

    assert result.exit_code == 0
    assert "Human-in-the-loop approvals enabled" in result.output


def test_memory_cli_add_list_search_delete(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXEC_AGENT_DATA_DIR", str(tmp_path))
    from exec_agent.config import get_settings

    get_settings.cache_clear()
    add_result = runner.invoke(app, ["memory", "add", "User prefers concise answers", "--tag", "preference"])
    assert add_result.exit_code == 0
    assert "Added memory 1" in add_result.output

    list_result = runner.invoke(app, ["memory", "list"])
    assert list_result.exit_code == 0
    assert "User prefers concise answers" in list_result.output
    assert "preference" in list_result.output

    search_result = runner.invoke(app, ["memory", "search", "concise"])
    assert search_result.exit_code == 0
    assert "User prefers concise answers" in search_result.output

    delete_result = runner.invoke(app, ["memory", "delete", "1"])
    assert delete_result.exit_code == 0
    assert "Deleted memory 1" in delete_result.output
    get_settings.cache_clear()
