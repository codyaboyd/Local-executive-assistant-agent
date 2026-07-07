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
