from pathlib import Path

import pytest

from app.tools import shell
from exec_agent.config import Settings, get_settings
from exec_agent.safety import UserFacingError


def configure(monkeypatch, tmp_path: Path, **extra: str) -> Path:
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("EXEC_AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EXEC_AGENT_SHELL_WORKDIR", str(workspace))
    monkeypatch.setenv("EXEC_AGENT_SHELL_ENABLED", "true")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return workspace


def test_shell_settings_defaults() -> None:
    settings = Settings()

    assert settings.shell_enabled is True
    assert settings.shell_workdir == Path("./workspace")
    assert settings.shell_timeout_seconds == 120
    assert settings.shell_max_output_chars == 20000
    assert "python" in settings.shell_allowlist
    assert "sudo" in settings.shell_denylist


def test_run_command_captures_output_and_history(monkeypatch, tmp_path, capsys) -> None:
    workspace = configure(monkeypatch, tmp_path)

    result = shell.run_command("python -c 'print(42)'")

    assert result.exit_code == 0
    assert result.cwd == str(workspace.resolve())
    assert "42" in result.stdout
    assert "42" in capsys.readouterr().out
    assert shell.history(limit=1)[0].command == "python -c 'print(42)'"


def test_run_command_blocks_shell_operators(monkeypatch, tmp_path) -> None:
    configure(monkeypatch, tmp_path)

    with pytest.raises(UserFacingError):
        shell.run_command("python -V; rm -rf .")


def test_run_command_stays_in_workspace(monkeypatch, tmp_path) -> None:
    configure(monkeypatch, tmp_path)

    with pytest.raises(UserFacingError):
        shell.run_command("pwd", cwd=tmp_path.parent)


def test_approval_gated_command_blocks_without_autonomy(monkeypatch, tmp_path) -> None:
    configure(monkeypatch, tmp_path)

    with pytest.raises(UserFacingError):
        shell.run_command("rm old.txt")


def test_output_listener_receives_chunks(monkeypatch, tmp_path) -> None:
    configure(monkeypatch, tmp_path)
    events = []
    shell.register_output_listener(events.append)
    try:
        shell.run_command("python -c 'print(123)'", timeout=5)
    finally:
        shell.clear_output_listeners()

    assert any("123" in str(event.get("chunk")) for event in events)


def test_shell_cli_run_and_history(monkeypatch, tmp_path) -> None:
    from typer.testing import CliRunner
    from exec_agent.cli import app

    configure(monkeypatch, tmp_path)
    runner = CliRunner()

    run_result = runner.invoke(app, ["shell", "run", "python -c 'print(456)'"])
    history_result = runner.invoke(app, ["shell", "history", "--limit", "1"])

    assert run_result.exit_code == 0
    assert "456" in run_result.output
    assert "Exit code" in run_result.output
    assert history_result.exit_code == 0
    assert "python -c 'print(456)'" in history_result.output
