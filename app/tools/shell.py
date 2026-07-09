"""Safe command-line execution tool for the local assistant.

Commands are parsed with :mod:`shlex` and executed with subprocess argument
arrays (``shell=False``) so shell metacharacters are passed as literal
arguments instead of being interpreted by a shell.
"""

from __future__ import annotations

import os
import shlex
import sqlite3
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Literal

from exec_agent.config import get_settings
from exec_agent.safety import UserFacingError

OutputStream = Literal["stdout", "stderr"]
OutputListener = Callable[[dict[str, object]], None]
_OUTPUT_LISTENERS: list[OutputListener] = []
_APPROVAL_AUTONOMY = {"autonomous_limited", "autonomous_full"}


@dataclass(frozen=True)
class CommandResult:
    """Captured result for one command execution."""

    id: int
    command: str
    argv: list[str]
    cwd: str
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    started_at: str
    finished_at: str
    approval_required: bool
    approval_reason: str


def register_output_listener(listener: OutputListener) -> None:
    """Register a callback used by terminal or web UI layers for live output."""

    _OUTPUT_LISTENERS.append(listener)


def clear_output_listeners() -> None:
    """Remove registered output listeners, primarily for tests."""

    _OUTPUT_LISTENERS.clear()


def default_shell_history_path() -> Path:
    """Return the default SQLite database path for shell command history."""

    return get_settings().expanded_data_dir / "shell_history.sqlite3"


def run_command(command: str, cwd: str | Path | None = None, timeout: int | float | None = None) -> CommandResult:
    """Run an allowlisted command inside the configured shell workspace.

    Potentially destructive or network/package-management commands are blocked
    unless the configured autonomy level allows pre-approved execution.
    """

    settings = get_settings()
    if not settings.shell_enabled:
        raise UserFacingError("Shell execution is disabled. Set EXEC_AGENT_SHELL_ENABLED=true to enable it.")
    argv = _parse_command(command)
    approval_required, approval_reason = _approval_requirement(argv)
    if approval_required and settings.autonomy_level not in _APPROVAL_AUTONOMY:
        raise UserFacingError(
            f"Command requires approval before execution ({approval_reason}): {command}. "
            "Set EXEC_AGENT_AUTONOMY_LEVEL=autonomous_limited or autonomous_full after approval."
        )
    _validate_policy(argv, allow_unlisted=approval_required)
    workdir = _resolve_workdir(cwd)
    requested_timeout = float(timeout if timeout is not None else settings.shell_timeout_seconds)
    max_output_chars = settings.shell_max_output_chars

    started = _now()
    start_time = time.monotonic()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    exit_code = 1
    timed_out = False

    process = subprocess.Popen(  # noqa: S603 - argv is validated and shell=False prevents shell injection.
        argv,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        env=os.environ.copy(),
        bufsize=1,
    )

    def consume(stream_name: OutputStream) -> None:
        stream = process.stdout if stream_name == "stdout" else process.stderr
        if stream is None:
            return
        for chunk in iter(stream.readline, ""):
            _append_limited(stdout_parts if stream_name == "stdout" else stderr_parts, chunk, max_output_chars)
            print(chunk, end="")
            _emit_output({"stream": stream_name, "chunk": chunk, "command": command, "cwd": str(workdir)})

    threads = [Thread(target=consume, args=("stdout",), daemon=True), Thread(target=consume, args=("stderr",), daemon=True)]
    for thread in threads:
        thread.start()
    try:
        exit_code = process.wait(timeout=requested_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        exit_code = -9
    for thread in threads:
        thread.join(timeout=1)
    if timed_out:
        message = f"Command timed out after {requested_timeout:g} seconds.\n"
        _append_limited(stderr_parts, message, max_output_chars)
        print(message, end="")
        _emit_output({"stream": "stderr", "chunk": message, "command": command, "cwd": str(workdir)})

    finished = _now()
    duration = time.monotonic() - start_time
    result = CommandResult(
        id=0,
        command=command,
        argv=argv,
        cwd=str(workdir),
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        exit_code=exit_code,
        duration_seconds=duration,
        started_at=started,
        finished_at=finished,
        approval_required=approval_required,
        approval_reason=approval_reason,
    )
    return _store_history(result)


def history(limit: int = 50) -> list[CommandResult]:
    """Return recent shell command history."""

    _init_db(default_shell_history_path())
    with sqlite3.connect(default_shell_history_path()) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM shell_commands ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_result(row) for row in rows]


def _parse_command(command: str) -> list[str]:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise UserFacingError(f"Could not parse command safely: {exc}") from exc
    if not argv:
        raise UserFacingError("Command cannot be empty.")
    if any(token in command for token in (";", "&&", "||", "|", "`", "$(", ">", "<")):
        raise UserFacingError("Shell operators and redirection are not supported. Pass a single command with arguments.")
    return argv


def _split_csv(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _approval_requirement(argv: list[str]) -> tuple[bool, str]:
    exe = Path(argv[0]).name
    allowlist = _split_csv(get_settings().shell_allowlist)
    if exe not in allowlist:
        return True, f"{exe!r} is not explicitly allowlisted"
    if exe in {"rm", "mv", "curl"}:
        return True, f"{exe} is approval-gated"
    if exe == "git" and len(argv) > 1 and argv[1] == "push":
        return True, "git push is approval-gated"
    if exe == "npm" and len(argv) > 1 and argv[1] == "install":
        return True, "npm install is approval-gated"
    if exe == "pip" and len(argv) > 1 and argv[1] == "install":
        return True, "pip install is approval-gated"
    return False, ""


def _validate_policy(argv: list[str], *, allow_unlisted: bool = False) -> None:
    denylist = _split_csv(get_settings().shell_denylist)
    exe = Path(argv[0]).name
    if exe in denylist:
        raise UserFacingError(f"Command is denied by policy: {exe}")
    if exe not in _split_csv(get_settings().shell_allowlist) and not allow_unlisted:
        raise UserFacingError(f"Command is not allowlisted: {exe}")


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve(strict=False)


def _resolve_workdir(cwd: str | Path | None) -> Path:
    root = _resolve_config_path(get_settings().shell_workdir)
    requested = root if cwd is None else _resolve_config_path(cwd)
    try:
        requested.relative_to(root)
    except ValueError as exc:
        raise UserFacingError(f"Shell working directory must stay inside {root}: {requested}") from exc
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def _append_limited(parts: list[str], chunk: str, max_chars: int) -> None:
    current = sum(len(part) for part in parts)
    if current >= max_chars:
        return
    parts.append(chunk[: max_chars - current])


def _emit_output(event: dict[str, object]) -> None:
    for listener in list(_OUTPUT_LISTENERS):
        try:
            listener(event)
        except Exception:
            continue


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS shell_commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                argv TEXT NOT NULL,
                cwd TEXT NOT NULL,
                stdout TEXT NOT NULL,
                stderr TEXT NOT NULL,
                exit_code INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                approval_required INTEGER NOT NULL,
                approval_reason TEXT NOT NULL
            )
            """
        )


def _store_history(result: CommandResult) -> CommandResult:
    db_path = default_shell_history_path()
    _init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO shell_commands
            (command, argv, cwd, stdout, stderr, exit_code, duration_seconds, started_at, finished_at, approval_required, approval_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (result.command, "\n".join(result.argv), result.cwd, result.stdout, result.stderr, result.exit_code, result.duration_seconds, result.started_at, result.finished_at, int(result.approval_required), result.approval_reason),
        )
        command_id = int(cursor.lastrowid)
    return CommandResult(command_id, result.command, result.argv, result.cwd, result.stdout, result.stderr, result.exit_code, result.duration_seconds, result.started_at, result.finished_at, result.approval_required, result.approval_reason)


def _row_to_result(row: sqlite3.Row) -> CommandResult:
    return CommandResult(
        id=int(row["id"]),
        command=str(row["command"]),
        argv=str(row["argv"]).split("\n"),
        cwd=str(row["cwd"]),
        stdout=str(row["stdout"]),
        stderr=str(row["stderr"]),
        exit_code=int(row["exit_code"]),
        duration_seconds=float(row["duration_seconds"]),
        started_at=str(row["started_at"]),
        finished_at=str(row["finished_at"]),
        approval_required=bool(row["approval_required"]),
        approval_reason=str(row["approval_reason"]),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
