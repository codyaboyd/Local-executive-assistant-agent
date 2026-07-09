"""Autonomous task execution loop with SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Literal, Protocol

from exec_agent.config import get_settings

AutonomyLevel = Literal["off", "suggest_only", "human_approved", "autonomous_limited", "autonomous_full"]
TaskStatus = Literal["running", "completed", "blocked", "cancelled", "failed"]

DANGEROUS_COMMAND_HINTS = ("rm ", "sudo", "mkfs", "shutdown", "reboot", "chmod -R", "chown -R", ":(){", "dd if=")


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    description: str
    autonomy_level: AutonomyLevel
    status: TaskStatus
    plan: list[str]
    final_summary: str
    error: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TaskStepRecord:
    id: int
    task_id: str
    step_number: int
    phase: str
    action: str
    tool_name: str
    result: str
    error: str
    created_at: str


@dataclass(frozen=True)
class ToolResult:
    output: str
    complete: bool = False
    blocked: bool = False
    error: str = ""


class TaskTool(Protocol):
    name: str
    dangerous: bool

    def run(self, task: TaskRecord, step_number: int, context: str) -> ToolResult: ...


class PlannerTool:
    name = "planner"
    dangerous = False

    def run(self, task: TaskRecord, step_number: int, context: str) -> ToolResult:
        del step_number, context
        plan = "\n".join(f"- {item}" for item in task.plan)
        return ToolResult(f"Plan prepared for goal: {task.description}\n{plan}")


class ReflectionTool:
    name = "reflect"
    dangerous = False

    def run(self, task: TaskRecord, step_number: int, context: str) -> ToolResult:
        if step_number >= len(task.plan):
            return ToolResult(f"Completed goal: {task.description}", complete=True)
        return ToolResult(f"Inspected progress after step {step_number}. Continue. Context: {context[-300:]}")


ProgressSink = Callable[[str], None]


def default_tasks_path() -> Path:
    return get_settings().expanded_data_dir / "tasks.sqlite3"


class TaskStore:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_tasks_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    autonomy_level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan TEXT NOT NULL DEFAULT '[]',
                    final_summary TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    action TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    result TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_tasks_updated_at ON tasks(updated_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_task_steps_task_id ON task_steps(task_id)")

    def create(self, description: str, autonomy_level: AutonomyLevel, plan: list[str]) -> TaskRecord:
        now = _now()
        task_id = uuid.uuid4().hex[:12]
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, description, autonomy_level, "running", json.dumps(plan), "", "", now, now),
            )
        return self.get(task_id)  # type: ignore[return-value]

    def get(self, task_id: str) -> TaskRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return _task_from_row(row) if row else None

    def list(self, limit: int = 20) -> list[TaskRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [_task_from_row(row) for row in rows]

    def latest(self) -> TaskRecord | None:
        tasks = self.list(limit=1)
        return tasks[0] if tasks else None

    def update_status(self, task_id: str, status: TaskStatus, final_summary: str = "", error: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET status = ?, final_summary = ?, error = ?, updated_at = ? WHERE task_id = ?",
                (status, final_summary, error, _now(), task_id),
            )

    def update_plan(self, task_id: str, plan: list[str]) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE tasks SET plan = ?, updated_at = ? WHERE task_id = ?", (json.dumps(plan), _now(), task_id))

    def add_step(self, task_id: str, step_number: int, phase: str, action: str, tool_name: str, result: str = "", error: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO task_steps (task_id, step_number, phase, action, tool_name, result, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, step_number, phase, action, tool_name, result, error, _now()),
            )

    def steps(self, task_id: str) -> list[TaskStepRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM task_steps WHERE task_id = ? ORDER BY step_number, id", (task_id,)).fetchall()
        return [_step_from_row(row) for row in rows]

    def cancel(self, task_id: str) -> bool:
        if self.get(task_id) is None:
            return False
        self.update_status(task_id, "cancelled", "Task cancelled by user.")
        return True


class AutonomousTaskRunner:
    def __init__(self, store: TaskStore | None = None, tools: Iterable[TaskTool] | None = None, progress: ProgressSink | None = None) -> None:
        self.store = store or TaskStore()
        self.tools = list(tools or [PlannerTool(), ReflectionTool()])
        self.progress = progress or (lambda message: None)

    def run(self, description: str, autonomy_level: AutonomyLevel | None = None) -> TaskRecord:
        settings = get_settings()
        level = autonomy_level or settings.autonomy_level
        plan = make_initial_plan(description)
        task = self.store.create(description, level, plan)
        self._emit(f"Task {task.task_id} started ({level})")
        if level in ("off", "suggest_only", "human_approved"):
            summary = "Autonomy requires human approval; suggested plan stored for review." if level == "human_approved" else "Autonomy disabled; suggested plan stored for review."
            self.store.update_status(task.task_id, "blocked", summary)
            self._emit(summary)
            return self.store.get(task.task_id)  # type: ignore[return-value]
        started = time.monotonic()
        context = ""
        repeated_results: dict[str, int] = {}
        try:
            for step_number in range(1, settings.max_autonomous_steps + 1):
                latest = self.store.get(task.task_id)
                if latest and latest.status == "cancelled":
                    self._emit(f"Task {task.task_id} cancelled by emergency stop.")
                    break
                if time.monotonic() - started > settings.task_timeout_seconds:
                    self.store.update_status(task.task_id, "blocked", error="Task timeout reached.")
                    break
                phase = LOOP_PHASES[(step_number - 1) % len(LOOP_PHASES)]
                tool = self._choose_tool(step_number)
                if settings.require_approval_for_dangerous_commands and tool.dangerous:
                    self.store.update_status(task.task_id, "blocked", error=f"Dangerous tool {tool.name!r} requires approval.")
                    break
                action = task.plan[min(step_number - 1, len(task.plan) - 1)] if task.plan else phase
                self._emit(f"[{phase}] {action} via {tool.name}")
                result = tool.run(task, step_number, context)
                self.store.add_step(task.task_id, step_number, phase, action, tool.name, result.output, result.error)
                if result.error:
                    self.store.update_status(task.task_id, "failed", error=result.error)
                    break
                fingerprint = f"{tool.name}:{result.output[:200]}"
                repeated_results[fingerprint] = repeated_results.get(fingerprint, 0) + 1
                if repeated_results[fingerprint] >= 3:
                    self.store.update_status(task.task_id, "blocked", error="Loop safeguard triggered: repeated result.")
                    break
                context = f"{context}\n{result.output}"[-4000:]
                if result.blocked:
                    self.store.update_status(task.task_id, "blocked", result.output)
                    break
                if result.complete:
                    self.store.update_status(task.task_id, "completed", result.output)
                    break
            else:
                self.store.update_status(task.task_id, "blocked", error="Maximum autonomous steps reached.")
        except Exception as exc:  # noqa: BLE001
            self.store.update_status(task.task_id, "failed", error=str(exc))
        final = self.store.get(task.task_id)
        self._emit(f"Task {task.task_id} {final.status if final else 'finished'}")
        return final  # type: ignore[return-value]

    def _choose_tool(self, step_number: int) -> TaskTool:
        return self.tools[min(step_number - 1, len(self.tools) - 1)]

    def _emit(self, message: str) -> None:
        self.progress(message)


LOOP_PHASES = ["understand goal", "make plan", "choose tools", "execute step", "inspect result", "revise plan"]


def make_initial_plan(description: str) -> list[str]:
    return [
        f"Understand goal: {description}",
        "Make a concise plan with success criteria.",
        "Choose the safest available tool for the next step.",
        "Execute one bounded step and capture output.",
        "Inspect the result for errors, completion, or blockers.",
        "Revise the plan before continuing.",
    ]


def is_dangerous_command(command: str) -> bool:
    lowered = command.lower()
    return any(hint in lowered for hint in DANGEROUS_COMMAND_HINTS)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _task_from_row(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(str(row["task_id"]), str(row["description"]), str(row["autonomy_level"]), str(row["status"]), json.loads(str(row["plan"] or "[]")), str(row["final_summary"]), str(row["error"]), str(row["created_at"]), str(row["updated_at"]))  # type: ignore[arg-type]


def _step_from_row(row: sqlite3.Row) -> TaskStepRecord:
    return TaskStepRecord(int(row["id"]), str(row["task_id"]), int(row["step_number"]), str(row["phase"]), str(row["action"]), str(row["tool_name"]), str(row["result"]), str(row["error"]), str(row["created_at"]))
