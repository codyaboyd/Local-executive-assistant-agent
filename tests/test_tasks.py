from exec_agent.tasks import AutonomousTaskRunner, TaskStore, ToolResult


class MockTool:
    name = "mock"
    dangerous = False

    def __init__(self):
        self.calls = 0

    def run(self, task, step_number, context):
        self.calls += 1
        return ToolResult(f"mock step {step_number}", complete=self.calls == 2)


def test_autonomous_task_execution_with_mocked_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("EXEC_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EXEC_AGENT_MAX_AUTONOMOUS_STEPS", "5")
    from exec_agent.config import get_settings

    get_settings.cache_clear()
    store = TaskStore(tmp_path / "tasks.sqlite3")
    messages = []
    task = AutonomousTaskRunner(store=store, tools=[MockTool()], progress=messages.append).run(
        "prepare board packet", autonomy_level="autonomous_full"
    )

    assert task.status == "completed"
    assert "Completed" not in task.error
    assert len(store.steps(task.task_id)) == 2
    assert any("started" in message for message in messages)
    get_settings.cache_clear()


def test_suggest_only_stores_plan_without_steps(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = AutonomousTaskRunner(store=store, tools=[MockTool()]).run("outline launch", autonomy_level="suggest_only")

    assert task.status == "blocked"
    assert task.plan[0].startswith("Understand goal")
    assert store.steps(task.task_id) == []


def test_loop_safeguard_blocks_repeated_results(tmp_path, monkeypatch):
    monkeypatch.setenv("EXEC_AGENT_MAX_AUTONOMOUS_STEPS", "5")
    from exec_agent.config import get_settings

    get_settings.cache_clear()

    class RepeatingTool:
        name = "repeat"
        dangerous = False

        def run(self, task, step_number, context):
            return ToolResult("same output")

    task = AutonomousTaskRunner(store=TaskStore(tmp_path / "tasks.sqlite3"), tools=[RepeatingTool()]).run(
        "repeat", autonomy_level="autonomous_full"
    )

    assert task.status == "blocked"
    assert "repeated result" in task.error
    get_settings.cache_clear()
