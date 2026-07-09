from exec_agent.config import get_settings
from exec_agent.safety import UserFacingError
from exec_agent.services import get_backend, safety_snapshot

import pytest


def test_backend_exposes_shared_safety_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("EXEC_AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EXEC_AGENT_ALLOWED_DIRS", str(tmp_path / "workspace"))
    monkeypatch.setenv("EXEC_AGENT_AUTONOMY_LEVEL", "human_approved")
    get_settings.cache_clear()

    snapshot = safety_snapshot()

    assert snapshot.autonomy_level == "human_approved"
    assert str(tmp_path / "workspace") in snapshot.allowed_dirs


def test_backend_filesystem_uses_same_allowed_directory_policy(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("shared backend", encoding="utf-8")
    monkeypatch.setenv("EXEC_AGENT_ALLOWED_DIRS", str(workspace))
    get_settings.cache_clear()

    backend = get_backend()

    assert backend.list_files(workspace) == ["note.txt"]
    assert backend.read_file(workspace / "note.txt") == "shared backend"
    outside = tmp_path / "outside.txt"
    outside.write_text("blocked", encoding="utf-8")
    with pytest.raises(UserFacingError):
        backend.read_file(outside)
