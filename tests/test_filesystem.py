from pathlib import Path

import pytest

from app.tools import filesystem
from exec_agent.config import get_settings
from exec_agent.safety import UserFacingError


def _configure(monkeypatch, tmp_path, *, autonomy="human_approved"):
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    monkeypatch.setenv("EXEC_AGENT_ALLOWED_DIRS", str(allowed))
    monkeypatch.setenv("EXEC_AGENT_READONLY_DIRS", "")
    monkeypatch.setenv("EXEC_AGENT_BLOCKED_PATHS", "/etc,/root,/home/*/.ssh,/home/*/.gnupg")
    monkeypatch.setenv("EXEC_AGENT_MAX_FILE_SIZE_MB", "1")
    monkeypatch.setenv("EXEC_AGENT_AUTONOMY_LEVEL", autonomy)
    get_settings.cache_clear()
    return allowed


def test_filesystem_read_list_search_allowed_dir(tmp_path, monkeypatch) -> None:
    allowed = _configure(monkeypatch, tmp_path)
    (allowed / "notes.txt").write_text("hello keyword", encoding="utf-8")

    assert filesystem.list_dir(allowed) == ["notes.txt"]
    assert filesystem.read_file(allowed / "notes.txt") == "hello keyword"
    assert filesystem.search_files("keyword", allowed) == [str(allowed / "notes.txt")]
    get_settings.cache_clear()


def test_filesystem_blocks_paths_outside_allowed_dir(tmp_path, monkeypatch) -> None:
    allowed = _configure(monkeypatch, tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(UserFacingError):
        filesystem.read_file(outside)
    with pytest.raises(UserFacingError):
        filesystem.read_file(allowed / ".." / "secret.txt")
    get_settings.cache_clear()


def test_filesystem_blocks_symlink_escape(tmp_path, monkeypatch) -> None:
    allowed = _configure(monkeypatch, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (allowed / "link.txt").symlink_to(outside)

    with pytest.raises(UserFacingError):
        filesystem.read_file(allowed / "link.txt")
    get_settings.cache_clear()


def test_filesystem_requires_autonomy_for_overwrite_delete_and_move(tmp_path, monkeypatch) -> None:
    allowed = _configure(monkeypatch, tmp_path)
    file_path = allowed / "a.txt"
    file_path.write_text("old", encoding="utf-8")

    with pytest.raises(UserFacingError):
        filesystem.write_file(file_path, "new")
    with pytest.raises(UserFacingError):
        filesystem.move_file(file_path, allowed / "b.txt")
    with pytest.raises(UserFacingError):
        filesystem.delete_file(file_path)
    get_settings.cache_clear()


def test_filesystem_allows_guarded_actions_with_autonomy(tmp_path, monkeypatch) -> None:
    allowed = _configure(monkeypatch, tmp_path, autonomy="autonomous_limited")
    file_path = allowed / "a.txt"

    filesystem.write_file(file_path, "old")
    filesystem.write_file(file_path, "new")
    moved = filesystem.move_file(file_path, allowed / "b.txt")
    assert moved.read_text(encoding="utf-8") == "new"
    filesystem.delete_file(moved)
    assert not moved.exists()
    get_settings.cache_clear()
