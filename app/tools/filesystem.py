"""Controlled filesystem access tools for predetermined local directories."""

from __future__ import annotations

import fnmatch
import logging
import shutil
from pathlib import Path
from typing import Any

from exec_agent.config import get_settings
from exec_agent.safety import UserFacingError

logger = logging.getLogger(__name__)

_WRITE_AUTONOMY = {"autonomous_limited", "autonomous_full"}


def _split_paths(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_config_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve(strict=False)


def _allowed_dirs() -> list[Path]:
    return [_resolve_config_path(path) for path in _split_paths(get_settings().allowed_dirs)]


def _readonly_dirs() -> list[Path]:
    return [_resolve_config_path(path) for path in _split_paths(get_settings().readonly_dirs)]


def _blocked_patterns() -> list[str]:
    return _split_paths(get_settings().blocked_paths)


def _max_file_bytes() -> int:
    return get_settings().max_file_size_mb * 1024 * 1024


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_user_path(path: str | Path, *, for_write: bool = False) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw

    if for_write and not raw.exists():
        parent = raw.parent.resolve(strict=True)
        resolved = parent / raw.name
    else:
        resolved = raw.resolve(strict=True)

    text = resolved.as_posix()
    for pattern in _blocked_patterns():
        absolute_pattern = _resolve_config_path(pattern).as_posix() if not Path(pattern).is_absolute() else pattern
        normalized_pattern = absolute_pattern.rstrip("/")
        if (
            fnmatch.fnmatch(text, normalized_pattern)
            or fnmatch.fnmatch(text, f"{normalized_pattern}/*")
            or (not any(char in normalized_pattern for char in "*?[") and text.startswith(f"{normalized_pattern}/"))
        ):
            raise UserFacingError(f"Path is blocked by policy: {resolved}")

    if not any(_is_relative_to(resolved, root) for root in _allowed_dirs()):
        allowed = ", ".join(str(path) for path in _allowed_dirs())
        raise UserFacingError(f"Path is outside allowed directories: {resolved}. Allowed: {allowed}")
    return resolved


def _ensure_writable(path: Path) -> None:
    for root in _readonly_dirs():
        if _is_relative_to(path, root):
            raise UserFacingError(f"Path is read-only by policy: {path}")


def _require_approval(action: str, target: Path) -> None:
    if get_settings().autonomy_level not in _WRITE_AUTONOMY:
        raise UserFacingError(f"{action} requires approval for {target}; set EXEC_AGENT_AUTONOMY_LEVEL=autonomous_limited or autonomous_full to allow non-interactive execution.")


def _check_size(path: Path) -> None:
    if path.is_file() and path.stat().st_size > _max_file_bytes():
        raise UserFacingError(f"File exceeds maximum size of {get_settings().max_file_size_mb} MiB: {path}")


def _log(action: str, **fields: Any) -> None:
    logger.info("filesystem action", extra={"tool": "filesystem", "event": action, **fields})


def list_dir(path: str | Path) -> list[str]:
    """List one allowed directory without following symlinks outside allowed roots."""

    root = _resolve_user_path(path)
    if not root.is_dir():
        raise UserFacingError(f"Expected a directory: {root}")
    entries = sorted(child.name + ("/" if child.is_dir() else "") for child in root.iterdir())
    _log("list_dir", path=str(root))
    return entries


def read_file(path: str | Path) -> str:
    """Read an allowed file within the configured size limit."""

    file_path = _resolve_user_path(path)
    if not file_path.is_file():
        raise UserFacingError(f"Expected a file: {file_path}")
    _check_size(file_path)
    content = file_path.read_text(encoding="utf-8")
    _log("read_file", path=str(file_path))
    return content


def write_file(path: str | Path, content: str) -> Path:
    """Write a UTF-8 file, requiring approval before overwriting existing files."""

    file_path = _resolve_user_path(path, for_write=True)
    _ensure_writable(file_path)
    if file_path.exists():
        if not file_path.is_file():
            raise UserFacingError(f"Expected a file: {file_path}")
        _require_approval("Overwrite", file_path)
    if len(content.encode("utf-8")) > _max_file_bytes():
        raise UserFacingError(f"Content exceeds maximum size of {get_settings().max_file_size_mb} MiB: {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    _log("write_file", path=str(file_path))
    return file_path


def edit_file(path: str | Path, patch: str | dict[str, str] | list[dict[str, str]]) -> Path:
    """Apply simple text replacement edits to a file.

    ``patch`` may be ``{"old": "...", "new": "..."}``, a list of those replacements,
    or a string appended to the file. Multiple replacements are treated as a bulk edit.
    """

    file_path = _resolve_user_path(path)
    _ensure_writable(file_path)
    if not file_path.is_file():
        raise UserFacingError(f"Expected a file: {file_path}")
    _check_size(file_path)
    replacements: list[dict[str, str]]
    if isinstance(patch, str):
        replacements = [{"old": "", "new": patch}]
    elif isinstance(patch, dict):
        replacements = [patch]
    else:
        replacements = patch
    if len(replacements) > 1:
        _require_approval("Bulk edit", file_path)
    content = file_path.read_text(encoding="utf-8")
    updated = content
    for replacement in replacements:
        old = replacement.get("old", "")
        new = replacement.get("new", "")
        if old:
            if old not in updated:
                raise UserFacingError(f"Patch text not found in {file_path}")
            updated = updated.replace(old, new, 1)
        else:
            updated += new
    if len(updated.encode("utf-8")) > _max_file_bytes():
        raise UserFacingError(f"Edited content exceeds maximum size of {get_settings().max_file_size_mb} MiB: {file_path}")
    file_path.write_text(updated, encoding="utf-8")
    _log("edit_file", path=str(file_path))
    return file_path


def copy_file(src: str | Path, dst: str | Path) -> Path:
    """Copy an allowed file to an allowed destination, requiring approval before overwrite."""

    source = _resolve_user_path(src)
    destination = _resolve_user_path(dst, for_write=True)
    _ensure_writable(destination)
    if not source.is_file():
        raise UserFacingError(f"Expected a file: {source}")
    _check_size(source)
    if destination.exists():
        _require_approval("Overwrite", destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    _log("copy_file", path=str(destination), src=str(source))
    return destination


def move_file(src: str | Path, dst: str | Path) -> Path:
    """Move an allowed file, requiring approval because it removes the source path."""

    source = _resolve_user_path(src)
    destination = _resolve_user_path(dst, for_write=True)
    _ensure_writable(source)
    _ensure_writable(destination)
    _require_approval("Move", source)
    if destination.exists():
        _require_approval("Overwrite", destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    _log("move_file", path=str(destination), src=str(source))
    return destination


def delete_file(path: str | Path) -> None:
    """Delete an allowed file after approval/autonomy checks."""

    file_path = _resolve_user_path(path)
    _ensure_writable(file_path)
    _require_approval("Delete", file_path)
    if not file_path.is_file():
        raise UserFacingError(f"Expected a file: {file_path}")
    file_path.unlink()
    _log("delete_file", path=str(file_path))


def search_files(query: str, root: str | Path = "./workspace") -> list[str]:
    """Search allowed files for a text query under an allowed root."""

    root_path = _resolve_user_path(root)
    if not root_path.is_dir():
        raise UserFacingError(f"Expected a directory: {root_path}")
    matches: list[str] = []
    for child in root_path.rglob("*"):
        try:
            resolved = child.resolve(strict=True)
            if not resolved.is_file() or not _is_relative_to(resolved, root_path):
                continue
            _check_size(resolved)
            if query in resolved.read_text(encoding="utf-8", errors="ignore"):
                matches.append(str(resolved))
        except (OSError, UnicodeError, UserFacingError):
            continue
    _log("search_files", path=str(root_path))
    return sorted(matches)
