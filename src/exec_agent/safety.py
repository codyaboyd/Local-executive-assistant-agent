"""Safety validation helpers for local file/tool boundaries."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

from exec_agent.config import get_settings


class UserFacingError(RuntimeError):
    """An actionable error message safe to show directly to users."""


def allowed_upload_extensions() -> set[str]:
    raw = get_settings().allowed_upload_extensions
    return {item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}" for item in raw.split(",") if item.strip()}


def validate_local_file(path: str | Path, *, allowed_extensions: Iterable[str] | None = None, purpose: str = "file") -> Path:
    """Resolve and validate a user-supplied local file path against size and extension limits."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"{purpose.title()} not found: {file_path}")
    if not file_path.is_file():
        raise UserFacingError(f"Expected a regular {purpose}, got: {file_path}")
    allowed = {ext.lower() for ext in (allowed_extensions or allowed_upload_extensions())}
    if allowed and file_path.suffix.lower() not in allowed:
        raise ValueError(
            f"Unsupported {purpose} type {file_path.suffix!r}. Allowed extensions: {', '.join(sorted(allowed))}."
        )
    max_bytes = get_settings().max_upload_bytes
    size = file_path.stat().st_size
    if size > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise UserFacingError(f"{purpose.title()} is too large ({size} bytes). Maximum allowed size is {mb:.1f} MiB.")
    return file_path
