"""Shared backend service layer for CLI and web UI entry points.

The CLI and FastAPI routes import this module instead of reaching directly into
individual tools.  Tool modules still own low-level IO, while these services are
where product-level policy (profile, autonomy, HITL, allowed paths, and shell
permissions) is consistently applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.memory.long_term import LongTermMemoryStore
from app.memory.vector_store import VectorStore
from app.tools import filesystem, web_fastcrw
from app.tools.docx import ingest_docx
from app.tools.image import ask_image, describe_image
from app.tools.pdf import chunk_text, ingest_pdf
from app.tools.shell import CommandResult, history as shell_history, run_command
from exec_agent.config import Settings, get_settings
from exec_agent.safety import UserFacingError, validate_local_file
from exec_agent.tasks import AutonomyLevel, AutonomousTaskRunner, TaskRecord, TaskStore

ToolKind = Literal["read", "write", "shell", "web", "model", "memory", "document", "image", "task"]
_WRITE_LEVELS = {"autonomous_limited", "autonomous_full"}
_SHELL_LEVELS = {"autonomous_limited", "autonomous_full"}
_WEB_LEVELS = {"human_approved", "autonomous_limited", "autonomous_full"}


@dataclass(frozen=True)
class SafetySnapshot:
    """Effective policy advertised to all clients before tool execution."""

    runtime_profile: str
    autonomy_level: str
    hitl: bool
    actions_hitl: bool
    allowed_dirs: str
    readonly_dirs: str
    shell_enabled: bool
    shell_workdir: str
    web_enabled: bool
    fastcrw_enabled: bool
    local_only: bool


def safety_snapshot(settings: Settings | None = None) -> SafetySnapshot:
    s = settings or get_settings()
    return SafetySnapshot(
        runtime_profile=s.runtime_profile,
        autonomy_level=s.autonomy_level,
        hitl=s.hitl,
        actions_hitl=s.actions_hitl,
        allowed_dirs=s.allowed_dirs,
        readonly_dirs=s.readonly_dirs,
        shell_enabled=s.shell_enabled,
        shell_workdir=str(s.shell_workdir),
        web_enabled=s.web_enabled,
        fastcrw_enabled=s.fastcrw_enabled,
        local_only=s.local_only,
    )


def require_tool(kind: ToolKind, *, action: str, autonomy_level: AutonomyLevel | None = None) -> None:
    """Apply shared product-level policy before a backend tool is invoked."""

    s = get_settings()
    level = autonomy_level or s.autonomy_level
    if s.local_only and kind == "web":
        raise UserFacingError(f"{action} is blocked because EXEC_AGENT_LOCAL_ONLY=true.")
    if kind == "web" and level not in _WEB_LEVELS:
        raise UserFacingError(f"{action} requires autonomy level human_approved or higher.")
    if kind == "shell" and not s.shell_enabled:
        raise UserFacingError(f"{action} is blocked because shell execution is disabled.")
    if kind == "write" and s.actions_hitl and level not in _WRITE_LEVELS:
        raise UserFacingError(f"{action} requires approval; raise autonomy to autonomous_limited or autonomous_full after review.")


class AssistantBackend:
    """Coherent backend API shared by terminal commands and the web UI."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def safety(self) -> SafetySnapshot:
        return safety_snapshot(self.settings)

    # Filesystem
    def list_files(self, path: str | Path) -> list[str]:
        require_tool("read", action="List files")
        return filesystem.list_dir(path)

    def read_file(self, path: str | Path) -> str:
        require_tool("read", action="Read file")
        return filesystem.read_file(path)

    def search_files(self, query: str, root: str | Path = "./workspace") -> list[str]:
        require_tool("read", action="Search files")
        return filesystem.search_files(query, root)

    def write_file(self, path: str | Path, content: str) -> Path:
        require_tool("write", action="Write file")
        return filesystem.write_file(path, content)

    # Shell
    def run_shell(self, command: str, cwd: str | Path | None = None, timeout: int | float | None = None) -> CommandResult:
        require_tool("shell", action="Run shell command")
        return run_command(command, cwd=cwd, timeout=timeout)

    def shell_history(self, limit: int = 50) -> list[CommandResult]:
        return shell_history(limit=limit)

    # Documents/images/memory/RAG
    def ingest_path(self, path: str | Path) -> int:
        file_path = validate_local_file(path, purpose="document")
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return ingest_pdf(file_path)
        if suffix == ".docx":
            return ingest_docx(file_path)
        if suffix in {".txt", ".md"}:
            content = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(content)
            VectorStore().add_documents(chunks, [{"source": str(file_path), "source_type": "document"} for _ in chunks])
            return len(chunks)
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            caption = describe_image(file_path)
            VectorStore().add_documents([caption], [{"source": str(file_path), "source_type": "image"}])
            return 1
        raise UserFacingError(f"Unsupported ingest type: {suffix}")

    def ask_image(self, path: str | Path, question: str, **kwargs: Any) -> str:
        require_tool("image", action="Analyze image")
        return ask_image(path, question, **kwargs)

    def memory_store(self) -> LongTermMemoryStore:
        require_tool("memory", action="Use memory")
        return LongTermMemoryStore()

    def vector_store(self) -> VectorStore:
        require_tool("document", action="Use vector store")
        return VectorStore()

    # Web research
    def search_web(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        require_tool("web", action="Search web")
        return web_fastcrw.search_web(query, max_results or self.settings.fastcrw_max_results)

    def scrape_url(self, url: str) -> web_fastcrw.WebPage:
        require_tool("web", action="Scrape web page")
        return web_fastcrw.scrape_url(url)

    def crawl_url(self, url: str, limit: int = 10) -> list[web_fastcrw.WebPage]:
        require_tool("web", action="Crawl website")
        return web_fastcrw.crawl_url(url, limit)

    def web_health(self) -> dict[str, Any]:
        require_tool("web", action="Check FastCRW health")
        return web_fastcrw.health_check()

    # Tasks
    def run_task(self, description: str, autonomy_level: AutonomyLevel | None = None, progress: Any | None = None) -> TaskRecord:
        level = autonomy_level or self.settings.autonomy_level
        return AutonomousTaskRunner(progress=progress).run(description, autonomy_level=level)

    def task_store(self) -> TaskStore:
        return TaskStore()


def get_backend() -> AssistantBackend:
    """Return a lightweight backend facade for the current settings."""

    return AssistantBackend()
