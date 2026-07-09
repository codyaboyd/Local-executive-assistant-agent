"""Command-line entry point for the FastAPI web UI."""

from __future__ import annotations

import uvicorn

from exec_agent.config import get_settings


def main() -> None:
    """Run the Bootstrap web UI with Uvicorn."""

    settings = get_settings()
    uvicorn.run("exec_agent.web:app", host=settings.web_host, port=settings.web_port, reload=False)
