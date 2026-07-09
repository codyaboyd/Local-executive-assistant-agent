"""Command-line entry point for the FastAPI web UI."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the Bootstrap web UI with Uvicorn."""

    uvicorn.run("exec_agent.web:app", host="0.0.0.0", port=8000, reload=False)
