#!/usr/bin/env python3
"""Generate and run the project Makefile targets from a temporary file.

This keeps product Makefile targets out of the repository while preserving a
familiar `make <target>` workflow for local development and smoke tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

MAKEFILE = r"""
FASTCRW_COMPOSE_FILE := docker-compose.fastcrw.yml
FASTCRW_PORT ?= 3002
COMPOSE_ENV_FILE := $(if $(wildcard .env),--env-file .env,)
ARGS ?= chat

.PHONY: install test run eval fastcrw-up fastcrw-down fastcrw-logs fastcrw-health

install:
	uv sync --extra dev

test:
	uv run pytest

run:
	uv run exec-agent $(ARGS)

eval:
	uv run exec-agent eval run

fastcrw-up:
	FASTCRW_PORT=$(FASTCRW_PORT) docker compose -f $(FASTCRW_COMPOSE_FILE) $(COMPOSE_ENV_FILE) up -d fastcrw

fastcrw-down:
	FASTCRW_PORT=$(FASTCRW_PORT) docker compose -f $(FASTCRW_COMPOSE_FILE) $(COMPOSE_ENV_FILE) down

fastcrw-logs:
	FASTCRW_PORT=$(FASTCRW_PORT) docker compose -f $(FASTCRW_COMPOSE_FILE) $(COMPOSE_ENV_FILE) logs -f fastcrw

fastcrw-health:
	curl -fsS http://127.0.0.1:$(FASTCRW_PORT)/health
""".lstrip()


def main(argv: list[str]) -> int:
    """Write the generated Makefile to a temporary path and delegate to make."""

    repo_root = Path(__file__).resolve().parents[1]
    args = argv or ["run"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="exec-agent-", suffix=".mk", delete=False) as handle:
        handle.write(MAKEFILE)
        temp_makefile = Path(handle.name)
    try:
        command = ["make", "-f", str(temp_makefile), *args]
        return subprocess.call(command, cwd=repo_root, env=os.environ.copy())
    finally:
        temp_makefile.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
