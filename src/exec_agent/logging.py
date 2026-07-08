"""Structured logging setup for the terminal agent."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render log records as compact JSON for terminal and file collection."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("tool", "intent", "node", "event", "path", "url"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, structured: bool = True) -> None:
    """Configure root logging once with a JSON formatter by default."""

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if structured else logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.handlers[:] = [handler]
