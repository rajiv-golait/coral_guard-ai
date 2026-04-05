"""Logging and shared helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# coralguard-api/ (project root when running uvicorn from this directory)
ROOT_DIR = Path(__file__).resolve().parents[1]

LOGGER_NAME = "coralguard"


def setup_logging(level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(LOGGER_NAME)
    if log.handlers:
        return log
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    log.addHandler(handler)
    log.setLevel(level.upper())
    return log


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_unlink(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def redact_secrets(settings: dict[str, Any]) -> dict[str, Any]:
    out = dict(settings)
    for k in list(out.keys()):
        if "key" in k.lower() or "secret" in k.lower() or "token" in k.lower():
            out[k] = "***" if out[k] else ""
    return out
