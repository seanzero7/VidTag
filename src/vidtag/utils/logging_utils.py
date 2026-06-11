"""Console/file logging and JSONL metric records."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def get_logger(name: str, logfile: str | None = None) -> logging.Logger:
    """INFO logger with one console handler and optionally one file handler.

    Idempotent: repeated calls with the same name/logfile add no duplicates.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    if logfile:
        logfile = os.path.abspath(logfile)
        attached = {
            h.baseFilename for h in logger.handlers if isinstance(h, logging.FileHandler)
        }
        if logfile not in attached:
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            filehandler = logging.FileHandler(logfile)
            filehandler.setFormatter(formatter)
            logger.addHandler(filehandler)
    return logger


def log_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append one JSON line to ``path``, creating parent dirs as needed."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
