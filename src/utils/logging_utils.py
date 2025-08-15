"""
Logging utilities for the project.

Provides centralized configuration for application logging with both
stdout and optional rotating file handlers.

Functions:
    configure_logging(
        log_level="INFO",
        file_mode="auto" | "fixed" | "none",
        log_subdir="logs",
        log_file=None,
        fmt=DEFAULT_FORMAT,
        max_bytes=5_000_000,
        backup_count=3,
        to_stdout=True
    ) -> Path | None
        - Configures the root logger.
        - Supports console output and rotating file logs under <project>/outputs/<log_subdir>.
        - file_mode:
            "auto"  → log file named after script (e.g., myscript.log)
            "fixed" → log file uses a given name
            "none"  → no file logging
        - Returns the path to the log file or None.

    get_logger(name: str) -> logging.Logger
        - Retrieves a named logger instance for modules.

Usage:
    from utils.logging_utils import configure_logging, get_logger

    configure_logging(log_level="DEBUG")
    logger = get_logger(__name__)
    logger.info("This is a log message")


Author: Tomasz Lasota
Date: 2025-08-15
Version: 1.1
"""


import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Optional
from src.utils.paths import OUTPUTS_DIR


DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(stage)s|%(run_id)s] %(name)s %(message)s"

class _ContextFilter(logging.Filter):
    """Injects run_id and stage into every record."""
    def __init__(self, run_id: Optional[str], stage: Optional[str]):
        super().__init__()
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        # Default stage = current script name (without .py)
        self.stage = stage or (Path(sys.argv[0]).stem or "app")

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id
        if not hasattr(record, "stage"):
            record.stage = self.stage
        return True

def configure_logging(
    log_level: str = "INFO",
    file_mode: str = "auto",          # "auto" | "fixed" | "none"
    log_subdir: str = "logs",
    log_file: str | Path | None = None,
    fmt: str = DEFAULT_FORMAT,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    to_stdout: bool = True,
    *,
    run_id: str | None = None,
    stage: str | None = None,
) -> Path | None:
    """
    Configure root logger with stdout and optional rotating file in outputs/logs.

    Where logs go:
      - Root logs directory = OUTPUTS_DIR / log_subdir  (e.g., <project>/outputs/logs)
      - file_mode="auto":  <outputs/logs>/<script_name>.log -> automatically determined
      - file_mode="fixed": <outputs/logs>/<log_file or 'app.log'> -> requires input of log file name
      - file_mode="none":  no file handler

    Returns:
      The resolved Path to the file log (or None if file_mode="none").
    """
    logs_root = (OUTPUTS_DIR / log_subdir).resolve()

    # Decide file path (if any)
    file_path: Path | None = None
    if file_mode == "auto":
        script_name = Path(sys.argv[0]).stem or "app"
        file_path = logs_root / f"{script_name}.log"
    elif file_mode == "fixed":
        file_path = logs_root / (Path(log_file).name if log_file else "app.log")
    elif file_mode == "none":
        file_path = None
    else:
        raise ValueError("file_mode must be one of: 'auto', 'fixed', 'none'")

    # Root logger setup
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = logging.Formatter(fmt)
    context_filter = _ContextFilter(run_id=run_id, stage=stage)

    # Console handler
    if to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        sh.addFilter(context_filter)
        root.addHandler(sh)

    # Rotating file handler
    if file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            filename=str(file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        fh.addFilter(context_filter)
        root.addHandler(fh)

    logging.getLogger(__name__).info(
        "Logging configured (mode=%s, file=%s)",
        file_mode,
        str(file_path) if file_path else None,
    )
    return file_path

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

