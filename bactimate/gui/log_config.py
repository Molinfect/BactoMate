"""
Structured logging for BactiMate.
Use get_logger() in modules; log_output() in MainWindow remains for user-facing status.
Crash tracebacks are written to project Logs/crash.log for debugging.
"""
import logging
import sys
from pathlib import Path

LOG_NAME = "bactimate"

# Logs dir is set when setup_logging runs (avoids circular import at module load)
_LOGS_DIR: Path | None = None


def get_logger() -> logging.Logger:
    """Return the BactiMate application logger."""
    return logging.getLogger(LOG_NAME)


def setup_logging(level: int = logging.INFO, logs_dir: Path | None = None) -> None:
    """Configure the bactimate logger (call once at startup, e.g. from main.py)."""
    global _LOGS_DIR
    if logs_dir is not None:
        _LOGS_DIR = logs_dir
    log = logging.getLogger(LOG_NAME)
    if log.handlers:
        return
    log.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(name)s:%(levelname)s: %(message)s"))
    log.addHandler(handler)


def get_crash_log_path() -> Path | None:
    """Return path to crash log file (project Logs/crash.log), or None if Logs dir not set."""
    if _LOGS_DIR is None:
        try:
            from bactimate.gui.constants import LOGS_DIR
            return LOGS_DIR / "crash.log"
        except Exception:
            return None
    return _LOGS_DIR / "crash.log"


def write_crash_log(text: str) -> None:
    """Append an uncaught exception traceback to project Logs/crash.log for debugging."""
    path = get_crash_log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    except Exception:
        pass


def log_exception(log: logging.Logger, msg: str = "Exception") -> None:
    """Log the current exception with traceback at ERROR level."""
    log.exception(msg)
