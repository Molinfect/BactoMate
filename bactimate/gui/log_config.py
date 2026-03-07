"""
Structured logging for BactiMate.
Three log types: session logs (JSON, keep 100), launch logs (launch_*.log, keep 20), debug (debug_*.log, keep 10).
Crash and Omnipose/debug lines go to the current run's debug_<timestamp>.log so debug logs don't grow indefinitely.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

LOG_NAME = "bactimate"

# Set at setup_logging: logs dir and current run's debug log path (one file per run, pruned to MAX_DEBUG_LOGS)
_LOGS_DIR: Path | None = None
_DEBUG_LOG_PATH: Path | None = None


def get_logger() -> logging.Logger:
    """Return the BactiMate application logger."""
    return logging.getLogger(LOG_NAME)


def _prune_old_logs(logs_dir: Path, pattern: str, keep: int) -> None:
    """Keep only the most recent `keep` files matching pattern (e.g. debug_*.log); delete older ones."""
    if keep <= 0 or not logs_dir.is_dir():
        return
    files = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep:]:
        try:
            p.unlink()
        except OSError:
            pass


def setup_logging(level: int = logging.INFO, logs_dir: Path | None = None) -> None:
    """Configure the bactimate logger and create this run's debug log (call once at startup, e.g. from main.py)."""
    global _LOGS_DIR, _DEBUG_LOG_PATH
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
    if _LOGS_DIR is not None:
        try:
            _LOGS_DIR.mkdir(parents=True, exist_ok=True)
            from bactimate.gui.constants import MAX_DEBUG_LOGS
            _prune_old_logs(_LOGS_DIR, "debug_*.log", max(0, MAX_DEBUG_LOGS - 1))  # before creating new one
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            _DEBUG_LOG_PATH = _LOGS_DIR / f"debug_{ts}.log"
            os.environ["BACTIMATE_OMNI_DEBUG_LOG"] = str(_DEBUG_LOG_PATH)
            with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BactiMate started; debug log: {_DEBUG_LOG_PATH}\n")
        except Exception:
            _DEBUG_LOG_PATH = _LOGS_DIR / "debug.log"  # fallback
            os.environ["BACTIMATE_OMNI_DEBUG_LOG"] = str(_DEBUG_LOG_PATH)


def get_crash_log_path() -> Path | None:
    """Return path to this run's debug log (crash tracebacks go here), or None if Logs dir not set."""
    if _DEBUG_LOG_PATH is not None:
        return _DEBUG_LOG_PATH
    if _LOGS_DIR is not None:
        return _LOGS_DIR / "debug.log"
    try:
        from bactimate.gui.constants import LOGS_DIR
        return LOGS_DIR / "debug.log"
    except Exception:
        return None


def get_omnipose_debug_log_path() -> Path | None:
    """Return path to this run's debug log (Omnipose/user-model debug lines go here), or None."""
    if _DEBUG_LOG_PATH is not None:
        return _DEBUG_LOG_PATH
    if _LOGS_DIR is not None:
        return _LOGS_DIR / "debug.log"
    try:
        from bactimate.gui.constants import LOGS_DIR
        return LOGS_DIR / "debug.log"
    except Exception:
        return None


def append_omnipose_debug(msg: str) -> None:
    """Append a timestamped line to this run's debug log (Omnipose/model load, etc.)."""
    path = get_omnipose_debug_log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def write_crash_log(text: str) -> None:
    """Append an uncaught exception traceback to this run's debug log for debugging."""
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
