"""
Structured logging for BactiMate.
Single GUI log: bactimate_gui_<timestamp>.log (launch script redirects stderr here; Python adds FileHandler, crash, Qt messages).
Session logs: *_session.json (keep 100). No separate launch or debug files.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

LOG_NAME = "bactimate"

# Set at setup_logging: logs dir and current run's GUI log path (same file used by launch script when BACTIMATE_GUI_LOG is set)
_LOGS_DIR: Path | None = None
_GUI_LOG_PATH: Path | None = None


def get_logger() -> logging.Logger:
    """Return the BactiMate application logger."""
    return logging.getLogger(LOG_NAME)


def _prune_old_logs(logs_dir: Path, pattern: str, keep: int) -> None:
    """Keep only the most recent `keep` files matching pattern; delete older ones."""
    if keep <= 0 or not logs_dir.is_dir():
        return
    files = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep:]:
        try:
            p.unlink()
        except OSError:
            pass


def setup_logging(level: int = logging.INFO, logs_dir: Path | None = None) -> None:
    """Configure the bactimate logger and use single GUI log (call once at startup).
    If BACTIMATE_GUI_LOG is set (e.g. by launch_gui.sh), use that path so launch + Python share one file.
    Otherwise create Logs/bactimate_gui_<timestamp>.log.
    """
    global _LOGS_DIR, _GUI_LOG_PATH
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

    env_path = os.environ.get("BACTIMATE_GUI_LOG")
    if env_path and env_path.strip():
        _GUI_LOG_PATH = Path(env_path.strip()).resolve()
    elif _LOGS_DIR is not None:
        try:
            _LOGS_DIR.mkdir(parents=True, exist_ok=True)
            from bactimate.gui.constants import MAX_GUI_LOGS
            _prune_old_logs(_LOGS_DIR, "bactimate_gui_*.log", max(0, MAX_GUI_LOGS - 1))
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            _GUI_LOG_PATH = _LOGS_DIR / f"bactimate_gui_{ts}.log"
        except Exception:
            _GUI_LOG_PATH = _LOGS_DIR / "bactimate_gui.log"
    else:
        _GUI_LOG_PATH = None

    if _GUI_LOG_PATH is not None:
        try:
            _GUI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(_GUI_LOG_PATH, encoding="utf-8", mode="a")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s:%(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            log.addHandler(file_handler)
            with open(_GUI_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BactiMate started; GUI log: {_GUI_LOG_PATH}\n")
        except Exception:
            pass
    os.environ["BACTIMATE_OMNI_DEBUG_LOG"] = str(_GUI_LOG_PATH) if _GUI_LOG_PATH else ""


def get_gui_log_path() -> Path | None:
    """Return path to this run's single GUI log (crash, Qt messages, Omnipose debug all go here), or None."""
    if _GUI_LOG_PATH is not None:
        return _GUI_LOG_PATH
    if _LOGS_DIR is not None:
        return _LOGS_DIR / "bactimate_gui.log"
    try:
        from bactimate.gui.constants import LOGS_DIR
        return LOGS_DIR / "bactimate_gui.log"
    except Exception:
        return None


def get_crash_log_path() -> Path | None:
    """Return path for crash tracebacks (same as GUI log)."""
    return get_gui_log_path()


def get_omnipose_debug_log_path() -> Path | None:
    """Return path for Omnipose/debug lines (same as GUI log)."""
    return get_gui_log_path()


def append_omnipose_debug(msg: str) -> None:
    """Append a timestamped line to the GUI log (Omnipose/model load, etc.)."""
    path = get_gui_log_path()
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
    """Append crash/exception traceback to the GUI log."""
    path = get_crash_log_path()
    try:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
                f.write("\n")
        else:
            sys.stderr.write("[BactiMate GUI log path not set] ")
            sys.stderr.write(text)
            sys.stderr.write("\n")
            sys.stderr.flush()
    except Exception:
        sys.stderr.write(text)
        sys.stderr.write("\n")
        sys.stderr.flush()


def write_gui_log_line(prefix: str, message: str) -> None:
    """Append a single timestamped line to the GUI log (e.g. for Qt messages)."""
    path = get_gui_log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{prefix}] {message}\n")
    except Exception:
        pass


def log_exception(log: logging.Logger, msg: str = "Exception") -> None:
    """Log the current exception with traceback at ERROR level."""
    log.exception(msg)
