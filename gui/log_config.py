"""
Structured logging for BactiMate.
Use get_logger() in modules; log_output() in MainWindow remains for user-facing status.
"""
import logging
import sys

LOG_NAME = "bactimate"


def get_logger() -> logging.Logger:
    """Return the BactiMate application logger."""
    return logging.getLogger(LOG_NAME)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the bactimate logger (call once at startup, e.g. from main.py)."""
    log = logging.getLogger(LOG_NAME)
    if log.handlers:
        return
    log.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(name)s:%(levelname)s: %(message)s"))
    log.addHandler(handler)


def log_exception(log: logging.Logger, msg: str = "Exception") -> None:
    """Log the current exception with traceback at ERROR level."""
    log.exception(msg)
