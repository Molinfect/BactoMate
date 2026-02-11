"""
Persistent preferences via QSettings (geometry, last used directories).
Makes the app feel stand-alone across restarts.
"""
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QByteArray, QSettings


_ORG = "BactiMate"
_APP = "BactiMate"

# Keys for last-used directory per tab (one per tab; used for both file and folder dialogs)
LAST_DIR_ND2 = "nd2"
LAST_DIR_SEGMENTATION = "segmentation"
LAST_DIR_ANALYZE_MASKS = "analyze_masks"
LAST_DIR_TRACKING = "tracking"
LAST_DIR_GROWTH_CURVE = "growth_curve_video"
LAST_DIR_SESSION_LOGS = "session_logs"


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def get_last_dir(key: str) -> str:
    """Return last used directory for the given tab key, or empty string."""
    val = _settings().value(f"last_dir/{key}", "")
    path_str = str(val) if val else ""
    if path_str and Path(path_str).is_dir():
        return path_str
    return ""


def set_last_dir(key: str, path: str) -> None:
    """Save last used directory for the given tab key. Pass a directory path."""
    if not path:
        return
    p = Path(path)
    if p.is_file():
        p = p.parent
    if p.is_dir():
        _settings().setValue(f"last_dir/{key}", str(p.resolve()))


def get_geometry() -> Optional[QByteArray]:
    """Return saved main window geometry, or None."""
    val = _settings().value("main_window/geometry")
    if isinstance(val, QByteArray):
        return val
    if isinstance(val, bytes):
        return QByteArray(val)
    return None


def set_geometry(data: QByteArray) -> None:
    """Save main window geometry."""
    _settings().setValue("main_window/geometry", data)


def get_state() -> Optional[QByteArray]:
    """Return saved main window state (toolbars, etc.), or None."""
    val = _settings().value("main_window/state")
    if isinstance(val, QByteArray):
        return val
    if isinstance(val, bytes):
        return QByteArray(val)
    return None


def set_state(data: QByteArray) -> None:
    """Save main window state."""
    _settings().setValue("main_window/state", data)
