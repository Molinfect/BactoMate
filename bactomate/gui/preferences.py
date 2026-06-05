"""
Persistent preferences via QSettings (geometry, last used directories).
Makes the app feel stand-alone across restarts.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt6.QtCore import QByteArray, QSettings


_ORG = "BactoMate"
_APP = "BactoMate"

# Keys for last-used directory per tab (one per tab; used for both file and folder dialogs)
LAST_DIR_ND2 = "nd2"
LAST_DIR_SEGMENTATION = "segmentation"
LAST_DIR_ANALYZE_MASKS = "analyze_masks"
LAST_DIR_TRACKING = "tracking"
LAST_DIR_GROWTH_CURVE = "growth_curve_video"
LAST_DIR_SESSION_LOGS = "session_logs"
LAST_DIR_UNET_SEGMENT = "unet_segment"


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def get_default_browse_dir() -> str:
    """Return the user's Desktop for file/folder dialogs when no last dir is set. Fallback to home if Desktop missing."""
    desktop = Path.home() / "Desktop"
    return str(desktop) if desktop.is_dir() else str(Path.home())


def get_last_dir(key: str) -> str:
    """Return last used directory for the given tab key, or the user's Desktop."""
    val = _settings().value(f"last_dir/{key}", "")
    path_str = str(val) if val else ""
    if path_str and Path(path_str).is_dir():
        return path_str
    desktop = Path.home() / "Desktop"
    return str(desktop) if desktop.is_dir() else str(Path.home())


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


# Preview image canvas: once user has used zoom/pan, we stop showing the long hover tooltip
def get_preview_zoom_pan_tip_seen() -> bool:
    """True if user has already used zoom or pan in any preview (we then suppress the long tooltip)."""
    val = _settings().value("preview_zoom_pan_tip_seen", False)
    if val is True or val == "true" or val == 1 or val == "1":
        return True
    return False


def set_preview_zoom_pan_tip_seen() -> None:
    """Mark that the user has used zoom/pan in a preview so we stop showing the tip."""
    _settings().setValue("preview_zoom_pan_tip_seen", True)


def get_building_unet_path() -> str:
    """Return saved path to Building U-NET project, or empty string."""
    val = _settings().value("building_unet_path", "")
    return str(val) if val else ""


def set_building_unet_path(path: str) -> None:
    """Save path to Building U-NET project."""
    _settings().setValue("building_unet_path", path)


def get_montage_presets() -> Dict[str, Any]:
    """Return saved montage figure presets keyed by name."""
    raw = _settings().value("montage_presets", "{}")
    try:
        data = json.loads(str(raw or "{}"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def set_montage_presets(presets: Dict[str, Any]) -> None:
    _settings().setValue("montage_presets", json.dumps(presets))


def list_montage_preset_names() -> list[str]:
    return sorted(get_montage_presets().keys(), key=lambda s: s.lower())


def save_montage_preset(name: str, payload: Dict[str, Any]) -> None:
    key = str(name or "").strip()
    if not key:
        return
    presets = get_montage_presets()
    presets[key] = payload
    set_montage_presets(presets)


def load_montage_preset(name: str) -> Optional[Dict[str, Any]]:
    data = get_montage_presets().get(str(name or "").strip())
    return data if isinstance(data, dict) else None


def delete_montage_preset(name: str) -> bool:
    key = str(name or "").strip()
    presets = get_montage_presets()
    if key not in presets:
        return False
    del presets[key]
    set_montage_presets(presets)
    return True
