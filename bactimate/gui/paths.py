"""
Single source of truth for app root, logs dir, and script paths.
Works in development (repo root), frozen .app bundle, and pip-installed layout.
"""
import os
import sys
from pathlib import Path


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False) or hasattr(sys, "__compiled__")


def _is_installed() -> bool:
    """True when running from an installed package (site-packages), not from repo."""
    if _is_frozen():
        return False
    try:
        resolved = Path(__file__).resolve()
        parts = resolved.parts
        return "site-packages" in parts or "dist-packages" in parts
    except Exception:
        return False


def _get_app_root_development() -> Path:
    """Repo root: directory containing main.py, bactimate/ (when running from source)."""
    # paths.py is at bactimate/gui/paths.py -> parent.parent = bactimate, parent.parent.parent = repo
    return Path(__file__).resolve().parent.parent.parent


def _get_app_root_installed() -> Path:
    """When pip-installed: package root (bactimate/), which contains scripts/."""
    try:
        import bactimate as _pkg
        return Path(_pkg.__file__).resolve().parent
    except Exception:
        return _get_app_root_development()


def _get_app_root_frozen() -> Path:
    """When frozen (e.g. .app): Resources folder contains project files (main.py, scripts/, etc.)."""
    try:
        from bactimate.gui import patches
        base = getattr(patches, "base_path", None)
        if base is None:
            return _get_app_root_development()
        base_path = Path(base)
        # macOS .app: executable is in Contents/MacOS, project files in Contents/Resources
        if sys.platform == "darwin" and "Contents/MacOS" in str(base_path):
            resources = base_path.parent / "Resources"
            if resources.is_dir():
                return resources
        # Fallback: treat base_path as app root (e.g. Windows or custom bundle)
        return base_path
    except Exception:
        return _get_app_root_development()


def get_app_root() -> Path:
    """Application root: directory containing scripts/ and (in dev) main.py, bactimate/."""
    if _is_frozen():
        return _get_app_root_frozen()
    if _is_installed():
        return _get_app_root_installed()
    return _get_app_root_development()


def get_logs_dir() -> Path:
    """Directory for session logs and crash log. When installed, use platform user data dir."""
    if _is_installed():
        return _get_user_data_logs_dir()
    return get_app_root() / "Logs"


def _get_user_data_logs_dir() -> Path:
    """Platform-specific user data dir for logs (persists across pip upgrades)."""
    if sys.platform == "darwin":
        base = Path(os.path.expanduser("~/Library/Application Support"))
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", os.path.expanduser("~")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")))
    return base / "BactiMate" / "Logs"


def _get_user_data_models_base() -> Path:
    """Platform-specific base dir for user-trained models (BactiMate/Models)."""
    if sys.platform == "darwin":
        base = Path(os.path.expanduser("~/Library/Application Support"))
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", os.path.expanduser("~")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")))
    return base / "BactiMate" / "Models"


def get_omnipose_user_models_dir() -> Path:
    """
    Directory for user-trained Omnipose models (.pth). Created if missing.
    - Installed: e.g. ~/Library/Application Support/BactiMate/Models/omnipose_user_models (macOS).
    - Dev/frozen: get_app_root() / "omnipose_user_models".
    """
    if _is_installed():
        path = _get_user_data_models_base() / "omnipose_user_models"
    else:
        path = get_app_root() / "omnipose_user_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_models_dir() -> Path:
    """Alias for get_omnipose_user_models_dir(). Prefer get_omnipose_user_models_dir() for clarity."""
    return get_omnipose_user_models_dir()


def get_unet_models_dir() -> Path:
    """
    Directory for user-trained U-Net models (.keras + patch_config.json). Created if missing.
    - Installed: e.g. ~/Library/Application Support/BactiMate/Models/unet_user_models (macOS).
    - Dev/frozen: get_app_root() / "unet_user_models".
    """
    if _is_installed():
        path = _get_user_data_models_base() / "unet_user_models"
    else:
        path = get_app_root() / "unet_user_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_scripts_dir() -> Path:
    """Directory containing backend scripts (convert_nd2_to_tif.py, batch_process_*.py, etc.)."""
    return get_app_root() / "scripts"


def get_script_path(basename: str) -> Path:
    """Path to a script by filename, e.g. get_script_path('convert_nd2_to_tif.py')."""
    return get_scripts_dir() / basename
