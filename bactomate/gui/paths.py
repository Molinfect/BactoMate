"""
Single source of truth for app root, logs dir, and script paths.
Works in development (repo root), frozen .app bundle, and pip-installed layout.
"""
import os
import sys
import shutil
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
    """Repo root: directory containing main.py, bactomate/ (when running from source)."""
    # paths.py is at bactomate/gui/paths.py -> parent.parent = bactomate, parent.parent.parent = repo
    return Path(__file__).resolve().parent.parent.parent


def _get_app_root_installed() -> Path:
    """When pip-installed: package root (bactomate/), which contains scripts/."""
    try:
        import bactomate as _pkg
        return Path(_pkg.__file__).resolve().parent
    except Exception:
        return _get_app_root_development()


def _get_app_root_frozen() -> Path:
    """When frozen (e.g. .app): Resources folder contains project files (main.py, scripts/, etc.)."""
    try:
        from bactomate.gui import patches
        base = getattr(patches, "base_path", None)
        if base is None:
            return _get_app_root_development()
        base_path = Path(base)
        # macOS .app: executable may be Contents/MacOS/BactoMate or Contents/Resources/venv/bin/python3
        if sys.platform == "darwin":
            if "Contents/MacOS" in str(base_path):
                resources = base_path.parent / "Resources"
                if resources.is_dir():
                    return resources
            # Launched via venv (launch_gui.sh -> venv/bin/python3): base_path is .../Resources/venv/bin
            # Walk up to Contents/Resources so icon and paths resolve correctly
            current = base_path
            for _ in range(8):
                if current.name == "Resources" and (current.parent / "MacOS").is_dir():
                    return current
                if (current / "main.py").is_file() and (current / "img").is_dir():
                    return current
                parent = current.parent
                if parent == current:
                    break
                current = parent
        # Fallback: treat base_path as app root (e.g. Windows or custom bundle)
        return base_path
    except Exception:
        return _get_app_root_development()


def get_app_root() -> Path:
    """Application root: directory containing scripts/ and (in dev) main.py, bactomate/."""
    if _is_frozen():
        return _get_app_root_frozen()
    if _is_installed():
        return _get_app_root_installed()
    return _get_app_root_development()


def _running_from_app_bundle() -> bool:
    """True when app root is inside a .app bundle (Contents/Resources), so logs should go to user dir."""
    try:
        root = get_app_root()
        return "Contents/Resources" in str(root) or (root.parent / "MacOS").is_dir()
    except Exception:
        return False


def configure_macos_bundle_subprocess_env(env: dict[str, str]) -> None:
    """
    Ensure child processes spawned from the .app bundle can load the bundled
    Python.framework (ND2 conversion, segmentation scripts, etc.).
    """
    if sys.platform != "darwin" or not _running_from_app_bundle():
        return
    try:
        resources = get_app_root()
        contents = resources.parent
        py_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
        framework_root = contents / "Frameworks" / "Python.framework"
        python_home = framework_root / "Versions" / py_mm
        if not python_home.is_dir():
            return
        # Do not set PYTHONHOME here: the bundled venv must keep its own site-packages.
        dyld_parts = [str(framework_root)]
        qt_fw = resources / "venv" / f"lib/python{py_mm}/site-packages/PyQt6/Qt6/lib"
        if qt_fw.is_dir():
            dyld_parts.append(str(qt_fw))
        env["DYLD_FRAMEWORK_PATH"] = os.pathsep.join(dyld_parts)
    except Exception:
        pass


def configure_windows_subprocess_env(env: dict[str, str]) -> None:
    """Keep child processes on the BactoMate venv; ignore stray PYTHONHOME / user site."""
    if sys.platform != "win32":
        return
    try:
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        venv = get_app_root() / "venv"
        if venv.is_dir():
            env["VIRTUAL_ENV"] = str(venv.resolve())
    except Exception:
        pass


def get_python_executable_for_subprocess() -> str:
    """
    Python executable to use when running scripts in a subprocess (e.g. batch_process_tiff_sequence.py).
    When running from the .app bundle, the main process is the embedded launcher (Contents/MacOS/BactoMate),
    which always runs main.py and ignores script args. Using it would open a second GUI. So we use the
    venv Python inside the bundle instead. In dev, use sys.executable.

    On Windows one-folder PyInstaller builds, ``sys.executable`` is ``BactoMate.exe`` (same as the GUI).
    WorkerThread detects that case and runs supported scripts in-process instead of spawning a child.
    """
    if _running_from_app_bundle():
        root = get_app_root()
        for name in ("python3", "BactoMate", "python3.11", "python3.12"):
            exe = root / "venv" / "bin" / name
            if exe.exists() and os.access(exe, os.X_OK):
                return str(exe.resolve())
    if sys.platform == "win32":
        root = get_app_root()
        for name in ("python.exe", "python3.exe", "python3.11.exe"):
            exe = root / "venv" / "Scripts" / name
            if exe.is_file():
                return str(exe.resolve())
    return sys.executable


def get_subprocess_pythonpath_root() -> str:
    """
    Directory that must be on PYTHONPATH so a child Python process can ``import bactomate``
    when running scripts by file path (e.g. batch_process_tiff_sequence.py).

    In development this is the repository root (parent of the ``bactomate`` package).
    When pip-installed it is ``site-packages`` (parent of the ``bactomate`` package dir).
    """
    import bactomate as _pkg

    return str(Path(_pkg.__file__).resolve().parent.parent)


def get_logs_dir() -> Path:
    """Directory for session logs (*_session.json) and GUI log (bactomate_gui_*.log). When running from .app, use Logs inside the app if writable."""
    if _is_installed():
        return _get_user_data_logs_dir()
    if sys.platform == "darwin" and _running_from_app_bundle():
        app_logs = get_app_root() / "Logs"
        try:
            app_logs.mkdir(parents=True, exist_ok=True)
            if os.access(app_logs, os.W_OK):
                return app_logs
        except (OSError, PermissionError):
            pass
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
    return base / "BactoMate" / "Logs"


def _get_user_data_models_base() -> Path:
    """Platform-specific base dir for user-trained models (BactoMate/Models)."""
    if sys.platform == "darwin":
        base = Path(os.path.expanduser("~/Library/Application Support"))
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", os.path.expanduser("~")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")))
    return base / "BactoMate" / "Models"


def _copy_legacy_model_files(legacy_dir: Path, target_dir: Path) -> None:
    """
    Best-effort migration for older frozen builds that stored models in the app bundle.

    Existing target files win; this avoids overwriting user-trained models after upgrades.
    """
    try:
        legacy = Path(legacy_dir)
        target = Path(target_dir)
        if legacy.resolve() == target.resolve() or not legacy.is_dir():
            return
        target.mkdir(parents=True, exist_ok=True)
        for src in legacy.iterdir():
            if not src.is_file():
                continue
            dest = target / src.name
            if dest.exists():
                continue
            try:
                shutil.copy2(src, dest)
            except OSError:
                pass
    except Exception:
        pass


def get_omnipose_user_models_dir() -> Path:
    """
    Directory for user-trained Omnipose models (.pth). Created if missing.
    - Installed/frozen: e.g. ~/Library/Application Support/BactoMate/Models/omnipose_user_models (macOS).
    - Dev: get_app_root() / "omnipose_user_models".
    """
    if _is_installed() or _is_frozen():
        path = _get_user_data_models_base() / "omnipose_user_models"
    else:
        path = get_app_root() / "omnipose_user_models"
    path.mkdir(parents=True, exist_ok=True)
    if _is_frozen():
        _copy_legacy_model_files(get_app_root() / "omnipose_user_models", path)
    return path


def get_omnipose_pretrained_cache_dir() -> Path:
    """
    Directory where BactoMate mirrors built-in Omnipose checkpoints after first download.
    Same layout as cellpose_omni's MODEL_DIR filenames (e.g. bact_phase_omnitorch_0).
    - Installed/frozen: .../BactoMate/Models/omnipose_pretrained
    - Dev: get_app_root() / "omnipose_pretrained"
    """
    if _is_installed() or _is_frozen():
        path = _get_user_data_models_base() / "omnipose_pretrained"
    else:
        path = get_app_root() / "omnipose_pretrained"
    path.mkdir(parents=True, exist_ok=True)
    if _is_frozen():
        _copy_legacy_model_files(get_app_root() / "omnipose_pretrained", path)
    return path


def get_user_models_dir() -> Path:
    """Alias for get_omnipose_user_models_dir(). Prefer get_omnipose_user_models_dir() for clarity."""
    return get_omnipose_user_models_dir()


def get_unet_models_dir() -> Path:
    """
    Directory for user-trained U-Net models (.keras + patch_config.json). Created if missing.
    - Installed/frozen: e.g. ~/Library/Application Support/BactoMate/Models/unet_user_models (macOS).
    - Dev: get_app_root() / "unet_user_models".
    """
    if _is_installed() or _is_frozen():
        path = _get_user_data_models_base() / "unet_user_models"
    else:
        path = get_app_root() / "unet_user_models"
    path.mkdir(parents=True, exist_ok=True)
    if _is_frozen():
        _copy_legacy_model_files(get_app_root() / "unet_user_models", path)
    return path


def get_scripts_dir() -> Path:
    """Directory containing backend scripts (convert_nd2_to_tif.py, batch_process_*.py, etc.)."""
    return get_app_root() / "scripts"


def get_script_path(basename: str) -> Path:
    """Path to a script by filename, e.g. get_script_path('convert_nd2_to_tif.py')."""
    return get_scripts_dir() / basename
