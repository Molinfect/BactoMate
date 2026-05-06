"""
Entry point for running BactiMate (python -m bactimate or pip-installed bactimate command).
Handles --version, --smoke-test, patches, logging, then launches the GUI.
"""
import sys
import traceback
import warnings
from time import perf_counter


def run() -> None:
    t0 = perf_counter()
    t_last = t0

    def _mark(log, label: str) -> None:
        nonlocal t_last
        try:
            now = perf_counter()
            dt_total_ms = (now - t0) * 1000.0
            dt_step_ms = (now - t_last) * 1000.0
            t_last = now
            log.info("[startup] %s step_ms=%.1f total_ms=%.1f", label, dt_step_ms, dt_total_ms)
        except Exception:
            pass

    # Show "BactiMate" in macOS menu bar (and Dock/Activity Monitor) when run from .app bundle
    try:
        import setproctitle
        setproctitle.setproctitle("BactiMate")
    except Exception:
        pass

    # --version / --smoke-test: exit before loading full GUI (used by CI / packaged app)
    if len(sys.argv) >= 2 and sys.argv[1] in ("--version", "-v"):
        from bactimate import __version__
        print(f"BactiMate {__version__}")
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "--smoke-test":
        try:
            import bactimate.gui.patches  # noqa: F401
            import numpy as np
            from PyQt6.QtCore import QCoreApplication
            import tifffile
            _ = np.zeros((1, 1))
            _ = QCoreApplication
            print("Smoke test OK")
            sys.exit(0)
        except Exception as e:
            print(f"Smoke test failed: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    # Patches (Qt paths, omnipose patch) must run before PyQt6
    import bactimate.gui.patches  # noqa: F401

    # Apply MPS (Apple Silicon) patches to site-packages when running from source, so the batch
    # subprocess (same Python env) loads patched omnipose/cellpose_omni and uses "mps" not "mps:True".
    try:
        from pathlib import Path
        from bactimate.gui.paths import get_app_root
        _patch_script = (get_app_root() / "setup" / "apply_all_patches.py").resolve()
        if _patch_script.exists():
            import importlib.util
            _spec = importlib.util.spec_from_file_location("_apply_patches", _patch_script)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.patch_omnipose_gpu_mps()
            _mod.patch_cellpose_omni_core_mps_device()
    except Exception:
        pass  # don't block startup; user can run setup/apply_all_patches.py manually

    # Windows-only: console encoding and UI fixes (ignored on macOS/Linux)
    if sys.platform == "win32":
        from bactimate.gui.windows_patches import apply_early
        apply_early()

    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*area_threshold.*")

    from bactimate.gui.constants import LOGS_DIR
    from bactimate.gui.log_config import setup_logging, get_logger, get_gui_log_path, write_crash_log
    import logging
    setup_logging(logging.INFO, logs_dir=LOGS_DIR)

    # In windowed PyInstaller builds there is no console. Enable faulthandler to
    # the GUI log so native crashes during heavy imports leave a traceback.
    try:
        import faulthandler
        _fault_path = get_gui_log_path()
        if _fault_path is not None:
            _fault_path.parent.mkdir(parents=True, exist_ok=True)
            _fault_file = open(_fault_path, "a", encoding="utf-8")
            faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception:
        pass

    try:
        log = get_logger()
        _mark(log, "logging_ready")
        log.info("Starting BactiMate")
        log.info("Loading GUI (PyQt6 and tabs; can take 1–2 min on first launch from app bundle)...")
        _mark(log, "before_import_main_window")
        from bactimate.gui.main_window import main
        _mark(log, "after_import_main_window")
        log.info("GUI modules loaded, creating window...")
        main()
    except ModuleNotFoundError as e:
        if "PyQt6" in str(e):
            print("PyQt6 is not installed.", file=sys.stderr)
            print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
            print("Or use a virtual environment and run: ./launch_gui.sh", file=sys.stderr)
            sys.exit(1)
        raise
    except BaseException:
        write_crash_log("Fatal startup error:\n" + traceback.format_exc())
        raise


def main() -> None:
    """Alias for run() for setuptools entry point."""
    run()
