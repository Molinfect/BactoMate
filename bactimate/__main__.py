"""
Entry point for running BactiMate (python -m bactimate or pip-installed bactimate command).
Handles --version, --smoke-test, patches, logging, then launches the GUI.
"""
import sys
import traceback
import warnings


def run() -> None:
    # --version / --smoke-test: exit before loading full GUI (used by CI / frozen builds)
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

    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*area_threshold.*")

    from bactimate.gui.constants import LOGS_DIR
    from bactimate.gui.log_config import setup_logging, get_logger
    import logging
    setup_logging(logging.INFO, logs_dir=LOGS_DIR)

    try:
        log = get_logger()
        log.info("Starting BactiMate")
        log.info("Loading GUI (PyQt6 and tabs; can take 1–2 min on first launch from app bundle)...")
        from bactimate.gui.main_window import main
        log.info("GUI modules loaded, creating window...")
        main()
    except ModuleNotFoundError as e:
        if "PyQt6" in str(e):
            print("PyQt6 is not installed.", file=sys.stderr)
            print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
            print("Or use a virtual environment and run: ./launch_gui.sh", file=sys.stderr)
            sys.exit(1)
        raise


def main() -> None:
    """Alias for run() for setuptools entry point."""
    run()
