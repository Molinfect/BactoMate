#!/usr/bin/env python3
"""
BactiMate – batch processing and analysis GUI for microscopy (Omnipose, tracking, SAM2).

Usage:
    python main.py
    python main.py --version
    python main.py --smoke-test
"""
import sys
import traceback
import warnings

# Structured logging (configured after patches; gui uses get_logger())
import logging

# --version / --smoke-test: exit before loading full GUI (used by CI / frozen builds)
if len(sys.argv) >= 2 and sys.argv[1] in ("--version", "-v"):
    print("BactiMate 1.0")
    sys.exit(0)
if len(sys.argv) >= 2 and sys.argv[1] == "--smoke-test":
    try:
        import gui.patches  # noqa: F401 – Qt paths must be set before PyQt6
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
import gui.patches  # noqa: F401

# Suppress noisy warnings from omnipose/pkg_resources and area_threshold
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*area_threshold.*")

# Structured logging for app and GUI
from gui.log_config import setup_logging, get_logger
setup_logging(logging.INFO)

if __name__ == "__main__":
    try:
        log = get_logger()
        log.info("Starting BactiMate")
        from gui.main_window import main
        main()
    except ModuleNotFoundError as e:
        if "PyQt6" in str(e):
            print("PyQt6 is not installed.", file=sys.stderr)
            print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
            print("Or use a virtual environment and run: ./launch_gui.sh", file=sys.stderr)
            sys.exit(1)
        raise
