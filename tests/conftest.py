"""Shared pytest configuration for BactoMate GUI tests."""
from __future__ import annotations

import os
import sys

import pytest

# Headless Qt for CI and local pytest without a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Single QApplication instance for widget tests (avoids segfaults on macOS)."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 not available")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
