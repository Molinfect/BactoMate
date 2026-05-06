"""
Windows-only patches for BactiMate. Loaded and applied only when sys.platform == 'win32'
(which includes both 32-bit and 64-bit Windows). On macOS and Linux this module is never imported.

Applied fixes:
- Console encoding: wrap stdout/stderr so Unicode symbols (e.g. ✓/✗) that Windows console
  (cp1252) cannot encode are replaced with ASCII before writing. Prevents
  "'charmap' codec can't encode character '\\u2717'" during conversion and other CLI output.
- UI: ensure push buttons get a minimum width so labels like "Browse File..." / "Browse Folder..."
  are not cut off on Windows (different font metrics / DPI than macOS).

DLL / PyTorch on Windows:
- The "DLL initialization routine failed" (e.g. c10.dll) is an environment issue, not fixable
  by code here. Typical causes: missing Microsoft Visual C++ Redistributable (64-bit), or
  using a CUDA PyTorch build without the matching CUDA/VC++ runtime. Recommend installing
  VC++ Redistributable and/or using a CPU-only PyTorch wheel on Windows if GPU is not needed.
"""
from __future__ import annotations

import sys
from typing import Any, Optional, TextIO


# Unicode symbols that often appear in our log/print output and cause cp1252 encode errors on Windows.
# Map to ASCII equivalents so the same code paths work everywhere.
_WINDOWS_SAFE_REPLACEMENTS = (
    ("\u2717", "[X]"),   # BALLOT X (✗)
    ("\u2713", "[OK]"),  # CHECK MARK (✓)
    ("\u2714", "[OK]"),  # HEAVY CHECK MARK (✔)
    ("\u2718", "[X]"),   # HEAVY BALLOT X (✘)
)


def _safe_encode_text(s: str) -> str:
    """Replace Unicode symbols that break Windows console (cp1252) with ASCII."""
    for old, new in _WINDOWS_SAFE_REPLACEMENTS:
        s = s.replace(old, new)
    return s


class _WindowsSafeStream:
    """Wraps a text stream and replaces Unicode symbols that fail on Windows console encoding."""

    def __init__(self, stream: Optional[TextIO]) -> None:
        self._stream = stream

    def write(self, s: str) -> int:
        if isinstance(s, str):
            s = _safe_encode_text(s)
        if self._stream is None:
            return len(s)
        return self._stream.write(s)

    def flush(self) -> None:
        if self._stream is None:
            return
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        if self._stream is None:
            raise AttributeError(name)
        return getattr(self._stream, name)


def apply_early() -> None:
    """
    Apply patches that must run before any other GUI or worker code.
    Call once at application startup when sys.platform == 'win32'.
    """
    # Wrap stdout/stderr so print() and log output that contains ✓/✗ don't raise
    # UnicodeEncodeError when the console uses cp1252.
    if not isinstance(sys.stdout, _WindowsSafeStream):
        sys.stdout = _WindowsSafeStream(sys.stdout)
    if not isinstance(sys.stderr, _WindowsSafeStream):
        sys.stderr = _WindowsSafeStream(sys.stderr)


def apply_after_ui(app: Any) -> None:
    """
    Apply UI patches after QApplication is created and stylesheet is set.
    Call once from main_window.main() when sys.platform == 'win32'.
    Ensures push buttons have a minimum width so labels like "Browse File..." are not cut off
    on Windows (different default font/DPI than macOS).
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        return
    if app is None:
        return
    # Append a minimal override so QPushButton gets a reasonable min width on Windows.
    # 140px fits "Browse Folder..." with typical Windows font; existing setStyleSheet is preserved.
    extra = """
    QPushButton { min-width: 140px; }
    """
    current = app.styleSheet() or ""
    app.setStyleSheet(current + extra)
