"""
Verify that the current interpreter is acceptable for BactoMate on Windows.

BactoMate requires python.org (or py launcher) Python 3.11, 64-bit — not Conda,
not the Microsoft Store stub, not Python 3.12+.

On success prints the absolute path to sys.executable (stdout only) and exits 0.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if sys.platform != "win32":
        _fail("This check is for Windows only.")

    exe = Path(sys.executable).resolve()
    exe_str = str(exe).lower()

    if "windowsapps" in exe_str:
        _fail(
            "Microsoft Store Python is not supported.\n"
            "  Fix: Install Python 3.11 (64-bit) from https://www.python.org/downloads/\n"
            "  Tick \"Add python.exe to PATH\" and disable the Store alias in"
            " Settings → Apps → Advanced app settings → App execution aliases."
        )

    if sys.version_info[:2] != (3, 11):
        _fail(
            f"Python 3.11 is required (found {sys.version_info.major}.{sys.version_info.minor}).\n"
            "  Fix: Install Python 3.11 from https://www.python.org/downloads/\n"
            "  Then run: py -3.11 setup\\install_windows.bat"
        )

    bits = struct.calcsize("P") * 8
    if bits != 64:
        _fail(
            f"64-bit Python is required (found {bits}-bit).\n"
            "  Fix: Uninstall 32-bit Python and install Python 3.11 (64-bit)."
        )

    prefix = Path(sys.prefix).resolve()
    base = Path(sys.base_prefix).resolve()
    for path in (prefix, base, exe.parent):
        if (path / "conda-meta").is_dir():
            _fail(
                "Conda Python is not supported for BactoMate.\n"
                "  Fix: Install plain Python 3.11 from https://www.python.org/downloads/\n"
                "  Do not create the venv with conda base; use py -3.11 -m venv venv."
            )
        lower = str(path).lower()
        if "miniconda" in lower or "anaconda" in lower or "\\conda\\" in lower:
            _fail(
                "Conda Python is not supported for BactoMate.\n"
                "  Fix: Install plain Python 3.11 from https://www.python.org/downloads/"
            )

    print(exe)


if __name__ == "__main__":
    main()
