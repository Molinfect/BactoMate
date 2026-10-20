"""
Verify that an interpreter is acceptable for a BactoMate macOS source install.

BactoMate requires plain CPython 3.11 (python.org or Homebrew). Conda,
Miniconda, Anaconda, and every other minor version are rejected before pip runs.

On success prints the absolute path to sys.executable (stdout only) and exits 0.
On failure prints a short message to stderr and exits 1.

Kept compatible with old interpreters (3.8+) because the installer runs this
against whatever python3 is on PATH before it knows the version is acceptable.
"""
import os
import sys
from typing import Optional, Tuple

REQUIRED_VERSION = (3, 11)
CONDA_MARKERS = ("miniconda", "anaconda")

_FIX = (
    "  Fix: Install plain Python 3.11 from https://www.python.org/downloads/\n"
    "       or run: brew install python@3.11\n"
    "  Lab users can instead download BactoMate.dmg from the GitHub release."
)


def check_python(
    version: Tuple[int, int], executable: str, prefix: str
) -> Optional[str]:
    """Return None if the interpreter is acceptable, else a short error message."""
    major, minor = version[0], version[1]
    if (major, minor) != REQUIRED_VERSION:
        return (
            f"Python 3.11 is required (found {major}.{minor}).\n"
            "  The macOS source installer accepts Python 3.11.\n" + _FIX
        )

    if _is_conda(executable, prefix):
        return (
            "Conda Python is not supported for BactoMate (found Conda/Miniconda/Anaconda).\n"
            "  Create the venv with Python 3.11 from python.org or Homebrew.\n" + _FIX
        )

    return None


def _is_conda(executable: str, prefix: str) -> bool:
    lowered = (executable.lower(), prefix.lower())
    if any(marker in path for path in lowered for marker in CONDA_MARKERS):
        return True
    return bool(prefix) and os.path.isdir(os.path.join(prefix, "conda-meta"))


def main() -> None:
    version = (sys.version_info.major, sys.version_info.minor)
    executable = os.path.abspath(sys.executable)
    real_executable = os.path.realpath(sys.executable)

    # A venv's own prefix looks clean even when its base interpreter is Conda,
    # so check base_prefix and the symlink-resolved executable as well.
    prefixes = {sys.prefix, getattr(sys, "base_prefix", sys.prefix)}
    candidates = [(exe, prefix) for exe in {executable, real_executable} for prefix in prefixes]
    for exe, prefix in candidates:
        error = check_python(version, exe, prefix)
        if error is not None:
            print(error, file=sys.stderr)
            raise SystemExit(1)

    print(executable)


if __name__ == "__main__":
    main()
