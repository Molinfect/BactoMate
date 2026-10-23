# 01 — macOS source install stops before a source build

**What to build:** A person who extracted the GitHub source zip and ran the macOS installer on Python 3.14, Python 3.10, Python 3.12, or Conda gets a short failure that names Python 3.11 and the DMG. No NumPy or scikit-image compile starts. A person who already has plain Python 3.11 gets a wheel install. If a virtual environment already exists, it stays recoverable until the new one passes the smoke test. The install log records the interpreter and, on failure, a one-line reason.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] The installer accepts only CPython 3.11 that is not Conda, Miniconda, or Anaconda. It does not fall through to `python3` when that interpreter is another version.
- [x] pip is pinned to `>=24.2,<26`, matching the Windows installer.
- [x] Native packages that failed in the lab logs (NumPy, SciPy, scikit-image) install from wheels only. A missing wheel stops the install with the package name. Pure-Python packages that publish no wheel may still install from an sdist.
- [x] An existing virtual environment is not deleted until the smoke test passes. On failure, the previous environment is still on disk and the message says how to restore it. A leftover previous environment from an earlier failed run is not deleted automatically.
- [x] The install log contains the interpreter path, the version, and on failure a short reason rather than only a compiler transcript.
- [x] An automated test accepts 3.11 and rejects 3.10, 3.12, 3.14, and a Conda prefix.
