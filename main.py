#!/usr/bin/env python3
"""
BactiMate – batch processing and analysis GUI for microscopy (Omnipose, tracking, SAM2).

Usage:
    python main.py
    python main.py --version
    python main.py --smoke-test
    python -m bactimate
    bactimate   (after pip install -e .)
"""
# Diagnostic for follow_build.py: record who is running (pid, parent, executable, argv0)
if __name__ == "__main__":
    try:
        import os
        import sys
        chain_log = os.path.expanduser("~/Library/Logs/BactiMate/launch_chain.log")
        with open(chain_log, "a") as f:
            f.write("MAIN_PY pid=%s ppid=%s executable=%s argv0=%s\n" % (
                os.getpid(), os.getppid(), getattr(sys, "executable", ""), sys.argv[0] if sys.argv else ""
            ))
    except Exception:
        pass

# Set process title first so macOS menu bar and Dock show "BactiMate" not "Python"
# (required when launched from .app bundle via python main.py; setproctitle must be in venv)
if __name__ == "__main__":
    try:
        import setproctitle
        setproctitle.setproctitle("BactiMate")
    except Exception:
        pass

if __name__ == "__main__":
    from bactimate.__main__ import run
    run()
