"""Exit 0 when torch.cuda.is_available(), else 1. Used by install_windows.bat."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except (ImportError, RuntimeError, OSError):
        return 1
    return 0 if torch.cuda.is_available() else 1


if __name__ == "__main__":
    sys.exit(main())
