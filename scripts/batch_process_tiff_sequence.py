#!/usr/bin/env python3
"""
Entry point for batch_process_tiff_sequence when running from repo root.
Delegates to bactimate.scripts.batch_process_tiff_sequence (single source of truth).
"""
import sys
from pathlib import Path

# Ensure project root is on path when running from repo
_script_dir = Path(__file__).resolve().parent
_root = _script_dir.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if __name__ == "__main__":
    from bactimate.scripts.batch_process_tiff_sequence import main
    main()
