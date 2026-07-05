#!/usr/bin/env python3
"""Print paths of omnipose and peakdetect without triggering full omnipose import (which pulls in peakdetect)."""
import sys
from pathlib import Path

for p in sys.path:
    p = Path(p)
    if not p.is_dir():
        continue
    gpu = p / "omnipose" / "gpu.py"
    peak = p / "peakdetect" / "peakdetect.py"
    if gpu.exists():
        print("omnipose/gpu.py:", gpu.resolve())
    if peak.exists():
        print("peakdetect/peakdetect.py:", peak.resolve())

if "omnipose" not in str(sys.path):
    print("(Run with the same Python you use for BactoMate so the paths match.)", file=sys.stderr)
