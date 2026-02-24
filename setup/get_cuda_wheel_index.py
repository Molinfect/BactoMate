"""
Print PyTorch wheel index suffix for current NVIDIA driver (e.g. cu128 for CUDA 12.8).
Run with: py -3.11 setup/get_cuda_wheel_index.py
Prints one line: cu128, cu126, cu124, cu121, cu118, or empty if no GPU / nvidia-smi failed.
"""
from __future__ import annotations

import re
import subprocess
import sys


def main() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("", end="")
        return 0
    if out.returncode != 0 or not out.stdout:
        print("", end="")
        return 0
    # e.g. "CUDA Version: 12.8" or "CUDA Version: 11.8"
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out.stdout)
    if not m:
        print("", end="")
        return 0
    major, minor = int(m.group(1)), int(m.group(2))
    # Map to known PyTorch indices
    if major >= 12:
        if minor >= 8:
            print("cu128", end="")
        elif minor >= 6:
            print("cu126", end="")
        elif minor >= 4:
            print("cu124", end="")
        elif minor >= 1:
            print("cu121", end="")
        else:
            print("cu121", end="")
    elif major == 11 and minor >= 8:
        print("cu118", end="")
    else:
        print("cu118", end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
