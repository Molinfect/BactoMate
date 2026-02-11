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
if __name__ == "__main__":
    from bactimate.__main__ import run
    run()
