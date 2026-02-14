#!/usr/bin/env python3
"""
Fix a user-trained .pth checkpoint so it has nchan and nclasses in the file.
Models saved by Omnipose directly (or copied/renamed) often lack these; then
BactiMate's loader can infer wrong architecture and the model fails to detect cells.
This script writes nchan/nclasses into the checkpoint in place.

Usage:
  python -m bactimate.scripts.fix_user_model_metadata path/to/model_omni.pth
  python -m bactimate.scripts.fix_user_model_metadata user_models/mf_ontop_bact_phase_omni.pth
  python -m bactimate.scripts.fix_user_model_metadata user_models/mf_ontop_bact_phase_omni.pth --nclasses 3   # force nclasses=3

After running, reload the model in BactiMate (e.g. reselect it in the dropdown).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add nchan/nclasses to a user .pth so it loads correctly in BactiMate."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to .pth file (e.g. user_models/my_model_omni.pth)")
    parser.add_argument("--nchan", type=int, default=None, help="Force nchan (e.g. 2)")
    parser.add_argument("--nclasses", type=int, default=None, help="Force nclasses (e.g. 3); use if Fix wrote wrong value")
    args = parser.parse_args()
    pth = args.checkpoint.expanduser().resolve()
    if not pth.exists():
        print(f"Error: file not found: {pth}", file=sys.stderr)
        sys.exit(1)
    from bactimate.gui.omnipose_utils import fix_user_model_metadata
    ok, msg = fix_user_model_metadata(pth, nchan_override=args.nchan, nclasses_override=args.nclasses)
    if ok:
        print(msg)
    else:
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
