#!/usr/bin/env python3
"""
Convert a BactiMate-style checkpoint (state_dict + nchan + nclasses) to a plain
state_dict-only .pth, like built-in Omnipose models (e.g. bact_phase_omnitorch_0).

Why: Some loaders expect torch.load(path) to return only the state_dict. BactiMate
patches cellpose_omni to unwrap automatically; if you use an unpatched environment
or segmentation still fails, convert and try the _plain.pth file.

Usage:
  python -m bactimate.scripts.convert_checkpoint_to_plain path/to/scratch_4000_omni.pth
  python -m bactimate.scripts.convert_checkpoint_to_plain path/to/model.pth -o path/to/model_plain.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save checkpoint as plain state_dict only (same format as built-in Omnipose models)."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to .pth file (BactiMate or Omnipose trained)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: <stem>_plain.pth)")
    args = parser.parse_args()
    pth = args.checkpoint.expanduser().resolve()
    if not pth.exists():
        print(f"Error: file not found: {pth}", file=sys.stderr)
        sys.exit(1)
    from bactimate.gui.omnipose_utils import convert_checkpoint_to_plain_state_dict
    ok, msg = convert_checkpoint_to_plain_state_dict(pth, out_path=args.output)
    if ok:
        print(msg)
    else:
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
