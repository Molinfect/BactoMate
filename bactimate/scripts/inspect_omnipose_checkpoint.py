#!/usr/bin/env python3
"""
Inspect an Omnipose/Cellpose .pth checkpoint and print architecture-related fields.
Use this to compare user-trained models with built-in ones and debug "no cells" issues.

Usage:
  python -m bactimate.scripts.inspect_omnipose_checkpoint path/to/model.pth
  python -m bactimate.scripts.inspect_omnipose_checkpoint --list-user-models   # show where BactiMate stores models and list .pth files

BactiMate user models folder:
  - Running from repo:  <repo_root>/user_models/
  - Installed (macOS):  ~/Library/Application Support/BactiMate/Models
  - Installed (Windows): %APPDATA%/BactiMate/Models

See: https://omnipose.readthedocs.io (Training, Settings).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _print_user_models_help() -> None:
    """Print BactiMate's user models directory and list .pth files there."""
    from bactimate.gui.paths import get_user_models_dir
    user_dir = get_user_models_dir()
    print(f"BactiMate user models folder:\n  {user_dir}")
    pth_files = sorted(user_dir.glob("*.pth"))
    if not pth_files:
        print("  (no .pth files found)")
    else:
        print(f"  .pth files ({len(pth_files)}):")
        for f in pth_files:
            print(f"    {f.name}")
    print("\nExample:")
    if pth_files:
        print(f"  python -m bactimate.scripts.inspect_omnipose_checkpoint \"{pth_files[0]}\"")
    else:
        print(f"  python -m bactimate.scripts.inspect_omnipose_checkpoint \"{user_dir}/your_model.pth\"")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Omnipose/Cellpose .pth checkpoint (keys, nchan, nclasses, first layer shape)."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=None,
        help="Path to .pth model file (optional if --list-user-models is set)",
    )
    parser.add_argument(
        "--list-user-models",
        action="store_true",
        help="Show BactiMate's user models folder and list .pth files; then exit",
    )
    args = parser.parse_args()

    if args.list_user_models:
        _print_user_models_help()
        sys.exit(0)

    if not args.checkpoint:
        print("Error: give a checkpoint path or use --list-user-models", file=sys.stderr)
        _print_user_models_help()
        sys.exit(1)

    pth = args.checkpoint.expanduser().resolve()
    if not pth.exists():
        print(f"Error: file not found: {pth}", file=sys.stderr)
        print("BactiMate user models folder and .pth files:", file=sys.stderr)
        _print_user_models_help()
        sys.exit(1)

    from bactimate.gui.omnipose_utils import inspect_omnipose_checkpoint, get_default_user_model_architecture

    info = inspect_omnipose_checkpoint(pth)
    if "error" in info:
        print(f"Error loading checkpoint: {info['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Checkpoint: {pth.name}")
    stem = pth.stem
    if "omni" not in stem.lower():
        print("*** WARNING: Filename stem does not contain 'omni'. ***")
        print("   cellpose_omni loads custom models with omni=True ONLY when 'omni' is in the filename.")
        print("   Otherwise it uses standard Cellpose (omni=False) -> wrong or random detections.")
        print("   Fix: Rename to e.g. mf_test_omni.pth, or use BactiMate (it now passes an omni-named path automatically).")
    print(f"Top-level keys ({len(info['keys'])}): {info['keys'][:20]}{'...' if len(info['keys']) > 20 else ''}")
    print(f"nchan (from checkpoint or inferred): {info.get('nchan')}")
    print(f"nclasses (from checkpoint): {info.get('nclasses')}")
    print(f"diam_mean (from checkpoint): {info.get('diam_mean')}")
    print(f"First conv weight shape (out, in, k, k): {info.get('first_layer_shape')}")
    default_nchan, default_nclasses = get_default_user_model_architecture()
    print(f"\nBactiMate default at inference: nchan={default_nchan}, nclasses={default_nclasses}")
    print("If your model was trained with different nchan/nclasses, inference must use the same.")


if __name__ == "__main__":
    main()
