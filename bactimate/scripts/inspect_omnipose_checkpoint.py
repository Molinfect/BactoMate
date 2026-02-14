#!/usr/bin/env python3
"""
Inspect an Omnipose/Cellpose .pth checkpoint and print architecture-related fields.
Use this to compare user-trained models with built-in ones and debug "no cells" issues.

Usage:
  python -m bactimate.scripts.inspect_omnipose_checkpoint path/to/model.pth
  python -m bactimate.scripts.inspect_omnipose_checkpoint path/to/model.pth --fix   # write nchan/nclasses into file, then inspect
  python -m bactimate.scripts.inspect_omnipose_checkpoint --list-user-models   # show BactiMate user models folder and list .pth files
  python -m bactimate.scripts.inspect_omnipose_checkpoint --list-builtin        # show cellpose_omni cache dir and built-in model paths (bact_phase_omni, etc.)

You can inspect any checkpoint by path: user_models/*.pth or the cache paths from --list-builtin.

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


def _get_builtin_model_dir() -> Path:
    """Return cellpose_omni cache dir without importing cellpose_omni (avoids broken omnipose.core in system install)."""
    import os
    env = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cellpose" / "models"


def _print_builtin_models_help() -> None:
    """Print cellpose_omni cache directory and paths to built-in models (bact_phase_omni, etc.)."""
    model_dir = _get_builtin_model_dir()
    print(f"Built-in Omnipose models cache (cellpose_omni):\n  {model_dir}")
    if not model_dir.exists():
        print("  (directory does not exist yet; run BactiMate or segment once to download models)")
        return
    # Known built-in types (BactiMate uses these); files are named e.g. bact_phase_omnitorch_0 (no .pth)
    builtin_types = [
        "bact_phase_omni",
        "bact_phase_affinity",
        "bact_omni",
        "bact_fluor_omni",
        "cyto2_omni",
        "cyto_omni",
    ]
    print("\nBuilt-in model paths (use these with inspect):")
    for name in builtin_types:
        path = model_dir / f"{name}torch_0"
        status = "✓" if path.exists() else "(not downloaded)"
        print(f"  {path}  {status}")
    print("\nExample:")
    p = model_dir / "bact_phase_omnitorch_0"
    if p.exists():
        print(f"  python3 -m bactimate.scripts.inspect_omnipose_checkpoint \"{p}\"")
    else:
        print(f"  python3 -m bactimate.scripts.inspect_omnipose_checkpoint \"{model_dir}/bact_phase_omnitorch_0\"")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Omnipose/Cellpose .pth checkpoint (keys, nchan, nclasses, first layer shape)."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=None,
        help="Path to checkpoint file (.pth or cache file from --list-builtin); optional if --list-* is set",
    )
    parser.add_argument(
        "--list-user-models",
        action="store_true",
        help="Show BactiMate's user models folder and list .pth files; then exit",
    )
    parser.add_argument(
        "--list-builtin",
        action="store_true",
        help="Show cellpose_omni cache directory and paths for built-in models (bact_phase_omni, etc.); then exit",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Write nchan/nclasses into the checkpoint (same as Fix metadata in BactiMate); then show inspect output",
    )
    args = parser.parse_args()

    if args.list_user_models:
        _print_user_models_help()
        sys.exit(0)
    if args.list_builtin:
        _print_builtin_models_help()
        sys.exit(0)

    if not args.checkpoint:
        print("Error: give a checkpoint path or use --list-user-models or --list-builtin", file=sys.stderr)
        _print_user_models_help()
        sys.exit(1)

    pth = args.checkpoint.expanduser().resolve()
    if not pth.exists():
        print(f"Error: file not found: {pth}", file=sys.stderr)
        print("BactiMate user models folder and .pth files:", file=sys.stderr)
        _print_user_models_help()
        sys.exit(1)

    from bactimate.gui.omnipose_utils import (
        inspect_omnipose_checkpoint,
        get_default_user_model_architecture,
        fix_user_model_metadata,
    )

    if args.fix:
        ok, msg = fix_user_model_metadata(pth)
        if ok:
            print(msg)
        else:
            print(f"Fix failed: {msg}", file=sys.stderr)
            sys.exit(1)

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
    print(f"nclasses (from checkpoint or inferred from output layer): {info.get('nclasses')}")
    print(f"diam_mean (from checkpoint): {info.get('diam_mean')}")
    print(f"First conv weight shape (out, in, k, k): {info.get('first_layer_shape')}")
    default_nchan, default_nclasses = get_default_user_model_architecture()
    print(f"\nBactiMate default at inference: nchan={default_nchan}, nclasses={default_nclasses}")
    print("If your model was trained with different nchan/nclasses, inference must use the same.")


if __name__ == "__main__":
    main()
