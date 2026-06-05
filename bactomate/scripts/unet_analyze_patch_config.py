#!/usr/bin/env python
"""
Analyze training data and write patch_config.json for U-Net patch-based training.
Uses in-repo minimal geometry analysis from image/mask dimensions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent so we can import from bactomate.gui when run as script
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _minimal_analyze(train_path: Path, img_filter: str, mask_filter: str) -> dict:
    """Minimal patch config from training folder (no scipy/matplotlib)."""
    try:
        from bactomate.gui.mask_utils import get_training_pairs
    except ImportError:
        pairs = []
        exts = (".tif", ".tiff")
        for f in train_path.iterdir():
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            base = f.stem.replace(img_filter, "") if img_filter else f.stem
            for ext in exts:
                mask_path = train_path / f"{base}{mask_filter}{ext}"
                if mask_path.exists():
                    pairs.append((f, mask_path))
                    break
    else:
        pairs = get_training_pairs(train_path, img_filter, mask_filter)

    if not pairs:
        return {}
    try:
        import numpy as np
        import tifffile
    except ImportError:
        return {"patch_height": 256, "patch_width": 256, "overlap": 8, "patches_per_image": 500}

    heights, widths = [], []
    for img_path, _ in pairs[:50]:
        try:
            arr = tifffile.imread(str(img_path))
            if arr.ndim == 3:
                if arr.shape[-1] in (2, 3, 4):
                    h, w = arr.shape[0], arr.shape[1]
                else:
                    h, w = arr.shape[1], arr.shape[2]
            else:
                h, w = arr.shape[0], arr.shape[1]
            heights.append(h)
            widths.append(w)
        except Exception:
            continue
    if not heights:
        return {"patch_height": 256, "patch_width": 256, "overlap": 8, "patches_per_image": 500}
    h_med = int(np.median(heights))
    w_med = int(np.median(widths))
    # Patch size divisible by 8; cap at 512
    patch_h = min(512, max(16, (h_med // 8) * 8))
    patch_w = min(512, max(16, (w_med // 8) * 8))
    return {
        "patch_height": patch_h,
        "patch_width": patch_w,
        "overlap": max(0, min(16, patch_h // 15)),
        "patches_per_image": 500,
        "min_cell_coverage": 0.20,
        "max_cell_coverage": 0.90,
        "max_empty_region_ratio": 0.90,
        "max_empty_region_dimension": 1.0,
        "forced_patch_size": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze training data for U-Net patch config.")
    parser.add_argument("--train-path", type=Path, required=True, help="Training directory (images + masks)")
    parser.add_argument("--output", type=Path, required=True, help="Output patch_config.json path")
    parser.add_argument("--img-filter", default="", help="Image filename suffix (e.g. _img)")
    parser.add_argument("--mask-filter", default="_mask", help="Mask filename suffix (e.g. _mask)")
    parser.add_argument("--force-patch-size", type=str, default=None, help="Force WIDTHxHEIGHT (e.g. 30x120)")
    parser.add_argument("--no-visualizations", action="store_true", help="Skip creating visualization files")
    args = parser.parse_args()

    train_path = args.train_path.resolve()
    if not train_path.is_dir():
        print(f"Error: Training path is not a directory: {train_path}", file=sys.stderr)
        return 1

    cfg = _minimal_analyze(train_path, args.img_filter, args.mask_filter)
    if args.force_patch_size:
        try:
            w_str, h_str = args.force_patch_size.lower().split("x")
            cfg["patch_width"] = int(w_str.strip())
            cfg["patch_height"] = int(h_str.strip())
            cfg["forced_patch_size"] = True
        except ValueError:
            print(f"Invalid --force-patch-size: {args.force_patch_size}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
