#!/usr/bin/env python3
"""
Assign persistent CellIDs to a mask stack (run once after segmentation).

Segmentation (Omnipose, U-Net) outputs per-frame instance labels (1, 2, 3, ... per frame),
so the same number in different frames is not the same cell. This script links cells
across frames by nearest-neighbor centroid distance and rewrites the mask stack so that
each cell keeps the same ID in every frame. Use it right after segmentation so that
Analyze Masks and Tracking can preserve the same CellID through the pipeline.

Usage:
    python -m bactimate.scripts.assign_persistent_ids masks.tif [-o output.tif] [--pixel-size 0.108]
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile

from bactimate.scripts.analyze_cell_areas import link_persistent_cell_ids_raw


def main():
    parser = argparse.ArgumentParser(
        description="Assign persistent CellIDs to a mask stack (run after segmentation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("masks", type=str, help="Path to mask TIFF (T, Y, X)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path (default: overwrite input, or masks_with_cellids.tif in same dir)")
    parser.add_argument("--pixel-size", type=float, default=0.108,
                        help="Pixel size in µm for linking distance (default: 0.108)")
    parser.add_argument("--max-distance", type=float, default=5.0,
                        help="Max distance in µm to link same cell between frames (default: 5.0)")
    args = parser.parse_args()

    masks_path = Path(args.masks)
    if not masks_path.exists():
        parser.error(f"Masks file not found: {masks_path}")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = masks_path.parent / f"{masks_path.stem}_with_cellids{masks_path.suffix}"

    print(f"Loading masks: {masks_path}")
    masks = tifffile.imread(str(masks_path))
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    num_frames = masks.shape[0]
    print(f"  Frames: {num_frames}")

    print("Linking cells across frames...")
    masks_with_cellids = link_persistent_cell_ids_raw(
        masks,
        pixel_size=args.pixel_size,
        max_distance_um=args.max_distance,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), masks_with_cellids)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
