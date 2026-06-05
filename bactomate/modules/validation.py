"""
Shared validation helpers for BactoMate.
Used before long runs to fail fast with clear error messages.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def paths_exist(paths: List[str | Path]) -> Tuple[bool, List[str]]:
    """
    Check which paths exist. Returns (all_exist, list_of_missing_paths).
    """
    missing: List[str] = []
    for p in paths:
        path = Path(p) if not isinstance(p, Path) else p
        if not path.exists():
            missing.append(str(path))
    return (len(missing) == 0, missing)


def mask_fi_spatial_shape_match(
    mask_path: str | Path,
    fi_path: str | Path,
) -> Tuple[bool, str | None]:
    """
    Check that the spatial dimensions (height, width) of the mask and FI image match.
    Loads only the first frame of each to avoid reading full stacks.

    Returns:
        (True, None) if shapes match.
        (False, error_message) if mismatch or read error.
    """
    try:
        import tifffile
    except ImportError:
        return (False, "tifffile is required for validation.")

    mask_path = Path(mask_path)
    fi_path = Path(fi_path)
    if not mask_path.exists():
        return (False, f"Mask file not found: {mask_path}")
    if not fi_path.exists():
        return (False, f"FI file not found: {fi_path}")

    try:
        masks = tifffile.imread(mask_path)
    except Exception as e:
        return (False, f"Could not read mask file: {e}")

    try:
        fi_images = tifffile.imread(fi_path)
    except Exception as e:
        return (False, f"Could not read FI file: {e}")

    # Mask: first frame shape (H, W)
    if masks.ndim == 2:
        mask_h, mask_w = masks.shape
    elif masks.ndim >= 3:
        mask_h, mask_w = masks.shape[-2], masks.shape[-1]
    else:
        return (False, f"Unexpected mask dimensions: {masks.ndim}")

    # FI: normalize to (T, H, W) and take first frame (H, W)
    if fi_images.ndim == 2:
        fi_h, fi_w = fi_images.shape
    elif fi_images.ndim == 4:
        # (T, C, Y, X)
        fi_h, fi_w = fi_images.shape[2], fi_images.shape[3]
    elif fi_images.ndim == 3:
        if fi_images.shape[-1] < 10:
            # (Y, X, C)
            fi_h, fi_w = fi_images.shape[0], fi_images.shape[1]
        else:
            # (T, Y, X)
            fi_h, fi_w = fi_images.shape[1], fi_images.shape[2]
    else:
        return (False, f"Unexpected FI image dimensions: {fi_images.ndim}")

    if (mask_h, mask_w) != (fi_h, fi_w):
        return (
            False,
            f"Mask and FI image spatial sizes do not match: mask {mask_h}×{mask_w}, FI {fi_h}×{fi_w}. "
            "Both must have the same height and width.",
        )
    return (True, None)
