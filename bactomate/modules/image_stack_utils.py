"""
Utilities to coerce FI/foci images into predictable shapes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def percentile_normalize_2d(image: np.ndarray, *, pmin: float = 1.0, pmax: float = 99.8) -> np.ndarray:
    """
    Percentile-based normalization for display/inference.

    Returns float32 in [0, 1] using robust percentiles (like Spotiflow's normalize).
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.astype(np.float32, copy=False)
    flat = arr[np.isfinite(arr)]
    if flat.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(flat, float(pmin)))
    hi = float(np.percentile(flat, float(pmax)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)


def coerce_plane_to_2d(image: np.ndarray) -> np.ndarray:
    """
    Return a single 2D plane from an image-like array.

    Heuristics match existing FI/foci behavior:
    - 2D stays unchanged.
    - 3D with small trailing dimension is treated as YXC and channel 0 is used.
    - Otherwise leading slices are peeled until 2D remains.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        return np.asarray(arr, dtype=np.float64).reshape((0, 0))
    while arr.ndim > 2:
        if arr.ndim == 3 and arr.shape[-1] < 10:
            arr = arr[:, :, 0]
        else:
            arr = arr[0]
    return np.asarray(arr)


def coerce_stack_to_tyx(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Coerce a FI/foci array to (T, Y, X) using conservative shape heuristics.

    Returns None when coercion is not possible.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        return None

    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim == 3:
        # YXC -> choose first channel and add T=1
        if arr.shape[-1] < 10 and arr.shape[0] >= 10 and arr.shape[1] >= 10:
            return arr[:, :, 0][np.newaxis, ...]
        # CYX -> choose first channel and add T=1
        if arr.shape[0] < 10 and arr.shape[1] >= 10 and arr.shape[2] >= 10:
            return arr[0][np.newaxis, ...]
        # Assume TYX
        return arr

    if arr.ndim == 4:
        # TCYX
        if arr.shape[1] < 10 and arr.shape[2] >= 10 and arr.shape[3] >= 10:
            return arr[:, 0, :, :]
        # TYXC
        if arr.shape[-1] < 10 and arr.shape[1] >= 10 and arr.shape[2] >= 10:
            return arr[:, :, :, 0]
        # CTYX
        if arr.shape[0] < 10 and arr.shape[2] >= 10 and arr.shape[3] >= 10:
            return arr[0, :, :, :]

    return None
