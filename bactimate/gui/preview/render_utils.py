from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class LutSpec:
    auto_lo_pct: float = 0.01
    auto_hi_pct: float = 99.99


def auto_lut(frame_2d: np.ndarray, *, lo_pct: float = 0.01, hi_pct: float = 99.99) -> tuple[float, float]:
    """Compute a wide percentile LUT band for a 2D frame."""
    arr = np.asarray(frame_2d, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 1.0)
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return (0.0, 1.0)
    try:
        lo = float(np.percentile(flat, float(lo_pct)))
        hi = float(np.percentile(flat, float(hi_pct)))
    except Exception:
        lo = float(np.min(flat))
        hi = float(np.max(flat))
    if not np.isfinite(lo):
        lo = float(np.min(flat))
    if not np.isfinite(hi):
        hi = float(np.max(flat))
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def apply_lut_to_uint8(frame_2d: Optional[np.ndarray], vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
    """
    Map a 2D frame to uint8 using vmin/vmax LUT.

    - If vmin/vmax are None or invalid, uses min/max of the frame.
    - Always returns a contiguous uint8 2D array.
    """
    if frame_2d is None:
        return np.zeros((1, 1), dtype=np.uint8)
    arr = np.asarray(frame_2d, dtype=np.float64)
    if arr.ndim != 2:
        arr = np.asarray(arr.squeeze())
        if arr.ndim != 2:
            return np.zeros((1, 1), dtype=np.uint8)
    flat = arr[np.isfinite(arr)]
    if flat.size == 0:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    mn = float(np.min(flat))
    mx = float(np.max(flat))
    if vmin is None or vmax is None:
        vmin = mn
        vmax = mx if mx > mn else mn + 1.0
    try:
        vmin_f = float(vmin)
        vmax_f = float(vmax)
    except Exception:
        vmin_f, vmax_f = mn, (mx if mx > mn else mn + 1.0)
    if not np.isfinite(vmin_f) or not np.isfinite(vmax_f) or vmax_f <= vmin_f:
        vmin_f, vmax_f = mn, (mx if mx > mn else mn + 1.0)
    vr = max(vmax_f - vmin_f, 1e-6)
    out = ((arr - vmin_f) / vr * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def draw_ring_markers(
    rgb_u8: np.ndarray,
    peaks: Iterable[dict],
    *,
    color: tuple[int, int, int] = (255, 165, 0),
    radius: int = 4,
    scale_from: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Draw small ring markers into an (H, W, 3) uint8 overlay.

    `peaks` expects dict-like items with keys 'y' and 'x' in the coordinates of `scale_from`
    (default: current rgb shape). If `scale_from=(h0,w0)` is provided, peaks are scaled to rgb size.
    """
    rgb = np.asarray(rgb_u8)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return rgb_u8
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if scale_from is not None:
        h0, w0 = int(scale_from[0]), int(scale_from[1])
        sx = (w / w0) if w0 else 1.0
        sy = (h / h0) if h0 else 1.0
    else:
        sx = sy = 1.0

    r = max(1, int(radius))
    c0, c1, c2 = int(color[0]), int(color[1]), int(color[2])
    for p in peaks or []:
        try:
            yy = int(round(float(p.get("y", 0)) * sy))
            xx = int(round(float(p.get("x", 0)) * sx))
        except Exception:
            continue
        if not (0 <= yy < h and 0 <= xx < w):
            continue
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d2 = dx * dx + dy * dy
                if d2 < (r - 1) * (r - 1) or d2 > r * r:
                    continue
                y2 = yy + dy
                x2 = xx + dx
                if 0 <= y2 < h and 0 <= x2 < w:
                    rgb[y2, x2, 0] = c0
                    rgb[y2, x2, 1] = c1
                    rgb[y2, x2, 2] = c2
    return rgb_u8

