"""Fast vectorized foci marker and blob overlay drawing for previews."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_FOCI_PREVIEW_RENDER_CAP = 500

_RING_OFFSETS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _ring_offsets(radius: int, *, inner_gap: int = 2) -> tuple[np.ndarray, np.ndarray]:
    r = max(2, int(radius))
    cached = _RING_OFFSETS.get(r)
    if cached is not None:
        return cached
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
    d2 = dy.astype(np.int32) ** 2 + dx.astype(np.int32) ** 2
    outer_sq = r * r
    inner_sq = max(0, (r - int(inner_gap)) * (r - int(inner_gap)))
    mask = (d2 <= outer_sq) & (d2 >= inner_sq)
    offsets = (dy[mask].astype(np.int32), dx[mask].astype(np.int32))
    _RING_OFFSETS[r] = offsets
    return offsets


def _scale_factors(
    mask_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[float, float]:
    h0, w0 = int(mask_shape[0]), int(mask_shape[1])
    hs, ws = int(target_shape[0]), int(target_shape[1])
    sx = (ws / w0) if w0 else 1.0
    sy = (hs / h0) if h0 else 1.0
    return sx, sy


def _scaled_peak_coords(
    peaks: Sequence[dict[str, Any]],
    *,
    mask_shape: tuple[int, int],
    target_shape: tuple[int, int],
    cap: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not peaks:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    items = list(peaks)
    if cap is not None and int(cap) > 0:
        items = items[: int(cap)]
    sx, sy = _scale_factors(mask_shape, target_shape)
    ys = np.array([int(round(float(p.get("y", 0)) * sy)) for p in items], dtype=np.int32)
    xs = np.array([int(round(float(p.get("x", 0)) * sx)) for p in items], dtype=np.int32)
    return ys, xs


def draw_peak_rings_on_rgb(
    target_rgb: np.ndarray,
    peaks: Sequence[dict[str, Any]],
    color: tuple[int, int, int],
    *,
    mask_shape: tuple[int, int],
    radius: int = 4,
    inner_gap: int = 2,
    cap: Optional[int] = None,
) -> None:
    """Draw hollow ring markers for many peaks using vectorized indexing."""
    hs, ws = int(target_rgb.shape[0]), int(target_rgb.shape[1])
    ys, xs = _scaled_peak_coords(
        peaks,
        mask_shape=mask_shape,
        target_shape=(hs, ws),
        cap=cap,
    )
    if ys.size == 0:
        return
    dy, dx = _ring_offsets(radius, inner_gap=inner_gap)
    cr, cg, cb = (int(color[0]), int(color[1]), int(color[2]))
    for yy, xx in zip(ys, xs):
        y2 = yy + dy
        x2 = xx + dx
        valid = (y2 >= 0) & (y2 < hs) & (x2 >= 0) & (x2 < ws)
        if not np.any(valid):
            continue
        target_rgb[y2[valid], x2[valid], 0] = cr
        target_rgb[y2[valid], x2[valid], 1] = cg
        target_rgb[y2[valid], x2[valid], 2] = cb


def draw_foci_blob_contours_on_rgb(
    target_rgb: np.ndarray,
    mask_2d: np.ndarray,
    peaks: Sequence[dict[str, Any]],
    plane: np.ndarray,
    settings: dict[str, Any],
    contour_rgb: tuple[int, int, int],
    *,
    cap: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Draw intensity-grown focus boundaries; batch boundary extraction when possible."""
    from skimage.segmentation import find_boundaries

    from bactomate.gui.foci_filter_pipeline import (
        foci_blob_max_radius,
        foci_blob_threshold_frac,
    )
    from bactomate.modules.foci_utils import get_focus_blob_mask

    h0, w0 = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    h1, w1 = int(target_rgb.shape[0]), int(target_rgb.shape[1])
    sx = (w1 / w0) if w0 else 1.0
    sy = (h1 / h0) if h0 else 1.0
    frac = foci_blob_threshold_frac(settings)
    max_r = foci_blob_max_radius(settings)
    label = np.asarray(mask_2d, dtype=np.int32)
    work = np.asarray(plane, dtype=np.float64)
    cr, cg, cb = (int(contour_rgb[0]), int(contour_rgb[1]), int(contour_rgb[2]))
    missed: list[dict[str, Any]] = []
    if not peaks:
        return missed

    items = list(peaks)
    if cap is not None and int(cap) > 0:
        items = items[: int(cap)]

    combined = np.zeros((h1, w1), dtype=bool)
    for p in items:
        try:
            hy = int(float(p.get("y", 0) or 0))
            hx = int(float(p.get("x", 0) or 0))
        except Exception:
            continue
        blob = get_focus_blob_mask(
            work, hy, hx, threshold_frac=frac, max_radius=max_r, labeled_mask=label
        )
        if blob is None or not np.any(blob):
            missed.append(p)
            continue
        ys, xs = np.nonzero(blob)
        y2 = np.clip(np.round(ys.astype(np.float64) * sy).astype(np.int32), 0, h1 - 1)
        x2 = np.clip(np.round(xs.astype(np.float64) * sx).astype(np.int32), 0, w1 - 1)
        combined[y2, x2] = True

    if np.any(combined):
        try:
            edge = find_boundaries(combined, mode="outer")
            if np.any(edge):
                target_rgb[edge, 0] = cr
                target_rgb[edge, 1] = cg
                target_rgb[edge, 2] = cb
        except Exception:
            pass
    else:
        missed.extend(items)
    return missed


def draw_foci_blob_fill_and_edges_on_rgb(
    target_rgb: np.ndarray,
    plane: np.ndarray,
    mask: np.ndarray,
    peaks: Sequence[dict[str, Any]],
    draw_color: np.ndarray,
    *,
    threshold_frac: float = 0.8,
    max_radius: int = 5,
    cap: Optional[int] = None,
) -> None:
    """Fill combined focus blobs once, then extract outer edges once (preview hot path)."""
    from skimage.segmentation import find_boundaries

    from bactomate.modules.foci_utils import get_focus_blob_mask

    if not peaks:
        return
    items = list(peaks)
    if cap is not None and int(cap) > 0:
        items = items[: int(cap)]
    label = np.asarray(mask, dtype=np.int32)
    work = np.asarray(plane, dtype=np.float32)
    h, w = int(target_rgb.shape[0]), int(target_rgb.shape[1])
    combined = np.zeros((h, w), dtype=bool)
    frac = float(np.clip(threshold_frac, 0.05, 1.0))
    max_r = max(1, int(max_radius))
    for p in items:
        try:
            yi = int(p["y"])
            xi = int(p["x"])
        except Exception:
            continue
        blob = get_focus_blob_mask(
            work, yi, xi, threshold_frac=frac, max_radius=max_r, labeled_mask=label
        )
        if blob is not None and np.any(blob):
            combined |= np.asarray(blob, dtype=bool)
    if not np.any(combined):
        return
    target_rgb[combined] = draw_color
    try:
        boundaries = find_boundaries(combined, mode="outer")
        if np.any(boundaries):
            target_rgb[boundaries, 0] = 0
            target_rgb[boundaries, 1] = 0
            target_rgb[boundaries, 2] = 0
    except Exception:
        pass
