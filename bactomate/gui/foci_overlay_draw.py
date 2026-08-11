"""Fast vectorized foci marker and blob overlay drawing for previews."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_FOCI_PREVIEW_RENDER_CAP = 500

# Uniform preview / overlay language: Spotiflow peak pixel + exclusive blob outline.
FOCI_PEAK_PIXEL_RGB: tuple[int, int, int] = (255, 30, 30)
FOCI_BLOB_OUTLINE_RGB: tuple[int, int, int] = (255, 200, 120)
FOCI_FAIL_OUTLINE_RGB: tuple[int, int, int] = (255, 140, 140)

_RING_OFFSETS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _exclusive_blob_outline_mask(blob: np.ndarray) -> np.ndarray:
    """Hollow 1 px ring around an exclusive blob (never fills the interior)."""
    m = np.asarray(blob, dtype=bool)
    if m.size == 0 or not np.any(m):
        return m
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(m, iterations=1) & ~m
    except Exception:
        from skimage.segmentation import find_boundaries

        return find_boundaries(m.astype(np.uint8), mode="outer")


def _paint_scaled_bool_mask_on_rgb(
    target_rgb: np.ndarray,
    mask_bool: np.ndarray,
    color: tuple[int, int, int],
    *,
    sy: float,
    sx: float,
) -> None:
    if mask_bool is None or not np.any(mask_bool):
        return
    h1, w1 = int(target_rgb.shape[0]), int(target_rgb.shape[1])
    ys, xs = np.nonzero(mask_bool)
    y2 = np.clip(np.round(ys.astype(np.float64) * sy).astype(np.int32), 0, h1 - 1)
    x2 = np.clip(np.round(xs.astype(np.float64) * sx).astype(np.int32), 0, w1 - 1)
    cr, cg, cb = (int(color[0]), int(color[1]), int(color[2]))
    target_rgb[y2, x2, 0] = cr
    target_rgb[y2, x2, 1] = cg
    target_rgb[y2, x2, 2] = cb


def _peak_hollow_ring_mask(
    shape: tuple[int, int],
    y: int,
    x: int,
    *,
    radius: int = 2,
) -> np.ndarray:
    """Fixed hollow ring around a peak when exclusive blobs are too small to outline."""
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=bool)
    r = max(1, int(radius))
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            d2 = (yy - y) * (yy - y) + (xx - x) * (xx - x)
            if d2 == 0:
                continue
            if (r - 1) * (r - 1) <= d2 <= r * r:
                out[yy, xx] = True
    return out


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


def draw_peak_pixels_on_rgb(
    target_rgb: np.ndarray,
    peaks: Sequence[dict[str, Any]],
    color: tuple[int, int, int],
    *,
    mask_shape: tuple[int, int],
    cap: Optional[int] = None,
    min_display_size: int = 1,
) -> None:
    """Paint each Spotiflow peak as a solid pixel (or tiny block when preview is upscaled).

    This is the detection coordinate itself — distinct from blob outlines — so overlapping
    foci remain countable as separate red centers.
    """
    hs, ws = int(target_rgb.shape[0]), int(target_rgb.shape[1])
    h0, w0 = int(mask_shape[0]), int(mask_shape[1])
    sx = (ws / w0) if w0 else 1.0
    sy = (hs / h0) if h0 else 1.0
    # Keep the painted spot ~one source pixel wide on screen when zoomed in.
    block = max(1, int(min_display_size), int(round(min(sx, sy))))
    half = max(0, (block - 1) // 2)
    ys, xs = _scaled_peak_coords(
        peaks,
        mask_shape=mask_shape,
        target_shape=(hs, ws),
        cap=cap,
    )
    if ys.size == 0:
        return
    cr, cg, cb = (int(color[0]), int(color[1]), int(color[2]))
    for yy, xx in zip(ys, xs):
        y0b = max(0, int(yy) - half)
        y1b = min(hs, int(yy) + half + 1)
        x0b = max(0, int(xx) - half)
        x1b = min(ws, int(xx) + half + 1)
        if y0b >= y1b or x0b >= x1b:
            continue
        target_rgb[y0b:y1b, x0b:x1b, 0] = cr
        target_rgb[y0b:y1b, x0b:x1b, 1] = cg
        target_rgb[y0b:y1b, x0b:x1b, 2] = cb


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


def draw_peak_qc_markers_on_rgb(
    target_rgb: np.ndarray,
    peaks: Sequence[dict[str, Any]],
    color: tuple[int, int, int],
    *,
    mask_shape: tuple[int, int],
    alpha: float = 0.55,
    plus: Optional[bool] = None,
) -> int:
    """Draw compact alpha-blended foci markers for folder / QC thumbnails.

    Rings on every peak become an opaque carpet on dense FOVs. This paints at most
    one marker per display pixel (occupancy), with light blending so phase/cells stay
    readable underneath. Returns the number of unique display pixels painted.

    ``plus`` defaults to auto: tiny + markers when sparse, single pixels when dense.
    """
    hs, ws = int(target_rgb.shape[0]), int(target_rgb.shape[1])
    ys, xs = _scaled_peak_coords(
        peaks,
        mask_shape=mask_shape,
        target_shape=(hs, ws),
        cap=None,
    )
    if ys.size == 0:
        return 0
    # Collapse peaks that map to the same thumbnail pixel (density, not carpet).
    packed = ys.astype(np.int64) * int(ws) + xs.astype(np.int64)
    _, uniq_idx = np.unique(packed, return_index=True)
    ys = ys[uniq_idx]
    xs = xs[uniq_idx]
    coverage = float(ys.size) / float(max(1, hs * ws))
    use_plus = bool(plus) if plus is not None else coverage < 0.06
    a = float(np.clip(alpha if use_plus else min(alpha, 0.42), 0.05, 1.0))
    inv = 1.0 - a
    cr, cg, cb = (float(color[0]), float(color[1]), float(color[2]))

    def _blend(y_idx: np.ndarray, x_idx: np.ndarray) -> None:
        if y_idx.size == 0:
            return
        valid = (y_idx >= 0) & (y_idx < hs) & (x_idx >= 0) & (x_idx < ws)
        if not np.any(valid):
            return
        yy = y_idx[valid]
        xx = x_idx[valid]
        bg = target_rgb[yy, xx].astype(np.float32)
        target_rgb[yy, xx, 0] = np.clip(bg[:, 0] * inv + cr * a, 0, 255).astype(np.uint8)
        target_rgb[yy, xx, 1] = np.clip(bg[:, 1] * inv + cg * a, 0, 255).astype(np.uint8)
        target_rgb[yy, xx, 2] = np.clip(bg[:, 2] * inv + cb * a, 0, 255).astype(np.uint8)

    _blend(ys, xs)
    if use_plus:
        # Tiny + keeps markers visible on ~200px tiles without the ring carpet.
        _blend(ys - 1, xs)
        _blend(ys + 1, xs)
        _blend(ys, xs - 1)
        _blend(ys, xs + 1)
    return int(ys.size)


def draw_foci_blob_contours_on_rgb(
    target_rgb: np.ndarray,
    mask_2d: np.ndarray,
    peaks: Sequence[dict[str, Any]],
    plane: np.ndarray,
    settings: dict[str, Any],
    contour_rgb: tuple[int, int, int],
    *,
    cap: Optional[int] = None,
    ownership_peaks: Optional[Sequence[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Draw exclusive focus boundaries (no shared pixels between peaks).

    When ``ownership_peaks`` is provided, exclusive assignment uses that full set
    (e.g. pass+fail together) while only ``peaks`` are outlined in ``contour_rgb``.
    """
    from bactomate.gui.foci_filter_pipeline import (
        foci_blob_max_radius,
        foci_blob_threshold_frac,
    )
    from bactomate.modules.foci_utils import exclusive_focus_blob_metrics

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

    paint_keys: set[tuple[int, int]] = set()
    for p in items:
        try:
            paint_keys.add((int(float(p.get("y", 0) or 0)), int(float(p.get("x", 0) or 0))))
        except Exception:
            continue
    if not paint_keys:
        return list(items)

    ownership = list(ownership_peaks) if ownership_peaks is not None else list(items)
    if ownership_peaks is not None and cap is not None and int(cap) > 0:
        ownership = list(ownership_peaks)

    measured = exclusive_focus_blob_metrics(
        work,
        ownership,
        threshold_frac=frac,
        max_radius=max_r,
        labeled_mask=label,
    )
    if not measured:
        return list(items)

    # Per-blob hollow rings. find_boundaries on a packed multi-label map paints almost
    # every pixel in tight clusters (looks "filled"); morph-outline stays hollow.
    edge_disp = np.zeros((h1, w1), dtype=bool)
    painted_keys: set[tuple[int, int]] = set()
    for peak, blob, _size, _mean in measured:
        try:
            key = (int(peak.get("y", 0)), int(peak.get("x", 0)))
        except Exception:
            continue
        if key not in paint_keys:
            continue
        if blob is None or not np.any(blob):
            missed.append(peak)
            continue
        area = int(np.sum(blob))
        if area >= 6:
            ring = _exclusive_blob_outline_mask(blob)
        else:
            ring = _peak_hollow_ring_mask(blob.shape[:2], key[0], key[1], radius=2)
        if not np.any(ring):
            missed.append(peak)
            continue
        ys, xs = np.nonzero(ring)
        y2 = np.clip(np.round(ys.astype(np.float64) * sy).astype(np.int32), 0, h1 - 1)
        x2 = np.clip(np.round(xs.astype(np.float64) * sx).astype(np.int32), 0, w1 - 1)
        edge_disp[y2, x2] = True
        painted_keys.add(key)

    if not painted_keys:
        return list(items)

    if np.any(edge_disp):
        target_rgb[edge_disp, 0] = cr
        target_rgb[edge_disp, 1] = cg
        target_rgb[edge_disp, 2] = cb

    for p in items:
        try:
            key = (int(float(p.get("y", 0) or 0)), int(float(p.get("x", 0) or 0)))
        except Exception:
            missed.append(p)
            continue
        if key not in painted_keys:
            missed.append(p)
    return missed


def draw_exclusive_foci_preview_on_rgb(
    target_rgb: np.ndarray,
    mask_2d: np.ndarray,
    peaks: Sequence[dict[str, Any]],
    plane: np.ndarray,
    *,
    settings: Optional[dict[str, Any]] = None,
    threshold_frac: float = 0.8,
    max_radius: int = 5,
    outline_rgb: tuple[int, int, int] = FOCI_BLOB_OUTLINE_RGB,
    peak_rgb: tuple[int, int, int] = FOCI_PEAK_PIXEL_RGB,
    cap: Optional[int] = None,
    ownership_peaks: Optional[Sequence[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Uniform foci look: light exclusive outline, then red Spotiflow peak pixels on top."""
    fs = dict(settings or {})
    fs.setdefault("foci_blob_threshold_frac", float(threshold_frac))
    fs.setdefault("foci_blob_max_radius", int(max_radius))
    missed = draw_foci_blob_contours_on_rgb(
        target_rgb,
        mask_2d,
        peaks,
        plane,
        fs,
        outline_rgb,
        cap=cap,
        ownership_peaks=ownership_peaks,
    )
    draw_peak_pixels_on_rgb(
        target_rgb,
        peaks,
        peak_rgb,
        mask_shape=tuple(np.asarray(mask_2d).shape[:2]),
        cap=cap,
    )
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
    """Fill exclusive focus blobs, then extract outer edges (preview hot path)."""
    from skimage.segmentation import find_boundaries

    from bactomate.modules.foci_utils import exclusive_focus_blob_masks

    if not peaks:
        return
    items = list(peaks)
    if cap is not None and int(cap) > 0:
        items = items[: int(cap)]
    label = np.asarray(mask, dtype=np.int32)
    work = np.asarray(plane, dtype=np.float32)
    h, w = int(target_rgb.shape[0]), int(target_rgb.shape[1])
    frac = float(np.clip(threshold_frac, 0.05, 1.0))
    max_r = max(1, int(max_radius))
    masks = exclusive_focus_blob_masks(
        work,
        items,
        threshold_frac=frac,
        max_radius=max_r,
        labeled_mask=label,
    )
    if not masks:
        return
    labeled = np.zeros((h, w), dtype=np.int32)
    for i, blob in enumerate(masks):
        if blob is not None and np.any(blob):
            labeled[np.asarray(blob, dtype=bool)] = i + 1
    combined = labeled > 0
    if not np.any(combined):
        return
    target_rgb[combined] = draw_color
    try:
        # Keep filled interior; outer edge only (inner would erase tiny filled blobs).
        boundaries = find_boundaries(labeled, mode="outer")
        if np.any(boundaries):
            target_rgb[boundaries, 0] = 0
            target_rgb[boundaries, 1] = 0
            target_rgb[boundaries, 2] = 0
    except Exception:
        pass
