"""
Foci detection and assignment for fluorescence images.

Uses skimage: enhance image, peak_local_max, optional size filter,
then assign peaks to cells via labeled mask. Used by Analyze Masks when "Run foci analysis" is on.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import gaussian, unsharp_mask
from skimage.morphology import disk, white_tophat
from skimage.segmentation import expand_labels

from bactomate.modules.image_stack_utils import coerce_plane_to_2d

# Default preprocessing (same idea as external foci_tool)
DEFAULT_SIGMA_SMOOTH = 1.0
DEFAULT_BG_RADIUS = 5
DEFAULT_SHARPEN_RADIUS = 1.0
DEFAULT_SHARPEN_AMOUNT = 1.5

_SPOTIFLOW_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_SPOTIFLOW_DETECT_CACHE: Dict[Any, List[Dict[str, Any]]] = {}
_SPOTIFLOW_DETECT_CACHE_MAX = 32


def clear_spotiflow_detect_cache() -> None:
    """Clear cached Spotiflow inference outputs. Intended for tests and memory cleanup."""
    _SPOTIFLOW_DETECT_CACHE.clear()


def _spotiflow_cache_key(model_name: str = "general", model_path: Optional[str] = None) -> Tuple[str, str]:
    mp = (model_path or "").strip()
    if mp:
        from pathlib import Path
        try:
            mp = str(Path(mp).expanduser().resolve())
        except Exception:
            pass
    return ((model_name or "general").strip() or "general", mp)


def clear_spotiflow_model_cache() -> None:
    """Clear cached Spotiflow model instances. Intended for tests and explicit memory cleanup."""
    _SPOTIFLOW_MODEL_CACHE.clear()
    clear_spotiflow_detect_cache()


def load_spotiflow_model(model_name: str = "general", model_path: Optional[str] = None) -> Any:
    """
    Load and cache a Spotiflow model by name or folder path.

    Spotiflow model construction is expensive; callers that process stacks should
    load once and pass the model to ``detect_peaks_spotiflow`` for each frame.
    """
    key = _spotiflow_cache_key(model_name=model_name, model_path=model_path)
    if key in _SPOTIFLOW_MODEL_CACHE:
        return _SPOTIFLOW_MODEL_CACHE[key]
    from spotiflow.model import Spotiflow

    _model_name, mp = key
    if mp:
        model = Spotiflow.from_folder(mp, map_location="auto")
    else:
        model = Spotiflow.from_pretrained(_model_name, map_location="auto")
    _SPOTIFLOW_MODEL_CACHE[key] = model
    return model


def enhance_foci_image(
    image: np.ndarray,
    sigma_smooth: float = DEFAULT_SIGMA_SMOOTH,
    bg_radius: int = DEFAULT_BG_RADIUS,
    sharpen_radius: float = DEFAULT_SHARPEN_RADIUS,
    sharpen_amount: float = DEFAULT_SHARPEN_AMOUNT,
) -> np.ndarray:
    """Reduce background and sharpen for foci/peak detection."""
    from skimage.util import img_as_float64
    img = np.asarray(image, dtype=np.float64)
    if img.size == 0:
        return img
    smoothed = gaussian(img, sigma=sigma_smooth, preserve_range=True)
    struc = disk(bg_radius)
    bg_sub = white_tophat(smoothed, struc)
    sharpened = unsharp_mask(
        bg_sub, radius=sharpen_radius, amount=sharpen_amount, preserve_range=True
    )
    return np.clip(sharpened, 0, None).astype(np.float64)


def focus_size_at_peak(
    image: np.ndarray,
    y: int,
    x: int,
    threshold_frac: float = 0.8,
    max_radius: int = 5,
) -> int:
    """Connected-component size (pixels) of the focus containing (y, x)."""
    size, _ = focus_size_and_mean_at_peak(image, y, x, threshold_frac, max_radius)
    return size


def focus_size_and_mean_at_peak(
    image: np.ndarray,
    y: int,
    x: int,
    threshold_frac: float = 0.8,
    max_radius: int = 5,
    labeled_mask: Optional[np.ndarray] = None,
) -> Tuple[int, float]:
    """
    Compute size (pixels) and mean intensity of the connected focus blob containing (y, x).
    Threshold in a window at peak_intensity * threshold_frac. Returns (size_px, mean_intensity).

    Area is a pixel count only; this module does not use pixel_size. Physical area in µm²
    is area_px * (pixel_size**2), applied by the caller (GUI/Excel) when pixel size is known.

    If labeled_mask is provided (same shape as image), only pixels in the same cell as (y,x)
    are counted, so size never includes pixels from another cell or background. Peak in
    background (label 0) is not restricted.
    """
    from skimage.measure import label as sk_label
    h, w = image.shape[:2]
    if y < 0 or y >= h or x < 0 or x >= w:
        return 1, 0.0
    peak_val = float(image[y, x])
    if not np.isfinite(peak_val) or peak_val <= 0:
        return 1, peak_val
    thresh = peak_val * threshold_frac
    y0 = max(0, y - max_radius)
    y1 = min(h, y + max_radius + 1)
    x0 = max(0, x - max_radius)
    x1 = min(w, x + max_radius + 1)
    window = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    binary = (window >= thresh).astype(np.uint8)
    if labeled_mask is not None and labeled_mask.shape[:2] == image.shape[:2]:
        cell_val = int(labeled_mask[y, x])
        if cell_val != 0:
            mask_window = np.asarray(labeled_mask[y0:y1, x0:x1], dtype=np.int32)
            binary = binary & (mask_window == cell_val).astype(np.uint8)
    labeled, n = sk_label(binary, connectivity=2, return_num=True)
    if n == 0:
        return 1, peak_val
    ly, lx = y - y0, x - x0
    if ly < 0 or ly >= labeled.shape[0] or lx < 0 or lx >= labeled.shape[1]:
        return 1, peak_val
    lab = int(labeled[ly, lx])
    if lab == 0:
        return 1, peak_val
    mask = labeled == lab
    size = int(np.sum(mask))
    mean_val = float(np.mean(window[mask])) if size > 0 else peak_val
    return size, mean_val


def get_focus_blob_mask(
    image: np.ndarray,
    y: int,
    x: int,
    threshold_frac: float = 0.8,
    max_radius: int = 5,
    labeled_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return a boolean mask (same shape as image) that is True on the connected focus blob
    containing (y, x). Uses the same logic as focus_size_and_mean_at_peak (threshold at
    peak * threshold_frac, same cell if labeled_mask given). Use for drawing the actual
    focus shape in the preview.
    """
    from skimage.measure import label as sk_label
    h, w = image.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    if y < 0 or y >= h or x < 0 or x >= w:
        return out
    peak_val = float(image[y, x])
    if not np.isfinite(peak_val) or peak_val <= 0:
        out[y, x] = True
        return out
    thresh = peak_val * threshold_frac
    y0 = max(0, y - max_radius)
    y1 = min(h, y + max_radius + 1)
    x0 = max(0, x - max_radius)
    x1 = min(w, x + max_radius + 1)
    window = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    binary = (window >= thresh).astype(np.uint8)
    if labeled_mask is not None and labeled_mask.shape[:2] == image.shape[:2]:
        cell_val = int(labeled_mask[y, x])
        if cell_val != 0:
            mask_window = np.asarray(labeled_mask[y0:y1, x0:x1], dtype=np.int32)
            binary = binary & (mask_window == cell_val).astype(np.uint8)
    labeled, n = sk_label(binary, connectivity=2, return_num=True)
    if n == 0:
        out[y, x] = True
        return out
    ly, lx = y - y0, x - x0
    if ly < 0 or ly >= labeled.shape[0] or lx < 0 or lx >= labeled.shape[1]:
        out[y, x] = True
        return out
    lab = int(labeled[ly, lx])
    if lab == 0:
        out[y, x] = True
        return out
    out[y0:y1, x0:x1] = (labeled == lab)
    return out


def detect_peaks_skimage(
    image: np.ndarray,
    min_rel: Optional[float] = 2.0,
    min_distance: int = 2,
    min_size: int = 1,
    max_size: Optional[int] = None,
    min_peak_raw: Optional[float] = None,
    min_focus_mean: Optional[float] = None,
    use_preprocessing: bool = True,
    sigma_smooth: float = DEFAULT_SIGMA_SMOOTH,
    bg_radius: int = DEFAULT_BG_RADIUS,
    sharpen_radius: float = DEFAULT_SHARPEN_RADIUS,
    sharpen_amount: float = DEFAULT_SHARPEN_AMOUNT,
    size_threshold_frac: float = 0.8,
    size_max_radius: int = 5,
    labeled_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Detect foci in a 2D image. Returns list of dicts with keys:
    y, x, intensity, size_px, mean_intensity (and optionally rel_intensity).

    When labeled_mask is provided, detection is cell-first: we iterate over each cell and find
    local maxima only inside that cell (no global frame scan). For each candidate peak we build
    the focus blob (connected pixels >= 80% of peak, same cell), then apply min_peak_raw,
    min_size, max_size, min_focus_mean. All intensity/size use the raw image.

    When labeled_mask is None, we run peak_local_max on the full (preprocessed) image with
    threshold min_rel * img_median, then apply the same blob and filters on raw.
    """
    raw_img = np.asarray(image, dtype=np.float64)
    if raw_img.ndim != 2 or raw_img.size == 0:
        return []
    raw_median = float(np.median(raw_img))
    if raw_median <= 0:
        raw_median = 1e-10
    h, w = raw_img.shape

    if labeled_mask is not None and labeled_mask.shape[:2] == (h, w):
        # Cell-first: elevated regions (connected components above cell_mean × min_rel), then 80% blob + filters
        peaks = _detect_peaks_skimage_per_cell(
            raw_img, labeled_mask, raw_median,
            min_rel, min_distance, min_size, max_size,
            min_peak_raw, min_focus_mean,
            size_threshold_frac, size_max_radius,
        )
        return peaks

    # Legacy path: full-image peak_local_max when no mask
    img = raw_img.copy()
    if use_preprocessing:
        img = enhance_foci_image(
            img, sigma_smooth, bg_radius, sharpen_radius, sharpen_amount
        )
    img_median = float(np.median(img))
    if img_median <= 0:
        img_median = 1e-10
    min_abs = (img_median * min_rel) if (min_rel is not None and min_rel > 0) else (img_median * 0.5)
    coords = peak_local_max(img, min_distance=min_distance, threshold_abs=min_abs)
    if coords.size == 0:
        return []
    raw_intensities = raw_img[coords[:, 0], coords[:, 1]]
    min_raw_abs = (raw_median * min_rel) if (min_rel is not None and min_rel > 0) else None
    peaks = []
    for k in range(len(coords)):
        yi, xi = int(coords[k, 0]), int(coords[k, 1])
        raw_int = float(raw_intensities[k])
        if min_raw_abs is not None and raw_int < min_raw_abs:
            continue
        if min_peak_raw is not None and min_peak_raw > 0 and raw_int < min_peak_raw:
            continue
        size_px, mean_int = focus_size_and_mean_at_peak(
            raw_img, yi, xi,
            threshold_frac=size_threshold_frac,
            max_radius=size_max_radius,
            labeled_mask=labeled_mask,
        )
        if min_size is not None and size_px < min_size:
            continue
        if max_size is not None and max_size > 0 and size_px > max_size:
            continue
        if min_focus_mean is not None and min_focus_mean > 0 and mean_int < min_focus_mean:
            continue
        rel = raw_int / raw_median
        peaks.append({
            "y": yi, "x": xi, "intensity": raw_int, "mean_intensity": mean_int,
            "rel_intensity": rel, "size_px": size_px,
        })
    return peaks


def _detect_peaks_skimage_per_cell(
    raw_img: np.ndarray,
    labeled_mask: np.ndarray,
    raw_median: float,
    min_rel: Optional[float],
    min_distance: int,
    min_size: int,
    max_size: Optional[int],
    min_peak_raw: Optional[float],
    min_focus_mean: Optional[float],
    size_threshold_frac: float,
    size_max_radius: int,
) -> List[Dict[str, Any]]:
    """
    Cell-first: find elevated regions (connected components with intensity >= cell_mean × min_rel),
    use each region's max pixel as seed, build 80% blob, apply filters. Merge candidates within min_distance.
    """
    from skimage.measure import label as sk_label
    h, w = raw_img.shape
    raw_median = raw_median if raw_median > 0 else 1e-10
    factor = float(min_rel) if min_rel is not None and min_rel > 0 else 1.0
    factor = max(0.5, factor)
    peaks: List[Dict[str, Any]] = []
    for lab in np.unique(labeled_mask):
        if lab == 0:
            continue
        cell_mask = (labeled_mask == lab)
        cell_mean = float(np.mean(raw_img[cell_mask]))
        if not np.isfinite(cell_mean) or cell_mean <= 0:
            continue
        thresh = cell_mean * factor
        elevated = (raw_img >= thresh) & cell_mask
        if not np.any(elevated):
            continue
        labeled = sk_label(elevated.astype(np.uint8), connectivity=2)
        for comp_id in np.unique(labeled):
            if comp_id == 0:
                continue
            comp_mask = (labeled == comp_id)
            n_px = int(np.sum(comp_mask))
            if n_px < 2:
                continue
            ys, xs = np.where(comp_mask)
            vals = raw_img[ys, xs]
            idx_max = int(np.argmax(vals))
            yi, xi = int(ys[idx_max]), int(xs[idx_max])
            raw_int = float(raw_img[yi, xi])
            if min_peak_raw is not None and min_peak_raw > 0 and raw_int < min_peak_raw:
                continue
            size_px, mean_int = focus_size_and_mean_at_peak(
                raw_img, yi, xi,
                threshold_frac=size_threshold_frac,
                max_radius=size_max_radius,
                labeled_mask=labeled_mask,
            )
            if min_size is not None and size_px < min_size:
                continue
            if max_size is not None and max_size > 0 and size_px > max_size:
                continue
            if min_focus_mean is not None and min_focus_mean > 0 and mean_int < min_focus_mean:
                continue
            rel = raw_int / raw_median
            peaks.append({
                "y": yi, "x": xi, "intensity": raw_int, "mean_intensity": mean_int,
                "rel_intensity": rel, "size_px": size_px,
            })
    if min_distance > 0 and len(peaks) > 1:
        peaks = _merge_nearby_peaks(peaks, merge_distance_px=float(min_distance))
    return peaks


def resolve_peak_cell_id(
    labeled_mask: np.ndarray,
    y: int,
    x: int,
    *,
    touch_radius: int = 2,
) -> int:
    """Mask label at the peak, or nearest cell within *touch_radius* when on background."""
    label = np.asarray(labeled_mask, dtype=np.int32)
    if label.ndim != 2:
        return 0
    h, w = int(label.shape[0]), int(label.shape[1])
    if not (0 <= int(y) < h and 0 <= int(x) < w):
        return 0
    at_peak = int(label[int(y), int(x)])
    if at_peak > 0:
        return at_peak
    r = max(1, int(touch_radius))
    best_id = 0
    best_d2: Optional[int] = None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            yy = int(y) + dy
            xx = int(x) + dx
            if not (0 <= yy < h and 0 <= xx < w):
                continue
            lid = int(label[yy, xx])
            if lid <= 0:
                continue
            d2 = dy * dy + dx * dx
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_id = lid
    return int(best_id)


def assign_peaks_to_cells(
    peaks: List[Dict[str, Any]],
    labeled_mask: np.ndarray,
    dilation_px: int = 1,
    frame_index: int = 0,
    *,
    touch_radius: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], List[int]]]:
    """
    Assign each peak to a cell (label) using the mask. Optionally dilate labels.

    Peaks on background but within *touch_radius* of a cell (edge foci) are assigned
    to the nearest cell label, matching Filtering preview / ``resolve_foci_cell_id_at_peak``.

    Returns:
        foci_rows: list of dicts with keys foci_id, cell_id, frame, y, x, intensity, size_px, mean_intensity, rel_intensity, in_cell.
        cell_foci_ids: (cell_id, frame) -> list of foci_id (for building foci_ids column).
    """
    if not peaks:
        return [], {}
    h, w = labeled_mask.shape[:2]
    label_plane = np.asarray(labeled_mask, dtype=np.int32)
    if dilation_px > 0:
        mask = expand_labels(label_plane, distance=dilation_px)
    else:
        mask = label_plane
    foci_rows: List[Dict[str, Any]] = []
    cell_foci_ids: Dict[Tuple[int, int], List[int]] = {}
    foci_id = 1
    for p in peaks:
        yi = int(p["y"])
        xi = int(p["x"])
        if yi < 0 or yi >= h or xi < 0 or xi >= w:
            foci_rows.append({
                "foci_id": foci_id,
                "cell_id": None,
                "frame": frame_index,
                "y": yi,
                "x": xi,
                "intensity": p.get("intensity", np.nan),
                "size_px": p.get("size_px", 1),
                "mean_intensity": p.get("mean_intensity", np.nan),
                "rel_intensity": p.get("rel_intensity", np.nan),
                "in_cell": False,
            })
            foci_id += 1
            continue
        cell_id = int(mask[yi, xi])
        if cell_id <= 0:
            cell_id = resolve_peak_cell_id(label_plane, yi, xi, touch_radius=touch_radius)
        in_cell = cell_id > 0
        foci_rows.append({
            "foci_id": foci_id,
            "cell_id": cell_id if in_cell else None,
            "frame": frame_index,
            "y": yi,
            "x": xi,
            "intensity": p.get("intensity", np.nan),
            "size_px": p.get("size_px", 1),
            "mean_intensity": p.get("mean_intensity", np.nan),
            "rel_intensity": p.get("rel_intensity", np.nan),
            "in_cell": in_cell,
        })
        if in_cell:
            key = (cell_id, frame_index)
            cell_foci_ids.setdefault(key, []).append(foci_id)
        foci_id += 1
    return foci_rows, cell_foci_ids


def _cell_cov_map(image: np.ndarray, labeled_mask: np.ndarray) -> Dict[int, float]:
    """Per-label coefficient of variation (std/mean) of pixel intensity inside each cell. Skips background (0)."""
    out: Dict[int, float] = {}
    raw = np.asarray(image, dtype=np.float64)
    if raw.ndim != 2 or labeled_mask is None or raw.shape != labeled_mask.shape:
        return out
    for lab in np.unique(labeled_mask):
        if lab == 0:
            continue
        pixels = raw[labeled_mask == lab]
        if pixels.size < 3:
            continue
        mu = float(np.mean(pixels))
        if mu <= 0 or not np.isfinite(mu):
            continue
        std = float(np.std(pixels))
        if np.isfinite(std) and std >= 0:
            out[int(lab)] = std / mu
    return out


def _cell_area_map(labeled_mask: np.ndarray) -> Dict[int, int]:
    """Per-label pixel count (area in px). Skips background (0)."""
    out: Dict[int, int] = {}
    if labeled_mask is None or labeled_mask.size == 0:
        return out
    for lab in np.unique(labeled_mask):
        if lab == 0:
            continue
        out[int(lab)] = int(np.sum(labeled_mask == lab))
    return out


def _cell_mean_map(image: np.ndarray, labeled_mask: np.ndarray) -> Dict[int, float]:
    """Per-label mean pixel intensity inside each cell. Skips background (0)."""
    out: Dict[int, float] = {}
    raw = np.asarray(image, dtype=np.float64)
    if raw.ndim != 2 or labeled_mask is None or raw.shape != labeled_mask.shape:
        return out
    for lab in np.unique(labeled_mask):
        if lab == 0:
            continue
        pixels = raw[labeled_mask == lab]
        if pixels.size < 2:
            continue
        mu = float(np.mean(pixels))
        if np.isfinite(mu) and mu > 0:
            out[int(lab)] = mu
    return out


def _cell_percentile_map(
    image: np.ndarray,
    labeled_mask: np.ndarray,
    percentile: float = 90.0,
) -> Dict[int, float]:
    """Per-label percentile of pixel intensity inside each cell (e.g. 90 = 90th percentile). No fixed multipliers."""
    out: Dict[int, float] = {}
    raw = np.asarray(image, dtype=np.float64)
    if raw.ndim != 2 or labeled_mask is None or raw.shape != labeled_mask.shape:
        return out
    for lab in np.unique(labeled_mask):
        if lab == 0:
            continue
        pixels = raw[labeled_mask == lab]
        if pixels.size < 5:
            continue
        pct = float(np.percentile(pixels, percentile))
        if np.isfinite(pct):
            out[int(lab)] = pct
    return out


def _cell_foci_thresholds(
    image: np.ndarray,
    labeled_mask: np.ndarray,
    cov_map: Dict[int, float],
    median_cov: float,
    percentile_uniform: float,
    percentile_heterogeneous: float,
    percentile_large_blob: float,
) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """
    Per-cell intensity thresholds for foci (no fixed multipliers). Returns label -> (thresh_standard, thresh_large).
    Standard threshold uses percentile_uniform (low CoV) or percentile_heterogeneous (high CoV). Large-blob threshold
    uses percentile_large_blob for diffuse/big foci. All from that cell's own pixel distribution.
    """
    raw = np.asarray(image, dtype=np.float64)
    if raw.ndim != 2 or labeled_mask is None or raw.shape != labeled_mask.shape:
        return {}
    out: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    for lab in np.unique(labeled_mask):
        if lab == 0:
            continue
        pixels = raw[labeled_mask == lab]
        if pixels.size < 5:
            continue
        cell_cov = cov_map.get(lab) if cov_map else None
        p = percentile_heterogeneous if (median_cov > 0 and cell_cov is not None and cell_cov >= median_cov) else percentile_uniform
        thresh_std = float(np.percentile(pixels, p))
        thresh_large = float(np.percentile(pixels, percentile_large_blob))
        if np.isfinite(thresh_std) and np.isfinite(thresh_large):
            out[int(lab)] = (thresh_std, thresh_large)
    return out


def _region_grow_above_threshold(
    image: np.ndarray,
    cell_mask: np.ndarray,
    seed_y: int,
    seed_x: int,
    threshold: float,
) -> np.ndarray:
    """
    Grow a connected region from (seed_y, seed_x) by adding 8-connected neighbors
    where image >= threshold, restricted to cell_mask. Returns boolean mask (same shape as image).
    """
    h, w = image.shape[:2]
    if seed_y < 0 or seed_y >= h or seed_x < 0 or seed_x >= w:
        return np.zeros_like(cell_mask, dtype=bool)
    if not cell_mask[seed_y, seed_x] or float(image[seed_y, seed_x]) < threshold:
        return np.zeros_like(cell_mask, dtype=bool)
    out = np.zeros_like(cell_mask, dtype=bool)
    out[seed_y, seed_x] = True
    img_flat = np.asarray(image, dtype=np.float64).ravel()
    cell_flat = np.asarray(cell_mask, dtype=bool).ravel()
    # 8-connected neighbors
    dy = (-1, -1, -1, 0, 0, 1, 1, 1)
    dx = (-1, 0, 1, -1, 1, -1, 0, 1)
    frontier = [(seed_y, seed_x)]
    while frontier:
        r, c = frontier.pop()
        for i in range(8):
            nr, nc = r + dy[i], c + dx[i]
            if 0 <= nr < h and 0 <= nc < w and cell_mask[nr, nc] and not out[nr, nc]:
                if float(image[nr, nc]) >= threshold:
                    out[nr, nc] = True
                    frontier.append((nr, nc))
    return out


def _merge_nearby_peaks(peaks: List[Dict[str, Any]], merge_distance_px: float = 11.0) -> List[Dict[str, Any]]:
    """Merge peaks within merge_distance_px, keeping the one with highest intensity. Use distance > 2*blob max_radius (e.g. 11 for max_radius=5) so focus regions do not overlap."""
    if len(peaks) <= 1 or merge_distance_px <= 0:
        return peaks
    used = [False] * len(peaks)
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(peaks):
        if used[i]:
            continue
        yi, xi = float(p.get("y", 0)), float(p.get("x", 0))
        intensity_i = float(p.get("intensity", 0) or 0)
        best = p
        best_int = intensity_i
        for j in range(i + 1, len(peaks)):
            if used[j]:
                continue
            q = peaks[j]
            yj, xj = float(q.get("y", 0)), float(q.get("x", 0))
            dist = np.sqrt((yi - yj) ** 2 + (xi - xj) ** 2)
            if dist <= merge_distance_px:
                used[j] = True
                q_int = float(q.get("intensity", 0) or 0)
                if q_int > best_int:
                    best_int = q_int
                    best = q
        out.append(best)
    return out


def detect_peaks_cell_first_region_grow(
    image: np.ndarray,
    labeled_mask: np.ndarray,
    grow_threshold_factor: float = 1.08,
    min_focus_mean_ratio: float = 1.25,
    max_focus_frac_of_cell: float = 0.4,
    merge_nearby_px: float = 11.0,
    foci_scope: str = "intracellular",
) -> List[Dict[str, Any]]:
    """
    Cell-first, no global preprocessing: for each cell get mean FI, find max pixel, grow region
    (neighbors with intensity >= cell_mean * grow_threshold_factor) so the boundary is "clearly
    elevated" and we don't absorb most of the cell. Keep region only if focus mean is clearly
    above cell mean and region is not most of the cell. Returns list of peak dicts.
    foci_scope: "intracellular" (default), "extracellular", or "both"; only intracellular is implemented.
    """
    raw = np.asarray(image, dtype=np.float64)
    if raw.ndim != 2 or raw.size == 0 or labeled_mask is None or raw.shape != labeled_mask.shape:
        return []
    if foci_scope not in ("intracellular", "extracellular", "both"):
        foci_scope = "intracellular"
    if foci_scope != "intracellular":
        # Placeholder for future extracellular/both logic
        pass
    h, w = raw.shape
    frame_median = float(np.median(raw))
    if frame_median <= 0:
        frame_median = 1e-10
    all_peaks: List[Dict[str, Any]] = []
    for lab in np.unique(labeled_mask):
        if lab == 0:
            continue
        cell_mask = (labeled_mask == lab)
        cell_area_px = int(np.sum(cell_mask))
        if cell_area_px < 3:
            continue
        cell_mean = float(np.mean(raw[cell_mask]))
        if not np.isfinite(cell_mean) or cell_mean <= 0:
            continue
        # Grow only into pixels >= cell_mean * grow_threshold_factor so we don't over-grow into cytoplasm
        grow_thresh = cell_mean * max(1.0, float(grow_threshold_factor))
        exclude = np.zeros_like(cell_mask, dtype=bool)
        while True:
            candidate = np.where(cell_mask & ~exclude)
            if not candidate[0].size:
                break
            vals = raw[candidate[0], candidate[1]]
            idx_max = np.argmax(vals)
            sy, sx = int(candidate[0][idx_max]), int(candidate[1][idx_max])
            peak_val = float(vals[idx_max])
            if peak_val < grow_thresh:
                break
            focus_mask = _region_grow_above_threshold(raw, cell_mask, sy, sx, grow_thresh)
            if not np.any(focus_mask):
                exclude[sy, sx] = True
                continue
            focus_size = int(np.sum(focus_mask))
            focus_mean = float(np.mean(raw[focus_mask]))
            focus_max = float(np.max(raw[focus_mask]))
            if focus_mean < cell_mean * min_focus_mean_ratio:
                exclude |= focus_mask
                continue
            if max_focus_frac_of_cell < 1.0 and focus_size > max_focus_frac_of_cell * cell_area_px:
                exclude |= focus_mask
                continue
            rel = focus_mean / frame_median
            all_peaks.append({
                "y": sy,
                "x": sx,
                "intensity": focus_max,
                "size_px": focus_size,
                "mean_intensity": focus_mean,
                "rel_intensity": rel,
            })
            exclude |= focus_mask
    if merge_nearby_px > 0 and len(all_peaks) > 1:
        all_peaks = _merge_nearby_peaks(all_peaks, merge_distance_px=merge_nearby_px)
    return all_peaks


def detect_peaks_skimage_auto(
    image: np.ndarray,
    min_distance: int = 2,
    labeled_mask: Optional[np.ndarray] = None,
    use_preprocessing: bool = True,
    min_rel_vs_cell: float = 1.0,
    min_rel_uniform_cell: float = 1.45,
    max_focus_frac_of_cell: float = 0.4,
    large_blob_min_px: int = 12,
    merge_nearby_px: float = 11.0,
    percentile_uniform: float = 92.0,
    percentile_heterogeneous: float = 85.0,
    percentile_large_blob: float = 80.0,
    z_score_min: float = 1.4,
) -> List[Dict[str, Any]]:
    """
    Fully per-cell, data-driven foci detection. No fixed multipliers, no frame-level intensity cut.
    (1) Get candidates (low bar: 0.5*img_median for peak_local_max only).
    (2) Per-cell gate: keep peak only if (peak >= cell_mean * min_rel_vs_cell);
        in uniform cells also require peak >= cell_mean * min_rel_uniform_cell (stricter bump to avoid homogeneous cytoplasm as "foci");
        reject if focus blob area > max_focus_frac_of_cell of cell area (foci must be focal, not most of the cell);
        and ((peak >= cell percentile) OR (peak >= cell_mean + z_score_min*cell_std)). Drop peaks outside cells.
    (3) Uniform cells only: drop peaks in the bottom 25% by score within that cell.
    (4) Merge nearby peaks so foci do not overlap (merge_nearby_px > 2*blob radius).
    Only global step: initial peak detection threshold (0.5*img_median); all decisions are per-cell.
    """
    # Get many candidates with low bar (no frame-median filter; per-cell gate does the work)
    candidates = detect_peaks_skimage(
        image,
        min_rel=None,
        min_distance=min_distance,
        min_size=1,
        max_size=None,
        min_peak_raw=None,
        min_focus_mean=None,
        use_preprocessing=use_preprocessing,
        labeled_mask=labeled_mask,
    )
    if not candidates:
        return []

    # Per-cell gate: min_rel_vs_cell, in uniform cells min_rel_uniform_cell, max focus fraction, then percentile or z-score
    if labeled_mask is not None and labeled_mask.size > 0:
        h, w = labeled_mask.shape
        cov_map = _cell_cov_map(image, labeled_mask)
        cell_mean_map = _cell_mean_map(image, labeled_mask)
        cell_area_map = _cell_area_map(labeled_mask)
        median_cov = float(np.median(list(cov_map.values()))) if cov_map else 0.0
        thresh_map = _cell_foci_thresholds(
            image, labeled_mask, cov_map, median_cov,
            percentile_uniform, percentile_heterogeneous, percentile_large_blob,
        )
        filtered = []
        for p in candidates:
            yi, xi = int(p.get("y", 0)), int(p.get("x", 0))
            if yi < 0 or yi >= h or xi < 0 or xi >= w:
                continue
            lab = int(labeled_mask[yi, xi])
            if lab == 0 or lab not in thresh_map:
                continue
            thresh_std, thresh_large = thresh_map[lab]
            peak_int = float(p.get("intensity", 0) or 0)
            size_px = int(p.get("size_px", 0) or 0)
            cell_mean = cell_mean_map.get(lab) if cell_mean_map else None
            cell_area_px = cell_area_map.get(lab) if cell_area_map else None
            cell_cov_val = cov_map.get(lab) if cov_map else None
            is_uniform = cell_cov_val is not None and median_cov > 0 and cell_cov_val < median_cov
            # Foci must be focal: reject if blob is most of the cell
            if cell_area_px is not None and cell_area_px > 0 and max_focus_frac_of_cell < 1.0:
                if size_px > max_focus_frac_of_cell * cell_area_px:
                    continue
            # Per-cell "not below background": peak >= min_rel_vs_cell * cell_mean
            if cell_mean is not None and cell_mean > 0 and peak_int < cell_mean * min_rel_vs_cell:
                continue
            # In uniform cells require a clear bump (e.g. 1.45× mean) so homogeneous cytoplasm is not called a focus
            if is_uniform and cell_mean is not None and cell_mean > 0 and peak_int < cell_mean * min_rel_uniform_cell:
                continue
            # Use more lenient (lower) threshold for large/diffuse blobs
            if size_px >= large_blob_min_px and thresh_large is not None:
                thresh = min(thresh_std, thresh_large) if thresh_std is not None else thresh_large
            else:
                thresh = thresh_std
            if thresh is None:
                continue
            passes_pct = peak_int >= thresh
            cell_std = (cell_cov_val * cell_mean) if (cell_cov_val is not None and cell_mean is not None and cell_mean > 0) else None
            passes_z = (
                z_score_min > 0
                and cell_std is not None
                and cell_std > 0
                and peak_int >= cell_mean + z_score_min * cell_std
            )
            if passes_pct or passes_z:
                filtered.append(p)
        candidates = filtered
        if len(candidates) <= 1:
            return _merge_nearby_peaks(candidates, merge_nearby_px) if merge_nearby_px > 0 and len(candidates) > 1 else candidates

    # Score per candidate (rel * log1p(size)); used only for per-cell pruning in uniform cells
    rel = np.array([float(p.get("rel_intensity", 0) or 0) for p in candidates], dtype=np.float64)
    sz = np.array([float(p.get("size_px", 1) or 1) for p in candidates], dtype=np.float64)
    rel = np.clip(rel, 1e-6, None)
    sz = np.clip(sz, 0.5, None)
    score = rel * np.log1p(sz)
    # Fully per-cell: heterogeneous cells keep all; uniform cells drop bottom 25% by score within that cell only
    if labeled_mask is None or labeled_mask.size == 0:
        result = candidates
    else:
        h, w = labeled_mask.shape
        cov_map = _cell_cov_map(image, labeled_mask)
        if not cov_map:
            result = candidates
        else:
            cov_values = np.array(list(cov_map.values()), dtype=np.float64)
            median_cov = float(np.median(cov_values))
            # Group candidate indices by cell label
            cell_to_indices: Dict[int, List[int]] = {}
            for i, p in enumerate(candidates):
                yi, xi = int(p.get("y", 0)), int(p.get("x", 0))
                if 0 <= yi < h and 0 <= xi < w:
                    lab = int(labeled_mask[yi, xi])
                    if lab != 0:
                        cell_to_indices.setdefault(lab, []).append(i)
            keep = np.ones(len(candidates), dtype=bool)
            for lab, indices in cell_to_indices.items():
                if not indices:
                    continue
                cell_cov = cov_map.get(lab)
                if cell_cov is None:
                    continue
                if cell_cov >= median_cov:
                    continue  # heterogeneous: keep all (no pruning)
                # uniform cell: drop bottom 25% by score within this cell
                cell_scores = score[indices]
                if len(cell_scores) < 2:
                    continue
                score_25_cell = float(np.percentile(cell_scores, 25))
                for idx in indices:
                    if score[idx] < score_25_cell:
                        keep[idx] = False
            result = [p for i, p in enumerate(candidates) if keep[i]]
    if merge_nearby_px > 0 and len(result) > 1:
        result = _merge_nearby_peaks(result, merge_distance_px=merge_nearby_px)
    return result


def probability_from_spotiflow_details(
    details: Any,
    spot_index: int,
    *,
    spots_row: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Per-spot confidence from Spotiflow ``predict()`` (``details.prob``, not ``spots`` coords).

    Spotiflow returns ``(points, details)`` where ``points`` is ``(n, 2)`` (y, x) and
    ``details.prob`` holds one probability per spot (see Spotiflow API / ``SimpleNamespace``).
    """
    if details is not None:
        prob_arr = getattr(details, "prob", None)
        if prob_arr is not None:
            try:
                flat = np.asarray(prob_arr, dtype=np.float64).ravel()
                if 0 <= int(spot_index) < int(flat.size):
                    val = float(flat[int(spot_index)])
                    if np.isfinite(val):
                        return val
            except Exception:
                pass
    if spots_row is not None:
        row = np.asarray(spots_row, dtype=np.float64).ravel()
        if row.size >= 3:
            try:
                val = float(row[2])
                if np.isfinite(val):
                    return val
            except Exception:
                pass
    return None


def spotiflow_inference_cache_key(
    image: np.ndarray,
    *,
    source_path: Optional[str] = None,
    frame_idx: Optional[int] = None,
    source_mtime: Optional[float] = None,
    **detect_kwargs: Any,
) -> tuple:
    """Stable cache key for Spotiflow inference (file/frame/settings/mtime when available)."""
    arr = np.asarray(coerce_plane_to_2d(image))
    h, w = int(arr.shape[0]), int(arr.shape[1])
    samples: tuple[float, ...] = ()
    if arr.size:
        pts = (
            (0, 0),
            (0, max(0, w - 1)),
            (max(0, h - 1), 0),
            (max(0, h - 1), max(0, w - 1)),
            (h // 2, w // 2),
        )
        samples = tuple(float(arr[y, x]) for y, x in pts)
    settings_items = []
    for key in sorted(detect_kwargs):
        if key in ("spotiflow_model", "verbose", "raise_on_error", "detect_cache", "detect_cache_key"):
            continue
        settings_items.append((key, repr(detect_kwargs[key])))
    path_token = ""
    if source_path:
        try:
            from pathlib import Path

            path_token = str(Path(source_path).expanduser().resolve())
        except Exception:
            path_token = str(source_path)
    return (
        path_token,
        int(frame_idx) if frame_idx is not None else None,
        float(source_mtime) if source_mtime is not None else None,
        h,
        w,
        samples,
        tuple(settings_items),
    )


def detect_peaks_spotiflow(
    image: np.ndarray,
    *,
    model_name: str = "general",
    model_path: Optional[str] = None,
    spotiflow_model: Any = None,
    use_optimized_threshold: bool = True,
    prob_thresh: Optional[float] = None,
    min_distance: int = 1,
    subpix: bool = True,
    normalize: bool = True,
    perc_low: float = 1.0,
    perc_high: float = 99.8,
    bg_subtract_radius: int = 0,
    blur_sigma: float = 0.0,
    auto_tiling: bool = True,
    n_tiles_y: int = 1,
    n_tiles_x: int = 1,
    verbose: bool = False,
    raise_on_error: bool = False,
    detect_cache_key: Optional[tuple] = None,
    detect_cache: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Optional: detect spots using Spotiflow (deep learning). Returns list of dicts
    with y, x, intensity, probability, size_px, mean_intensity, rel_intensity, or None if spotiflow unavailable.

    Spotiflow ``predict()`` returns coordinates in ``spots`` (n×2) and per-spot scores in
    ``details.prob``; those scores are copied to each peak's ``probability`` for CSV export and filtering.

    When ``use_optimized_threshold`` is True, ``prob_thresh`` is not passed to ``predict`` (library uses
    tuned thresholds). Otherwise ``prob_thresh`` is used (0–1).

    ``blur_sigma`` and ``bg_subtract_radius`` are applied to a working copy of the image before inference.
    """
    cache = detect_cache if detect_cache is not None else _SPOTIFLOW_DETECT_CACHE
    if detect_cache_key is not None:
        cached = cache.get(detect_cache_key)
        if cached is not None:
            return [dict(p) for p in cached]

    sf_normalize = None
    if normalize:
        try:
            from spotiflow.utils import normalize as sf_normalize
        except ImportError:
            if raise_on_error:
                raise RuntimeError("Spotiflow import failed. Install spotiflow in the active environment.")
            return None
    raw = np.asarray(coerce_plane_to_2d(image), dtype=np.float64)
    if raw.ndim != 2 or raw.size == 0:
        return []
    raw_median = float(np.median(raw))
    if raw_median <= 0:
        raw_median = 1e-10

    img_work = np.asarray(raw, dtype=np.float32)
    if blur_sigma and float(blur_sigma) > 0:
        img_work = gaussian(img_work, sigma=float(blur_sigma), preserve_range=True).astype(np.float32)
    if bg_subtract_radius and int(bg_subtract_radius) > 0:
        r = int(bg_subtract_radius)
        img_work = white_tophat(img_work, footprint=disk(r)).astype(np.float32)

    if normalize:
        pl = float(perc_low)
        ph = float(perc_high)
        normalizer = (lambda x, _pl=pl, _ph=ph: sf_normalize(np.asarray(x, dtype=np.float32), pmin=_pl, pmax=_ph))
    else:
        normalizer = None

    prob_kw: Optional[float]
    if use_optimized_threshold:
        prob_kw = None
    else:
        prob_kw = float(prob_thresh) if prob_thresh is not None else 0.5

    n_tiles: Optional[Tuple[int, int]]
    if auto_tiling:
        n_tiles = None
    else:
        ny = max(1, int(n_tiles_y))
        nx = max(1, int(n_tiles_x))
        n_tiles = (ny, nx)

    try:
        model = spotiflow_model
        if model is None:
            model = load_spotiflow_model(model_name=model_name, model_path=model_path)
        spots, details = model.predict(
            img_work,
            prob_thresh=prob_kw,
            n_tiles=n_tiles,
            min_distance=int(min_distance),
            subpix=bool(subpix),
            normalizer=normalizer if normalize else None,
            verbose=verbose,
        )
    except ImportError as e:
        if raise_on_error:
            raise RuntimeError("Spotiflow import failed. Install spotiflow in the active environment.") from e
        return None
    except Exception as e:
        if raise_on_error:
            raise RuntimeError(f"Spotiflow prediction failed: {e}") from e
        return None
    if spots is None or getattr(spots, "size", 0) == 0:
        return []
    spots = np.asarray(spots)
    if spots.ndim != 2 or spots.shape[1] < 2:
        return []
    peaks = []
    h, w = raw.shape
    for i in range(spots.shape[0]):
        yy, xx = int(round(spots[i, 0])), int(round(spots[i, 1]))
        if 0 <= yy < h and 0 <= xx < w:
            prob = probability_from_spotiflow_details(
                details, i, spots_row=spots[i] if spots.shape[0] > i else None
            )
            intensity = float(raw[yy, xx])
            rel = intensity / raw_median
            p = {
                "y": yy,
                "x": xx,
                "intensity": intensity,
                "probability": prob,
                "size_px": 1,
                "mean_intensity": intensity,
                "rel_intensity": rel,
            }
            peaks.append(p)
    if detect_cache_key is not None:
        cache[detect_cache_key] = [dict(p) for p in peaks]
        while len(cache) > _SPOTIFLOW_DETECT_CACHE_MAX:
            try:
                cache.pop(next(iter(cache)))
            except StopIteration:
                break
    return peaks


def refine_spotiflow_peaks(
    image: np.ndarray,
    peaks: List[Dict[str, Any]],
    *,
    labeled_mask: Optional[np.ndarray] = None,
    threshold_frac: float = 0.8,
    max_radius: int = 5,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    min_peak_raw: Optional[float] = None,
    min_focus_mean: Optional[float] = None,
    per_cell_ratio: Optional[float] = None,
    accepted_labels: Optional[set[int]] = None,
    exclude_outside: bool = False,
) -> List[Dict[str, Any]]:
    """
    Refine Spotiflow point detections with blob metrics and optional post-filters.

    All thresholds operate on the raw image.
    """
    if not peaks:
        return []
    raw = np.asarray(coerce_plane_to_2d(image), dtype=np.float64)
    if raw.ndim != 2 or raw.size == 0:
        return []
    h, w = raw.shape
    raw_median = float(np.median(raw))
    if raw_median <= 0:
        raw_median = 1e-10
    label_img = None
    if labeled_mask is not None and labeled_mask.shape[:2] == raw.shape[:2]:
        label_img = np.asarray(labeled_mask, dtype=np.int32)
    accepted_set = {int(x) for x in accepted_labels} if accepted_labels else None

    try:
        threshold_frac = float(threshold_frac)
    except (TypeError, ValueError):
        threshold_frac = 0.8
    if not np.isfinite(threshold_frac):
        threshold_frac = 0.8
    threshold_frac = float(np.clip(threshold_frac, 0.05, 1.0))

    out: List[Dict[str, Any]] = []
    for p in peaks:
        yi, xi = int(p.get("y", -1)), int(p.get("x", -1))
        if yi < 0 or yi >= h or xi < 0 or xi >= w:
            continue
        cell_id = int(label_img[yi, xi]) if label_img is not None else 0
        if exclude_outside and label_img is not None:
            if cell_id <= 0:
                continue
            if accepted_set is not None and cell_id not in accepted_set:
                continue

        intensity = p.get("intensity")
        if intensity is None or not np.isfinite(float(intensity)):
            intensity = float(raw[yi, xi])
        intensity = float(intensity)
        if min_peak_raw is not None and float(min_peak_raw) > 0 and intensity < float(min_peak_raw):
            continue

        size_px, mean_int = focus_size_and_mean_at_peak(
            raw,
            yi,
            xi,
            threshold_frac=threshold_frac,
            max_radius=max_radius,
            labeled_mask=label_img,
        )
        if min_size is not None and int(min_size) > 0 and size_px < int(min_size):
            continue
        if max_size is not None and int(max_size) > 0 and size_px > int(max_size):
            continue
        if min_focus_mean is not None and float(min_focus_mean) > 0 and mean_int < float(min_focus_mean):
            continue

        if per_cell_ratio is not None and float(per_cell_ratio) > 0 and label_img is not None and cell_id > 0:
            blob_mask = get_focus_blob_mask(
                raw,
                yi,
                xi,
                threshold_frac=threshold_frac,
                max_radius=max_radius,
                labeled_mask=label_img,
            )
            rest_of_cell = (label_img == cell_id) & ~blob_mask
            n_rest = int(np.sum(rest_of_cell))
            if n_rest == 0:
                continue
            rest_mean = float(np.mean(raw[rest_of_cell]))
            if not np.isfinite(rest_mean) or rest_mean <= 0:
                continue
            ratio = float(mean_int) / rest_mean
            if ratio < float(per_cell_ratio):
                continue

        q = dict(p)
        q["intensity"] = intensity
        q["size_px"] = int(size_px)
        q["mean_intensity"] = float(mean_int)
        q["rel_intensity"] = float(intensity) / raw_median
        out.append(q)
    return out
