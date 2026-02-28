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

# Default preprocessing (same idea as external foci_tool)
DEFAULT_SIGMA_SMOOTH = 1.0
DEFAULT_BG_RADIUS = 5
DEFAULT_SHARPEN_RADIUS = 1.0
DEFAULT_SHARPEN_AMOUNT = 1.5


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
    threshold_frac: float = 0.5,
    max_radius: int = 5,
) -> int:
    """
    Compute connected-component size (pixels) of the focus containing (y, x).
    Threshold in a window around the peak at peak_intensity * threshold_frac.
    """
    from skimage.measure import label as sk_label
    h, w = image.shape[:2]
    if y < 0 or y >= h or x < 0 or x >= w:
        return 0
    peak_val = float(image[y, x])
    if not np.isfinite(peak_val) or peak_val <= 0:
        return 1
    thresh = peak_val * threshold_frac
    y0 = max(0, y - max_radius)
    y1 = min(h, y + max_radius + 1)
    x0 = max(0, x - max_radius)
    x1 = min(w, x + max_radius + 1)
    window = image[y0:y1, x0:x1]
    binary = (window >= thresh).astype(np.uint8)
    labeled, n = sk_label(binary, connectivity=2, return_num=True)
    if n == 0:
        return 1
    ly, lx = y - y0, x - x0
    if ly < 0 or ly >= labeled.shape[0] or lx < 0 or lx >= labeled.shape[1]:
        return 1
    lab = int(labeled[ly, lx])
    if lab == 0:
        return 1
    return int(np.sum(labeled == lab))


def detect_peaks_skimage(
    image: np.ndarray,
    min_rel: float = 2.0,
    min_distance: int = 2,
    min_size: int = 1,
    max_size: Optional[int] = None,
    use_preprocessing: bool = True,
    sigma_smooth: float = DEFAULT_SIGMA_SMOOTH,
    bg_radius: int = DEFAULT_BG_RADIUS,
    sharpen_radius: float = DEFAULT_SHARPEN_RADIUS,
    sharpen_amount: float = DEFAULT_SHARPEN_AMOUNT,
    size_threshold_frac: float = 0.5,
    size_max_radius: int = 5,
) -> List[Dict[str, Any]]:
    """
    Detect foci in a 2D image. Returns list of dicts with keys:
    y, x, intensity, size_px (and optionally rel_intensity).

    Min relative intensity filter uses the raw (pre-processing) image: peaks are kept only
    where raw_intensity >= raw_median * min_rel, so the threshold is in the same units as
    the user's image. Peak detection still uses the preprocessed image for better localization.
    """
    raw_img = np.asarray(image, dtype=np.float64)
    if raw_img.ndim != 2 or raw_img.size == 0:
        return []
    raw_median = float(np.median(raw_img))
    if raw_median <= 0:
        raw_median = 1e-10

    img = raw_img.copy()
    if use_preprocessing:
        img = enhance_foci_image(
            img, sigma_smooth, bg_radius, sharpen_radius, sharpen_amount
        )
    img_median = float(np.median(img))
    if img_median <= 0:
        img_median = 1e-10
    # Threshold for peak_local_max on preprocessed image (reduces candidate count)
    if min_rel is not None and min_rel > 0:
        min_abs = img_median * min_rel
    else:
        min_abs = None
    coords = peak_local_max(img, min_distance=min_distance, threshold_abs=min_abs)
    if coords.size == 0:
        return []
    # Raw intensity at each peak (for user-facing filter and output)
    raw_intensities = raw_img[coords[:, 0], coords[:, 1]]
    min_raw_abs = (raw_median * min_rel) if (min_rel is not None and min_rel > 0) else None
    peaks: List[Dict[str, Any]] = []
    for k in range(len(coords)):
        yi, xi = int(coords[k, 0]), int(coords[k, 1])
        raw_int = float(raw_intensities[k])
        if min_raw_abs is not None and raw_int < min_raw_abs:
            continue
        size_px = focus_size_at_peak(
            img, yi, xi,
            threshold_frac=size_threshold_frac,
            max_radius=size_max_radius,
        )
        if min_size is not None and size_px < min_size:
            continue
        if max_size is not None and max_size > 0 and size_px > max_size:
            continue
        rel = raw_int / raw_median
        peaks.append({
            "y": yi,
            "x": xi,
            "intensity": raw_int,
            "rel_intensity": rel,
            "size_px": size_px,
        })
    return peaks


def assign_peaks_to_cells(
    peaks: List[Dict[str, Any]],
    labeled_mask: np.ndarray,
    dilation_px: int = 1,
    frame_index: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], List[int]]]:
    """
    Assign each peak to a cell (label) using the mask. Optionally dilate labels.

    Returns:
        foci_rows: list of dicts with keys foci_id, cell_id, frame, y, x, intensity, size_px, in_cell.
        cell_foci_ids: (cell_id, frame) -> list of foci_id (for building foci_ids column).
    """
    if not peaks:
        return [], {}
    h, w = labeled_mask.shape[:2]
    if dilation_px > 0:
        mask = expand_labels(labeled_mask.astype(np.int32), distance=dilation_px)
    else:
        mask = np.asarray(labeled_mask, dtype=np.int32)
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
                "in_cell": False,
            })
            foci_id += 1
            continue
        cell_id = int(mask[yi, xi])
        in_cell = cell_id > 0
        foci_rows.append({
            "foci_id": foci_id,
            "cell_id": cell_id if in_cell else None,
            "frame": frame_index,
            "y": yi,
            "x": xi,
            "intensity": p.get("intensity", np.nan),
            "size_px": p.get("size_px", 1),
            "in_cell": in_cell,
        })
        if in_cell:
            key = (cell_id, frame_index)
            cell_foci_ids.setdefault(key, []).append(foci_id)
        foci_id += 1
    return foci_rows, cell_foci_ids
