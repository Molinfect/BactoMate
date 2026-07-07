"""Per-cell focus / sharpness scores from phase images (out-of-focus rejection)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def laplacian_variance_in_mask(plane: np.ndarray, cell_mask: np.ndarray) -> float:
    """Variance of Laplacian inside ``cell_mask`` — higher means sharper (more in focus)."""
    if cell_mask is None or not np.any(cell_mask):
        return float("nan")
    img = np.asarray(plane, dtype=np.float64)
    lap = ndimage.laplace(img)
    vals = lap[cell_mask]
    if vals.size == 0:
        return float("nan")
    return float(np.var(vals))


def score_cells_in_frame(
    phase_plane: np.ndarray,
    mask_plane: np.ndarray,
    labels: list[int] | set[int],
) -> dict[int, float]:
    """Return Laplacian-variance focus score per cell label in one frame."""
    mask_plane = np.asarray(mask_plane, dtype=np.int32)
    out: dict[int, float] = {}
    for lab in labels:
        lab_i = int(lab)
        if lab_i <= 0:
            continue
        cell_mask = mask_plane == lab_i
        out[lab_i] = laplacian_variance_in_mask(phase_plane, cell_mask)
    return out


def filter_labels_by_focus(
    scores: dict[int, float],
    *,
    min_absolute: float | None = None,
    min_fraction_of_median: float | None = None,
) -> set[int]:
    """
    Keep labels that pass focus thresholds.

    ``min_fraction_of_median``: e.g. 0.5 keeps cells with score >= 50% of frame median.
    """
    if not scores:
        return set()
    finite = {k: v for k, v in scores.items() if v is not None and np.isfinite(v)}
    if not finite:
        return set(scores.keys())
    accepted = set(finite.keys())
    if min_absolute is not None and float(min_absolute) > 0:
        accepted = {k for k in accepted if finite[k] >= float(min_absolute)}
    if min_fraction_of_median is not None and float(min_fraction_of_median) > 0:
        med = float(np.median(list(finite.values())))
        if med > 0:
            cutoff = med * float(min_fraction_of_median)
            accepted = {k for k in accepted if finite[k] >= cutoff}
    return accepted
