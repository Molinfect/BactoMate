"""Per-cell focus / sharpness scores for out-of-focus rejection.

Phase contrast and bright-field need different notions of “in focus”:

- **Phase** (tuned on real bacterial phase stacks): score the radial intensity
  profile across the cell edge. In-focus rods have a steep rise into a halo that
  peaks close to the mask; defocus pushes the halo peak outward and softens the
  rise. Neighbor cells (and their halos) are excluded from the profile rings.
- **Bright-field**: high-frequency content / edge energy inside the cell (classic
  autofocus-style sharpness).

Frame scoring precomputes normalize once and scores each cell on a tight crop.
"""
from __future__ import annotations

from typing import Callable, Iterable, Literal

import numpy as np
from scipy import ndimage

FocusModality = Literal["phase", "brightfield"]
ProgressCallback = Callable[[int, int], None]

_PHASE_PAD = 10  # room for profile rings out to ~8 px + neighbor exclusion
_MIN_PHASE_AREA = 80  # ignore debris / tiny fragments (unstable profiles)
_NEIGHBOR_EXCLUDE_R = 4


def normalize_focus_modality(value: object | None) -> FocusModality:
    """Map UI / settings values to a supported focus modality (default: phase)."""
    raw = str(value or "phase").strip().lower().replace("-", "").replace(" ", "")
    if raw in {"bf", "brightfield", "brightfld", "transmission", "trans"}:
        return "brightfield"
    return "phase"


def _robust_normalize_plane(plane: np.ndarray) -> np.ndarray:
    """Median/MAD normalize so scores are roughly intensity-scale invariant."""
    img = np.asarray(plane, dtype=np.float64)
    if img.size == 0:
        return img
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.std(img))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return (img - med) / scale


def _binary_disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (yy * yy + xx * xx) <= (r * r)


def _tenengrad_energy(plane: np.ndarray) -> np.ndarray:
    sx = ndimage.sobel(plane, axis=1)
    sy = ndimage.sobel(plane, axis=0)
    return sx * sx + sy * sy


def laplacian_variance_in_mask(plane: np.ndarray, cell_mask: np.ndarray) -> float:
    """Variance of Laplacian inside ``cell_mask`` — higher means sharper (bright-field)."""
    if cell_mask is None or not np.any(cell_mask):
        return float("nan")
    img = np.asarray(plane, dtype=np.float64)
    lap = ndimage.laplace(img)
    vals = lap[cell_mask]
    if vals.size == 0:
        return float("nan")
    return float(np.var(vals))


def _padded_slices(
    slc: tuple[slice, slice],
    shape: tuple[int, ...],
    pad: int,
) -> tuple[slice, slice]:
    h, w = int(shape[0]), int(shape[1])
    y0 = max(0, int(slc[0].start) - pad)
    y1 = min(h, int(slc[0].stop) + pad)
    x0 = max(0, int(slc[1].start) - pad)
    x1 = min(w, int(slc[1].stop) + pad)
    return slice(y0, y1), slice(x0, x1)


def _phase_score_from_crops(
    img: np.ndarray,
    cell_mask: np.ndarray,
    bg_ok: np.ndarray,
) -> float:
    """
    Phase score from the radial intensity profile at the cell edge.

    Uses mean intensity in successive 1-px rings outside the mask (excluding
    neighbor cells/halos). A steep rise into a near halo scores high; a soft
    rise / far-out halo peak scores low.
    """
    if cell_mask is None or not np.any(cell_mask):
        return float("nan")
    mask = np.asarray(cell_mask, dtype=bool)
    if int(mask.sum()) < _MIN_PHASE_AREA:
        return float("nan")
    bg = np.asarray(bg_ok, dtype=bool) & ~mask
    if not np.any(bg):
        bg = ~mask

    dist_out = ndimage.distance_transform_edt(~mask)
    dist_in = ndimage.distance_transform_edt(mask)
    deep = mask & (dist_in >= 2.0)
    if not np.any(deep):
        deep = mask & (dist_in >= 1.0)
    if not np.any(deep):
        deep = mask
    mu_in = float(np.mean(img[deep]))

    rings: list[float] = []
    for k in range(1, 7):
        band = bg & (dist_out > (k - 0.5)) & (dist_out <= (k + 0.5))
        if int(band.sum()) < 5:
            rings.append(float("nan"))
        else:
            rings.append(float(np.mean(img[band])))
    rings_a = np.asarray(rings, dtype=np.float64)
    if not np.any(np.isfinite(rings_a)):
        return float("nan")

    seq = np.asarray([mu_in, *rings_a], dtype=np.float64)
    diffs = np.diff(seq)
    diffs = np.where(np.isfinite(diffs), diffs, -np.inf)
    max_rise = float(np.max(diffs)) if np.any(np.isfinite(diffs)) else 0.0
    if max_rise <= 0.0:
        return 0.0

    peak_i = int(np.nanargmax(rings_a))
    peak_val = float(rings_a[peak_i])
    far = bg & (dist_out > 5.0) & (dist_out <= 9.0)
    if int(far.sum()) < 20:
        far = bg & (dist_out > 4.0)
    if np.any(far):
        mu_far = float(np.mean(img[far]))
    else:
        mu_far = float(np.nanmedian(rings_a))

    half = mu_far + 0.5 * (peak_val - mu_far)
    width = 6.0
    for k in range(peak_i, len(rings_a)):
        if np.isfinite(rings_a[k]) and float(rings_a[k]) <= half:
            width = float(k - peak_i + 1)
            break
    peak_dist = float(peak_i + 1)

    return float(
        (max_rise ** 2)
        / ((1.0 + 0.35 * (peak_dist - 1.0)) ** 2)
        / ((1.0 + 0.5 * (width - 1.0)) ** 2)
    )


def _brightfield_score_from_crops(
    img: np.ndarray,
    energy: np.ndarray,
    lap: np.ndarray,
    cell_mask: np.ndarray,
) -> float:
    if cell_mask is None or not np.any(cell_mask):
        return float("nan")
    mask = np.asarray(cell_mask, dtype=bool)
    se1 = _binary_disk(1)
    eroded = ndimage.binary_erosion(mask, structure=se1)
    interior = eroded if np.any(eroded) else mask
    lap_vals = lap[interior]
    if lap_vals.size == 0:
        return float("nan")
    lap_var = float(np.var(lap_vals))

    dilated = ndimage.binary_dilation(mask, structure=se1)
    ring = dilated & ~eroded
    if not np.any(ring):
        ring = mask
    edge_vals = energy[ring]
    edge = float(np.mean(edge_vals)) if edge_vals.size else 0.0
    return float(np.log1p(max(0.0, lap_var + 0.25 * edge)))


def _neighbor_free_bg(mask_plane: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
    """Unlabeled pixels away from other cells (so neighbor halos do not poison rings)."""
    others = (mask_plane > 0) & ~cell_mask
    if not np.any(others):
        return mask_plane == 0
    blocked = ndimage.binary_dilation(others, structure=_binary_disk(_NEIGHBOR_EXCLUDE_R))
    return (mask_plane == 0) & ~blocked


def phase_focus_score(
    plane: np.ndarray,
    cell_mask: np.ndarray,
    *,
    free_bg: np.ndarray | None = None,
) -> float:
    """
    Phase-contrast focus score for a single mask (convenience API).

    Prefer ``score_cells_in_frame`` when scoring many cells.
    """
    if cell_mask is None or not np.any(cell_mask):
        return float("nan")
    mask = np.asarray(cell_mask, dtype=bool)
    img = _robust_normalize_plane(plane)
    if free_bg is None:
        bg_ok = ~mask
    else:
        bg_ok = np.asarray(free_bg, dtype=bool) & ~mask
    return _phase_score_from_crops(img, mask, bg_ok)


def brightfield_focus_score(plane: np.ndarray, cell_mask: np.ndarray) -> float:
    """Bright-field focus score for a single mask (convenience API)."""
    if cell_mask is None or not np.any(cell_mask):
        return float("nan")
    mask = np.asarray(cell_mask, dtype=bool)
    img = _robust_normalize_plane(plane)
    energy = _tenengrad_energy(img)
    lap = ndimage.laplace(img)
    return _brightfield_score_from_crops(img, energy, lap, mask)


def score_cell_focus(
    plane: np.ndarray,
    cell_mask: np.ndarray,
    *,
    modality: object | None = "phase",
    free_bg: np.ndarray | None = None,
) -> float:
    """Score one cell mask for the given imaging modality."""
    mode = normalize_focus_modality(modality)
    if mode == "brightfield":
        return brightfield_focus_score(plane, cell_mask)
    return phase_focus_score(plane, cell_mask, free_bg=free_bg)


def score_cells_in_frame(
    phase_plane: np.ndarray,
    mask_plane: np.ndarray,
    labels: Iterable[int] | set[int],
    *,
    modality: object | None = "phase",
    progress_callback: ProgressCallback | None = None,
) -> dict[int, float]:
    """Return focus score per cell label in one frame (shared precompute)."""
    mask_plane = np.asarray(mask_plane, dtype=np.int32)
    mode = normalize_focus_modality(modality)
    label_list = [int(lab) for lab in labels if int(lab) > 0]
    n_lab = len(label_list)
    if n_lab == 0:
        return {}

    img = _robust_normalize_plane(phase_plane)
    energy = _tenengrad_energy(img) if mode == "brightfield" else None
    lap = ndimage.laplace(img) if mode == "brightfield" else None

    max_lab = int(mask_plane.max()) if mask_plane.size else 0
    objects = ndimage.find_objects(mask_plane, max_label=max_lab) if max_lab > 0 else []

    out: dict[int, float] = {}
    for i, lab_i in enumerate(label_list):
        if progress_callback is not None and (i == 0 or i + 1 == n_lab or i % 25 == 0):
            try:
                progress_callback(i, n_lab)
            except Exception:
                pass

        slc = objects[lab_i - 1] if 1 <= lab_i <= len(objects) else None
        if slc is None:
            out[lab_i] = float("nan")
            continue
        crop_sl = _padded_slices(slc, img.shape, _PHASE_PAD)
        img_c = img[crop_sl]
        mask_c = mask_plane[crop_sl] == lab_i
        if mode == "brightfield":
            assert energy is not None and lap is not None
            out[lab_i] = _brightfield_score_from_crops(
                img_c, energy[crop_sl], lap[crop_sl], mask_c
            )
        else:
            bg_c = _neighbor_free_bg(mask_plane[crop_sl], mask_c)
            out[lab_i] = _phase_score_from_crops(img_c, mask_c, bg_c)

    if progress_callback is not None:
        try:
            progress_callback(n_lab, n_lab)
        except Exception:
            pass
    return out


def filter_labels_by_focus(
    scores: dict[int, float],
    *,
    min_absolute: float | None = None,
    min_fraction_of_median: float | None = None,
    reference_percentile: float = 80.0,
) -> set[int]:
    """
    Keep labels that pass focus thresholds.

    ``min_fraction_of_median``: fraction of a high within-frame reference score
    (default: 80th percentile). Using the median fails when most cells are OOF —
    the median is then also OOF and almost nothing can be rejected. A high
    percentile tracks the sharper cells so soft ones can be gated out.
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
        vals = np.asarray(list(finite.values()), dtype=np.float64)
        pct = float(np.clip(reference_percentile, 50.0, 100.0))
        ref = float(np.percentile(vals, pct))
        if ref > 0:
            cutoff = ref * float(min_fraction_of_median)
            accepted = {k for k in accepted if finite[k] >= cutoff}
    return accepted


def focus_reference_score(
    scores: dict[int, float],
    *,
    reference_percentile: float = 80.0,
) -> float | None:
    """High within-frame reference used by the relative focus gate (for UI)."""
    finite = [v for v in scores.values() if v is not None and np.isfinite(v)]
    if not finite:
        return None
    pct = float(np.clip(reference_percentile, 50.0, 100.0))
    ref = float(np.percentile(np.asarray(finite, dtype=np.float64), pct))
    return ref if ref > 0 else None
