from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from bactomate.modules.cell_focus_score import (
    filter_labels_by_focus,
    focus_reference_score,
    normalize_focus_modality,
    phase_focus_score,
    score_cells_in_frame,
)


def _disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (yy * yy + xx * xx) <= (r * r)


def _paint_rod(plane: np.ndarray, mask: np.ndarray, cy: int, cx: int, length: int, width: int, label: int) -> None:
    y0, y1 = cy - width // 2, cy + (width + 1) // 2
    x0, x1 = cx - length // 2, cx + (length + 1) // 2
    mask[y0:y1, x0:x1] = label
    plane[y0:y1, x0:x1] = 40.0


def test_normalize_focus_modality():
    assert normalize_focus_modality("phase") == "phase"
    assert normalize_focus_modality("Bright-field") == "brightfield"
    assert normalize_focus_modality("bf") == "brightfield"
    assert normalize_focus_modality(None) == "phase"


def test_focus_filter_keeps_sharper_cell_brightfield():
    rng = np.random.default_rng(0)
    plane = np.zeros((64, 64), dtype=np.float32)
    plane[20:30, 20:30] = rng.normal(100.0, 8.0, (10, 10))
    plane[40:50, 40:50] = rng.normal(40.0, 0.2, (10, 10))
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[20:30, 20:30] = 1
    mask[40:50, 40:50] = 2
    scores = score_cells_in_frame(plane, mask, [1, 2], modality="brightfield")
    kept = filter_labels_by_focus(scores, min_fraction_of_median=0.8)
    assert 1 in kept
    assert 2 not in kept


def test_phase_prefers_near_halo_over_far_soft_halo():
    """Steep near-rim halo (in focus) must outrank a farther, softer defocus halo."""
    plane = np.full((120, 120), 120.0, dtype=np.float64)
    mask = np.zeros((120, 120), dtype=np.int32)

    # Cell 1: in-focus-like — dark body, bright rim peaking at 1 px
    _paint_rod(plane, mask, cy=40, cx=40, length=28, width=10, label=1)
    m1 = mask == 1
    rim1 = ndimage.binary_dilation(m1, structure=_disk(1)) & ~m1
    plane[rim1] = 180.0

    # Cell 2: defocus-like — softer body, halo mass farther out
    _paint_rod(plane, mask, cy=85, cx=85, length=28, width=10, label=2)
    m2 = mask == 2
    plane[m2] = 70.0
    far_halo = ndimage.binary_dilation(m2, structure=_disk(4)) & ~ndimage.binary_dilation(
        m2, structure=_disk(2)
    )
    plane[far_halo] = 175.0
    region = ndimage.binary_dilation(m2, structure=_disk(5))
    plane[region] = ndimage.gaussian_filter(plane, sigma=1.6)[region]

    scores = score_cells_in_frame(plane, mask, [1, 2], modality="phase")
    assert scores[1] > scores[2], f"near-halo should win: {scores}"


def test_phase_rejects_blurred_cell_vs_sharp():
    plane = np.full((100, 100), 110.0, dtype=np.float64)
    mask = np.zeros((100, 100), dtype=np.int32)
    _paint_rod(plane, mask, cy=30, cx=30, length=26, width=10, label=1)
    _paint_rod(plane, mask, cy=70, cx=70, length=26, width=10, label=2)
    for lab in (1, 2):
        m = mask == lab
        rim = ndimage.binary_dilation(m, structure=_disk(1)) & ~m
        plane[rim] = 165.0
    region = ndimage.binary_dilation(mask == 2, structure=_disk(4))
    plane[region] = ndimage.gaussian_filter(plane, sigma=2.2)[region]

    scores = score_cells_in_frame(plane, mask, [1, 2], modality="phase")
    assert scores[1] > scores[2]
    kept = filter_labels_by_focus(scores, min_fraction_of_median=0.7)
    assert 1 in kept
    assert 2 not in kept


def test_phase_cluster_scores_similar_to_isolated_peer():
    plane = np.full((140, 140), 120.0, dtype=np.float64)
    mask = np.zeros((140, 140), dtype=np.int32)
    _paint_rod(plane, mask, cy=45, cx=40, length=22, width=9, label=1)
    _paint_rod(plane, mask, cy=45, cx=58, length=22, width=9, label=2)
    _paint_rod(plane, mask, cy=105, cx=105, length=22, width=9, label=3)
    for lab in (1, 2, 3):
        m = mask == lab
        rim = ndimage.binary_dilation(m, structure=_disk(1)) & ~(mask > 0)
        plane[rim] = 170.0

    scores = score_cells_in_frame(plane, mask, [1, 2, 3], modality="phase")
    ratio_1 = scores[1] / max(scores[3], 1e-9)
    ratio_2 = scores[2] / max(scores[3], 1e-9)
    assert 0.45 <= ratio_1 <= 2.2, f"cluster cell1 vs isolated: {scores}"
    assert 0.45 <= ratio_2 <= 2.2, f"cluster cell2 vs isolated: {scores}"


def test_phase_focus_score_nan_on_empty_or_tiny_mask():
    plane = np.zeros((32, 32), dtype=np.float64)
    assert np.isnan(phase_focus_score(plane, np.zeros((32, 32), dtype=bool)))
    tiny = np.zeros((32, 32), dtype=bool)
    tiny[10:14, 10:14] = True  # 16 px < min area
    assert np.isnan(phase_focus_score(plane, tiny))


def test_real_phase_stack_ranks_tight_halo_above_soft_far_halo():
    """Optional regression on the user's Desktop test_outoffocus pair."""
    root = Path("/Users/philipppopp/Desktop/test_outoffocus")
    phase_path = root / "TH437_Mi(+)__Phase60x.tif"
    mask_path = root / "TH437_Mi(+)__Phase60x_masks.tif"
    if not phase_path.is_file() or not mask_path.is_file():
        return
    from tifffile import imread

    phase = np.asarray(imread(str(phase_path)))
    mask = np.asarray(imread(str(mask_path)), dtype=np.int32)
    if phase.ndim == 3:
        phase = phase[0]
    if mask.ndim == 3:
        mask = mask[0]
    labels = [int(x) for x in np.unique(mask) if int(x) > 0]
    scores = score_cells_in_frame(phase, mask, labels, modality="phase")
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    ref = focus_reference_score(finite)
    assert ref is not None and ref > 0
    # Lab 28: sharp left-side rod in this stack; labs 20/29: soft far-halo examples.
    assert finite[28] / ref >= 0.9
    assert finite[20] / ref < 0.4
    assert finite[29] / ref < 0.4
    kept = filter_labels_by_focus(finite, min_fraction_of_median=0.6)
    assert 28 in kept
    assert 20 not in kept
    assert 29 not in kept
