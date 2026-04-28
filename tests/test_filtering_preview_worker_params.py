"""Verify the Filtering Preview worker really uses the params it was given.

If the preview overlay isn't reflecting changes the user made to the filter
parameters, it could be because:

1. The cache check rejects the new result (covered by
   test_filtering_preview_cache.py).
2. The worker runs but with stale params, so accepted/rejected don't change.

These tests pin contract #2: when called with two different parameter sets
on the same mask, the worker MUST emit different (accepted, rejected) lists.
The worker is exercised directly (without a QThread) so we can keep the test
deterministic and headless.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

try:
    from PyQt6.QtCore import QObject

    from bactimate.gui.analyze_masks_preview_worker import AnalyzeMasksPreviewWorker
except Exception as e:  # pragma: no cover
    pytest.skip(f"PyQt6 import failed: {e}", allow_module_level=True)


def _make_mask_with_mixed_aspect_ratios() -> np.ndarray:
    """Build a synthetic mask with a few elongated cells and a few square cells.

    Square cells should be rejected by min_aspect_ratio >= 3.0.
    """

    mask = np.zeros((128, 128), dtype=np.int32)
    # Three long thin "rod" cells: aspect ratio ~6
    mask[10:14, 10:34] = 1
    mask[40:44, 10:34] = 2
    mask[70:74, 10:34] = 3
    # Three square "blob" cells: aspect ratio ~1
    mask[10:22, 60:72] = 4
    mask[40:52, 60:72] = 5
    mask[70:82, 60:72] = 6
    return mask


def _run_worker_sync(mask: np.ndarray, params: dict) -> tuple[list, list, list]:
    """Run the worker synchronously and capture its emitted (accepted, rejected, all)."""

    captured = {}

    def _on_result_ready(*args):
        # Match the result_ready signal's positional layout:
        # rid, fid, acc, rej, all, n_obj, err, vis, peaks, n_foci, n_foci_drawn,
        # raw_median_fi, min_rel, mode_txt
        captured["accepted"] = list(args[2] or [])
        captured["rejected"] = list(args[3] or [])
        captured["all"] = list(args[4] or [])
        captured["err"] = str(args[6] or "")

    worker = AnalyzeMasksPreviewWorker(
        request_id=1,
        frame_idx=0,
        mask_cleaned=mask,
        params=params,
        fi_image=None,
        display_params={"do_fi_contour": False, "fi_contour_thickness_val": 1, "run_foci": False, "foci_snapshot": None},
        skip_vis=True,
    )
    worker.result_ready.connect(_on_result_ready)
    worker.run()
    return captured.get("accepted", []), captured.get("rejected", []), captured.get("all", [])


def _params(min_aspect_ratio: float) -> dict:
    return {
        "pixel_size": 0.108,
        "exclude_border": False,
        "min_aspect_ratio": float(min_aspect_ratio),
        "max_aspect_ratio": 50.0,
        "min_eccentricity": 0.0,
        "min_solidity": 0.0,
        "min_area": None,
        "max_area": None,
        "min_width": None,
        "max_width": None,
        "min_length": None,
    }


def _ensure_qapp():
    """Ensure a QApplication exists (Qt signal connections require an event loop owner)."""

    try:
        from PyQt6.QtWidgets import QApplication
    except Exception:
        pytest.skip("PyQt6 widgets not available")
        return None
    return QApplication.instance() or QApplication(sys.argv)


def test_worker_emits_different_accept_reject_for_different_min_aspect_ratio():
    _ensure_qapp()
    mask = _make_mask_with_mixed_aspect_ratios()

    # Permissive: every labelled cell should pass.
    acc_loose, rej_loose, _ = _run_worker_sync(mask, _params(min_aspect_ratio=1.0))

    # Strict: only the long thin rods should pass (~6:1), squares (~1:1) should be rejected.
    acc_strict, rej_strict, _ = _run_worker_sync(mask, _params(min_aspect_ratio=3.0))

    assert len(rej_loose) == 0, f"Permissive filter rejected unexpectedly: {rej_loose}"
    assert len(rej_strict) > 0, "Strict filter should reject the square cells"
    assert set(acc_strict) != set(acc_loose), (
        "Different min_aspect_ratio values must produce different accepted sets — "
        f"loose={sorted(acc_loose)} strict={sorted(acc_strict)}"
    )


def test_worker_sees_pixel_size_change_through_size_filter():
    """If pixel_size changes, the same area threshold maps to different physical sizes,
    so size filtering should change."""

    _ensure_qapp()
    mask = _make_mask_with_mixed_aspect_ratios()

    p_small = _params(min_aspect_ratio=1.0)
    p_small["pixel_size"] = 0.05
    p_small["min_area"] = 1.0  # 1 µm² — at 0.05 µm/px each cell is huge in µm² units

    p_large = _params(min_aspect_ratio=1.0)
    p_large["pixel_size"] = 1.0
    p_large["min_area"] = 1.0  # 1 µm² — at 1.0 µm/px the small cells are below threshold

    acc_small, _, _ = _run_worker_sync(mask, p_small)
    acc_large, _, _ = _run_worker_sync(mask, p_large)

    assert set(acc_small) != set(acc_large), (
        "pixel_size change must propagate through filter_by_size — "
        f"small_px={sorted(acc_small)} large_px={sorted(acc_large)}"
    )
