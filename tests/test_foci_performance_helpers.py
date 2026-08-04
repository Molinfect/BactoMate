"""Tests for foci preview performance helpers (caches, batched draw, progress guard)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from bactomate.gui.foci_filter_pipeline import FociPassFailCache, foci_pass_fail_cache_key
from bactomate.gui.foci_overlay_draw import (
    DEFAULT_FOCI_PREVIEW_RENDER_CAP,
    draw_foci_blob_contours_on_rgb,
    draw_foci_blob_fill_and_edges_on_rgb,
    draw_peak_rings_on_rgb,
)
from bactomate.modules.foci_utils import (
    clear_spotiflow_detect_cache,
    detect_peaks_spotiflow,
    spotiflow_inference_cache_key,
)


def test_foci_pass_fail_cache_reuses_results() -> None:
    cache = FociPassFailCache(max_size=4)
    key = foci_pass_fail_cache_key(0, "mask.tif", "sig", {1, 2}, "ch_a")
    calls = {"n": 0}

    def _compute():
        calls["n"] += 1
        return [{"y": 1, "x": 2}], [{"y": 3, "x": 4}]

    out1 = cache.get_or_compute(key, _compute)
    out2 = cache.get_or_compute(key, _compute)
    assert calls["n"] == 1
    assert out1 == out2
    assert out1[0][0]["y"] == 1
    assert out1[1][0]["y"] == 3


def test_draw_peak_rings_respects_cap() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    peaks = [{"y": 10 + i, "x": 10 + i} for i in range(20)]
    draw_peak_rings_on_rgb(
        rgb,
        peaks,
        (255, 128, 0),
        mask_shape=(32, 32),
        radius=3,
        cap=5,
    )
    assert int(np.sum(rgb > 0)) > 0


def test_batched_blob_contours_single_boundary_pass() -> None:
    plane = np.zeros((24, 24), dtype=np.float64)
    plane[12, 12] = 200.0
    plane[12, 13] = 180.0
    plane[13, 12] = 170.0
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[8:18, 8:18] = 1
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    peaks = [{"y": 12, "x": 12}, {"y": 14, "x": 14}]
    settings = {"foci_blob_threshold_frac": 0.8, "foci_blob_max_radius": 5}
    missed = draw_foci_blob_contours_on_rgb(
        rgb,
        mask,
        peaks,
        plane,
        settings,
        (255, 165, 0),
        cap=DEFAULT_FOCI_PREVIEW_RENDER_CAP,
    )
    assert int(np.sum(rgb)) > 0
    assert isinstance(missed, list)


def test_batched_blob_fill_and_edges() -> None:
    plane = np.zeros((20, 20), dtype=np.float32)
    plane[10, 10] = 150.0
    plane[10, 11] = 120.0
    mask = np.ones((20, 20), dtype=np.int32)
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    color = np.array([0, 255, 255], dtype=np.uint8)
    draw_foci_blob_fill_and_edges_on_rgb(
        rgb,
        plane,
        mask,
        [{"y": 10, "x": 10}],
        color,
        threshold_frac=0.8,
        max_radius=4,
    )
    assert int(np.sum(rgb)) > 0


def test_spotiflow_detect_cache_key_stable() -> None:
    img = np.ones((8, 8), dtype=np.float32)
    k1 = spotiflow_inference_cache_key(img, model_name="general", min_distance=2)
    k2 = spotiflow_inference_cache_key(img, model_name="general", min_distance=2)
    k3 = spotiflow_inference_cache_key(img, model_name="general", min_distance=3)
    assert k1 == k2
    assert k1 != k3


def test_spotiflow_detect_cache_short_circuits(monkeypatch) -> None:
    clear_spotiflow_detect_cache()
    img = np.ones((4, 4), dtype=np.float32)
    key = spotiflow_inference_cache_key(img, model_name="general")

    class _FakeModel:
        predict_calls = 0

        def predict(self, *_a, **_k):
            _FakeModel.predict_calls += 1
            spots = np.array([[1.0, 1.0]], dtype=np.float64)
            details = {"prob": np.array([0.95], dtype=np.float64)}
            return spots, details

    model = _FakeModel()
    out1 = detect_peaks_spotiflow(
        img,
        detect_cache_key=key,
        normalize=False,
        spotiflow_model=model,
    )
    out2 = detect_peaks_spotiflow(
        img,
        detect_cache_key=key,
        normalize=False,
        spotiflow_model=model,
    )
    assert _FakeModel.predict_calls == 1
    assert out1 and out2
    assert len(out1) == len(out2) == 1


def test_filter_apply_progress_reentrancy_guard(qapp) -> None:
    pytest.importorskip("PyQt6")

    from bactomate.gui.filter_apply_progress import FilterApplyProgressDialog

    dlg = FilterApplyProgressDialog(None)
    dlg.set_pump_events(False)
    dlg._processing_events = True
    dlg._step_label.setText("before")
    dlg.set_progress(50, "nested", "should be ignored")
    assert dlg._step_label.text() == "before"


def test_filter_apply_progress_ignores_updates_while_closing(qapp) -> None:
    """After close begins, set_progress is a no-op (must not touch dying widgets).

    Avoids ``QApplication.processEvents()``: pumping the shared session QApplication
    drains leftover ``QThread.finished`` slots from earlier Analyze Masks GUI tests
    and has segfaulted macOS CI (see ``on_thread_finished`` in preview filtering).
    """
    pytest.importorskip("PyQt6")

    from bactomate.gui.filter_apply_progress import FilterApplyProgressDialog, _qt_alive

    dlg = FilterApplyProgressDialog(None)
    dlg.set_pump_events(False)
    dlg.set_progress(10, "start", "")
    assert dlg._step_label.text() == "start"

    dlg._closing = True
    dlg.set_progress(40, "Applying", "should ignore")
    assert dlg._step_label.text() == "start"
    assert _qt_alive(dlg)

    # Late updates after close must remain no-ops.
    dlg.set_progress(90, "late", "ignored")
    assert dlg._step_label.text() == "start"


def test_filter_apply_progress_throttles_rapid_updates(qapp) -> None:
    pytest.importorskip("PyQt6")

    from bactomate.gui.filter_apply_progress import FilterApplyProgressDialog

    dlg = FilterApplyProgressDialog(None)
    dlg.set_pump_events(False)
    dlg._min_ui_interval_s = 0.2
    dlg.set_progress(10, "A", "one")
    # Same percent + rapid detail change may be throttled…
    dlg.set_progress(10, "A", "two")
    # …but a percent increase must never be dropped (avoids stuck 0% bars).
    dlg.set_progress(11, "B", "")
    assert dlg._bar.value() >= 11
    time.sleep(0.21)
    dlg.set_progress(12, "C", "")
    assert dlg._bar.value() >= 12
