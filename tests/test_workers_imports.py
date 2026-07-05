"""Regression: public worker symbols resolve after package split."""
from __future__ import annotations

from bactomate.gui.workers import (
    DEFAULT_GROWTH_GRAPH_STYLE,
    DEFAULT_GROWTH_Y_COLUMN,
    DEFAULT_VIDEO_FI_LAYOUT,
    DEFAULT_VIDEO_OVERLAY_COLORS,
    GrowthCurveVideoWorker,
    TrackingPreviewResult,
    TrackingWorker,
    WorkerThread,
    apply_video_montage_overlays,
    compose_growth_curve_combined_frame,
    list_growth_curve_excel_columns,
    load_growth_curve_summary_data,
    load_growth_curve_video_source,
    normalize_video_overlay_colors,
)


def test_worker_thread_import() -> None:
    assert WorkerThread is not None


def test_tracking_worker_import() -> None:
    assert TrackingWorker is not None
    assert TrackingPreviewResult is not None


def test_growth_curve_worker_import() -> None:
    assert GrowthCurveVideoWorker is not None
    assert load_growth_curve_video_source is not None
    assert compose_growth_curve_combined_frame is not None


def test_growth_curve_defaults_import() -> None:
    assert DEFAULT_GROWTH_GRAPH_STYLE
    assert DEFAULT_GROWTH_Y_COLUMN
    assert DEFAULT_VIDEO_FI_LAYOUT
    assert DEFAULT_VIDEO_OVERLAY_COLORS


def test_video_overlay_helpers_import() -> None:
    assert apply_video_montage_overlays is not None
    assert normalize_video_overlay_colors is not None
    assert list_growth_curve_excel_columns is not None
    assert load_growth_curve_summary_data is not None
