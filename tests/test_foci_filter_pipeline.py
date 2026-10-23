"""Tests for shared foci CSV filtering and blob refinement pipeline."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bactomate.gui.foci_filter_pipeline import (
    accepted_cell_labels_for_foci,
    cell_filters_active_for_foci,
    classify_foci_cell_relation,
    format_foci_cell_id_display,
    resolve_foci_cell_id_at_peak,
    _foci_table_cell_id,
    compute_foci_pass_fail_for_frame,
    filter_foci_csv_rows,
    foci_blob_threshold_frac,
    foci_filter_failure_reasons,
    measure_foci_rows_metrics,
    refine_foci_rows_with_blob,
    write_filtered_foci_csv,
)

try:
    from PyQt6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None  # type: ignore[misc, assignment]


@pytest.fixture(scope="module", autouse=True)
def _qapp_module():
    if QApplication is None:
        yield None
        return
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.mark.parametrize('value', [None, '', 'invalid', float('nan'), float('inf')])
def test_foci_blob_threshold_invalid_values_use_default(value):
    assert foci_blob_threshold_frac({'foci_blob_threshold_frac': value}) == 0.99


@pytest.mark.parametrize(
    ('settings', 'expected_area'),
    [
        ({}, 2),
        ({'foci_blob_threshold_frac': 0.0}, 9),
        ({'foci_blob_threshold_frac': 0.0001}, 5),
        ({'foci_blob_threshold_frac': 0.8999}, 3),
        ({'foci_blob_threshold_frac': 0.9}, 2),
        ({'foci_blob_threshold_frac': 0.99}, 2),
        ({'foci_blob_threshold_frac': 1.0}, 1),
    ],
)
def test_foci_blob_threshold_range_and_precision(settings, expected_area):
    foci = np.array([[0.0, 0.01, 0.0], [89.98, 100.0, 89.995], [0.0, 99.0, 0.0]])
    rows = [{'y': 1, 'x': 1, 'intensity': 100.0}]
    refined = refine_foci_rows_with_blob(rows, foci, settings=settings)
    measured = measure_foci_rows_metrics(rows, foci, settings=settings)
    assert refined[0]['area_px'] == expected_area
    assert measured[(1, 1)]['area_px'] == expected_area


def test_filter_foci_csv_rows_probability_and_intensity():
    rows = [
        {"y": 1, "x": 2, "intensity": 100.0, "probability": 0.9},
        {"y": 3, "x": 4, "intensity": 50.0, "probability": 0.2},
    ]
    settings = {
        "foci_prob_enabled": True,
        "foci_prob_thresh": 0.5,
        "foci_min_int_enabled": True,
        "foci_min_int": 80.0,
        "foci_exclude_outside_cells": False,
    }
    out = filter_foci_csv_rows(rows, settings)
    assert len(out) == 1
    assert out[0]["y"] == 1 and out[0]["x"] == 2


def test_refine_foci_rows_adds_blob_metrics():
    foci = np.zeros((8, 8), dtype=np.float64)
    foci[4, 4] = 100.0
    foci[3, 4] = 85.0
    rows = [{"y": 4, "x": 4, "intensity": 100.0, "probability": 0.9}]
    settings = {"foci_blob_threshold_frac": 0.8, "foci_blob_max_radius": 5}
    out = refine_foci_rows_with_blob(rows, foci, settings=settings)
    assert len(out) == 1
    assert int(out[0].get("area_px", 0)) >= 1
    assert float(out[0].get("mean_intensity", 0)) > 0


def test_refine_foci_rows_adds_per_cell_ratio():
    foci = np.full((7, 7), 10.0, dtype=np.float64)
    foci[3, 3] = 100.0
    foci[3, 2] = 90.0
    mask = np.ones((7, 7), dtype=np.int32)
    rows = [{"y": 3, "x": 3, "intensity": 100.0, "probability": 0.9}]
    settings = {"foci_blob_threshold_frac": 0.8, "foci_blob_max_radius": 5}
    out = refine_foci_rows_with_blob(rows, foci, label_plane=mask, settings=settings)
    assert len(out) == 1
    assert float(out[0].get("per_cell_ratio", 0.0)) > 1.0


def test_write_filtered_foci_csv_columns(tmp_path: Path):
    out = tmp_path / "filtered.csv"
    rows = [
        {
            "frame": 0,
            "y": 1,
            "x": 2,
            "intensity": 10.0,
            "probability": 0.5,
            "area_px": 3,
            "mean_intensity": 8.0,
            "rel_intensity": 1.2,
            "cell_id": 5,
        }
    ]
    write_filtered_foci_csv(out, rows, include_frame=True)
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "frame",
            "y",
            "x",
            "intensity",
            "probability",
            "area_px",
            "mean_intensity",
            "rel_intensity",
            "cell_id",
        ]
        row = next(reader)
    assert row["area_px"] == "3"
    assert row["cell_id"] == "5"


def test_filter_foci_rows_by_min_area_px():
    rows = [
        {"y": 1, "x": 1, "area_px": 2},
        {"y": 2, "x": 2, "area_px": 8},
    ]
    settings = {"foci_min_area_px_enabled": True, "foci_min_area_px": 5}
    from bactomate.gui.foci_filter_pipeline import filter_foci_rows_by_min_area_px

    out = filter_foci_rows_by_min_area_px(rows, settings)
    assert len(out) == 1
    assert out[0]["area_px"] == 8


def test_populate_foci_table_uses_fail_peak_metrics():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QTableWidget

    table = QTableWidget()
    frame_rows = [
        {"y": 2, "x": 2, "intensity": 200.0, "probability": 0.9},
        {"y": 0, "x": 0, "intensity": 5.0, "probability": 0.1},
    ]
    pass_peaks = [
        {
            "y": 2,
            "x": 2,
            "intensity": 200.0,
            "probability": 0.9,
            "area_px": 4,
            "mean_intensity": 180.0,
            "rel_intensity": 1.5,
            "cell_id": 1,
        }
    ]
    fail_peaks = [
        {
            "y": 0,
            "x": 0,
            "intensity": 5.0,
            "probability": 0.1,
            "area_px": 3,
            "mean_intensity": 6.0,
            "rel_intensity": 0.8,
            "cell_id": 0,
        }
    ]
    from bactomate.gui.foci_filter_pipeline import populate_foci_preview_table

    populate_foci_preview_table(
        table,
        frame_rows=frame_rows,
        pass_peaks=pass_peaks,
        fail_peaks=fail_peaks,
    )
    fail_row = None
    for row_i in range(table.rowCount()):
        if table.item(row_i, 0) and table.item(row_i, 0).text() == "No":
            fail_row = row_i
            break
    assert fail_row is not None
    assert table.item(fail_row, 6).text() == "3"
    assert table.item(fail_row, 7).text() == "6"
    assert table.item(fail_row, 8).text() == "0.80"


def test_foci_table_header_double_click_sorts_numerically():
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QTableWidget

    from bactomate.gui.foci_filter_pipeline import (
        apply_foci_table_header_sort,
        populate_foci_preview_table,
    )

    table = QTableWidget()
    frame_rows = [
        {"y": 1, "x": 1, "intensity": 9.0, "probability": 0.2},
        {"y": 2, "x": 2, "intensity": 100.0, "probability": 0.9},
        {"y": 3, "x": 3, "intensity": 50.0, "probability": 0.5},
    ]
    pass_peaks = [{"y": r["y"], "x": r["x"], "intensity": r["intensity"], "probability": r["probability"]} for r in frame_rows]
    populate_foci_preview_table(table, frame_rows=frame_rows, pass_peaks=pass_peaks, fail_peaks=[])

    # Int. column = 4
    apply_foci_table_header_sort(table, 4)
    ints = [float(table.item(r, 4).text()) for r in range(table.rowCount())]
    assert ints == sorted(ints)

    apply_foci_table_header_sort(table, 4)
    ints_desc = [float(table.item(r, 4).text()) for r in range(table.rowCount())]
    assert ints_desc == sorted(ints, reverse=True)

    # Persists across rebuild
    populate_foci_preview_table(table, frame_rows=frame_rows, pass_peaks=pass_peaks, fail_peaks=[])
    ints_again = [float(table.item(r, 4).text()) for r in range(table.rowCount())]
    assert ints_again == sorted(ints, reverse=True)
    hdr = table.horizontalHeader()
    assert hdr is not None and hdr.isSortIndicatorShown()
    assert int(hdr.sortIndicatorSection()) == 4
    assert hdr.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder


def test_cell_filters_active_for_foci_only_size_and_border() -> None:
    assert not cell_filters_active_for_foci({})
    assert not cell_filters_active_for_foci({"min_area_um2": 0.0, "max_area_um2": None})
    assert cell_filters_active_for_foci({"min_area_um2": 0.5})
    assert cell_filters_active_for_foci({"exclude_border": True})


def test_foci_in_rejected_cell_passes_without_active_cell_filters() -> None:
    foci = np.zeros((6, 6), dtype=np.float64)
    foci[2, 2] = 200.0
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[2, 2] = 5
    frame_rows = [{"y": 2, "x": 2, "intensity": 200.0, "probability": 0.9}]
    settings = {
        "foci_prob_enabled": False,
        "foci_min_int_enabled": False,
        "foci_exclude_outside_cells": False,
        "foci_blob_threshold_frac": 0.8,
        "foci_blob_max_radius": 3,
        "foci_min_area_px_enabled": False,
    }
    passing, failing = compute_foci_pass_fail_for_frame(
        frame_rows,
        settings,
        mask_2d=mask,
        foci_plane=foci,
        accepted_cell_labels={1},
    )
    assert len(passing) == 0
    assert len(failing) == 1

    passing2, failing2 = compute_foci_pass_fail_for_frame(
        frame_rows,
        settings,
        mask_2d=mask,
        foci_plane=foci,
        accepted_cell_labels=None,
    )
    assert len(passing2) == 1
    assert len(failing2) == 0


def test_filter_foci_csv_rows_respects_accepted_cell_labels():
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[2, 2] = 1
    mask[0, 0] = 2
    rows = [
        {"y": 2, "x": 2, "intensity": 100.0, "probability": 0.9},
        {"y": 0, "x": 0, "intensity": 100.0, "probability": 0.9},
    ]
    out = filter_foci_csv_rows(
        rows,
        {"foci_exclude_outside_cells": False},
        mask_2d=mask,
        accepted_cell_labels={1},
    )
    assert len(out) == 1
    assert out[0]["y"] == 2 and out[0]["x"] == 2


def test_filter_foci_csv_rows_rejects_peak_adjacent_to_rejected_cell():
    """Peak on background next to rejected cell must not pass (uses nearest cell)."""
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[3:6, 3:6] = 4
    rows = [{"y": 2, "x": 4, "intensity": 100.0, "probability": 0.9}]
    out = filter_foci_csv_rows(
        rows,
        {"foci_exclude_outside_cells": False},
        mask_2d=mask,
        accepted_cell_labels={1},
    )
    assert out == []


def test_sync_foci_peaks_moves_rejected_cell_peak_to_fail():
    from bactomate.gui.foci_filter_pipeline import sync_foci_peaks_with_accepted_cells

    mask = np.zeros((10, 10), dtype=np.int32)
    mask[3:6, 3:6] = 4
    peak = {"y": 2, "x": 4, "intensity": 100.0}
    passing, failing = sync_foci_peaks_with_accepted_cells(
        [peak],
        [],
        mask_2d=mask,
        accepted_cell_labels={1},
        settings={},
    )
    assert passing == []
    assert len(failing) == 1
    assert any("rejected cell" in str(r).lower() for r in failing[0].get("fail_reasons", []))


def test_measure_foci_rows_metrics_never_drops_peaks():
    foci = np.zeros((6, 6), dtype=np.float64)
    foci[0, 0] = 5.0
    rows = [{"y": 0, "x": 0, "intensity": 5.0, "probability": 0.1}]
    metrics = measure_foci_rows_metrics(rows, foci, settings={"foci_blob_threshold_frac": 0.8})
    assert (0, 0) in metrics
    assert int(metrics[(0, 0)].get("area_px", 0)) >= 1
    assert float(metrics[(0, 0)].get("mean_intensity", 0)) >= 0


def test_compute_foci_fail_includes_metrics_and_reasons():
    foci = np.zeros((6, 6), dtype=np.float64)
    foci[2, 2] = 200.0
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[2, 2] = 1
    frame_rows = [
        {"y": 2, "x": 2, "intensity": 200.0, "probability": 0.9},
        {"y": 0, "x": 0, "intensity": 5.0, "probability": 0.1},
    ]
    settings = {
        "foci_prob_enabled": True,
        "foci_prob_thresh": 0.5,
        "foci_min_int_enabled": False,
        "foci_exclude_outside_cells": True,
        "foci_blob_threshold_frac": 0.8,
        "foci_blob_max_radius": 3,
        "foci_min_area_px_enabled": False,
    }
    passing, failing = compute_foci_pass_fail_for_frame(
        frame_rows, settings, mask_2d=mask, foci_plane=foci
    )
    assert len(passing) == 1
    assert len(failing) == 1
    fail = failing[0]
    assert fail["y"] == 0
    assert int(fail.get("area_px", 0)) >= 1
    assert "mean_intensity" in fail
    assert isinstance(fail.get("fail_reasons"), list)
    assert any("probability" in r.lower() for r in fail["fail_reasons"])


def test_compute_foci_pass_fail_for_frame():
    foci = np.zeros((6, 6), dtype=np.float64)
    foci[2, 2] = 200.0
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[2, 2] = 1
    frame_rows = [
        {"y": 2, "x": 2, "intensity": 200.0, "probability": 0.9},
        {"y": 0, "x": 0, "intensity": 5.0, "probability": 0.1},
    ]
    settings = {
        "foci_prob_enabled": True,
        "foci_prob_thresh": 0.5,
        "foci_min_int_enabled": False,
        "foci_exclude_outside_cells": True,
        "foci_blob_threshold_frac": 0.8,
        "foci_blob_max_radius": 3,
    }
    passing, failing = compute_foci_pass_fail_for_frame(
        frame_rows, settings, mask_2d=mask, foci_plane=foci
    )
    assert len(passing) == 1
    assert len(failing) == 1
    assert failing[0]["y"] == 0
    assert int(failing[0].get("area_px", 0) or 0) >= 1
    assert "mean_intensity" in failing[0]


def test_compute_foci_pass_fail_respects_min_area_px():
    foci = np.zeros((6, 6), dtype=np.float64)
    foci[2, 2] = 200.0
    foci[2, 3] = 180.0
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[2, 2] = 1
    mask[2, 3] = 1
    frame_rows = [
        {"y": 2, "x": 2, "intensity": 200.0, "probability": 0.9},
        {"y": 2, "x": 3, "intensity": 180.0, "probability": 0.9},
    ]
    settings = {
        "foci_prob_enabled": False,
        "foci_min_int_enabled": False,
        "foci_exclude_outside_cells": False,
        "foci_blob_threshold_frac": 0.9,
        "foci_blob_max_radius": 2,
        "foci_min_area_px_enabled": True,
        "foci_min_area_px": 4,
    }
    passing, failing = compute_foci_pass_fail_for_frame(
        frame_rows, settings, mask_2d=mask, foci_plane=foci
    )
    assert len(passing) + len(failing) == 2
    for p in passing:
        assert int(p.get("area_px", 0)) >= 4


def test_filter_foci_csv_rows_keeps_outside_when_exclude_disabled() -> None:
    mask = np.zeros((6, 6), dtype=np.int32)
    mask[2, 2] = 1
    rows = [
        {"y": 0, "x": 0, "intensity": 100.0, "probability": 0.9},
        {"y": 2, "x": 2, "intensity": 100.0, "probability": 0.9},
    ]
    out = filter_foci_csv_rows(
        rows,
        {"foci_exclude_outside_cells": False},
        mask_2d=mask,
        accepted_cell_labels={1},
    )
    keys = {(int(r["y"]), int(r["x"])) for r in out}
    assert (0, 0) in keys
    assert (2, 2) in keys


def test_foci_in_rejected_cell_fails_without_explicit_size_filter() -> None:
    """Morphology-only rejected cells must fail foci after Apply (no min_area required)."""
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[5, 5] = 1
    mask[15, 15] = 2
    foci_plane = np.zeros((20, 20), dtype=np.float64)
    foci_plane[5, 5] = 100.0
    foci_plane[15, 15] = 100.0
    rows = [
        {"y": 5, "x": 5, "intensity": 100.0, "probability": 0.9},
        {"y": 15, "x": 15, "intensity": 100.0, "probability": 0.9},
    ]
    settings = {}  # no explicit size/border foci filters
    passing, failing = compute_foci_pass_fail_for_frame(
        rows,
        settings,
        mask_2d=mask,
        foci_plane=foci_plane,
        accepted_cell_labels=accepted_cell_labels_for_foci({1}, cell_filter_applied=True),
    )
    pass_keys = {(int(p["y"]), int(p["x"])) for p in passing}
    fail_keys = {(int(p["y"]), int(p["x"])) for p in failing}
    assert (5, 5) in pass_keys
    assert (15, 15) in fail_keys
    assert not cell_filters_active_for_foci(settings)


def test_foci_failure_reason_outside_vs_rejected_cell() -> None:
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[15, 15] = 5
    outside_off = foci_filter_failure_reasons(
        {"y": 0, "x": 0, "intensity": 100.0},
        {"foci_exclude_outside_cells": False},
        mask_2d=mask,
        accepted_cell_labels={1},
    )
    assert "outside" not in " ".join(outside_off).lower()

    outside_on = foci_filter_failure_reasons(
        {"y": 0, "x": 0, "intensity": 100.0},
        {"foci_exclude_outside_clear": True},
        mask_2d=mask,
        accepted_cell_labels={1},
    )
    assert any("outside" in r.lower() for r in outside_on)

    edge_on = foci_filter_failure_reasons(
        {"y": 14, "x": 15, "intensity": 100.0},
        {"foci_exclude_outside_touching": True},
        mask_2d=mask,
        accepted_cell_labels={1},
    )
    assert any("touching" in r.lower() for r in edge_on)

    rejected = foci_filter_failure_reasons(
        {"y": 15, "x": 15, "intensity": 100.0},
        {},
        mask_2d=mask,
        accepted_cell_labels={1},
    )
    assert any("rejected" in r.lower() for r in rejected)
    assert "Outside cells" not in rejected


def test_foci_outside_relation_labels() -> None:
    from bactomate.gui.foci_filter_pipeline import foci_outside_relation_label

    assert foci_outside_relation_label("inside") == "inside"
    assert foci_outside_relation_label("outside_clear") == "outside"
    assert foci_outside_relation_label("outside_touching") == "edge"


def test_distribute_foci_column_widths_never_exceeds_avail() -> None:
    from bactomate.gui.foci_filter_pipeline import _distribute_foci_column_widths

    weights = [120, 80, 60, 50, 40, 35, 30, 28, 26, 24]
    avail = 280
    widths = _distribute_foci_column_widths(weights, avail)
    assert len(widths) == len(weights)
    assert sum(widths) == avail
    assert all(w >= 1 for w in widths)


def test_edge_foci_cell_column_shows_touching_cell_id() -> None:
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[15, 15] = 7
    assert classify_foci_cell_relation(mask, 14, 15) == "outside_touching"
    assert resolve_foci_cell_id_at_peak(mask, 14, 15) == 7
    assert (
        _foci_table_cell_id({"y": 14, "x": 15, "cell_id": 0}, mask) == "7"
    )
    assert _foci_table_cell_id({"y": 0, "x": 0, "cell_id": 0}, mask) == "—"
    assert (
        format_foci_cell_id_display(
            {"y": 0, "x": 0, "cell_id": 0},
            mask,
        )
        == "—"
    )


def test_classify_foci_cell_relation_inside_clear_and_edge() -> None:
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[15, 15] = 1
    assert classify_foci_cell_relation(mask, 15, 15) == "inside"
    assert classify_foci_cell_relation(mask, 0, 0) == "outside_clear"
    assert classify_foci_cell_relation(mask, 14, 15) == "outside_touching"


def test_filter_foci_csv_rows_outside_clear_vs_touching() -> None:
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[15, 15] = 1
    rows = [
        {"y": 0, "x": 0, "intensity": 10.0},
        {"y": 14, "x": 15, "intensity": 10.0},
        {"y": 15, "x": 15, "intensity": 10.0},
    ]
    clear_only = filter_foci_csv_rows(
        rows,
        {"foci_exclude_outside_clear": True, "foci_exclude_outside_touching": False},
        mask_2d=mask,
    )
    keys_clear = {(int(r["y"]), int(r["x"])) for r in clear_only}
    assert (0, 0) not in keys_clear
    assert (14, 15) in keys_clear
    assert (15, 15) in keys_clear

    touch_only = filter_foci_csv_rows(
        rows,
        {"foci_exclude_outside_clear": False, "foci_exclude_outside_touching": True},
        mask_2d=mask,
    )
    keys_touch = {(int(r["y"]), int(r["x"])) for r in touch_only}
    assert (0, 0) in keys_touch
    assert (14, 15) not in keys_touch
    assert (15, 15) in keys_touch


def test_foci_failure_reason_lists_multiple_causes() -> None:
    mask = np.zeros((6, 6), dtype=np.int32)
    reasons = foci_filter_failure_reasons(
        {"y": 0, "x": 0, "intensity": 50.0, "probability": 0.1},
        {
            "foci_prob_enabled": True,
            "foci_prob_thresh": 0.5,
            "foci_min_int_enabled": True,
            "foci_min_int": 80.0,
            "foci_exclude_outside_clear": True,
        },
        mask_2d=mask,
        accepted_cell_labels={1},
    )
    joined = " ".join(reasons).lower()
    assert "outside" in joined
    assert "probability" in joined
    assert "intensity" in joined
    assert len(reasons) >= 3


def test_exclusive_blob_metrics_disjoint_for_tight_cluster():
    """Close peaks on one bright patch must not share pixels; areas stay disjoint."""
    from bactomate.modules.foci_utils import (
        exclusive_focus_blob_masks,
        exclusive_focus_blob_metrics,
        get_focus_blob_mask,
    )

    foci = np.full((24, 24), 10.0, dtype=np.float64)
    # Flat bright island so independent growth claims the same pixels.
    foci[8:16, 8:16] = 100.0
    foci[11, 10] = 120.0
    foci[11, 13] = 119.0
    label = np.ones((24, 24), dtype=np.int32)
    peaks = [
        {"y": 11, "x": 10, "intensity": 120.0},
        {"y": 11, "x": 13, "intensity": 119.0},
    ]
    # Independent growth overlaps on this patch.
    independent = [
        get_focus_blob_mask(
            foci, int(p["y"]), int(p["x"]), threshold_frac=0.7, max_radius=6, labeled_mask=label
        )
        for p in peaks
    ]
    overlap = independent[0] & independent[1]
    assert int(np.sum(overlap)) > 0

    masks = exclusive_focus_blob_masks(
        foci, peaks, threshold_frac=0.7, max_radius=6, labeled_mask=label
    )
    assert len(masks) == 2
    assert int(np.sum(masks[0] & masks[1])) == 0

    measured = exclusive_focus_blob_metrics(
        foci, peaks, threshold_frac=0.7, max_radius=6, labeled_mask=label
    )
    areas = [int(size) for _p, _m, size, _mean in measured]
    assert all(a >= 1 for a in areas)
    assert sum(areas) == int(np.sum(masks[0] | masks[1]))

    settings = {"foci_blob_threshold_frac": 0.7, "foci_blob_max_radius": 6}
    refined = refine_foci_rows_with_blob(peaks, foci, label_plane=label, settings=settings)
    assert len(refined) == 2
    refined_areas = [int(r.get("area_px", 0)) for r in refined]
    assert sum(refined_areas) == sum(areas)


def test_exclusive_overlay_draws_n_peak_markers_in_cluster():
    """After exclusive contour draw, N peak rings remain for N clustered peaks."""
    from bactomate.gui.foci_overlay_draw import (
        draw_foci_blob_contours_on_rgb,
        draw_peak_rings_on_rgb,
    )

    h, w = 40, 40
    plane = np.zeros((h, w), dtype=np.float64)
    plane[15:22, 15:25] = 90.0
    coords = [(17, 17), (17, 20), (17, 23), (20, 18), (20, 21)]
    for y, x in coords:
        plane[y, x] = 130.0
    mask = np.ones((h, w), dtype=np.int32)
    peaks = [{"y": y, "x": x, "intensity": 130.0} for y, x in coords]
    settings = {"foci_blob_threshold_frac": 0.8, "foci_blob_max_radius": 5}
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    draw_foci_blob_contours_on_rgb(rgb, mask, peaks, plane, settings, (255, 165, 0))
    draw_peak_rings_on_rgb(
        rgb, peaks, (255, 165, 0), mask_shape=(h, w), radius=3, inner_gap=2
    )
    visible = 0
    for y, x in coords:
        patch = rgb[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4]
        if np.any(patch > 0):
            visible += 1
    assert visible == len(coords)

    metrics = measure_foci_rows_metrics(peaks, plane, label_plane=mask, settings=settings)
    assert len(metrics) == len(coords)
    areas = [int(metrics[(y, x)]["area_px"]) for y, x in coords]
    assert all(a >= 1 for a in areas)
    assert max(areas) < int(np.sum(plane >= 90))


def test_exclusive_overlay_draws_variable_contours_not_only_rings():
    """Exclusive overlay must paint irregular blob edges, not only identical peak rings."""
    from bactomate.gui.foci_overlay_draw import draw_foci_blob_contours_on_rgb

    h, w = 48, 48
    plane = np.full((h, w), 15.0, dtype=np.float64)
    yy, xx = np.ogrid[:h, :w]
    coords = [(20, 18), (20, 24), (26, 21)]
    for y, x in coords:
        plane += 110.0 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 2.2**2))
    mask = np.ones((h, w), dtype=np.int32)
    peaks = [{"y": y, "x": x, "intensity": float(plane[y, x])} for y, x in coords]
    settings = {"foci_blob_threshold_frac": 0.75, "foci_blob_max_radius": 6}
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    draw_foci_blob_contours_on_rgb(rgb, mask, peaks, plane, settings, (255, 200, 120))
    orange = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 100) & (rgb[:, :, 1] < 220)
    assert int(np.sum(orange)) > len(coords) * 4
    away = 0
    for y, x in coords:
        ring = orange.copy()
        ring[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = False
        away += int(np.sum(ring[max(0, y - 6) : y + 7, max(0, x - 6) : x + 7]))
    assert away > 0


def test_draw_peak_pixels_marks_each_spotiflow_center():
    """Red peak pixels remain distinct even when exclusive outlines touch."""
    from bactomate.gui.foci_overlay_draw import (
        draw_foci_blob_contours_on_rgb,
        draw_peak_pixels_on_rgb,
    )

    h, w = 32, 32
    plane = np.full((h, w), 10.0, dtype=np.float64)
    yy, xx = np.ogrid[:h, :w]
    coords = [(12, 12), (12, 15), (15, 13)]
    for y, x in coords:
        plane += 100.0 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.6**2))
    mask = np.ones((h, w), dtype=np.int32)
    peaks = [{"y": y, "x": x, "intensity": float(plane[y, x])} for y, x in coords]
    settings = {"foci_blob_threshold_frac": 0.8, "foci_blob_max_radius": 5}
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    draw_foci_blob_contours_on_rgb(rgb, mask, peaks, plane, settings, (255, 200, 120))
    draw_peak_pixels_on_rgb(rgb, peaks, (255, 30, 30), mask_shape=(h, w))
    for y, x in coords:
        assert int(rgb[y, x, 0]) == 255
        assert int(rgb[y, x, 1]) == 30
        assert int(rgb[y, x, 2]) == 30
