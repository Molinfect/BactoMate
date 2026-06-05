"""Tests for shared foci CSV filtering and blob refinement pipeline."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from bactomate.gui.foci_filter_pipeline import (
    accepted_cell_labels_for_foci,
    cell_filters_active_for_foci,
    classify_foci_cell_relation,
    format_foci_cell_id_display,
    resolve_foci_cell_id_at_peak,
    _foci_table_cell_id,
    compute_foci_pass_fail_for_frame,
    filter_foci_csv_rows,
    foci_filter_failure_reasons,
    measure_foci_rows_metrics,
    refine_foci_rows_with_blob,
    write_filtered_foci_csv,
)


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
    from PyQt6.QtWidgets import QApplication, QTableWidget

    QApplication.instance() or QApplication([])

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
