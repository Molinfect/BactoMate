"""Preview, canonical reports and legacy exports share calibrated track metrics."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bactomate.gui.tabs.tracking import TrackingTab
from bactomate.gui.tracking_filters import compute_track_metrics
from bactomate.gui.workers import TrackingWorker


def test_mean_speed_um_s():
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "frame": [0, 1, 2],
            "y": [0.0, 0.0, 0.0],
            "x": [0.0, 10.0, 20.0],
        }
    )
    # path = 20 px, pixel 0.325 µm/px → 6.5 µm; 3 frames at 1 s/frame → 2 s total
    metrics = TrackingTab._compute_track_metrics(tracks, pixel_size_um=0.325, time_per_frame_min=1.0 / 60.0)
    assert len(metrics) == 1
    assert abs(metrics.iloc[0]["mean_speed_um_s"] - 3.25) < 0.01


@pytest.mark.parametrize(
    "frames, xs, duration, gaps, speed, straight_speed, straightness",
    [
        ([0, 1, 2], [0, 10, 20], 3, 0, 10, 10, 1),
        ([2, 0], [20, 0], 3, 1, 10, 10, 1),
        ([0, 1, 2], [0, 10, 0], 3, 0, 10, 0, 0),
        ([5], [10], 1, 0, 0, 0, 0),
        ([0, 1], [10, 10], 2, 0, 0, 0, 0),
    ],
)
def test_preview_and_worker_metrics_match(frames, xs, duration, gaps, speed, straight_speed, straightness):
    tracks = pd.DataFrame({"track_id": 42.0, "frame": frames, "y": 0.0, "x": xs, "Cell_ID": 901})
    original = tracks.copy(deep=True)
    preview = TrackingTab._compute_track_metrics(tracks, 0.325, 1 / 60)
    worker = TrackingWorker("mask.tif", None, None, "general_2d", "greedy", "cpu")
    exported = worker._compute_track_metrics_table(tracks, 0.325, 1 / 60)

    pd.testing.assert_frame_equal(preview, exported)
    pd.testing.assert_frame_equal(tracks, original)
    m = preview.iloc[0]
    assert m["track_id"] == 42
    assert m["trajectory_length"] == len(frames)
    assert m["duration_frames"] == duration
    assert m["duration_min"] == pytest.approx(duration / 60)
    assert m["gap_frames"] == gaps
    assert m["mean_speed_px_frame"] == speed
    assert m["mean_speed_um_s"] == pytest.approx(speed * 0.325)
    assert m["straight_line_speed_px_frame"] == straight_speed
    assert m["straight_line_speed_um_s"] == pytest.approx(straight_speed * 0.325)
    assert m["straightness"] == straightness


@pytest.mark.parametrize("pixel_size, time_per_frame", [(0, 0), (0.325, 0), (0, 1 / 60)])
def test_missing_calibration_preserves_pixel_fallback(pixel_size, time_per_frame):
    tracks = pd.DataFrame({"track_id": [7, 7], "frame": [1, 3], "y": [0, 0], "x": [0, 20]})
    metrics = compute_track_metrics(tracks, pixel_size, time_per_frame).iloc[0]
    assert metrics["mean_speed_px_frame"] == 10
    assert metrics["mean_speed_um_s"] == 10
    assert metrics["straight_line_speed_um_s"] == 10
    assert metrics["path_length_um"] == pytest.approx(20 * pixel_size if pixel_size > 0 else 20)


@pytest.mark.parametrize("tracks", [None, pd.DataFrame(), pd.DataFrame({"track_id": [1]}), pd.DataFrame({"frame": [1]})])
def test_empty_or_missing_required_columns_have_stable_schema(tracks):
    worker = TrackingWorker("mask.tif", None, None, "general_2d", "greedy", "cpu")
    preview = TrackingTab._compute_track_metrics(tracks, 0, 0)
    exported = worker._compute_track_metrics_table(tracks, 0, 0)
    assert preview.empty
    assert "track_id" in preview.columns
    assert "mean_speed_um_s" in preview.columns
    pd.testing.assert_frame_equal(preview, exported)


def test_missing_coordinates_keep_duration_and_zero_motion():
    tracks = pd.DataFrame({"track_id": [2, 2, 8], "frame": [0, 2, 4]})
    metrics = compute_track_metrics(tracks, 0, 0).set_index("track_id")
    assert metrics.index.tolist() == [2, 8]
    assert metrics["duration_frames"].tolist() == [3, 1]
    assert metrics["path_length_px"].tolist() == [0, 0]


@pytest.mark.parametrize(
    "threshold, physical, cutoff",
    [(name, physical, 3.25 if physical else 10)
     for name in ("min_speed", "min_displacement", "min_path_length") for physical in (True, False)],
)
def test_report_filters_do_not_round_before_threshold_comparison(threshold, physical, cutoff):
    tracks = pd.DataFrame({"track_id": [7, 7], "frame": [0, 1], "y": [0, 0], "x": [0, 9.9999999]})
    params = {"use_physical_units": physical, "use_speed_physical": physical, threshold: cutoff}
    preview = TrackingTab._compute_track_metrics(tracks, 0.325, 1 / 60)
    assert TrackingTab._apply_track_filters(preview, params) == set()
    worker = TrackingWorker("mask.tif", None, None, "general_2d", "greedy", "cpu")
    worker._cal_ps, worker._cal_tp = 0.325, 1 / 60
    tables = worker._build_trajectory_tables(tracks, np.zeros((2, 1, 11), dtype=np.uint16))
    raw, filtered = worker._tracks_with_pass_column(tables["track_information"], params)
    assert raw["Pass"].tolist() == ["No"]
    assert filtered.empty


def test_report_straightness_filter_keeps_calculation_precision():
    tracks = pd.DataFrame({"track_id": [7, 7, 7], "frame": [0, 1, 2], "y": [0, 0.0001, 0], "x": [0, 5, 10]})
    params = {"min_straightness": 1.0}
    preview = compute_track_metrics(tracks, 0, 0)
    assert TrackingTab._apply_track_filters(preview, params) == set()
    worker = TrackingWorker("mask.tif", None, None, "general_2d", "greedy", "cpu")
    info = worker._build_track_information_table(tracks, preview)
    raw, filtered = worker._tracks_with_pass_column(info, params)
    assert raw["Pass"].tolist() == ["No"]
    assert filtered.empty


def test_preview_report_and_legacy_export_agree_without_changing_source_files(tmp_path, qapp):
    tracks = pd.DataFrame({
        "track_id": [7, 7, 7, 99], "frame": [0, 1, 2, 1],
        "y": [0, 0, 0, 0], "x": [0, 10, 20, 0],
        "Cell_ID": [901, 901, 901, 902], "Cell_ID_Source": ["analysis"] * 4,
    })
    csv_path = tmp_path / "sample_tracks.csv"
    tracks.to_csv(csv_path, index=False)
    # A historical report can retain its old speed; only explicit recomputation changes it.
    old_report = tmp_path / "sample_report.xlsx"
    pd.DataFrame({"TrackID": [7], "Cell_ID": [901], "mean_speed_um_s": [6.5 / 3]}).to_excel(
        old_report, sheet_name="raw", index=False,
    )
    snapshots = {path: path.read_bytes() for path in (csv_path, old_report)}
    params = {"pixel_size": 0.325, "time_per_frame": 1 / 60, "use_physical_units": True,
              "use_speed_physical": True, "min_speed": 3.0}
    preview = TrackingTab._compute_track_metrics(pd.read_csv(csv_path), 0.325, 1 / 60)
    preview_ids = TrackingTab._apply_track_filters(preview, params)
    assert preview_ids == {7}

    worker = TrackingWorker("mask.tif", None, None, "general_2d", "greedy", "cpu", filter_params=params)
    worker._cal_ps, worker._cal_tp = 0.325, 1 / 60
    tables = worker._build_trajectory_tables(pd.read_csv(csv_path), np.zeros((3, 1, 21), dtype=np.uint16))
    report = tmp_path / "recomputed_report.xlsx"
    worker._write_report_workbook(report, tables, filter_params=params)
    raw = pd.read_excel(report, sheet_name="raw")
    filtered = pd.read_excel(report, sheet_name="filtered")
    assert set(raw.loc[raw["Pass"] == "Yes", "TrackID"]) == preview_ids
    assert set(filtered["TrackID"]) == preview_ids
    assert filtered.iloc[0]["mean_speed_um_s"] == pytest.approx(3.25)
    assert tables["track_summary"].set_index("track_id").loc[7, "mean_speed_um_s"] == pytest.approx(3.25)
    run_info = pd.read_excel(report, sheet_name="Run info").set_index("parameter")["value"]
    assert run_info["track_speed_time_basis"] == "elapsed_frame_intervals"
    tables_dir, files = worker._write_canonical_tables(tmp_path / "recomputed", tables)
    metadata = json.loads((tables_dir / files["metadata"]).read_text())
    assert metadata["track_speed_time_basis"] == "elapsed_frame_intervals"
    summary_path = tables_dir / files["track_summary"]
    summary = pd.read_parquet(summary_path) if summary_path.suffix == ".parquet" else pd.read_csv(summary_path)
    assert summary.set_index("track_id").loc[7, "mean_speed_um_s"] == pytest.approx(3.25)

    tab = SimpleNamespace(
        parent_window=SimpleNamespace(log_output=lambda _message: None),
        _get_filter_params=lambda: params,
        _compute_track_metrics=TrackingTab._compute_track_metrics,
        _apply_track_filters=TrackingTab._apply_track_filters,
    )
    assert TrackingTab._export_filtered_tracks_for_paths(tab, [tmp_path / "sample_trajectories.tif"], silent=True) == 1
    legacy = pd.read_excel(tmp_path / "sample_filtered_tracks.xlsx", sheet_name="Filtered tracks")
    assert set(legacy["TrackID"]) == preview_ids
    assert legacy.iloc[0]["mean_speed_um_s"] == pytest.approx(3.25)
    for frame in (filtered, legacy):
        assert frame.iloc[0]["Cell_ID"] == 901
        assert frame.iloc[0]["ID_Source"] == "analysis"
        assert frame.iloc[0]["first_frame"] == 0
        assert frame.iloc[0]["last_frame"] == 2
        assert frame.iloc[0]["duration_frames"] == 3
    stored_raw, stored_filtered = worker._tracks_with_pass_column(pd.read_excel(old_report, sheet_name="raw"), params)
    assert stored_raw.iloc[0]["mean_speed_um_s"] == pytest.approx(6.5 / 3)
    assert stored_raw["Pass"].tolist() == ["No"]
    assert stored_filtered.empty
    for path, contents in snapshots.items():
        assert path.read_bytes() == contents
