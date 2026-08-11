"""Tracking report workbook: Run info, raw, filtered sheets."""

import pandas as pd

from bactomate.gui.workers import TrackingWorker


def test_report_workbook_raw_and_filtered_sheets(tmp_path):
    worker = TrackingWorker(
        "mask.tif",
        None,
        None,
        "general_2d",
        "greedy",
        "cpu",
        filter_params={
            "pixel_size": 0.0,
            "time_per_frame": 0.0,
            "use_physical_units": False,
            "use_speed_physical": False,
            "min_length": 2,
        },
    )
    track_info = pd.DataFrame(
        {
            "DisplayID": ["C10", "T2"],
            "ID_Source": ["analysis", "track"],
            "Cell_ID": [10, 20],
            "TrackID": [1, 2],
            "trajectory_length": [5, 1],
            "duration_frames": [5, 1],
            "gap_frames": [0, 0],
            "duration_min": [0.0, 0.0],
            "first_frame": [0, 1],
            "last_frame": [4, 1],
            "displacement_px": [1.0, 0.0],
            "displacement_um": [1.0, 0.0],
            "path_length_px": [2.0, 0.0],
            "path_length_um": [2.0, 0.0],
            "mean_speed_px_frame": [0.4, 0.0],
            "mean_speed_um_min": [0.0, 0.0],
            "mean_speed_um_s": [0.0, 0.0],
            "straight_line_speed_px_frame": [0.2, 0.0],
            "straight_line_speed_um_min": [0.0, 0.0],
            "straight_line_speed_um_s": [0.0, 0.0],
            "straightness": [0.5, 0.0],
        }
    )
    out = tmp_path / "T7_report.xlsx"
    worker._write_report_workbook(
        out,
        {"track_information": track_info, "track_summary": pd.DataFrame()},
        run_metadata={"mask_file": "/data/mask.tif", "trackastra_model": "general_2d"},
        filter_params=worker.filter_params,
    )
    sheets = pd.read_excel(out, sheet_name=None)
    assert "Run info" in sheets
    assert "raw" in sheets
    assert "filtered" in sheets
    assert "track_summary" not in sheets
    assert list(sheets["raw"]["Pass"]) == ["Yes", "No"]
    assert len(sheets["filtered"]) == 1
    assert str(sheets["raw"].iloc[0]["DisplayID"]) == "C10"
    assert str(sheets["filtered"].iloc[0]["ID_Source"]) == "analysis"
    assert int(sheets["filtered"].iloc[0]["TrackID"]) == 1
