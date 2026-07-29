from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd


def test_write_run_info_sheet_is_first(tmp_path: Path) -> None:
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return

    from bactomate.gui.analysis_workbook import RUN_INFO_SHEET, write_run_info_sheet

    xlsx = tmp_path / "sample_analysis.xlsx"
    pd.DataFrame({"Frame": [0], "Number of Cells": [1]}).to_excel(
        xlsx, sheet_name="Summary", index=False
    )
    write_run_info_sheet(
        xlsx,
        {
            "Filtered mask file used": "/tmp/sample_masks.tif",
            "Analyze FI": "Yes",
            "Include foci from detection CSVs": "Yes",
        },
    )
    wb = openpyxl.load_workbook(xlsx)
    assert wb.sheetnames[0] == RUN_INFO_SHEET
    info = pd.read_excel(xlsx, sheet_name=RUN_INFO_SHEET)
    assert "Filtered mask file used" in info["Setting"].values
    wb.close()


def test_per_cell_foci_lists_match_ids() -> None:
    from bactomate.gui.analysis_workbook import _update_per_cell_foci_detail_columns

    per_cell = pd.DataFrame([{"Frame": 0, "Cell_ID": 1}])
    foci_table = [
        {
            "foci_id": 798,
            "cell_id": 1,
            "frame": 0,
            "channel_name": "GFP",
            "intensity": 5000.0,
            "mean_intensity": 5100.0,
            "size_px": 2,
            "in_cell": True,
        },
        {
            "foci_id": 806,
            "cell_id": 1,
            "frame": 0,
            "channel_name": "GFP",
            "intensity": 6000.0,
            "mean_intensity": 6100.0,
            "size_px": 3,
            "in_cell": True,
        },
    ]
    channel_rows = [("GFP", [])]
    out, added = _update_per_cell_foci_detail_columns(
        per_cell,
        channel_rows=channel_rows,
        foci_table=foci_table,
        frame_shift=0,
    )
    assert "GFP_foci_intensities" in added
    assert out.loc[0, "foci_ids_GFP"] == "798,806"
    assert out.loc[0, "GFP_foci_intensities"] == "5000, 6000"
    assert out.loc[0, "GFP_foci_mean_intensities"] == "5100, 6100"
    assert float(out.loc[0, "GFP_foci_mean_intensity"]) == 5600.0
    assert float(out.loc[0, "GFP_foci_total_intensity"]) == 11000.0
    assert float(out.loc[0, "GFP_foci_max_intensity"]) == 6000.0


def test_apply_foci_summary_columns_single_when_all_in_cell() -> None:
    from bactomate.gui.analysis_workbook import apply_foci_summary_columns

    summary = pd.DataFrame([{"Frame": 0, "Number of Cells": 10}])
    summary = apply_foci_summary_columns(
        summary,
        label="GFP",
        totals_by_frame={0: 832},
        within_by_frame={0: 832},
    )
    assert "Total Foci Count (GFP)" in summary.columns
    assert int(summary.loc[0, "Total Foci Count (GFP)"]) == 832
    assert "Total Foci Count within Cells (GFP)" not in summary.columns
    assert "Foci outside cells (GFP)" not in summary.columns


def test_apply_foci_summary_columns_adds_outside_when_csv_has_extras() -> None:
    from bactomate.gui.analysis_workbook import apply_foci_summary_columns

    summary = pd.DataFrame([{"Frame": 0}])
    summary = apply_foci_summary_columns(
        summary,
        label="GFP",
        totals_by_frame={0: 100},
        within_by_frame={0: 80},
    )
    assert int(summary.loc[0, "Total Foci Count (GFP)"]) == 80
    assert int(summary.loc[0, "Foci outside cells (GFP)"]) == 20


def test_outcome_to_manifest_lists_csv_paths() -> None:
    from bactomate.gui.analysis_workbook import (
        FociChannelArtifact,
        FociIntegrationOutcome,
        outcome_to_manifest_entries,
    )

    outcome = FociIntegrationOutcome(
        performed=True,
        channels=[
            FociChannelArtifact(
                channel_label="GFP",
                foci_stack=Path("/data/sample_foci_GFP.tif"),
                csv_path=Path("/data/_seg/foci_filtered/sample_foci_filtered_GFP.csv"),
                csv_folder="foci_filtered",
            )
        ],
        in_cell_foci=12,
        per_cell_columns=["GFP_foci_count", "foci_ids_GFP"],
    )
    manifest = outcome_to_manifest_entries(outcome)
    assert manifest["Foci analysis performed"] == "Yes"
    assert manifest["Foci channels"] == "GFP"
    assert "foci_filtered" in manifest["Foci CSV files used"]
    assert "GFP_foci_count" in manifest["Per-Cell foci columns"]


def test_read_foci_csv_rows_preserves_blob_columns(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import _read_foci_csv_rows

    csv_path = tmp_path / "foci.csv"
    csv_path.write_text(
        "frame,y,x,intensity,probability,area_px,mean_intensity,rel_intensity,cell_id\n"
        "0,10,20,100.0,0.9,7,95.0,1.2,3\n",
        encoding="utf-8",
    )
    rows = _read_foci_csv_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["size_px"] == 7
    assert float(rows[0]["mean_intensity"]) == 95.0
    assert rows[0]["cell_id"] == 3


def test_refresh_workbook_channel_intensities_keeps_existing_columns_when_no_updates(tmp_path: Path) -> None:
    import numpy as np
    import tifffile

    from bactomate.gui.analysis_workbook import refresh_workbook_channel_intensities

    mask_path = tmp_path / "sample_mask_filtered.tif"
    tifffile.imwrite(mask_path, np.ones((1, 4, 4), dtype=np.uint16))
    analysis_path = tmp_path / "sample_analysis.xlsx"
    with pd.ExcelWriter(analysis_path, engine="openpyxl") as writer:
        pd.DataFrame({"Frame": [1], "Cell_ID": [1], "Area (µm²)": [2.5]}).to_excel(
            writer, sheet_name="Per-Cell Data", index=False
        )

    class _Tab:
        pass

    per_cell, logs, cols = refresh_workbook_channel_intensities(_Tab(), mask_path, analysis_path)
    assert logs == []
    assert cols == []
    assert per_cell is not None
    assert list(per_cell.columns) == ["Frame", "Cell_ID", "Area (µm²)"]

    written = pd.read_excel(analysis_path, sheet_name="Per-Cell Data")
    assert list(written.columns) == ["Frame", "Cell_ID", "Area (µm²)"]


def test_integrate_foci_csv_into_workbook_writes_mask_file_and_focus_area(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import integrate_foci_csv_into_workbook

    mask_path = tmp_path / "sample_mask_filtered.tif"
    mask_path.write_bytes(b"stub")
    analysis_path = tmp_path / "sample_analysis.xlsx"
    with pd.ExcelWriter(analysis_path, engine="openpyxl") as writer:
        pd.DataFrame({"Frame": [0], "Number of Cells": [1]}).to_excel(
            writer, sheet_name="Summary", index=False
        )
        pd.DataFrame({"Frame": [0], "Cell_ID": [1]}).to_excel(
            writer, sheet_name="Per-Cell Data", index=False
        )

    class _Spin:
        def value(self) -> float:
            return 0.5

    class _Tab:
        pixel_size = _Spin()

    channel_rows = [
        (
            "GFP",
            [
                {
                    "frame": 0,
                    "cell_id": 1,
                    "channel_name": "GFP",
                    "y": 10,
                    "x": 20,
                    "intensity": 100.0,
                    "size_px": 4,
                    "mean_intensity": 80.0,
                    "rel_intensity": 1.2,
                    "in_cell": True,
                }
            ],
        )
    ]

    def _fake_refresh(_tab, _mask_path, _analysis_path, *, artifacts=None, frame_shift=0, per_cell=None, write_workbook=True):
        return per_cell, [], []

    with patch(
        "bactomate.gui.analysis_workbook.collect_channel_rows_for_mask",
        return_value=(channel_rows, {}, [], []),
    ), patch(
        "bactomate.gui.tabs._analyze_masks_colocalization._persist_workbook_count_updates",
        return_value=None,
    ), patch(
        "bactomate.gui.tabs._analyze_masks_colocalization._infer_frame_shift",
        return_value=0,
    ), patch(
        "bactomate.gui.analysis_workbook.refresh_workbook_channel_intensities",
        side_effect=_fake_refresh,
    ):
        outcome = integrate_foci_csv_into_workbook(_Tab(), mask_path, analysis_path)

    assert outcome.performed is True
    per_cell = pd.read_excel(analysis_path, sheet_name="Per-Cell Data")
    assert "Mask file" not in per_cell.columns
    foci = pd.read_excel(analysis_path, sheet_name="Foci")
    assert "Mask file" not in foci.columns
    assert float(foci.loc[0, "Foci Area (µm²)"]) == 1.0
