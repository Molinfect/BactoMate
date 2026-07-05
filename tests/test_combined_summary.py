"""
Tests for combined analysis summary (Analyze Masks folder batch).
Run with: pytest tests/test_combined_summary.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest_import = None
try:
    import pytest

    pytest_import = pytest
except ImportError:
    pass

try:
    from PyQt6.QtWidgets import QApplication
    from bactomate.gui.tabs.analyze_masks import AnalyzeTab

    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


def _write_analysis_workbook(
    path: Path,
    *,
    sample_frames: list[int],
    area: list[float],
    num_cells: list[int],
    per_cell: pd.DataFrame | None = None,
    foci: pd.DataFrame | None = None,
    extra_summary_cols: dict | None = None,
) -> None:
    summary = pd.DataFrame(
        {
            "Frame": sample_frames,
            "Total Cell Area (µm²)": area,
            "Total Cell Area (Relative)": [a / area[0] for a in area],
            "Number of Cells": num_cells,
        }
    )
    if extra_summary_cols:
        for key, values in extra_summary_cols.items():
            summary[key] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        if per_cell is not None:
            per_cell.to_excel(writer, sheet_name="Per-Cell Data", index=False)
        if foci is not None:
            foci.to_excel(writer, sheet_name="Foci", index=False)


def test_combined_summary_long_and_wide_include_all_metrics(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import (
        COMBINED_FOCI_SHEET,
        COMBINED_PER_CELL_SHEET,
        COMBINED_SUMMARY_SHEET_LONG,
        COMBINED_SUMMARY_SHEET_WIDE,
        MASTER_PER_CELL_SAMPLE_COLUMN,
        write_combined_analysis_summary,
    )

    p1 = tmp_path / "file1_analysis.xlsx"
    p2 = tmp_path / "file2_analysis.xlsx"
    _write_analysis_workbook(
        p1,
        sample_frames=[1, 2],
        area=[100.0, 110.0],
        num_cells=[10, 11],
        extra_summary_cols={"Total Foci": [5, 6], "Mean Foci per Cell": [0.5, 0.55]},
        per_cell=pd.DataFrame({"Frame": [1], "Cell_ID": [1], "Number of Foci": [2]}),
        foci=pd.DataFrame({"foci_id": [1], "cell_id": [1], "frame": [1]}),
    )
    _write_analysis_workbook(
        p2,
        sample_frames=[1, 2],
        area=[200.0, 210.0],
        num_cells=[20, 21],
        extra_summary_cols={"Total Foci": [8, 9], "Mean Foci per Cell": [0.4, 0.43]},
        per_cell=pd.DataFrame({"Frame": [1], "Cell_ID": [2], "Number of Foci": [3]}),
    )

    out_path = tmp_path / "folder_combined_summary.xlsx"
    assert write_combined_analysis_summary([p1, p2], out_path) is True
    assert out_path.is_file()

    with pd.ExcelFile(out_path) as xl:
        assert COMBINED_SUMMARY_SHEET_LONG in xl.sheet_names
        assert COMBINED_SUMMARY_SHEET_WIDE in xl.sheet_names
        assert COMBINED_PER_CELL_SHEET in xl.sheet_names
        assert COMBINED_FOCI_SHEET in xl.sheet_names
        long_df = pd.read_excel(xl, sheet_name=COMBINED_SUMMARY_SHEET_LONG)
        wide_df = pd.read_excel(xl, sheet_name=COMBINED_SUMMARY_SHEET_WIDE)
        per_cell = pd.read_excel(xl, sheet_name=COMBINED_PER_CELL_SHEET)
        foci = pd.read_excel(xl, sheet_name=COMBINED_FOCI_SHEET)

    assert len(long_df) == 4
    assert set(long_df[MASTER_PER_CELL_SAMPLE_COLUMN]) == {"file1", "file2"}
    assert "Total Foci" in long_df.columns
    assert "Mean Foci per Cell" in long_df.columns

    assert "file1_Total Foci" in wide_df.columns
    assert "file2_Mean Foci per Cell" in wide_df.columns
    assert "file1_Total Cell Area (Relative)" in wide_df.columns

    assert len(per_cell) == 2
    assert len(foci) == 1


def test_combined_summary_empty_paths_returns_false(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import write_combined_analysis_summary

    out_path = tmp_path / "out.xlsx"
    assert write_combined_analysis_summary([], out_path) is False
    assert not out_path.exists()


def test_combined_summary_duplicate_frames_no_error(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import COMBINED_SUMMARY_SHEET_WIDE, write_combined_analysis_summary

    p1 = tmp_path / "dup_analysis.xlsx"
    summary = pd.DataFrame(
        {
            "Frame": [1, 1, 2],
            "Total Cell Area (µm²)": [100.0, 100.0, 110.0],
            "Total Cell Area (Relative)": [1.0, 1.0, 1.1],
            "Number of Cells": [10, 10, 11],
        }
    )
    with pd.ExcelWriter(p1, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)

    out_path = tmp_path / "combined.xlsx"
    assert write_combined_analysis_summary([p1], out_path) is True
    wide = pd.read_excel(out_path, sheet_name=COMBINED_SUMMARY_SHEET_WIDE)
    assert len(wide) == 2
    assert list(wide["Frame"]) == [1, 2]


def test_analyze_tab_wrapper_delegates_to_workbook(tmp_path: Path) -> None:
    if not QT_AVAILABLE:
        if pytest_import:
            pytest_import.skip("PyQt6 or AnalyzeMasksTab not available")
        return
    QApplication.instance() or QApplication(sys.argv)

    p1 = tmp_path / "alpha_analysis.xlsx"
    _write_analysis_workbook(
        p1,
        sample_frames=[1],
        area=[50.0],
        num_cells=[3],
    )
    out_path = tmp_path / "combined_summary.xlsx"
    tab = AnalyzeTab(parent=None)
    tab.parent_window = None
    ok = tab._build_combined_summary_excel([str(p1)], str(out_path), log_callback=None)
    assert ok is True
    assert out_path.exists()
