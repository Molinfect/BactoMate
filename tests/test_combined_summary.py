"""
Phase 1.3: Minimal tests for combined summary build (Analyze Masks).
Tests _build_combined_summary_excel: given Excel paths with Summary sheets, builds wide table and writes output.
Run with: pytest tests/test_combined_summary.py -v
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Qt required for AnalyzeMasksTab
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


def _make_summary_excel(path: Path, file_stem: str, frames: list, area: list, num_cells: list) -> None:
    """Write a minimal Summary sheet to path (Frame, Total Cell Area, Number of Cells)."""
    df = pd.DataFrame({
        "Frame": frames,
        "Total Cell Area (µm²)": area,
        "Total Cell Area (Relative)": [1.0] * len(frames),
        "Number of Cells": num_cells,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Summary", index=False)


def test_combined_summary_builds_wide_sheet():
    """Given 2 Excel files with Summary sheets, _build_combined_summary_excel produces output with wide sheet and expected columns."""
    if not QT_AVAILABLE:
        if pytest_import:
            pytest_import.skip("PyQt6 or AnalyzeMasksTab not available")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p1 = tmp / "file1_analysis.xlsx"
        p2 = tmp / "file2_analysis.xlsx"
        _make_summary_excel(p1, "file1", [1, 2, 3], [100.0, 110.0, 120.0], [10, 11, 12])
        _make_summary_excel(p2, "file2", [1, 2, 3], [200.0, 210.0, 220.0], [20, 21, 22])
        out_path = tmp / "combined_summary.xlsx"
        tab = AnalyzeTab(parent=None)
        tab.parent_window = None
        ok = tab._build_combined_summary_excel(
            [str(p1), str(p2)], str(out_path), match_log=None, log_callback=None
        )
        assert ok is True
        assert out_path.exists()
        with pd.ExcelFile(out_path) as xl:
            assert "Combined Summary (wide)" in xl.sheet_names
            wide = pd.read_excel(xl, sheet_name="Combined Summary (wide)")
        assert "Frame" in wide.columns
        assert list(wide["Frame"]) == [1, 2, 3]
        # At least one file's columns present (e.g. file1_Total Cell Area or file2_Number of Cells)
        has_file_col = any("file1_" in c or "file2_" in c for c in wide.columns)
        assert has_file_col


def test_combined_summary_empty_paths_returns_false():
    """Empty excel_paths returns False."""
    if not QT_AVAILABLE:
        if pytest_import:
            pytest_import.skip("PyQt6 or AnalyzeMasksTab not available")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.xlsx"
        tab = AnalyzeTab(parent=None)
        tab.parent_window = None
        ok = tab._build_combined_summary_excel([], str(out_path), log_callback=None)
        assert ok is False
        assert not out_path.exists()


def test_combined_summary_duplicate_frames_no_error():
    """Duplicate Frame rows per file are handled (drop_duplicates) and build succeeds."""
    if not QT_AVAILABLE:
        if pytest_import:
            pytest_import.skip("PyQt6 or AnalyzeMasksTab not available")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p1 = tmp / "dup_analysis.xlsx"
        # Two rows for frame 1 (duplicate)
        df = pd.DataFrame({
            "Frame": [1, 1, 2],
            "Total Cell Area (µm²)": [100.0, 100.0, 110.0],
            "Total Cell Area (Relative)": [1.0, 1.0, 1.0],
            "Number of Cells": [10, 10, 11],
        })
        with pd.ExcelWriter(p1, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Summary", index=False)
        out_path = tmp / "combined.xlsx"
        tab = AnalyzeTab(parent=None)
        tab.parent_window = None
        ok = tab._build_combined_summary_excel([str(p1)], str(out_path), log_callback=None)
        assert ok is True
        wide = pd.read_excel(out_path, sheet_name="Combined Summary (wide)")
        assert len(wide) == 2  # frames 1 and 2
        assert list(wide["Frame"]) == [1, 2]


if __name__ == "__main__":
    test_combined_summary_builds_wide_sheet()
    test_combined_summary_empty_paths_returns_false()
    test_combined_summary_duplicate_frames_no_error()
    print("All combined_summary tests passed.")
