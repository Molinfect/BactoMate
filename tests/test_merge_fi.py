"""
Phase 1.3: Minimal tests for merge phase+FI into one Excel (Analyze Masks).
Tests _merge_fi_channels_into_one_excel: phase + FI Per-Cell Data merged into one file with Summary + Per-Cell Data.
Run with: pytest tests/test_merge_fi.py -v
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    pytest = None

try:
    from PyQt6.QtWidgets import QApplication
    from bactimate.gui.tabs.analyze_masks import AnalyzeMasksTab
    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


def test_merge_phase_and_fi_produces_one_excel():
    """Given one phase and one FI Excel (same stem), merge produces one Excel with Summary + Per-Cell Data and FI column."""
    if not QT_AVAILABLE:
        if pytest:
            pytest.skip("PyQt6 or AnalyzeMasksTab not available")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stem = "m"
        analysis_dir = tmp / "_analysis"
        analysis_dir.mkdir(parents=True)
        mask_path = tmp / "m_masks.tif"
        mask_path.touch()
        fi_path = tmp / "m_fi_mScarlet.tif"
        fi_path.touch()
        # Phase Excel: Per-Cell with Frame, Cell_ID, morphology
        phase_path = analysis_dir / "m_analysis_phase.xlsx"
        df_phase = pd.DataFrame({
            "Frame": [1, 1, 2],
            "Cell_ID": [1, 2, 1],
            "Area (µm²)": [10.0, 11.0, 12.0],
        })
        df_summary = pd.DataFrame({
            "Frame": [1, 2],
            "Total Cell Area (µm²)": [21.0, 12.0],
            "Total Cell Area (Relative)": [1.0, 1.0],
            "Number of Cells": [2, 1],
        })
        with pd.ExcelWriter(phase_path, engine="openpyxl") as w:
            df_summary.to_excel(w, sheet_name="Summary", index=False)
            df_phase.to_excel(w, sheet_name="Per-Cell Data", index=False)
        # FI Excel: same Frame/Cell_ID + Mean FI Intensity
        fi_path_excel = analysis_dir / "m_analysis_mScarlet.xlsx"
        df_fi = pd.DataFrame({
            "Frame": [1, 1, 2],
            "Cell_ID": [1, 2, 1],
            "Mean FI Intensity": [100.0, 110.0, 120.0],
        })
        with pd.ExcelWriter(fi_path_excel, engine="openpyxl") as w:
            df_summary.to_excel(w, sheet_name="Summary", index=False)
            df_fi.to_excel(w, sheet_name="Per-Cell Data", index=False)
        merged_path = analysis_dir / "m_analysis.xlsx"
        # Tab with mock parent
        tab = AnalyzeMasksTab(parent=None)
        tab.parent_window = type("Mock", (), {"log_output": lambda self, msg: None})()
        tab.folder_fi_tasks = [
            (mask_path, None, phase_path),
            (mask_path, fi_path, fi_path_excel),
        ]
        tab.folder_results = [True, True]
        tab.folder_output_paths = [str(phase_path), str(fi_path_excel)]
        tab.folder_phase_per_mask = {}
        tab._merge_fi_channels_into_one_excel()
        assert merged_path.exists()
        xl = pd.ExcelFile(merged_path)
        assert "Summary" in xl.sheet_names
        assert "Per-Cell Data" in xl.sheet_names
        per_cell = pd.read_excel(merged_path, sheet_name="Per-Cell Data")
        assert "Frame" in per_cell.columns
        assert "Cell_ID" in per_cell.columns
        # Should have morphology (Area) and FI (Mean FI Intensity (mScarlet) or similar)
        has_fi_col = any("FI" in c or "Mean" in c for c in per_cell.columns)
        assert has_fi_col
        assert len(per_cell) == 3


if __name__ == "__main__":
    test_merge_phase_and_fi_produces_one_excel()
    print("All merge_fi tests passed.")
