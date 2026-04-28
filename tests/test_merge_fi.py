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
    from bactimate.gui.tabs.analyze_masks import AnalyzeTab
    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


def test_merge_phase_and_fi_produces_one_excel():
    """Given one phase and one FI Excel (same stem), merge produces one workbook with Summary, Per-Cell Data, and Foci channel sheet."""
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
        # FI Excel: same Frame/Cell_ID + Mean FI Intensity + foci columns + Foci sheet
        fi_path_excel = analysis_dir / "m_analysis_mScarlet.xlsx"
        df_fi = pd.DataFrame({
            "Frame": [1, 1, 2],
            "Cell_ID": [1, 2, 1],
            "Mean FI Intensity": [100.0, 110.0, 120.0],
            "mScarlet_foci_count": [1, 0, 1],
            "foci_ids_mScarlet": ["1", "", "2"],
        })
        df_foci = pd.DataFrame({
            "foci_id": [1, 2],
            "cell_id": [1, 1],
            "frame": [0, 1],
            "channel_name": ["mScarlet", "mScarlet"],
            "y": [10, 12],
            "x": [10, 15],
            "intensity": [250, 270],
            "in_cell": [True, True],
        })
        with pd.ExcelWriter(fi_path_excel, engine="openpyxl") as w:
            df_summary.to_excel(w, sheet_name="Summary", index=False)
            df_fi.to_excel(w, sheet_name="Per-Cell Data", index=False)
            df_foci.to_excel(w, sheet_name="Foci", index=False)
        merged_path = analysis_dir / "m_analysis.xlsx"
        foci_book_path = analysis_dir / "m_foci_analysis.xlsx"
        # Tab with mock parent
        tab = AnalyzeTab(parent=None)
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
        assert "Foci [mScarlet]" in xl.sheet_names
        per_cell = pd.read_excel(merged_path, sheet_name="Per-Cell Data")
        assert "Frame" in per_cell.columns
        assert "Cell_ID" in per_cell.columns
        # Should have morphology (Area) and FI (Mean FI Intensity (mScarlet) or similar)
        has_fi_col = any("FI" in c or "Mean" in c for c in per_cell.columns)
        assert has_fi_col
        assert "num foci [mScarlet]" in per_cell.columns
        assert "foci id [mScarlet]" in per_cell.columns
        # Per-Cell Data should not be cluttered with verbose foci aggregate columns
        assert all("Foci Mean Intensity" not in str(c) for c in per_cell.columns)
        assert len(per_cell) == 3
        summary = pd.read_excel(merged_path, sheet_name="Summary")
        # Multi-channel merge should add frame-mean FI columns to Summary for each channel.
        assert any("Mean FI Intensity" in str(c) and "(mScarlet)" in str(c) for c in summary.columns)
        foci_sheet = pd.read_excel(merged_path, sheet_name="Foci [mScarlet]")
        assert "cell_id" in foci_sheet.columns
        assert not foci_book_path.exists()


def test_merge_two_fi_channels_adds_coloc_per_cell_columns():
    """Two FI channels with matching foci should create per-cell directional num_coloc_foci columns in merged Per-Cell Data."""
    if not QT_AVAILABLE:
        if pytest:
            pytest.skip("PyQt6 or AnalyzeMasksTab not available")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        analysis_dir = tmp / "_analysis"
        analysis_dir.mkdir(parents=True)
        mask_path = tmp / "m_masks.tif"
        mask_path.touch()
        fi_gfp = tmp / "m_fi_GFP.tif"
        fi_ms = tmp / "m_fi_mScarlet.tif"
        fi_gfp.touch()
        fi_ms.touch()

        df_summary = pd.DataFrame({
            "Frame": [1],
            "Total Cell Area (µm²)": [21.0],
            "Total Cell Area (Relative)": [1.0],
            "Number of Cells": [1],
        })
        df_phase = pd.DataFrame({
            "Frame": [1],
            "Cell_ID": [1],
            "Area (µm²)": [21.0],
        })
        phase_path = analysis_dir / "m_analysis_phase.xlsx"
        with pd.ExcelWriter(phase_path, engine="openpyxl") as w:
            df_summary.to_excel(w, sheet_name="Summary", index=False)
            df_phase.to_excel(w, sheet_name="Per-Cell Data", index=False)

        df_gfp = pd.DataFrame({
            "Frame": [1],
            "Cell_ID": [1],
            "Mean FI Intensity": [100.0],
            "GFP_foci_count": [1],
            "foci_ids_GFP": ["1"],
        })
        df_gfp_foci = pd.DataFrame({
            "foci_id": [1],
            "cell_id": [1],
            "frame": [0],
            "channel_name": ["GFP"],
            "y": [10],
            "x": [10],
            "intensity": [220],
            "in_cell": [True],
        })
        gfp_path = analysis_dir / "m_analysis_GFP.xlsx"
        with pd.ExcelWriter(gfp_path, engine="openpyxl") as w:
            df_summary.to_excel(w, sheet_name="Summary", index=False)
            df_gfp.to_excel(w, sheet_name="Per-Cell Data", index=False)
            df_gfp_foci.to_excel(w, sheet_name="Foci", index=False)

        df_ms = pd.DataFrame({
            "Frame": [1],
            "Cell_ID": [1],
            "Mean FI Intensity": [120.0],
            "mScarlet_foci_count": [1],
            "foci_ids_mScarlet": ["1"],
        })
        df_ms_foci = pd.DataFrame({
            "foci_id": [1],
            "cell_id": [1],
            "frame": [0],
            "channel_name": ["mScarlet"],
            "y": [11],
            "x": [10],
            "intensity": [260],
            "in_cell": [True],
        })
        ms_path = analysis_dir / "m_analysis_mScarlet.xlsx"
        with pd.ExcelWriter(ms_path, engine="openpyxl") as w:
            df_summary.to_excel(w, sheet_name="Summary", index=False)
            df_ms.to_excel(w, sheet_name="Per-Cell Data", index=False)
            df_ms_foci.to_excel(w, sheet_name="Foci", index=False)

        merged_path = analysis_dir / "m_analysis.xlsx"
        tab = AnalyzeTab(parent=None)
        tab.parent_window = type("Mock", (), {"log_output": lambda self, msg: None})()
        tab.folder_fi_tasks = [
            (mask_path, None, phase_path),
            (mask_path, fi_gfp, gfp_path),
            (mask_path, fi_ms, ms_path),
        ]
        tab.folder_results = [True, True, True]
        tab.folder_output_paths = [str(phase_path), str(gfp_path), str(ms_path)]
        tab.folder_phase_per_mask = {}

        tab._merge_fi_channels_into_one_excel()
        assert merged_path.exists()
        per_cell = pd.read_excel(merged_path, sheet_name="Per-Cell Data")
        assert "num_coloc_foci_GFP_with_mScarlet" in per_cell.columns
        assert "num_coloc_foci_mScarlet_with_GFP" in per_cell.columns
        row0 = per_cell.iloc[0]
        assert int(row0["num_coloc_foci_GFP_with_mScarlet"]) >= 1
        assert int(row0["num_coloc_foci_mScarlet_with_GFP"]) >= 1


def test_coloc_pair_sheet_contains_pair_level_distance_and_ids():
    """Post-analysis colocalization embeds detailed pair rows (ids + distances) into *_analysis.xlsx."""
    if not QT_AVAILABLE:
        if pytest:
            pytest.skip("PyQt6 or AnalyzeMasksTab not available")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        analysis_dir = tmp / "_analysis"
        analysis_dir.mkdir(parents=True)
        mask_path = tmp / "m_masks.tif"
        mask_path.touch()
        analysis_path = analysis_dir / "m_analysis.xlsx"
        with pd.ExcelWriter(analysis_path, engine="openpyxl") as w:
            pd.DataFrame({"Frame": [1], "Number of Cells": [1]}).to_excel(w, sheet_name="Summary", index=False)
            pd.DataFrame({"Frame": [1], "Cell_ID": [1]}).to_excel(w, sheet_name="Per-Cell Data", index=False)

        tab = AnalyzeTab(parent=None)
        tab.parent_window = type("Mock", (), {"log_output": lambda self, msg: None})()
        # Avoid real IO/Spotiflow in this unit test
        tab._foci_channel_entries_for_colocalization = lambda _mp: [("GFP", Path("a.tif")), ("mScarlet", Path("b.tif"))]
        pair_df = pd.DataFrame(
            {
                "foci_id": [1],
                "colocalized_foci_id": [2],
                "distance": [1.3],
                "foci_channel": ["GFP"],
                "colocalized_channel": ["mScarlet"],
            }
        )
        tab._write_colocalization_workbook_for_pair = lambda *args, **kwargs: (True, pair_df)

        n_ok, _ = tab._run_all_colocalization_pair_workbooks_for_mask(mask_path)
        assert n_ok == 1
        xl = pd.ExcelFile(analysis_path)
        sheet = next((s for s in xl.sheet_names if s.startswith("Colocalization ")), None)
        assert sheet is not None
        d = pd.read_excel(analysis_path, sheet_name=sheet)
        assert "foci_id" in d.columns
        assert "colocalized_foci_id" in d.columns
        assert "distance" in d.columns


if __name__ == "__main__":
    test_merge_phase_and_fi_produces_one_excel()
    test_merge_two_fi_channels_adds_coloc_per_cell_columns()
    test_coloc_pair_sheet_contains_pair_level_distance_and_ids()
    print("All merge_fi tests passed.")
