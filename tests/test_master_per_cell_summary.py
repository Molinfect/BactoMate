from __future__ import annotations

from pathlib import Path

import pandas as pd
def _write_workbook(path: Path, *, per_cell: pd.DataFrame | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"Frame": [1], "Number of Cells": [2]}).to_excel(
            writer, sheet_name="Summary", index=False
        )
        if per_cell is not None:
            per_cell.to_excel(writer, sheet_name="Per-Cell Data", index=False)


def test_build_master_per_cell_stacks_with_source_columns(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import (
        MASTER_PER_CELL_SAMPLE_COLUMN,
        MASTER_PER_CELL_SOURCE_COLUMN,
        build_master_per_cell_dataframe,
    )

    p1 = tmp_path / "alpha_analysis.xlsx"
    p2 = tmp_path / "beta_masks_analysis.xlsx"
    _write_workbook(
        p1,
        per_cell=pd.DataFrame({"Frame": [1], "Cell_ID": [10], "Area (µm²)": [1.5]}),
    )
    _write_workbook(
        p2,
        per_cell=pd.DataFrame({"Frame": [2], "Cell_ID": [20], "Area (µm²)": [2.5]}),
    )

    df, warnings = build_master_per_cell_dataframe([p1, p2])
    assert warnings == []
    assert df is not None
    assert len(df) == 2
    assert list(df.columns[:2]) == [MASTER_PER_CELL_SOURCE_COLUMN, MASTER_PER_CELL_SAMPLE_COLUMN]
    assert set(df[MASTER_PER_CELL_SOURCE_COLUMN]) == {p1.name, p2.name}
    assert set(df[MASTER_PER_CELL_SAMPLE_COLUMN]) == {"alpha", "beta"}


def test_write_master_per_cell_summary_creates_named_file(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import (
        MASTER_PER_CELL_SUMMARY_FILENAME,
        MASTER_PER_CELL_SUMMARY_SHEET,
        write_master_per_cell_summary,
    )

    p1 = tmp_path / "one_analysis.xlsx"
    _write_workbook(
        p1,
        per_cell=pd.DataFrame({"Frame": [1], "Cell_ID": [1]}),
    )

    assert write_master_per_cell_summary(tmp_path) is True
    out = tmp_path / MASTER_PER_CELL_SUMMARY_FILENAME
    assert out.is_file()
    xl = pd.ExcelFile(out)
    assert MASTER_PER_CELL_SUMMARY_SHEET in xl.sheet_names
    sheet = pd.read_excel(out, sheet_name=MASTER_PER_CELL_SUMMARY_SHEET)
    assert len(sheet) == 1


def test_write_master_per_cell_summary_skips_aggregate_files(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import (
        MASTER_PER_CELL_SUMMARY_FILENAME,
        write_master_per_cell_summary,
    )

    p1 = tmp_path / "only_analysis.xlsx"
    _write_workbook(
        p1,
        per_cell=pd.DataFrame({"Frame": [1], "Cell_ID": [1]}),
    )
    assert write_master_per_cell_summary(tmp_path) is True

    # Stale master must not be read as input
    master = tmp_path / MASTER_PER_CELL_SUMMARY_FILENAME
    assert master.is_file()

    p2 = tmp_path / "second_analysis.xlsx"
    _write_workbook(
        p2,
        per_cell=pd.DataFrame({"Frame": [1], "Cell_ID": [2]}),
    )
    assert write_master_per_cell_summary(tmp_path) is True
    sheet = pd.read_excel(master, sheet_name="@summary_per-cell-data")
    assert len(sheet) == 2


def test_build_master_per_cell_merges_generic_foci_into_gfp_columns(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import build_master_per_cell_dataframe

    gfp = tmp_path / "numbered_analysis.xlsx"
    generic = tmp_path / "double_underscore_analysis.xlsx"
    _write_workbook(
        gfp,
        per_cell=pd.DataFrame(
            {
                "Frame": [1],
                "Cell_ID": [1],
                "GFP_foci_count": [3],
                "foci_ids_GFP": ["1,2,3"],
            }
        ),
    )
    _write_workbook(
        generic,
        per_cell=pd.DataFrame(
            {
                "Frame": [1],
                "Cell_ID": [2],
                "foci_foci_count": [5],
                "foci_ids_foci": ["4,5"],
            }
        ),
    )

    df, warnings = build_master_per_cell_dataframe([gfp, generic])
    assert warnings == []
    assert df is not None
    assert "foci_foci_count" not in df.columns
    assert "GFP_foci_count" in df.columns
    assert int(df.loc[df["Cell_ID"] == 1, "GFP_foci_count"].iloc[0]) == 3
    assert int(df.loc[df["Cell_ID"] == 2, "GFP_foci_count"].iloc[0]) == 5


def test_write_master_per_cell_summary_no_data_returns_false(tmp_path: Path) -> None:
    from bactomate.gui.analysis_workbook import write_master_per_cell_summary

    p1 = tmp_path / "empty_analysis.xlsx"
    _write_workbook(p1, per_cell=None)
    assert write_master_per_cell_summary(tmp_path) is False
