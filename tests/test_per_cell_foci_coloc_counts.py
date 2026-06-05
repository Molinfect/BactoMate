from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_persist_workbook_count_updates_adds_columns(tmp_path: Path):
    try:
        import openpyxl  # noqa: F401
    except Exception:
        # Optional dependency in some environments
        return

    from bactomate.gui.tabs._analyze_masks_colocalization import _persist_workbook_count_updates

    xlsx = tmp_path / "out.xlsx"
    df = pd.DataFrame(
        [
            {"Frame": 0, "Cell_ID": 1, "Area": 10.0},
            {"Frame": 0, "Cell_ID": 2, "Area": 12.0},
        ]
    )
    summary = pd.DataFrame(
        [
            {"Frame": 0, "Number of Cells": 2},
        ]
    )
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Per-Cell Data", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)

    channel_rows = [
        (
            "GFP",
            [
                {"cell_id": 1, "in_cell": True, "frame": 0},
                {"cell_id": 1, "in_cell": True, "frame": 0},
                {"cell_id": 2, "in_cell": True, "frame": 0},
            ],
        ),
        ("dsRed", [{"cell_id": 2, "in_cell": True, "frame": 0}]),
    ]
    raw_counts = {("GFP", 0): 5, ("dsRed", 0): 3}
    pair_df = pd.DataFrame(
        [
            {"cell_id_primary": 1, "cell_id_secondary": 1, "same_cell": True, "is_valid": True},
            {"cell_id_primary": 2, "cell_id_secondary": 2, "same_cell": True, "is_valid": True},
            {"cell_id_primary": 2, "cell_id_secondary": 2, "same_cell": True, "is_valid": False},
        ]
    )
    pair_tables = [("GFP", "dsRed", pair_df, len(pair_df))]

    _persist_workbook_count_updates(
        analysis_path=xlsx,
        channel_rows=channel_rows,
        raw_counts=raw_counts,
        pair_tables=pair_tables,
        same_cell_only=True,
    )

    out = pd.read_excel(xlsx, sheet_name="Per-Cell Data")
    assert "GFP_foci_count" in out.columns
    assert "dsRed_foci_count" in out.columns
    assert "coloc_pairs_count" in out.columns
    # GFP: cell1=2, cell2=1
    assert int(out.loc[out["Cell_ID"] == 1, "GFP_foci_count"].iloc[0]) == 2
    assert int(out.loc[out["Cell_ID"] == 2, "GFP_foci_count"].iloc[0]) == 1
    # dsRed: cell1=0, cell2=1
    assert int(out.loc[out["Cell_ID"] == 1, "dsRed_foci_count"].iloc[0]) == 0
    assert int(out.loc[out["Cell_ID"] == 2, "dsRed_foci_count"].iloc[0]) == 1
    # coloc valid same-cell: cell1=1, cell2=1 (invalid excluded)
    assert int(out.loc[out["Cell_ID"] == 1, "coloc_pairs_count"].iloc[0]) == 1
    assert int(out.loc[out["Cell_ID"] == 2, "coloc_pairs_count"].iloc[0]) == 1

    sm = pd.read_excel(xlsx, sheet_name="Summary")
    assert "Total Foci Count (GFP)" in sm.columns
    assert "Total Foci Count within Cells (GFP)" not in sm.columns
    assert int(sm.loc[0, "Total Foci Count (GFP)"]) == 3
    assert int(sm.loc[0, "Foci outside cells (GFP)"]) == 2
    assert int(sm.loc[0, "Total Foci Count (dsRed)"]) == 1
    assert int(sm.loc[0, "Foci outside cells (dsRed)"]) == 2
    assert "Valid Pairs Count (GFP×dsRed)" in sm.columns
    assert int(sm.loc[0, "Valid Pairs Count (GFP×dsRed)"]) == 2


def test_persist_coloc_single_frame_one_based_per_cell(tmp_path: Path) -> None:
    """Frameless coloc tables must use the same frame as Summary (e.g. Per-Cell Frame=1, CSV frame=0)."""
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return

    from bactomate.gui.tabs._analyze_masks_colocalization import _persist_workbook_count_updates

    xlsx = tmp_path / "one_based.xlsx"
    per_cell = pd.DataFrame(
        [
            {"Frame": 1, "Cell_ID": 10, "Area": 5.0},
            {"Frame": 1, "Cell_ID": 17, "Area": 6.0},
            {"Frame": 1, "Cell_ID": 99, "Area": 7.0},
        ]
    )
    summary = pd.DataFrame([{"Frame": 1, "Number of Cells": 3}])
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        per_cell.to_excel(w, sheet_name="Per-Cell Data", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)

    channel_rows = [
        ("dsRed", [{"cell_id": 10, "in_cell": True, "frame": 0}]),
        ("GFP", [{"cell_id": 17, "in_cell": True, "frame": 0}]),
    ]
    pair_df = pd.DataFrame(
        [
            {"cell_id_primary": 10, "same_cell": True, "is_valid": True},
            {"cell_id_primary": 17, "same_cell": True, "is_valid": True},
            {"cell_id_primary": 99, "same_cell": True, "is_valid": False},
            {"cell_id_primary": 99, "same_cell": False, "is_valid": True},
        ]
    )
    _persist_workbook_count_updates(
        analysis_path=xlsx,
        channel_rows=channel_rows,
        raw_counts={("dsRed", 0): 1, ("GFP", 0): 1},
        pair_tables=[("dsRed", "GFP", pair_df, len(pair_df))],
        same_cell_only=True,
    )

    out = pd.read_excel(xlsx, sheet_name="Per-Cell Data")
    assert int(out.loc[out["Cell_ID"] == 10, "coloc_pairs_count"].iloc[0]) == 1
    assert int(out.loc[out["Cell_ID"] == 17, "coloc_pairs_count"].iloc[0]) == 1
    assert int(out.loc[out["Cell_ID"] == 99, "coloc_pairs_count"].iloc[0]) == 0

    sm = pd.read_excel(xlsx, sheet_name="Summary")
    assert int(sm.loc[0, "Valid Pairs Count (GFP×dsRed)"]) == 2
