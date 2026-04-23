from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_update_per_cell_data_counts_adds_columns(tmp_path: Path):
    try:
        import openpyxl  # noqa: F401
    except Exception:
        # Optional dependency in some environments
        return

    from bactimate.gui.tabs._analyze_masks_colocalization import _update_per_cell_data_counts

    xlsx = tmp_path / "out.xlsx"
    df = pd.DataFrame(
        [
            {"Frame": 0, "Cell_ID": 1, "Area": 10.0},
            {"Frame": 0, "Cell_ID": 2, "Area": 12.0},
        ]
    )
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Per-Cell Data", index=False)
        pd.DataFrame([{"x": 1}]).to_excel(w, sheet_name="Summary", index=False)

    channel_rows = [
        ("GFP", [{"cell_id": 1, "in_cell": True}, {"cell_id": 1, "in_cell": True}, {"cell_id": 2, "in_cell": True}]),
        ("dsRed", [{"cell_id": 2, "in_cell": True}]),
    ]
    pair_df = pd.DataFrame(
        [
            {"cell_id_primary": 1, "cell_id_secondary": 1, "same_cell": True, "is_valid": True},
            {"cell_id_primary": 2, "cell_id_secondary": 2, "same_cell": True, "is_valid": True},
            {"cell_id_primary": 2, "cell_id_secondary": 2, "same_cell": True, "is_valid": False},
        ]
    )
    pair_tables = [("GFP", "dsRed", pair_df, len(pair_df))]

    _update_per_cell_data_counts(analysis_path=xlsx, channel_rows=channel_rows, pair_tables=pair_tables)

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

