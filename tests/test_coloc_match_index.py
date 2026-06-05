from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "coloc, a, b, expected",
    [
        (0, 0, 0, "no foci"),
        (0, 2, 0, "no correlation"),
        (0, 0, 2, "no correlation"),
        (0, 0, 3, "no correlation"),
        (1, 1, 6, "17%"),
        (4, 4, 4, "100%"),
        (2, 2, 3, "67%"),
        (0, 2, 3, "0%"),
        (1, 2, 2, "50%"),
    ],
)
def test_coloc_match_index_value(coloc: int, a: int, b: int, expected: str) -> None:
    from bactomate.modules.coloc_match_index import coloc_match_index_value

    assert coloc_match_index_value(coloc, a, b) == expected


def test_persist_workbook_adds_coloc_match_index(tmp_path) -> None:
    try:
        import openpyxl  # noqa: F401
    except Exception:
        pytest.skip("openpyxl not installed")
    import pandas as pd

    from bactomate.gui.tabs._analyze_masks_colocalization import _persist_workbook_count_updates

    xlsx = tmp_path / "out.xlsx"
    per_cell = pd.DataFrame(
        [
            {"Frame": 0, "Cell_ID": 1, "Area": 10.0},
            {"Frame": 0, "Cell_ID": 2, "Area": 12.0},
            {"Frame": 0, "Cell_ID": 3, "Area": 8.0},
        ]
    )
    summary = pd.DataFrame([{"Frame": 0, "Number of Cells": 3}])
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        per_cell.to_excel(w, sheet_name="Per-Cell Data", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)

    channel_rows = [
        (
            "488_nm",
            [
                {"cell_id": 1, "in_cell": True, "frame": 0},
                {"cell_id": 2, "in_cell": True, "frame": 0},
                {"cell_id": 2, "in_cell": True, "frame": 0},
            ],
        ),
        (
            "647",
            [
                {"cell_id": 1, "in_cell": True, "frame": 0},
                {"cell_id": 1, "in_cell": True, "frame": 0},
                {"cell_id": 1, "in_cell": True, "frame": 0},
                {"cell_id": 1, "in_cell": True, "frame": 0},
                {"cell_id": 1, "in_cell": True, "frame": 0},
                {"cell_id": 1, "in_cell": True, "frame": 0},
            ],
        ),
    ]
    pair_df = pd.DataFrame(
        [
            {"cell_id_primary": 1, "same_cell": True, "is_valid": True},
            {"cell_id_primary": 2, "same_cell": True, "is_valid": True},
            {"cell_id_primary": 2, "same_cell": True, "is_valid": True},
        ]
    )
    _persist_workbook_count_updates(
        analysis_path=xlsx,
        channel_rows=channel_rows,
        raw_counts={("488_nm", 0): 3, ("647", 0): 6},
        pair_tables=[("488_nm", "647", pair_df, len(pair_df))],
        same_cell_only=True,
    )

    out = pd.read_excel(xlsx, sheet_name="Per-Cell Data")
    assert "coloc_match_index" in out.columns
    assert out.loc[out["Cell_ID"] == 1, "coloc_match_index"].iloc[0] == "17%"
    assert out.loc[out["Cell_ID"] == 2, "coloc_match_index"].iloc[0] == "no correlation"
    assert out.loc[out["Cell_ID"] == 3, "coloc_match_index"].iloc[0] == "no foci"
