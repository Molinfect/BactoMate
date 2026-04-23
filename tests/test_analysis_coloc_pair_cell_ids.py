from __future__ import annotations


def test_coloc_pair_records_include_cell_ids():
    from bactimate.modules.analysis_coloc import ColocSettings, analyze_colocalization_foci_table

    settings = ColocSettings(
        coloc_max_distance=5.0,
        coloc_threshold=5.0,
        coloc_filter_boundaries=True,
    )

    # Two channels, one colocated pair in same cell_id=7.
    rows = [
        {"channel_name": "GFP", "frame": 0, "y": 10, "x": 10, "cell_id": 7, "foci_id": 1},
        {"channel_name": "dsRed", "frame": 0, "y": 11, "x": 10, "cell_id": 7, "foci_id": 1},
    ]

    out = analyze_colocalization_foci_table(rows, settings, foci_channels=("GFP", "dsRed"), frame=0)
    assert out.pair_records, "Expected at least one pair record"

    pr = out.pair_records[0]
    assert "cell_id_primary" in pr
    assert "cell_id_secondary" in pr
    assert pr["cell_id_primary"] == 7
    assert pr["cell_id_secondary"] == 7
    assert pr.get("same_cell") is True

