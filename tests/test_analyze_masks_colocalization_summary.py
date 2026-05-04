from __future__ import annotations

import pandas as pd


def test_infer_frame_shift_prefers_plus_one_when_overlap_improves() -> None:
    from bactimate.gui.tabs._analyze_masks_colocalization import _infer_frame_shift

    per_cell_frames = {1, 2}
    channel_rows = [
        ("A", [{"frame": 0, "in_cell": True, "cell_id": 1}]),
    ]
    pair_tables: list = []
    assert _infer_frame_shift(per_cell_frames=per_cell_frames, channel_rows=channel_rows, pair_tables=pair_tables) == 1


def test_infer_frame_shift_zero_when_frames_align() -> None:
    from bactimate.gui.tabs._analyze_masks_colocalization import _infer_frame_shift

    per_cell_frames = {0, 1}
    channel_rows = [
        ("A", [{"frame": 0, "in_cell": True, "cell_id": 1}]),
    ]
    assert _infer_frame_shift(per_cell_frames=per_cell_frames, channel_rows=channel_rows, pair_tables=[]) == 0


def test_pair_column_name_sorted() -> None:
    from bactimate.gui.tabs._analyze_masks_colocalization import _pair_column_name

    assert _pair_column_name("Zebra", "alpha") == "Valid Pairs Count (alpha×Zebra)"


def test_valid_pairs_counts_single_frame_no_frame_column() -> None:
    from bactimate.gui.tabs._analyze_masks_colocalization import _valid_pairs_counts_by_summary_frame

    df = pd.DataFrame(
        [
            {"cell_id_primary": 1, "same_cell": True, "is_valid": True},
            {"cell_id_primary": 1, "same_cell": True, "is_valid": False},
        ]
    )
    rows_a = [{"frame": 0, "in_cell": True, "cell_id": 1}]
    rows_b = [{"frame": 0, "in_cell": True, "cell_id": 1}]
    out = _valid_pairs_counts_by_summary_frame(
        df=df,
        frame_shift=1,
        same_cell_only=True,
        rows_a=rows_a,
        rows_b=rows_b,
        summary_frames={1},
    )
    assert out == {1: 1}
