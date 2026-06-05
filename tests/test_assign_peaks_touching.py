from __future__ import annotations

import numpy as np


def test_assign_peaks_to_cells_assigns_edge_peaks_like_filter_preview() -> None:
    from bactomate.modules.foci_utils import assign_peaks_to_cells

    mask = np.zeros((20, 20), dtype=np.int32)
    mask[15, 15] = 1
    peaks = [{"y": 14, "x": 15, "intensity": 1.0}]
    rows, _ = assign_peaks_to_cells(peaks, mask, dilation_px=1, frame_index=0)
    assert len(rows) == 1
    assert rows[0]["in_cell"] is True
    assert rows[0]["cell_id"] == 1


def test_build_foci_table_resets_foci_id_per_channel() -> None:
    from bactomate.gui.analysis_workbook import build_foci_table_from_channel_rows

    channel_rows = [
        (
            "Channel_1",
            [
                {
                    "in_cell": True,
                    "cell_id": 1,
                    "frame": 0,
                    "y": 1,
                    "x": 2,
                    "channel_name": "Channel_1",
                },
            ],
        ),
        (
            "Channel_2",
            [
                {
                    "in_cell": True,
                    "cell_id": 1,
                    "frame": 0,
                    "y": 3,
                    "x": 4,
                    "channel_name": "Channel_2",
                },
            ],
        ),
    ]
    table = build_foci_table_from_channel_rows(channel_rows)
    assert [r["foci_id"] for r in table] == [1, 1]
    assert [r["channel_name"] for r in table] == ["Channel_1", "Channel_2"]
