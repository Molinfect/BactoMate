from __future__ import annotations

import numpy as np


def test_assign_peaks_to_cells_assigns_edge_peaks_like_filter_preview() -> None:
    from bactomate.modules.foci_utils import assign_peaks_to_cells

    mask = np.zeros((20, 20), dtype=np.int32)
    mask[15, 15] = 1
    peaks = [{"y": 14, "x": 15, "intensity": 1.0}]
    rows, _, next_id = assign_peaks_to_cells(peaks, mask, dilation_px=1, frame_index=0)
    assert len(rows) == 1
    assert rows[0]["in_cell"] is True
    assert rows[0]["cell_id"] == 1
    assert rows[0]["foci_id"] == 1
    assert next_id == 2


def test_assign_peaks_to_cells_global_foci_id_across_frames() -> None:
    from bactomate.modules.foci_utils import assign_peaks_to_cells

    mask = np.zeros((20, 20), dtype=np.int32)
    mask[10, 10] = 1
    peaks_f0 = [{"y": 10, "x": 10, "intensity": 1.0}]
    peaks_f1 = [{"y": 11, "x": 11, "intensity": 2.0}]
    rows0, _, next_id = assign_peaks_to_cells(peaks_f0, mask, frame_index=0, start_foci_id=1)
    rows1, _, _ = assign_peaks_to_cells(peaks_f1, mask, frame_index=1, start_foci_id=next_id)
    assert rows0[0]["foci_id"] == 1
    assert rows1[0]["foci_id"] == 2


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
