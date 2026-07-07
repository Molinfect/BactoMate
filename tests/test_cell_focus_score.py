from __future__ import annotations

import numpy as np

from bactomate.modules.cell_focus_score import filter_labels_by_focus, score_cells_in_frame


def test_focus_filter_keeps_sharper_cell():
    rng = np.random.default_rng(0)
    plane = np.zeros((64, 64), dtype=np.float32)
    plane[20:30, 20:30] = rng.normal(100.0, 8.0, (10, 10))
    plane[40:50, 40:50] = rng.normal(40.0, 0.2, (10, 10))
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[20:30, 20:30] = 1
    mask[40:50, 40:50] = 2
    scores = score_cells_in_frame(plane, mask, [1, 2])
    kept = filter_labels_by_focus(scores, min_fraction_of_median=0.8)
    assert 1 in kept
    assert 2 not in kept
