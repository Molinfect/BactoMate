"""Display-only downsample for tracking preview (full-res tracking stays intact)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bactomate.gui.workers.tracking_io import (
    prepare_preview_display_stacks,
    preview_display_scale_and_steps,
    scale_tracks_df_for_display,
)


def test_preview_display_scale_noop_when_small():
    sy, sx, step_y, step_x = preview_display_scale_and_steps(200, 300, max_size=384)
    assert (sy, sx, step_y, step_x) == (1.0, 1.0, 1, 1)


def test_prepare_preview_display_stacks_downsamples_large():
    t, h, w = 3, 2304, 2304
    imgs = np.arange(t * h * w, dtype=np.float32).reshape(t, h, w)
    masks_in = np.zeros((t, h, w), dtype=np.int32)
    masks_tr = np.zeros((t, h, w), dtype=np.int32)
    _, _, step_y, step_x = preview_display_scale_and_steps(h, w, max_size=384)
    # Place labels on the integer-slice lattice so they survive ::step downsampling.
    y0, x0 = step_y * 10, step_x * 20
    masks_in[0, y0, x0] = 7
    masks_tr[0, y0, x0] = 3
    imgs_d, mi_d, mt_d, sy, sx = prepare_preview_display_stacks(
        imgs, masks_in, masks_tr, max_size=384
    )
    assert imgs_d.shape[0] == t
    assert max(imgs_d.shape[1], imgs_d.shape[2]) <= 384
    assert mi_d.shape == imgs_d.shape
    assert mt_d.shape == imgs_d.shape
    assert sy < 1.0 and sx < 1.0
    assert int(mi_d[0, y0 // step_y, x0 // step_x]) == 7
    assert int(mt_d[0, y0 // step_y, x0 // step_x]) == 3


def test_scale_tracks_df_for_display():
    df = pd.DataFrame(
        {"track_id": [1, 1], "frame": [0, 1], "y": [100.0, 200.0], "x": [300.0, 400.0]}
    )
    out = scale_tracks_df_for_display(df, 0.5, 0.25)
    assert list(out["y"]) == [50.0, 100.0]
    assert list(out["x"]) == [75.0, 100.0]
    # Original unchanged
    assert list(df["y"]) == [100.0, 200.0]


def test_scale_tracks_noop_at_unity():
    df = pd.DataFrame({"track_id": [1], "frame": [0], "y": [10.0], "x": [20.0]})
    out = scale_tracks_df_for_display(df, 1.0, 1.0)
    assert out is df
