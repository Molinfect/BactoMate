"""Display-only downsample for tracking preview (full-res tracking stays intact)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bactomate.gui.workers.tracking_io import (
    TRACKING_PREVIEW_DOWNSAMPLE_MAX_SIZE,
    prepare_preview_display_stacks,
    preview_display_scale_and_steps,
    scale_tracks_df_for_display,
)


def test_tracking_preview_max_size_matches_analyze_preview():
    assert TRACKING_PREVIEW_DOWNSAMPLE_MAX_SIZE == 2048


def test_preview_display_scale_noop_when_small():
    sy, sx, step_y, step_x = preview_display_scale_and_steps(200, 300, max_size=384)
    assert (sy, sx, step_y, step_x) == (1.0, 1.0, 1, 1)


def test_prepare_preview_display_stacks_noop_under_default_cap():
    t, h, w = 2, 800, 800
    imgs = np.zeros((t, h, w), dtype=np.float32)
    masks_in = np.zeros((t, h, w), dtype=np.int32)
    masks_tr = np.zeros((t, h, w), dtype=np.int32)
    imgs_d, mi_d, mt_d, sy, sx = prepare_preview_display_stacks(imgs, masks_in, masks_tr)
    assert imgs_d.shape == (t, h, w)
    assert mi_d.shape == imgs_d.shape
    assert mt_d.shape == imgs_d.shape
    assert sy == 1.0 and sx == 1.0


def test_prepare_preview_display_stacks_downsamples_large():
    t, h, w = 3, 2304, 2304
    max_size = 384
    imgs = np.arange(t * h * w, dtype=np.float32).reshape(t, h, w)
    masks_in = np.zeros((t, h, w), dtype=np.int32)
    masks_tr = np.zeros((t, h, w), dtype=np.int32)
    sy, sx, _, _ = preview_display_scale_and_steps(h, w, max_size=max_size)
    y0, x0 = 600, 1200
    masks_in[0, y0, x0] = 7
    masks_tr[0, y0, x0] = 3
    imgs_d, mi_d, mt_d, out_sy, out_sx = prepare_preview_display_stacks(
        imgs, masks_in, masks_tr, max_size=max_size
    )
    assert imgs_d.shape[0] == t
    assert max(imgs_d.shape[1], imgs_d.shape[2]) <= max_size
    assert mi_d.shape == imgs_d.shape
    assert mt_d.shape == imgs_d.shape
    assert out_sy < 1.0 and out_sx < 1.0
    ey = int(round(y0 * sy))
    ex = int(round(x0 * sx))
    ey = min(max(0, ey), imgs_d.shape[1] - 1)
    ex = min(max(0, ex), imgs_d.shape[2] - 1)
    assert int(mi_d[0, ey, ex]) == 7
    assert int(mt_d[0, ey, ex]) == 3


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
