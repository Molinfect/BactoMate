"""Tests for TIFF stack I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from bactomate.gui.workers.tracking_io import should_write_tracking_visualization
from bactomate.modules.tiff_stack_io import count_tiff_pages, read_tiff_stack


def test_count_tiff_pages(tmp_path: Path) -> None:
    path = tmp_path / "stack.tif"
    stack = np.zeros((3, 8, 8), dtype=np.uint16)
    tifffile.imwrite(path, stack)
    assert count_tiff_pages(path) == 3


def test_read_tiff_stack_preview_limits_frames(tmp_path: Path) -> None:
    path = tmp_path / "stack.tif"
    stack = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
    tifffile.imwrite(path, stack)
    arr, total, partial = read_tiff_stack(path, frame_limit=2, preview_mode=True)
    assert partial is True
    assert total == 5
    assert arr.shape[0] == 2


def test_should_write_tracking_visualization_for_long_stack(tmp_path: Path) -> None:
    path = tmp_path / "long_masks.tif"
    stack = np.zeros((200, 4, 4), dtype=np.uint16)
    tifffile.imwrite(path, stack)
    assert should_write_tracking_visualization(path, user_enabled=True, track_action="movement") is True
    assert should_write_tracking_visualization(path, user_enabled=False, track_action="movement") is True
    assert should_write_tracking_visualization(path, user_enabled=True, track_action="preview") is False
