"""Segmentation tab: algorithm (backend) switch must use the previous panel index."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from bactomate.gui.tabs.segmentation import SegmentationTab


@pytest.fixture
def seg_tab_minimal():
    """SegmentationTab with mocked child tabs (no full GUI)."""
    parent = MagicMock()
    omnipose = MagicMock()
    omnipose.file_path.text.return_value = "/data/folder"
    omnipose._embedded_segmentation_preview = MagicMock()
    omnipose._embedded_segmentation_preview._segmentation_input_path = "/data/folder/a.tif"
    omnipose._embedded_unet_preview = None

    unet = MagicMock()
    unet.file_path.text.return_value = "/data/folder"
    unet._embedded_segmentation_preview = None
    unet._embedded_unet_preview = None

    cellpose = MagicMock()
    cellpose.file_path.text.return_value = "/data/folder"
    cellpose._embedded_segmentation_preview = None
    cellpose._embedded_unet_preview = None

    filtering = MagicMock()
    analyze = MagicMock()

    tab = SegmentationTab.__new__(SegmentationTab)
    tab.parent_window = parent
    tab.omnipose_tab = omnipose
    tab.unet_tab = unet
    tab._cellpose_owner = cellpose
    tab.BACKENDS = SegmentationTab.BACKENDS
    tab._backend_combo_prev_index = 0
    return tab


def test_segmentation_owner_for_index(seg_tab_minimal):
    tab = seg_tab_minimal
    assert tab._segmentation_owner_for_index(0) is tab.omnipose_tab
    assert tab._segmentation_owner_for_index(1) is tab._cellpose_owner
    assert tab._segmentation_owner_for_index(2) is tab.unet_tab
