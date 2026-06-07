"""Workflow replay: active segmentation backend output folder conventions."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from bactomate.gui.tabs.segmentation import SegmentationTab


@pytest.fixture
def seg_tab_minimal():
    parent = MagicMock()
    omnipose = MagicMock()
    omnipose.file_path.text.return_value = "/data/Tif_files"
    omnipose.get_segmentation_output_folder_name.return_value = "_segmentation_bact_phase_omni"
    omnipose._embedded_segmentation_preview = None
    omnipose._embedded_unet_preview = None

    cellpose = MagicMock()
    cellpose.file_path.text.return_value = "/data/Tif_files"
    cellpose.get_segmentation_output_folder_name.return_value = "_segmentation_cellpose_sam"
    cellpose._embedded_segmentation_preview = None
    cellpose._embedded_unet_preview = None

    unet = MagicMock()
    unet.file_path.text.return_value = "/data/Tif_files"
    unet._embedded_segmentation_preview = None
    unet._embedded_unet_preview = None

    filtering = MagicMock()
    analyze = MagicMock()

    tab = SegmentationTab.__new__(SegmentationTab)
    tab.parent_window = parent
    tab.omnipose_tab = omnipose
    tab.unet_tab = unet
    tab._cellpose_owner = cellpose
    tab.filtering_tab = filtering
    tab.analyze_tab = analyze
    tab.BACKENDS = SegmentationTab.BACKENDS
    tab._BACKEND_KEYS = SegmentationTab._BACKEND_KEYS
    tab._backend_combo_prev_index = 0
    tab.backend_combo = MagicMock()
    tab.stack = MagicMock()
    return tab


def _set_backend(tab: SegmentationTab, index: int) -> None:
    tab.backend_combo.currentIndex.return_value = index


def test_infer_segment_output_omnipose_folder(seg_tab_minimal):
    tab = seg_tab_minimal
    _set_backend(tab, 0)
    out = tab.infer_segment_output_path("/data/Tif_files")
    assert out == Path("/data/Tif_files/_segmentation_bact_phase_omni")


def test_infer_segment_output_cellpose_folder(seg_tab_minimal):
    tab = seg_tab_minimal
    _set_backend(tab, 1)
    out = tab.infer_segment_output_path("/data/Tif_files")
    assert out == Path("/data/Tif_files/_segmentation_cellpose_sam")


def test_infer_segment_output_unet_folder(seg_tab_minimal):
    tab = seg_tab_minimal
    _set_backend(tab, 2)
    out = tab.infer_segment_output_path("/data/Tif_files")
    assert out == Path("/data/Tif_files/_segmentation_unet")


def test_infer_segment_output_unet_single_file_in_tif_folder(seg_tab_minimal):
    tab = seg_tab_minimal
    _set_backend(tab, 2)
    out = tab.infer_segment_output_path("/data/Tif_files/sample_phase.tif")
    assert out == Path("/data/Tif_files/_segmentation_unet")


def test_infer_segment_output_unet_single_file_outside_tif_folder(seg_tab_minimal):
    tab = seg_tab_minimal
    _set_backend(tab, 2)
    out = tab.infer_segment_output_path("/data/raw/sample.tif")
    assert out == Path("/data/raw/sample_unet_segmentation")


def test_file_path_proxy_syncs_all_backends(seg_tab_minimal):
    from bactomate.gui.tabs.segmentation import _SegmentationFilePathProxy

    tab = seg_tab_minimal
    tab._sync_segmentation_input_path = MagicMock()
    proxy = _SegmentationFilePathProxy(tab)
    proxy.setText("/data/Tif_files")
    tab._sync_segmentation_input_path.assert_called_once_with("/data/Tif_files")


def test_get_session_params_includes_backend(seg_tab_minimal):
    tab = seg_tab_minimal
    _set_backend(tab, 2)
    tab.unet_tab.get_session_params.return_value = {"unet_model": "m1", "performed": True}
    params = tab.get_session_params()
    assert params["segmentation_backend"] == "unet"
    assert params["unet_model"] == "m1"
    assert params["performed"] is True
