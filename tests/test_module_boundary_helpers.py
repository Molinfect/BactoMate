"""Boundary tests for large GUI tab helpers (pure functions only)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from bactomate.gui.tabs.file_conversion import ND2ConversionTab
from bactomate.gui.tabs.omnipose_seg import OmniposeSegTab
from bactomate.gui.tabs.post_processing import PostProcessingTab


def test_file_conversion_frame_range_single_frame() -> None:
    tab = ND2ConversionTab.__new__(ND2ConversionTab)
    tab.frame_extract_mode = MagicMock()
    tab.frame_extract_mode.currentIndex.return_value = 1
    tab.single_frame_spin = MagicMock()
    tab.single_frame_spin.value.return_value = 3
    tab.max_frames = MagicMock()
    assert tab._get_frame_range(10) == (2, 3)


def test_file_conversion_frame_range_first_n() -> None:
    tab = ND2ConversionTab.__new__(ND2ConversionTab)
    tab.frame_extract_mode = MagicMock()
    tab.frame_extract_mode.currentIndex.return_value = 0
    tab.single_frame_spin = MagicMock()
    tab.max_frames = MagicMock()
    tab.max_frames.value.return_value = 5
    assert tab._get_frame_range(10) == (0, 5)


def test_post_processing_parse_video_paths_pipe_separated() -> None:
    paths = PostProcessingTab._parse_growth_curve_video_paths("/a.tif|/b.tif")
    assert [str(p) for p in paths] == ["/a.tif", "/b.tif"]


def test_post_processing_video_tile_key_normalizes() -> None:
    assert PostProcessingTab._video_tile_key(" Phase ") == "phase"


def test_omnipose_segmentation_folder_name_from_model() -> None:
    tab = OmniposeSegTab.__new__(OmniposeSegTab)
    tab.model_combo = MagicMock()
    tab.model_combo.currentText.return_value = "bact_phase_omni"
    tab._resolve_model_selection = MagicMock(return_value=("bact_phase_omni", None))
    assert tab._get_omnipose_segmentation_folder_name() == "_segmentation_bact_phase_omni"
