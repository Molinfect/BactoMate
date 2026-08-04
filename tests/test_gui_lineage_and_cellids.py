"""GUI smoke tests: lineage preflight gate and post-segmentation Cell IDs."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
import tifffile

pytest.importorskip("PyQt6")

from bactomate.gui.tabs.omnipose_seg import OmniposeSegTab
from bactomate.gui.tabs.tracking import TrackingTab
from bactomate.gui.trackastra_utils import validate_trackastra_model_choice
from bactomate.gui.workers.tracking import TrackingWorker
from bactomate.modules.lineage_integration import LineagePreflightResult


@pytest.fixture
def tracking_tab_minimal(qapp):
    tab = TrackingTab.__new__(TrackingTab)
    tab.mode_combo = MagicMock()
    tab.mode_combo.currentText.return_value = "greedy"
    tab._selected_trackastra_model = MagicMock(return_value="general_2d")
    parent = MagicMock()
    tab.parent_window = parent
    tab._main_window = MagicMock(return_value=parent)
    return tab


@pytest.fixture
def omnipose_tab(qapp):
    tab = OmniposeSegTab(None)
    tab.parent_window = MagicMock()
    tab.parent_window.log_output = MagicMock()
    tab.parent_window.shared_metadata = {"pixel_size": 1.0}
    return tab


def test_lineage_preflight_blocks_on_error(tracking_tab_minimal, tmp_path: Path) -> None:
    mask = tmp_path / "sample_masks.tif"
    mask.write_bytes(b"\x00")
    pf = LineagePreflightResult(ok=False, errors=["No analysis workbook found under _analysis/."])
    with patch("bactomate.modules.lineage_integration.preflight_lineage_run", return_value=pf):
        with patch("bactomate.gui.tabs.tracking.show_message_dialog") as show_dialog:
            ok = tracking_tab_minimal._lineage_preflight_check(mask, title="Run lineage tracking")
    assert ok is False
    show_dialog.assert_called_once()


def test_lineage_preflight_passes_without_confirm_when_clean(tracking_tab_minimal, tmp_path: Path) -> None:
    mask = tmp_path / "sample_masks.tif"
    mask.write_bytes(b"\x00")
    pf = LineagePreflightResult(ok=True, n_frames=3, mask_path=mask, resolved_mask_path=mask)
    with patch("bactomate.modules.lineage_integration.preflight_lineage_run", return_value=pf):
        with patch("bactomate.gui.tabs.tracking.show_confirm_dialog") as confirm:
            ok = tracking_tab_minimal._lineage_preflight_check(mask, title="Preview lineage QC")
    assert ok is True
    confirm.assert_not_called()


def test_lineage_preflight_uses_strict_flags_for_full_run(tracking_tab_minimal, tmp_path: Path) -> None:
    mask = tmp_path / "sample_masks.tif"
    mask.write_bytes(b"\x00")
    pf = LineagePreflightResult(ok=True, n_frames=3, mask_path=mask, resolved_mask_path=mask)
    with patch("bactomate.modules.lineage_integration.preflight_lineage_run", return_value=pf) as preflight:
        with patch("bactomate.gui.tabs.tracking.show_confirm_dialog", return_value=True):
            tracking_tab_minimal._lineage_preflight_check(mask, title="Run lineage tracking", always_confirm=True)
            tracking_tab_minimal._lineage_preflight_check(mask, title="Preview lineage QC")
    assert preflight.call_args_list[0].kwargs["require_analysis_workbook"] is True
    assert preflight.call_args_list[0].kwargs["strict_lineage"] is True
    assert preflight.call_args_list[1].kwargs["require_analysis_workbook"] is False
    assert preflight.call_args_list[1].kwargs["strict_lineage"] is False


def test_validate_trackastra_model_choice_blocks_ilp_without_motile() -> None:
    with patch("bactomate.gui.trackastra_utils.ilp_dependencies_installed", return_value=False):
        ok, msg = validate_trackastra_model_choice("general_2d", linking_mode="ilp")
    assert ok is False
    assert msg is not None
    assert "motile" in msg.lower()


def test_ensure_trackastra_model_available_checks_current_mode(tracking_tab_minimal) -> None:
    tracking_tab_minimal.mode_combo.currentText.return_value = "ilp"
    with patch(
        "bactomate.gui.tabs.tracking.validate_trackastra_model_choice",
        return_value=(False, "missing motile"),
    ) as validate:
        with patch("bactomate.gui.tabs.tracking.show_message_dialog") as show_dialog:
            ok = tracking_tab_minimal._ensure_trackastra_model_available("general_2d", title="Preview tracking")
    assert ok is False
    assert validate.call_args.kwargs["linking_mode"] == "ilp"
    show_dialog.assert_called_once()


def test_prune_long_jump_edges_splits_track_and_relabels_masks() -> None:
    masks = np.zeros((2, 16, 16), dtype=np.uint16)
    masks[0, 2:4, 2:4] = 1
    masks[1, 12:14, 12:14] = 1
    graph = nx.DiGraph()
    graph.add_node("a", time=0, label=1, coords=(2.5, 2.5))
    graph.add_node("b", time=1, label=1, coords=(12.5, 12.5))
    graph.add_edge("a", "b")
    pruned_graph, masks_pruned, n_removed = TrackingWorker._prune_long_jump_edges(
        graph,
        masks,
        filter_params={"track_max_jump": 5.0, "pixel_size": 0.0, "use_physical_units": False},
        original_shape=masks.shape,
        debug_output=[],
    )
    assert n_removed == 1
    assert pruned_graph.number_of_edges() == 0
    assert int(masks_pruned[0].max()) == 1
    assert int(masks_pruned[1].max()) == 2


def test_apply_jump_prune_noop_preserves_tracked_masks() -> None:
    """Default max-jump (0) must not replace Trackastra masks with raw segmentation."""
    masks_input = np.zeros((2, 8, 8), dtype=np.uint16)
    masks_input[0, 1:3, 1:3] = 1
    masks_input[1, 1:3, 1:3] = 2  # per-frame labels (raw)
    masks_tracked = np.zeros((2, 8, 8), dtype=np.uint16)
    masks_tracked[0, 1:3, 1:3] = 7
    masks_tracked[1, 1:3, 1:3] = 7  # same track ID across frames
    graph = nx.DiGraph()
    graph.add_node("a", time=0, label=7, coords=(1.5, 1.5))
    graph.add_node("b", time=1, label=7, coords=(1.5, 1.5))
    graph.add_edge("a", "b")
    out_graph, out_masks, n_removed = TrackingWorker._apply_jump_prune(
        graph,
        masks_tracked,
        masks_input,
        filter_params={"track_max_jump": 0.0, "pixel_size": 0.0, "use_physical_units": False},
        original_shape=masks_input.shape,
        debug_output=[],
    )
    assert n_removed == 0
    assert out_graph is graph or out_graph.number_of_edges() == 1
    assert int(out_masks[0].max()) == 7
    assert int(out_masks[1].max()) == 7
    assert out_masks is masks_tracked


def test_apply_jump_prune_updates_masks_when_edges_removed() -> None:
    masks_input = np.zeros((2, 16, 16), dtype=np.uint16)
    masks_input[0, 2:4, 2:4] = 1
    masks_input[1, 12:14, 12:14] = 1
    masks_tracked = np.zeros((2, 16, 16), dtype=np.uint16)
    masks_tracked[0, 2:4, 2:4] = 1
    masks_tracked[1, 12:14, 12:14] = 1
    graph = nx.DiGraph()
    graph.add_node("a", time=0, label=1, coords=(2.5, 2.5))
    graph.add_node("b", time=1, label=1, coords=(12.5, 12.5))
    graph.add_edge("a", "b")
    _g, out_masks, n_removed = TrackingWorker._apply_jump_prune(
        graph,
        masks_tracked,
        masks_input,
        filter_params={"track_max_jump": 5.0, "pixel_size": 0.0, "use_physical_units": False},
        original_shape=masks_input.shape,
        debug_output=[],
    )
    assert n_removed == 1
    assert int(out_masks[0].max()) == 1
    assert int(out_masks[1].max()) == 2


def test_write_persistent_cellid_masks_creates_output(omnipose_tab, tmp_path: Path) -> None:
    mask = tmp_path / "sample_masks.tif"
    stack = np.zeros((2, 16, 16), dtype=np.uint16)
    stack[0, 4:8, 4:8] = 1
    stack[1, 4:8, 4:8] = 1
    tifffile.imwrite(mask, stack)
    omnipose_tab._write_persistent_cellid_masks(mask)
    # Sidecar is no longer written; IDs live in the primary mask.
    assert not (tmp_path / "sample_masks_cellids.tif").exists()
    primary = tifffile.imread(str(mask))
    assert primary.shape == stack.shape
    assert int(primary.max()) >= 1
    omnipose_tab.parent_window.log_output.assert_called()


def test_write_persistent_cellid_masks_always_runs(omnipose_tab, tmp_path: Path) -> None:
    mask = tmp_path / "sample_masks.tif"
    tifffile.imwrite(mask, np.zeros((2, 8, 8), dtype=np.uint16))
    omnipose_tab.assign_persistent_cellids.setChecked(False)
    omnipose_tab._write_persistent_cellid_masks(mask)
    assert not (tmp_path / "sample_masks_cellids.tif").exists()
    assert mask.is_file()


@pytest.fixture
def unet_tab(qapp):
    from bactomate.gui.tabs.unet_seg import UNetSegmentTab

    tab = UNetSegmentTab(None)
    tab.parent_window = MagicMock()
    tab.parent_window.log_output = MagicMock()
    tab.parent_window.shared_metadata = {"pixel_size": 1.0}
    return tab


def test_unet_write_persistent_cellid_masks_creates_output(unet_tab, tmp_path: Path) -> None:
    mask = tmp_path / "sample_unet_masks.tif"
    stack = np.zeros((2, 16, 16), dtype=np.uint16)
    stack[0, 4:8, 4:8] = 1
    stack[1, 4:8, 4:8] = 1
    tifffile.imwrite(mask, stack)
    unet_tab._write_persistent_cellid_masks(mask)
    assert not (tmp_path / "sample_unet_masks_cellids.tif").exists()
    written = tifffile.imread(str(mask))
    assert written.shape == stack.shape
    assert int(written.max()) >= 1
    unet_tab.parent_window.log_output.assert_called()


def test_unet_write_persistent_cellid_masks_always_runs(unet_tab, tmp_path: Path) -> None:
    mask = tmp_path / "sample_unet_masks.tif"
    tifffile.imwrite(mask, np.zeros((2, 8, 8), dtype=np.uint16))
    unet_tab.assign_persistent_cellids.setChecked(False)
    unet_tab._write_persistent_cellid_masks(mask)
    assert not (tmp_path / "sample_unet_masks_cellids.tif").exists()
    assert mask.is_file()
