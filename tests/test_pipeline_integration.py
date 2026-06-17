"""Slow integration test: masks → persistent IDs → analyze → lineage preflight → join."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _isolate_analyze_script_logging():
    """Reset script logger handlers when other tests (e.g. PyQt) touch logging."""
    from bactomate.scripts.analyze_cell_areas import SCRIPT_LOG, _setup_script_logging

    SCRIPT_LOG.handlers.clear()
    _setup_script_logging(log_to_file=False)
    yield
    SCRIPT_LOG.handlers.clear()


def _write_moving_cell_stack(path: Path, n_frames: int = 2) -> None:
    stack = np.zeros((n_frames, 32, 32), dtype=np.uint16)
    for t in range(n_frames):
        y0, x0 = 8 + t, 8 + t
        stack[t, y0 : y0 + 4, x0 : x0 + 4] = 1
    tifffile.imwrite(path, stack)


@pytest.mark.slow
def test_pipeline_masks_analyze_lineage_join(tmp_path: Path) -> None:
    from bactomate.gui.analysis_paths import analysis_dir_from_base, base_dir_for_mask, stem_from_mask_path
    from bactomate.modules.cell_id_linking import write_persistent_cellid_mask_file
    from bactomate.modules.lineage_integration import (
        join_analysis_to_lineage_points,
        preflight_lineage_run,
    )
    from bactomate.scripts.analyze_cell_areas import analyze_masks, save_to_excel

    base = tmp_path / "Tif_001"
    seg = base / "_segmentation_omni"
    seg.mkdir(parents=True)
    mask_path = seg / "sample_masks.tif"
    _write_moving_cell_stack(mask_path, n_frames=2)

    cellid_path = write_persistent_cellid_mask_file(mask_path, pixel_size=0.1)
    assert cellid_path is not None and cellid_path.is_file()

    results, all_frame_data, masks_with_cellids, _masks_raw, foci_table = analyze_masks(
        str(cellid_path),
        pixel_size=0.1,
        min_aspect_ratio=1.0,
        max_aspect_ratio=20.0,
        min_eccentricity=0.0,
        min_solidity=0.0,
    )
    assert results
    assert all_frame_data

    stem = stem_from_mask_path(cellid_path)
    analysis_dir = analysis_dir_from_base(base_dir_for_mask(cellid_path))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    workbook = analysis_dir / f"{stem}_analysis.xlsx"
    save_to_excel(results, workbook, detailed_data=all_frame_data, foci_table=foci_table, pixel_size=0.1)
    assert workbook.is_file()

    pf = preflight_lineage_run(
        cellid_path,
        linker_mode="greedy",
        model_name="general_2d",
        require_analysis_workbook=True,
        strict_lineage=False,
    )
    assert pf.ok, pf.errors
    assert pf.analysis_workbook == workbook

    lineage_pts = pd.DataFrame(
        {
            "track_id": [1, 1],
            "frame": [0, 1],
            "y": [10.0, 11.0],
            "x": [10.0, 11.0],
            "Cell_ID": [1, 1],
        }
    )
    per_cell = pd.read_excel(workbook, sheet_name="Per-Cell Data")
    joined = join_analysis_to_lineage_points(lineage_pts, per_cell)
    assert "Area (µm²)" in joined.columns or len(joined) >= 1
