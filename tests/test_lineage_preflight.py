"""Tests for lineage preflight validation (mask, workbook, frame counts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile


def _write_mask_stack(path: Path, n_frames: int) -> None:
    stack = np.zeros((n_frames, 8, 8), dtype=np.uint16)
    for t in range(n_frames):
        stack[t, 2:4, 2:4] = 1
    tifffile.imwrite(path, stack)


def _write_analysis_workbook(path: Path, n_frames: int, *, one_indexed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = 1 if one_indexed else 0
    frames = list(range(start, start + n_frames))
    summary = pd.DataFrame({"Frame": frames, "Cell Count": [1] * n_frames})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)


def test_preflight_strict_requires_analysis_workbook(tmp_path: Path) -> None:
    from bactomate.modules.lineage_integration import preflight_lineage_run

    mask = tmp_path / "sample_masks.tif"
    _write_mask_stack(mask, 3)
    pf = preflight_lineage_run(
        mask,
        linker_mode="greedy",
        model_name="general_2d",
        require_analysis_workbook=True,
        strict_lineage=True,
    )
    assert not pf.ok
    assert any("workbook" in e.lower() for e in pf.errors)


def test_preflight_strict_allows_raw_when_no_filtered_mask(tmp_path: Path) -> None:
    from bactomate.modules.lineage_integration import preflight_lineage_run

    raw = tmp_path / "sample_masks.tif"
    _write_mask_stack(raw, 2)
    wb = tmp_path / "_analysis" / "sample_analysis.xlsx"
    _write_analysis_workbook(wb, 2)

    pf = preflight_lineage_run(
        raw,
        linker_mode="greedy",
        model_name="general_2d",
        require_analysis_workbook=True,
        strict_lineage=True,
    )
    assert pf.ok
    assert not pf.mask_is_filtered
    assert not any("filtered" in e.lower() for e in pf.errors)


def test_preflight_auto_resolves_filtered_when_available(tmp_path: Path) -> None:
    from bactomate.modules.lineage_integration import preflight_lineage_run

    filt_dir = tmp_path / "filtered_masks"
    filt_dir.mkdir()
    raw = tmp_path / "sample_masks.tif"
    filtered = filt_dir / "sample_mask_filtered.tif"
    _write_mask_stack(raw, 2)
    _write_mask_stack(filtered, 2)

    pf = preflight_lineage_run(
        raw,
        linker_mode="greedy",
        model_name="general_2d",
        strict_lineage=True,
    )
    assert pf.ok
    assert pf.mask_is_filtered
    assert pf.resolved_mask_path == filtered


def test_preflight_strict_blocks_forced_raw_when_filtered_exists(tmp_path: Path) -> None:
    from bactomate.modules.lineage_integration import preflight_lineage_run

    filt_dir = tmp_path / "filtered_masks"
    filt_dir.mkdir()
    raw = tmp_path / "sample_masks.tif"
    filtered = filt_dir / "sample_mask_filtered.tif"
    _write_mask_stack(raw, 2)
    _write_mask_stack(filtered, 2)

    pf = preflight_lineage_run(
        raw,
        linker_mode="greedy",
        model_name="general_2d",
        prefer_filtered_mask=False,
        strict_lineage=True,
    )
    assert not pf.ok
    assert any("filtered" in e.lower() for e in pf.errors)


def test_preflight_frame_count_mismatch_warns(tmp_path: Path) -> None:
    from bactomate.modules.lineage_integration import preflight_lineage_run

    base = tmp_path / "Tif_001"
    seg = base / "_segmentation_omni"
    seg.mkdir(parents=True)
    mask = seg / "sample_masks.tif"
    _write_mask_stack(mask, 5)
    analysis_dir = base / "_analysis"
    wb = analysis_dir / "sample_analysis.xlsx"
    _write_analysis_workbook(wb, 3)

    pf = preflight_lineage_run(
        mask,
        linker_mode="greedy",
        model_name="general_2d",
        strict_lineage=False,
    )
    assert pf.ok
    assert pf.frame_count_mismatch
    assert pf.workbook_frames == 3
    assert pf.n_frames == 5
    assert any("frame" in w.lower() for w in pf.warnings)


def test_preflight_frame_count_match(tmp_path: Path) -> None:
    from bactomate.modules.lineage_integration import preflight_lineage_run

    base = tmp_path / "Tif_001"
    seg = base / "_segmentation_omni"
    seg.mkdir(parents=True)
    mask = seg / "sample_masks.tif"
    _write_mask_stack(mask, 4)
    wb = base / "_analysis" / "sample_analysis.xlsx"
    _write_analysis_workbook(wb, 4)

    pf = preflight_lineage_run(
        mask,
        linker_mode="greedy",
        model_name="general_2d",
    )
    assert pf.ok
    assert not pf.frame_count_mismatch
    assert pf.workbook_frames == 4
    assert pf.n_frames == 4


def test_workbook_summary_frame_count_helper(tmp_path: Path) -> None:
    from bactomate.modules.lineage_integration import _workbook_summary_frame_count

    wb = tmp_path / "test_analysis.xlsx"
    _write_analysis_workbook(wb, 6, one_indexed=True)
    count, is_1_indexed = _workbook_summary_frame_count(wb)
    assert count == 6
    assert is_1_indexed
