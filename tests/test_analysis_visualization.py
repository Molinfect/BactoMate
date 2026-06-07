"""Tests for combined analysis visualization helpers."""
from pathlib import Path

import numpy as np

from bactomate.gui.analysis_visualization import (
    _normalize_fluorescence_u8,
    _phase_stem_from_mask,
    _render_foci_panel_frame,
    _separate_label_positions,
    resolve_phase_image_for_mask,
    write_combined_analysis_visualization,
)


def test_phase_stem_from_mask_strips_suffix():
    p = Path("/data/ssb-mNG_cl1_001_Phase60X_mask_filtered.tif")
    assert _phase_stem_from_mask(p) == "ssb-mNG_cl1_001_Phase60X"


def test_resolve_phase_exact_match_not_shared_prefix(tmp_path):
    tif_dir = tmp_path / "Tif_files"
    tif_dir.mkdir()
    phase_a = tif_dir / "ssb-mNG_cl1__Phase60X.tif"
    phase_b = tif_dir / "ssb-mNG_cl1_001_Phase60X.tif"
    phase_a.write_bytes(b"x")
    phase_b.write_bytes(b"y")

    mask_a = tmp_path / "filtered_masks" / "ssb-mNG_cl1__Phase60X_mask_filtered.tif"
    mask_b = tmp_path / "filtered_masks" / "ssb-mNG_cl1_001_Phase60X_mask_filtered.tif"
    mask_a.parent.mkdir(parents=True, exist_ok=True)
    mask_b.parent.mkdir(parents=True, exist_ok=True)
    mask_a.write_bytes(b"m")
    mask_b.write_bytes(b"m")

    # Simulate base_dir_for_mask returning Tif_files parent
    import bactomate.gui.analysis_visualization as av

    original = av.base_dir_for_mask

    def _fake_base(mask_path: Path) -> Path:
        return tif_dir

    av.base_dir_for_mask = _fake_base
    try:
        assert resolve_phase_image_for_mask(mask_a) == phase_a
        assert resolve_phase_image_for_mask(mask_b) == phase_b
    finally:
        av.base_dir_for_mask = original


def test_fluorescence_lut_brighter_than_zeros():
    plane = np.zeros((64, 64), dtype=np.float32)
    plane[20:40, 20:40] = 50.0
    plane[30, 30] = 500.0
    out = _normalize_fluorescence_u8(plane)
    assert int(out.mean()) > 5
    assert int(out[30, 30]) > int(out[0, 0])


def test_separate_label_positions_spreads_nearby():
    labels = [("1", 10, 10), ("2", 11, 11), ("3", 12, 12)]
    out = _separate_label_positions(labels, min_sep_px=12)
    assert len(out) == 3
    assert out[0][1:] != out[1][1:] or out[1][1:] != out[2][1:]


def test_foci_panel_draws_after_cell_labels():
    """Cell labels use PIL; foci markers must still be drawable (writable array)."""
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[8:24, 8:24] = 1
    foci = np.random.RandomState(0).randint(0, 200, (32, 32), dtype=np.uint8)
    rows = [{"foci_id": 1, "y": 16, "x": 16}]
    panel = _render_foci_panel_frame(foci, mask, rows, title="GFP")
    assert panel.shape == (32, 32, 3)
    assert panel.dtype == np.uint8


def test_combined_visualization_loads_foci_stack_and_maps_generic_channel(tmp_path):
    """Legacy workbook channel=foci should draw markers on the discovered GFP panel."""
    import pandas as pd
    import tifffile
    import bactomate.gui.analysis_visualization as av

    mask_path = tmp_path / "sample_mCherry_mask_filtered.tif"
    foci_path = tmp_path / "sample_foci_GFP.tif"
    workbook = tmp_path / "sample_mCherry_mask_filtered_analysis.xlsx"
    out_path = tmp_path / "sample_mCherry_mask_filtered_analysis_visualization.tif"

    mask = np.zeros((1, 32, 32), dtype=np.int32)
    mask[:, 8:24, 8:24] = 1
    foci = np.zeros((1, 32, 32), dtype=np.uint16)
    foci[:, 16, 16] = 2000
    tifffile.imwrite(mask_path, mask)
    tifffile.imwrite(foci_path, foci)
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(
            [{"foci_id": 7, "cell_id": 1, "frame": 0, "channel_name": "foci", "y": 16, "x": 16}]
        ).to_excel(writer, sheet_name="Foci", index=False)

    class _Tab:
        def _resolve_foci_paths_for_mask(self, _mask_path):
            return [foci_path]

        def _extract_channel_tag_for_mask_stem(self, _file_stem, _mask_stem):
            return "GFP"

    import bactomate.gui.analysis_workbook as aw

    original = aw.discover_foci_csv_artifacts_for_mask

    def _fake_discover(_tab, _mask_path, prefer_filtered=True):
        from bactomate.gui.analysis_workbook import FociChannelArtifact

        csv_path = tmp_path / "sample_foci_filtered_GFP.csv"
        csv_path.write_text("frame,y,x,intensity,probability\n0,16,16,2000,0.9\n")
        return [FociChannelArtifact("GFP", foci_path, csv_path, "foci_filtered")]

    aw.discover_foci_csv_artifacts_for_mask = _fake_discover
    try:
        assert write_combined_analysis_visualization(
            _Tab(),
            mask_path=mask_path,
            output_path=out_path,
            analysis_workbook=workbook,
        )
    finally:
        aw.discover_foci_csv_artifacts_for_mask = original

    rendered = tifffile.imread(out_path)
    frame = rendered[0] if rendered.ndim == 4 else rendered
    foci_panel = frame[:, 34:, :]
    assert int(foci_panel[..., 0].max()) > 0
    assert np.any(np.all(foci_panel == np.array([255, 255, 0], dtype=np.uint8), axis=-1))
