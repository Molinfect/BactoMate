"""Unit tests for folder preview helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bactomate.gui.folder_preview.constants import (
    FOLDER_PREVIEW_MAX_FILES,
    cap_folder_preview_paths,
    folder_preview_enabled,
)
from bactomate.gui.folder_preview.fingerprint import settings_fingerprint


def test_folder_preview_enabled() -> None:
    assert folder_preview_enabled(0) is False
    assert folder_preview_enabled(1) is False
    assert folder_preview_enabled(2) is True
    assert folder_preview_enabled(100) is True


def test_cap_folder_preview_paths_preserves_order_and_reports_total() -> None:
    paths = [Path(f"f{i}.tif") for i in range(60)]
    capped, total = cap_folder_preview_paths(paths, max_files=FOLDER_PREVIEW_MAX_FILES)
    assert total == 60
    assert len(capped) == FOLDER_PREVIEW_MAX_FILES
    assert capped[0].name == "f0.tif"
    assert capped[-1].name == f"f{FOLDER_PREVIEW_MAX_FILES - 1}.tif"


def test_settings_fingerprint_stable_and_sensitive() -> None:
    a = {"diameter": 30.0, "flow": 0.4, "nested": {"x": 1}}
    b = {"nested": {"x": 1}, "flow": 0.4, "diameter": 30.0}
    c = {"diameter": 30.0, "flow": 0.5, "nested": {"x": 1}}
    assert settings_fingerprint(a) == settings_fingerprint(b)
    assert settings_fingerprint(a) != settings_fingerprint(c)


def test_folder_preview_grid_double_click(qapp) -> None:
    from PyQt6.QtGui import QPixmap

    from bactomate.gui.folder_preview.grid_widget import FolderPreviewGrid

    _ = qapp
    grid = FolderPreviewGrid()
    paths = [Path("a.tif"), Path("b.tif"), Path("c.tif")]
    grid.set_paths(paths, showing_of_total=(3, 3))
    seen: list[str] = []
    grid.tile_activated.connect(seen.append)
    pix = QPixmap(64, 64)
    pix.fill()
    grid.set_tile_result(paths[1], pix)
    tile = grid._tiles[str(paths[1])]
    tile.activated.emit(str(paths[1]))
    assert seen == [str(paths[1])]
    grid.close()
    grid.deleteLater()


def test_folder_preview_detail_view(qapp) -> None:
    from PyQt6.QtGui import QPixmap

    from bactomate.gui.folder_preview.grid_widget import FolderPreviewGrid

    _ = qapp
    grid = FolderPreviewGrid()
    paths = [Path("a.tif")]
    grid.set_paths(paths)
    pix = QPixmap(80, 60)
    pix.fill()
    grid.set_tile_result(paths[0], pix)
    assert not grid.is_detail_visible()
    grid.show_tile_detail(paths[0], pix)
    assert grid.is_detail_visible()
    grid.show_overview()
    assert not grid.is_detail_visible()
    grid.close()
    grid.deleteLater()


def test_render_segmentation_tile_draws_foci_markers() -> None:
    from bactomate.gui.folder_preview.seg_compute import render_segmentation_tile_rgb

    plane = np.zeros((64, 64), dtype=np.float32)
    plane[20:40, 20:40] = 100.0
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[20:40, 20:40] = 1
    peaks = [{"y": 30, "x": 30}]
    rgb = render_segmentation_tile_rgb(
        plane, mask, foci_layers=[(peaks, (255, 165, 0), (64, 64))], max_display=64
    )
    assert rgb.shape[2] == 3
    # Orange QC marker should tint some pixels near the peak.
    assert int(rgb[28:33, 28:33, 0].max()) >= 120


def test_render_segmentation_tile_draws_all_peaks_not_first_n_only() -> None:
    """Folder tiles must not truncate Spotiflow peaks to the first 500 (raster cutoff)."""
    from bactomate.gui.folder_preview.seg_compute import render_segmentation_tile_rgb

    h = w = 128
    plane = np.ones((h, w), dtype=np.float32) * 50.0
    mask = np.ones((h, w), dtype=np.uint16)
    # >500 peaks spanning top→bottom; a first-N=500 cap would leave the bottom empty.
    peaks = [{"y": float(y % h), "x": float(8 + (y // h) * 4)} for y in range(0, 600)]
    assert len(peaks) > 500
    rgb = render_segmentation_tile_rgb(
        plane, mask, foci_layers=[(peaks, (255, 165, 0), (h, w))], max_display=128
    )
    # Bottom band should still carry orange after drawing every peak.
    bottom = rgb[h - 8 :, :, 0]
    assert int(bottom.max()) >= 100


def test_folder_foci_qc_markers_do_not_paint_solid_carpet() -> None:
    """Dense peaks must not obliterate the underlying phase like radius-3 rings did."""
    from bactomate.gui.folder_preview.seg_compute import render_segmentation_tile_rgb
    from bactomate.gui.foci_overlay_draw import draw_peak_rings_on_rgb

    h = w = 64
    plane = np.full((h, w), 80.0, dtype=np.float32)
    mask = np.ones((h, w), dtype=np.uint16)
    peaks = [{"y": float(y), "x": float(x)} for y in range(0, h, 2) for x in range(0, w, 2)]
    rgb_qc = render_segmentation_tile_rgb(
        plane, mask, foci_layers=[(peaks, (255, 165, 0), (h, w))], max_display=64
    )
    rgb_rings = render_segmentation_tile_rgb(plane, mask, foci_layers=None, max_display=64)
    draw_peak_rings_on_rgb(
        rgb_rings, peaks, (255, 165, 0), mask_shape=(h, w), radius=3, cap=None
    )
    orange_qc = (rgb_qc[..., 0].astype(np.int16) - rgb_qc[..., 2].astype(np.int16)) > 80
    orange_rings = (rgb_rings[..., 0].astype(np.int16) - rgb_rings[..., 2].astype(np.int16)) > 80
    assert float(np.mean(orange_qc)) < float(np.mean(orange_rings)) * 0.75
    assert float(np.mean(orange_qc)) < 0.45


def test_rgb_to_qpixmap_roundtrip(qapp) -> None:
    from bactomate.gui.folder_preview.workers import rgb_to_qpixmap

    _ = qapp
    rgb = np.zeros((32, 40, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200
    pix = rgb_to_qpixmap(rgb)
    assert not pix.isNull()
    assert pix.width() == 40
    assert pix.height() == 32
