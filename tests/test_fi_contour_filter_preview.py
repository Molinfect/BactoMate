from __future__ import annotations

import numpy as np


def test_paint_contour_fi_ring_marks_accepted_cell() -> None:
    from bactomate.gui.analyze_masks_preview_vis import paint_contour_fi_ring_on_display_rgb

    mask = np.zeros((12, 12), dtype=np.int32)
    mask[4:9, 4:9] = 1
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    paint_contour_fi_ring_on_display_rgb(rgb, mask, {1}, thickness_px=1)
    assert int(rgb.max()) > 0


def test_auto_display_lut_range_uses_moderate_percentiles() -> None:
    from bactomate.gui.fi_contour_filter_preview import auto_display_lut_range

    plane = np.zeros((100, 100), dtype=np.float32)
    plane[40:60, 40:60] = 200.0
    plane += 5.0
    lo, hi = auto_display_lut_range(plane)
    assert lo >= 0.0
    assert hi <= 255.0
    assert hi > lo


def test_paint_contour_fi_ring_separates_touching_cells() -> None:
    """Touching labels must each get a contour, not one hull around the cluster."""
    from bactomate.gui.analyze_masks_preview_vis import (
        FI_CONTOUR_PREVIEW_RGB,
        paint_contour_fi_ring_on_display_rgb,
    )

    mask = np.zeros((10, 14), dtype=np.int32)
    mask[2:8, 1:7] = 1
    mask[2:8, 7:13] = 2
    rgb = np.full((10, 14, 3), 128, dtype=np.uint8)
    paint_contour_fi_ring_on_display_rgb(rgb, mask, {1, 2}, thickness_px=1)
    r0, g0, b0 = FI_CONTOUR_PREVIEW_RGB
    tinted = (rgb[:, :, 0] == r0) & (rgb[:, :, 1] == g0) & (rgb[:, :, 2] == b0)
    # Shared vertical edge between the two rods (not only the cluster outline).
    assert bool(np.any(tinted[3:7, 5:8]))


def test_high_contrast_contour_darkens_halo() -> None:
    from bactomate.gui.analyze_masks_preview_vis import paint_contour_fi_ring_on_display_rgb

    mask = np.zeros((12, 12), dtype=np.int32)
    mask[4:9, 4:9] = 1
    rgb_plain = np.full((12, 12, 3), 220, dtype=np.uint8)
    rgb_hc = rgb_plain.copy()
    paint_contour_fi_ring_on_display_rgb(rgb_plain, mask, {1}, thickness_px=1)
    paint_contour_fi_ring_on_display_rgb(
        rgb_hc, mask, {1}, thickness_px=1, high_contrast_outline=True
    )
    assert int(rgb_hc.min()) < int(rgb_plain.min())


def test_fi_contour_preview_button_requires_masks() -> None:
    from bactomate.gui.fi_contour_filter_preview import (
        accepted_cell_labels_from_mask,
        fi_contour_preview_button_enabled,
    )

    class _Ck:
        def __init__(self, checked: bool) -> None:
            self._checked = checked

        def isChecked(self) -> bool:
            return self._checked

    class _Tab:
        analyze_fi = _Ck(True)
        analyze_fi_contour = _Ck(True)

        def _get_mask_files_for_feedback(self):
            return []

    enabled, _ = fi_contour_preview_button_enabled(_Tab())
    assert enabled is False

    mask = np.zeros((8, 8), dtype=np.int32)
    mask[2:6, 2:6] = 3
    assert accepted_cell_labels_from_mask(mask) == {3}
