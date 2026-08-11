"""Pure compute helpers for filtering folder-preview tiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile

from bactomate.gui.analyze_masks_preview_vis import build_preview_vis
from bactomate.gui.image_utils import scale_for_preview
from bactomate.modules.image_stack_utils import coerce_plane_to_2d, coerce_stack_to_tyx


def load_mask_plane0(path: Path) -> np.ndarray:
    raw = tifffile.imread(str(path), key=0)
    plane = coerce_plane_to_2d(np.asarray(raw))
    if plane.ndim != 2:
        stack = coerce_stack_to_tyx(np.asarray(tifffile.imread(str(path))))
        if stack is None or stack.size == 0:
            raise ValueError(f"Could not read mask plane from {path.name}")
        plane = coerce_plane_to_2d(stack[0])
    return np.asarray(plane)


def _find_phase_companion(mask_path: Path) -> Optional[Path]:
    """Best-effort phase/BF companion next to a mask TIFF."""
    stem = mask_path.name
    lower = stem.lower()
    for token in ("_masks_filtered", "_masks", "_mask"):
        idx = lower.rfind(token)
        if idx > 0:
            base = stem[:idx]
            break
    else:
        base = mask_path.stem
    parent = mask_path.parent
    candidates = [
        parent / f"{base}_phase.tif",
        parent / f"{base}_phase.tiff",
        parent / f"{base}.tif",
        parent / f"{base}.tiff",
    ]
    # Also search one level up common layouts.
    for folder in (parent, parent.parent):
        if not folder.is_dir():
            continue
        for p in folder.glob(f"{base}*phase*.tif*"):
            if p.is_file():
                return p
        for p in folder.glob(f"{base}*.tif*"):
            name = p.name.lower()
            if "_masks" in name or "_foci" in name or "_overlay" in name:
                continue
            if p.is_file():
                return p
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_plane0_optional(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not Path(path).is_file():
        return None
    try:
        raw = tifffile.imread(str(path), key=0)
        return coerce_plane_to_2d(np.asarray(raw))
    except Exception:
        return None


def apply_cell_filters_plane(
    mask: np.ndarray,
    params: dict[str, Any],
    *,
    phase_plane: Optional[np.ndarray] = None,
) -> tuple[set[int], set[int], int]:
    """Return (accepted, rejected, n_objects) for one mask plane."""
    from skimage.morphology import remove_small_objects

    from bactomate.scripts.analyze_cell_areas import (
        calculate_cell_properties,
        filter_by_size,
        filter_rod_shaped_cells,
    )

    work = np.asarray(mask)
    min_size = int(params.get("min_size_pixels") or 10)
    if bool(params.get("remove_small", True)) and min_size > 0:
        try:
            work = remove_small_objects(work, min_size=min_size)
        except Exception:
            work = np.asarray(mask)

    properties = calculate_cell_properties(
        work,
        pixel_size=float(params.get("pixel_size") or 0.108),
        exclude_border=bool(params.get("exclude_border", False)),
    )
    rod = filter_rod_shaped_cells(
        properties,
        min_aspect_ratio=float(params.get("min_aspect_ratio") or 0.0),
        max_aspect_ratio=params.get("max_aspect_ratio"),
        min_eccentricity=float(params.get("min_eccentricity") or 0.0),
        min_solidity=float(params.get("min_solidity") or 0.0),
    )
    filtered = filter_by_size(
        rod,
        min_area_um2=params.get("min_area"),
        max_area_um2=params.get("max_area"),
        min_width_um=params.get("min_width"),
        max_width_um=params.get("max_width"),
        min_length_um=params.get("min_length"),
        max_length_um=params.get("max_length"),
    )
    accepted = {int(cell["label"]) for cell in filtered}
    if bool(params.get("focus_filter_enabled")) and accepted and phase_plane is not None:
        try:
            from bactomate.modules.cell_focus_score import filter_labels_by_focus, score_cells_in_frame

            scores = score_cells_in_frame(
                np.asarray(phase_plane, dtype=np.float64),
                work,
                accepted,
                modality=params.get("focus_modality", "phase"),
            )
            frac = float(params.get("focus_min_fraction_of_median") or 0.5)
            accepted = filter_labels_by_focus(scores, min_fraction_of_median=frac)
        except Exception:
            pass
    all_labels = {int(x) for x in np.unique(work) if int(x) > 0}
    rejected = all_labels - accepted
    return accepted, rejected, len(all_labels)


def render_filter_tile_rgb(
    mask: np.ndarray,
    accepted: set[int],
    rejected: set[int],
    *,
    fi_plane: Optional[np.ndarray] = None,
    max_display: int = 512,
) -> np.ndarray:
    vis, *_rest = build_preview_vis(
        np.asarray(mask),
        accepted,
        rejected,
        fi_plane,
        do_fi_contour=False,
        fi_contour_thickness_val=1,
        run_foci=False,
        foci_snapshot=None,
    )
    # Downscale for grid: scale_for_preview needs image+mask; use vis channels as image.
    gray = vis.mean(axis=2).astype(np.uint8)
    scaled_img, _ = scale_for_preview(gray, np.asarray(mask), max_size=max_display)
    # Rebuild small vis from scaled mask proportions via simple resize of RGB.
    h, w = scaled_img.shape[:2]
    if vis.shape[0] == h and vis.shape[1] == w:
        return np.ascontiguousarray(vis)
    try:
        from scipy.ndimage import zoom

        zh, zw = h / vis.shape[0], w / vis.shape[1]
        out = zoom(vis.astype(np.float32), (zh, zw, 1), order=0).astype(np.uint8)
        return np.ascontiguousarray(out)
    except Exception:
        y = np.linspace(0, vis.shape[0] - 1, h).astype(np.intp)
        x = np.linspace(0, vis.shape[1] - 1, w).astype(np.intp)
        return np.ascontiguousarray(vis[np.ix_(y, x)])


def compute_filter_tile(path: Path, params: dict[str, Any]) -> tuple[Optional[np.ndarray], str]:
    """Load mask frame 0, apply cell filters, return (rgb, error)."""
    try:
        mask_path = Path(path)
        mask = load_mask_plane0(mask_path)
        phase_path = _find_phase_companion(mask_path)
        phase = _load_plane0_optional(phase_path)
        accepted, rejected, _n = apply_cell_filters_plane(mask, params, phase_plane=phase)
        rgb = render_filter_tile_rgb(mask, accepted, rejected, fi_plane=phase)
        return rgb, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
