"""
Standard pipeline QC artifacts: overlays, filter visualizations, PNG previews.

These outputs are always written by BactoMate (not gated on optional UI checkboxes).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import tifffile

from bactomate.modules.image_stack_utils import coerce_stack_to_tyx

LogFn = Callable[[str], None] | None
_logger = logging.getLogger(__name__)


def write_png_preview_from_tiff(
    tiff_path: Path | str,
    *,
    png_path: Path | str | None = None,
    frame: int | str = "mid",
) -> Path | None:
    """
    Write a single-frame PNG preview from a multi-page TIFF stack.

    ``frame`` may be an int index or ``"mid"`` (default) for the middle frame.
    """
    tiff_path = Path(tiff_path)
    if not tiff_path.is_file():
        return None
    if png_path is None:
        png_path = tiff_path.with_suffix(".png")
    else:
        png_path = Path(png_path)
    try:
        stack = tifffile.imread(str(tiff_path))
        if stack.ndim == 2:
            plane = stack
        elif stack.ndim >= 3:
            n = int(stack.shape[0])
            if frame == "mid":
                idx = n // 2
            else:
                idx = int(frame)
                idx = max(0, min(idx, n - 1))
            plane = stack[idx]
        else:
            return None
        if plane.ndim == 3 and plane.shape[-1] in (3, 4):
            rgb = np.asarray(plane[..., :3], dtype=np.uint8)
        elif plane.ndim == 2:
            from bactomate.scripts.analyze_cell_areas import _normalize_plane_for_visualization_u8

            gray = _normalize_plane_for_visualization_u8(plane)
            if gray is None:
                return None
            rgb = np.stack([gray, gray, gray], axis=-1)
        else:
            return None
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(str(png_path))
        except ImportError:
            import imageio.v2 as imageio

            imageio.imwrite(str(png_path), rgb)
        return png_path
    except Exception as exc:
        _logger.warning("PNG preview failed for %s: %s", tiff_path, exc)
        return None


def write_png_previews_for_tiffs(paths: Iterable[Path | str], log: LogFn = None) -> list[Path]:
    """Write PNG previews for each existing TIFF path; return paths written."""
    written: list[Path] = []
    for p in paths:
        out = write_png_preview_from_tiff(p)
        if out is not None:
            written.append(out)
            if log is not None:
                log(f"PNG preview: {Path(p).name} → {out.name}\n")
    return written


def regenerate_segmentation_overlay(
    mask_path: Path | str,
    phase_path: Path | str | None = None,
    *,
    log: LogFn = None,
) -> Path | None:
    """
    Regenerate ``overlays/{stem}_overlay.tif`` from mask + phase with Cell ID labels.

    Uses the mask pixel values as Cell ID labels (persistent IDs when assigned).
    """
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        return None
    if phase_path is None:
        try:
            from bactomate.gui.analysis_visualization import resolve_phase_image_for_mask

            phase_path = resolve_phase_image_for_mask(mask_path)
        except Exception:
            phase_path = None
    if phase_path is None or not Path(phase_path).is_file():
        if log is not None:
            log(f"[Overlay] No phase image found for {mask_path.name}; skipped overlay regeneration.\n")
        return None
    phase_path = Path(phase_path)
    stem = mask_path.stem
    if stem.endswith("_masks"):
        stem = stem[: -len("_masks")]
    elif stem.endswith("_unet_masks"):
        stem = stem[: -len("_unet_masks")]
    # Keep overlays under overlays/ (same as the segment CLI) — avoid a second copy at folder root.
    overlay_dir = mask_path.parent / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"{stem}_overlay.tif"
    try:
        from bactomate.scripts.batch_process_tiff_sequence import overlay_masks_on_images

        images = tifffile.imread(str(phase_path))
        masks = tifffile.imread(str(mask_path))
        images_tyx = coerce_stack_to_tyx(images)
        masks_tyx = coerce_stack_to_tyx(masks)
        if images_tyx is None or masks_tyx is None:
            return None
        n = min(images_tyx.shape[0], masks_tyx.shape[0])
        overlay_masks_on_images(images_tyx[:n], masks_tyx[:n], str(overlay_path))
        write_png_preview_from_tiff(overlay_path)
        # Drop legacy root-level overlay copies left by older builds.
        for legacy in (
            mask_path.parent / f"{stem}_overlay.tif",
            mask_path.parent / f"{stem}_overlay.png",
        ):
            try:
                if legacy.is_file() and legacy.resolve() != overlay_path.resolve():
                    legacy.unlink()
            except Exception:
                pass
        if log is not None:
            log(f"Segmentation overlay (Cell IDs): {mask_path.name} → overlays/{overlay_path.name}\n")
        return overlay_path
    except Exception as exc:
        _logger.warning("Overlay regeneration failed for %s: %s", mask_path, exc)
        if log is not None:
            log(f"[Overlay] Failed for {mask_path.name}: {exc}\n")
        return None


def build_filter_frame_data_from_masks(
    original_masks: np.ndarray,
    filtered_masks: np.ndarray,
    *,
    pixel_size: float = 0.108,
    exclude_border: bool = False,
) -> list[dict[str, Any]]:
    """Build minimal per-frame data for filter visualization from mask stacks."""
    from bactomate.scripts.analyze_cell_areas import calculate_cell_properties

    all_frame_data: list[dict[str, Any]] = []
    n = min(int(original_masks.shape[0]), int(filtered_masks.shape[0]))
    for fi in range(n):
        orig = np.asarray(original_masks[fi], dtype=np.int32)
        filt = np.asarray(filtered_masks[fi], dtype=np.int32)
        try:
            props = calculate_cell_properties(orig, pixel_size=float(pixel_size), exclude_border=exclude_border)
        except Exception:
            props = []
        all_labels = {int(p["label"]) for p in props if int(p.get("label", 0)) > 0}
        accepted_labels = {int(v) for v in np.unique(filt) if int(v) > 0}
        rejected_labels = all_labels - accepted_labels
        cells = []
        for lab in sorted(accepted_labels):
            cells.append({"cell_id": int(lab), "label": int(lab)})
        all_frame_data.append(
            {
                "frame": fi + 1,
                "cells": cells,
                "accepted_labels": accepted_labels,
                "rejected_labels": rejected_labels,
            }
        )
    return all_frame_data


def write_filter_stage_visualization(
    *,
    mask_path: Path | str,
    original_masks: np.ndarray,
    filtered_masks: np.ndarray,
    pixel_size: float = 0.108,
    exclude_border: bool = False,
    foci_table: Sequence[dict[str, Any]] | None = None,
    log: LogFn = None,
) -> Path | None:
    """
    Write ``visualizations/{stem}_filter_visualization.tif`` (+ PNG) after Filtering.
    """
    mask_path = Path(mask_path)
    try:
        from bactomate.gui.analysis_paths import (
            analysis_dir_from_base,
            analysis_visualizations_dir,
            base_dir_for_mask,
        )
        from bactomate.gui.analysis_visualization import resolve_phase_image_for_mask
        from bactomate.scripts.analyze_cell_areas import create_filter_visualization

        analysis_dir = analysis_dir_from_base(base_dir_for_mask(mask_path))
        vis_dir = analysis_visualizations_dir(analysis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_stem = mask_path.stem.replace("_masks", "").replace("_unet_masks", "")
        vis_output = vis_dir / f"{vis_stem}_filter_visualization.tif"

        all_frame_data = build_filter_frame_data_from_masks(
            original_masks,
            filtered_masks,
            pixel_size=pixel_size,
            exclude_border=exclude_border,
        )

        bg_images = None
        phase_path = resolve_phase_image_for_mask(mask_path)
        if phase_path is not None and phase_path.is_file():
            try:
                bg_images = coerce_stack_to_tyx(tifffile.imread(str(phase_path)))
            except Exception:
                bg_images = None

        create_filter_visualization(
            filtered_masks,
            all_frame_data,
            str(vis_output),
            masks_original=original_masks,
            foci_table=list(foci_table) if foci_table else None,
            background_images=bg_images,
        )
        write_png_preview_from_tiff(vis_output)
        if log is not None:
            log(f"[Filtering] Wrote filter visualization: {vis_output.name}\n")
        return vis_output
    except Exception as exc:
        _logger.warning("Filter stage visualization failed for %s: %s", mask_path, exc)
        if log is not None:
            log(f"[Filtering] Filter visualization failed for {mask_path.name}: {exc}\n")
        return None
