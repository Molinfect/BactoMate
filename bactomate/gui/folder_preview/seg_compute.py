"""Pure compute helpers for segmentation folder-preview tiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import tifffile

from bactomate.gui.foci_overlay_draw import draw_peak_qc_markers_on_rgb
from bactomate.gui.image_utils import (
    PREVIEW_SEG_MAX_SIZE,
    build_segmentation_overlay,
    downsample_for_preview_seg,
    scale_for_preview,
    upsample_mask_to_shape,
)
from bactomate.modules.image_stack_utils import coerce_plane_to_2d, coerce_stack_to_tyx

# Match single-file segmentation preview ring colors.
FOCI_TILE_RING_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 165, 0),
    (0, 220, 255),
    (255, 80, 220),
    (120, 255, 120),
    (255, 230, 80),
)

FindFociSourcesFn = Callable[[Path], Sequence[Path]]


def load_first_plane(path: Path) -> np.ndarray:
    """Load frame 0 as float32 2D plane from a TIFF path."""
    raw = tifffile.imread(str(path), key=0)
    plane = coerce_plane_to_2d(np.asarray(raw))
    if plane.ndim != 2:
        stack = coerce_stack_to_tyx(np.asarray(tifffile.imread(str(path))))
        if stack is None or stack.size == 0:
            raise ValueError(f"Could not read 2D plane from {path.name}")
        plane = coerce_plane_to_2d(stack[0])
    return np.asarray(plane, dtype=np.float32)


def _to_display_u8(plane: np.ndarray) -> np.ndarray:
    arr = np.asarray(plane, dtype=np.float64)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def _to_three_channel(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return np.concatenate([arr, arr, arr], axis=-1)
        if arr.shape[-1] >= 3:
            return arr[..., :3]
    return np.stack([arr.squeeze(), arr.squeeze(), arr.squeeze()], axis=-1)


def segment_plane_with_model(
    plane: np.ndarray,
    model: Any,
    *,
    diameter: float,
    flow_threshold: float,
    mask_threshold: float,
    cluster: bool,
    is_cellpose_sam: bool = False,
    affinity_seg: bool = False,
    is_frozen: bool = False,
) -> np.ndarray:
    """Run Omnipose/Cellpose-SAM eval on one plane; return label mask at full plane shape."""
    from bactomate.gui.omnipose_utils import omnipose_eval_first_mask

    (seg_frame, seg_scale, (orig_h, orig_w)) = downsample_for_preview_seg(
        plane, max_size=PREVIEW_SEG_MAX_SIZE
    )
    effective = float(diameter) if diameter and float(diameter) > 0 else 30.0
    diameter_seg = max(1.0, effective * seg_scale)

    if is_cellpose_sam:
        img_3ch = _to_three_channel(seg_frame)
        result = model.eval(
            [img_3ch],
            diameter=diameter_seg if diameter_seg > 0 else None,
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(mask_threshold),
            batch_size=1,
        )
        batch_masks = result[0] if isinstance(result, tuple) else result
        if isinstance(batch_masks, np.ndarray) and batch_masks.ndim == 3:
            mask = batch_masks[0].astype(np.uint16)
        else:
            mask = np.asarray(batch_masks[0], dtype=np.uint16)
    else:
        kwargs = dict(
            channels=[0, 0],
            diameter=diameter_seg,
            flow_threshold=float(flow_threshold),
            mask_threshold=float(mask_threshold),
            cellprob_threshold=0.0,
            do_3D=False,
            omni=True,
            affinity_seg=bool(affinity_seg),
            cluster=bool(cluster),
            net_avg=False,
            resample=False,
            verbose=False,
        )
        if is_frozen:
            try:
                mask = omnipose_eval_first_mask(model.eval(seg_frame, **kwargs))
            except TypeError:
                kwargs["diameter"] = 30.0
                mask = omnipose_eval_first_mask(model.eval(seg_frame, **kwargs))
        else:
            mask = omnipose_eval_first_mask(model.eval(seg_frame, **kwargs))

    if seg_scale < 1.0 and mask is not None:
        mask = upsample_mask_to_shape(mask, orig_h, orig_w)
    return np.asarray(mask)


def _spotiflow_kwargs_from_params(sf: dict[str, Any]) -> dict[str, Any]:
    use_opt = bool(sf.get("use_optimized_threshold", True))
    prob = float(sf.get("prob_thresh") if sf.get("prob_thresh") is not None else 0.5)
    return {
        "use_optimized_threshold": use_opt,
        "prob_thresh": None if use_opt else float(prob),
        "min_distance": int(sf.get("min_distance", 1) or 1),
        "subpix": bool(sf.get("subpix", True)),
        "normalize": bool(sf.get("normalize", True)),
        "perc_low": float(sf.get("perc_low", 1.0) or 1.0),
        "perc_high": float(sf.get("perc_high", 99.8) or 99.8),
        "bg_subtract_radius": int(sf.get("bg_subtract_radius", 0) or 0),
        "blur_sigma": float(sf.get("blur_sigma", 0.0) or 0.0),
        "auto_tiling": bool(sf.get("auto_tiling", True)),
        "n_tiles_y": int(sf.get("n_tiles_y", 1) or 1),
        "n_tiles_x": int(sf.get("n_tiles_x", 1) or 1),
        "verbose": False,
        "raise_on_error": False,
    }


def detect_foci_peaks_on_plane(
    plane: np.ndarray,
    sf: dict[str, Any],
    *,
    spotiflow_model: Any = None,
) -> list[dict[str, Any]]:
    """Run Spotiflow on one plane; return peak list (empty on failure)."""
    from bactomate.modules.foci_utils import detect_peaks_spotiflow, load_spotiflow_model

    model = spotiflow_model
    if model is None:
        model = load_spotiflow_model(
            model_name=str(sf.get("model_name") or "general"),
            model_path=str(sf.get("model_path") or "").strip() or None,
        )
    peaks = detect_peaks_spotiflow(
        np.asarray(plane, dtype=np.float64),
        spotiflow_model=model,
        **_spotiflow_kwargs_from_params(sf),
    )
    if peaks is None:
        return []
    return list(peaks)


def collect_foci_layers_for_input(
    input_path: Path,
    params: dict[str, Any],
    *,
    find_foci_sources: FindFociSourcesFn,
    spotiflow_model: Any = None,
) -> list[tuple[list[dict[str, Any]], tuple[int, int, int], tuple[int, int]]]:
    """Discover foci stacks for ``input_path`` and detect peaks on frame 0.

    Returns list of ``(peaks, rgb, peak_coord_shape)`` where ``peak_coord_shape`` is the
    foci-plane ``(H, W)`` Spotiflow used (may differ from the phase tile).
    """
    from bactomate.gui.filter_preview_utils import foci_channel_entry_for_path

    if not bool(params.get("detect_foci", False)):
        return []
    try:
        sources = list(find_foci_sources(Path(input_path)) or [])
    except Exception:
        return []
    if not sources:
        return []

    default_sf = dict(params.get("spotiflow") or {})
    chmap = params.get("spotiflow_channels") if isinstance(params.get("spotiflow_channels"), dict) else {}
    csv_minp = float(params.get("foci_csv_min_probability", 0.0) or 0.0)
    layers: list[tuple[list[dict[str, Any]], tuple[int, int, int], tuple[int, int]]] = []
    for si, fp in enumerate(sources):
        try:
            foci_plane = load_first_plane(Path(fp))
            sf = dict(default_sf)
            # Canonical key, falling back to a legacy (pre-canonical) session key.
            _tag, ch = foci_channel_entry_for_path(chmap, Path(fp)) if isinstance(chmap, dict) else ("", None)
            if isinstance(ch, dict):
                sf.update(ch)
            peaks = detect_foci_peaks_on_plane(foci_plane, sf, spotiflow_model=spotiflow_model)
            if csv_minp > 0:
                peaks = [
                    p
                    for p in peaks
                    if p.get("probability") is None or float(p.get("probability") or 0.0) >= csv_minp
                ]
            color = FOCI_TILE_RING_RGB[si % len(FOCI_TILE_RING_RGB)]
            peak_shape = (int(foci_plane.shape[0]), int(foci_plane.shape[1]))
            layers.append((peaks, color, peak_shape))
        except Exception:
            continue
    return layers


def render_segmentation_tile_rgb(
    plane: np.ndarray,
    mask: np.ndarray,
    *,
    foci_layers: Optional[
        Sequence[
            tuple[Sequence[dict[str, Any]], tuple[int, int, int]]
            | tuple[Sequence[dict[str, Any]], tuple[int, int, int], tuple[int, int]]
        ]
    ] = None,
    max_display: int = 512,
) -> np.ndarray:
    """Build RGB overlay for a folder tile (downscaled for grid display)."""
    disp = _to_display_u8(plane)
    empty = np.zeros_like(mask, dtype=mask.dtype)
    disp_s, mask_s = scale_for_preview(disp, mask if mask is not None else empty, max_size=max_display)
    has_foci = bool(foci_layers) and any(bool(layer[0]) for layer in foci_layers)
    if mask is None:
        overlay = build_segmentation_overlay(disp_s, np.zeros(disp_s.shape[:2], dtype=np.uint16), alpha=0.0)
    else:
        # Softer cell tint when foci QC markers are present so dots stay readable.
        overlay = build_segmentation_overlay(disp_s, mask_s, alpha=0.32 if has_foci else 0.5)
    if foci_layers:
        phase_shape = (int(plane.shape[0]), int(plane.shape[1]))
        for layer in foci_layers:
            if len(layer) >= 3:
                peaks, color, peak_shape = layer[0], layer[1], layer[2]
            else:
                peaks, color = layer[0], layer[1]
                peak_shape = phase_shape
            if not peaks:
                continue
            # QC markers (not full preview rings): dense Spotiflow hits otherwise paint an
            # opaque orange carpet on downscaled tiles and hide the cells underneath.
            draw_peak_qc_markers_on_rgb(
                overlay,
                peaks,
                color,
                mask_shape=(int(peak_shape[0]), int(peak_shape[1])),
                alpha=0.55,
                plus=None,
            )
    return np.ascontiguousarray(overlay)


def compute_segmentation_tile(
    path: Path,
    model: Any,
    params: dict[str, Any],
    *,
    spotiflow_model: Any = None,
    find_foci_sources: Optional[FindFociSourcesFn] = None,
) -> tuple[Optional[np.ndarray], str]:
    """Load TIFF, segment frame 0, optionally detect foci, return (rgb, error)."""
    try:
        plane = load_first_plane(Path(path))
        mask = segment_plane_with_model(
            plane,
            model,
            diameter=float(params.get("diameter") or 0.0),
            flow_threshold=float(params.get("flow_threshold") or 0.4),
            mask_threshold=float(params.get("mask_threshold") or 0.0),
            cluster=bool(params.get("cluster", False)),
            is_cellpose_sam=bool(params.get("is_cellpose_sam", False)),
            affinity_seg=bool(params.get("affinity_seg", False)),
            is_frozen=bool(params.get("is_frozen", False)),
        )
        foci_layers: list[tuple[list[dict[str, Any]], tuple[int, int, int]]] = []
        if bool(params.get("detect_foci", False)) and find_foci_sources is not None:
            foci_layers = collect_foci_layers_for_input(
                Path(path),
                params,
                find_foci_sources=find_foci_sources,
                spotiflow_model=spotiflow_model,
            )
        rgb = render_segmentation_tile_rgb(plane, mask, foci_layers=foci_layers)
        return rgb, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
