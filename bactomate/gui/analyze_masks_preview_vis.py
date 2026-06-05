"""
Pure (no Qt) helper to build the Analyze Masks filter-preview overlay in a background thread.
Returns (vis, peaks_for_hover, n_foci_total, n_foci_drawn, raw_median_fi, min_rel, mode_txt).

Note: Filtering-stage foci CSV peak-filter summaries are computed in
`AnalyzeMasksPreviewWorker` and reported via the same `(n_foci_total, n_foci_drawn)`
fields; this module only handles visualization/drawing when `run_foci=True`.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Optional

from bactomate.modules.image_stack_utils import coerce_plane_to_2d

FOCI_CAP = 500

FOCI_MULTI_COLORS = [
    np.array([255, 165, 0], dtype=np.uint8),
    np.array([255, 0, 255], dtype=np.uint8),
    np.array([0, 255, 255], dtype=np.uint8),
    np.array([255, 255, 0], dtype=np.uint8),
    np.array([0, 255, 128], dtype=np.uint8),
]

# Contour-FI preview ring on filter preview (high contrast on phase / foci backgrounds).
FI_CONTOUR_PREVIEW_RGB = (255, 60, 255)


def paint_contour_fi_ring_on_display_rgb(
    rgb: np.ndarray,
    mask_2d: np.ndarray,
    accepted_labels: set[int] | frozenset[int],
    *,
    thickness_px: int = 1,
    blend_alpha: float = 0.95,
    high_contrast_outline: bool = False,
) -> None:
    """
    Draw the Contour-FI measurement ring on *rgb* (display resolution).

    Uses the same erosion-based membrane ring as ``analyze_cell_areas.calculate_fi_intensities``
    (not ``find_boundaries`` inner), scaled to match the preview canvas.
    """
    if rgb is None or getattr(rgb, "size", 0) == 0:
        return
    label = np.asarray(mask_2d, dtype=np.int32)
    if label.ndim != 2 or not accepted_labels:
        return
    accepted_arr = np.array(list(accepted_labels), dtype=np.int32)
    if accepted_arr.size == 0:
        return
    from skimage.morphology import binary_dilation, binary_erosion

    h0, w0 = int(label.shape[0]), int(label.shape[1])
    hs, ws = int(rgb.shape[0]), int(rgb.shape[1])
    if h0 == hs and w0 == ws:
        scaled_mask = label
    else:
        try:
            from scipy.ndimage import zoom

            sy = hs / h0 if h0 else 1.0
            sx = ws / w0 if w0 else 1.0
            scaled_mask = zoom(label.astype(np.float32), (sy, sx), order=0).astype(np.int32)
        except Exception:
            return

    accepted_binary = np.isin(scaled_mask, accepted_arr)
    if not np.any(accepted_binary):
        return
    thick = max(1, int(thickness_px))
    # Per-cell erosion rings (same as analyze_cell_areas.calculate_fi_intensities).
    # A single erosion on the union of labels would outline only the outer hull of
    # touching cells, not each bacterium.
    ring = np.zeros(scaled_mask.shape, dtype=bool)
    for lab in accepted_arr:
        cell_binary = scaled_mask == int(lab)
        if not np.any(cell_binary):
            continue
        eroded = binary_erosion(cell_binary)
        cell_ring = cell_binary & (~eroded)
        for _ in range(thick - 1):
            cell_ring = binary_dilation(cell_ring) & cell_binary
        ring |= cell_ring
    # Extra pixel at preview scale so a 1 px analysis thickness stays visible when downscaled.
    if max(h0, w0) > max(hs, ws) and thick == 1:
        ring = binary_dilation(ring) & accepted_binary
    if not np.any(ring):
        return

    alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    r0, g0, b0 = FI_CONTOUR_PREVIEW_RGB
    plane = np.asarray(rgb, dtype=np.float32)
    if high_contrast_outline:
        halo = binary_dilation(ring) & accepted_binary
        plane[halo, 0] *= 0.35
        plane[halo, 1] *= 0.35
        plane[halo, 2] *= 0.35
    plane[ring, 0] = plane[ring, 0] * (1.0 - alpha) + float(r0) * alpha
    plane[ring, 1] = plane[ring, 1] * (1.0 - alpha) + float(g0) * alpha
    plane[ring, 2] = plane[ring, 2] * (1.0 - alpha) + float(b0) * alpha
    rgb[:] = plane.astype(np.uint8)


def _fi_to_2d(fi_image: np.ndarray) -> np.ndarray:
    return np.asarray(coerce_plane_to_2d(np.asarray(fi_image)), dtype=np.float64)


def _foci_detect_draw_channel(
    vis: np.ndarray,
    fi_work: np.ndarray,
    mask: np.ndarray,
    accepted_labels: set[int],
    foci_snapshot: Optional[dict[str, Any]],
    draw_color: np.ndarray,
    peak_cap: int,
    channel_label: Optional[str],
) -> tuple[int, int, list[dict], Optional[float], Optional[float], str]:
    """One foci detection + draw pass. peak_cap limits peaks drawn (split across multi-channel preview)."""
    peaks_out: list[dict] = []
    raw_median_fi: Optional[float] = None
    min_rel: Optional[float] = None
    mode_txt = ""
    n_drawn = 0
    sn = foci_snapshot or {}
    detection_mode = int(sn.get("detection_mode", 0))
    try:
        from bactomate.modules.foci_utils import (
            detect_peaks_skimage as _detect_foci,
            detect_peaks_skimage_auto as _detect_foci_auto,
            detect_peaks_cell_first_region_grow as _detect_foci_cell_first,
            detect_peaks_spotiflow as _detect_foci_spotiflow,
            refine_spotiflow_peaks,
            get_focus_blob_mask,
        )
    except ImportError as e:
        if detection_mode == 2:
            raise RuntimeError(f"Spotiflow dependencies are unavailable: {e}") from e
        return (0, 0, [], None, None, "")
    if _detect_foci is None and _detect_foci_auto is None:
        return (0, 0, [], None, None, "")
    if fi_work.shape[:2] != vis.shape[:2]:
        return (0, 0, [], None, None, "")
    try:
        raw_median_fi = float(np.median(fi_work))
        rel_mode = (sn.get("mode") or "per_cell").strip().lower()
        min_sep = sn.get("min_sep", 2)
        blob_thresh = sn.get("blob_threshold", 0.8)
        try:
            blob_thresh = float(blob_thresh)
        except (TypeError, ValueError):
            blob_thresh = 0.8
        if not np.isfinite(blob_thresh):
            blob_thresh = 0.8
        blob_thresh = float(np.clip(blob_thresh, 0.05, 1.0))
        peaks = None
        if detection_mode == 2:
            if _detect_foci_spotiflow is None:
                raise RuntimeError("Spotiflow detector is unavailable in this environment.")
            sf_kw = dict(sn.get("spotiflow") or {})
            sf_kw["raise_on_error"] = True
            try:
                peaks = _detect_foci_spotiflow(fi_work, **sf_kw)
            except TypeError:
                peaks = _detect_foci_spotiflow(
                    fi_work,
                    model_name=str(sf_kw.get("model_name") or "general"),
                    raise_on_error=True,
                )
            if peaks is None:
                raise RuntimeError("Spotiflow detection returned no result.")
            sf = dict(sn.get("spotiflow") or {})
            pf_on = bool(sf.get("postfilters_enabled", False))
            min_sz = int(sf.get("pf_min_size_px", 0) or 0) if pf_on and bool(sf.get("pf_area_enabled", False)) else None
            max_sz = int(sf.get("pf_max_size_px", 0) or 0) if pf_on and bool(sf.get("pf_area_enabled", False)) else None
            if max_sz is not None and max_sz <= 0:
                max_sz = None
            min_peak_raw = (
                float(sf.get("pf_min_peak_raw", 0.0) or 0.0)
                if pf_on and bool(sf.get("pf_min_peak_raw_enabled", False))
                else None
            )
            min_focus_mean = (
                float(sf.get("pf_min_focus_mean", 0.0) or 0.0)
                if pf_on and bool(sf.get("pf_min_focus_mean_enabled", False))
                else None
            )
            per_cell_ratio = (
                float(sf.get("pf_cell_ratio_factor", 2.0) or 2.0)
                if pf_on and bool(sf.get("pf_per_cell_ratio_enabled", False))
                else None
            )
            peaks = refine_spotiflow_peaks(
                fi_work,
                peaks,
                labeled_mask=mask,
                threshold_frac=blob_thresh,
                max_radius=5,
                min_size=min_sz,
                max_size=max_sz,
                min_peak_raw=min_peak_raw,
                min_focus_mean=min_focus_mean,
                per_cell_ratio=per_cell_ratio,
            )
        elif detection_mode == 1:
            if _detect_foci_cell_first is not None:
                peaks = _detect_foci_cell_first(fi_work, labeled_mask=mask)
            elif _detect_foci_auto is not None:
                peaks = _detect_foci_auto(fi_work, min_distance=min_sep, labeled_mask=mask)
        elif _detect_foci is not None:
            mode_txt = rel_mode
            if rel_mode in ("per_cell", "per cell"):
                min_rel = None
            else:
                min_rel = sn.get("min_rel")
                if min_rel is None:
                    min_rel = 2.0
            min_sz = sn.get("min_sz")
            max_sz = sn.get("max_sz")
            if min_sz is None or max_sz is None:
                min_sz = 1
                max_sz = 500
            min_peak_raw = sn.get("min_peak_raw") or 0
            min_focus_mean = sn.get("min_focus_mean") or 0
            peaks = _detect_foci(
                fi_work,
                min_rel=min_rel,
                min_distance=min_sep,
                min_size=min_sz,
                max_size=max_sz,
                size_threshold_frac=blob_thresh,
                min_peak_raw=min_peak_raw if min_peak_raw and float(min_peak_raw) > 0 else None,
                min_focus_mean=min_focus_mean if min_focus_mean and float(min_focus_mean) > 0 else None,
                labeled_mask=mask,
            )
        if detection_mode != 0:
            mode_txt = "auto" if detection_mode == 1 else "spotiflow"
        if peaks is None:
            peaks = []
        exclude_outside = sn.get("exclude_outside", False)
        if exclude_outside:
            peaks = [
                p
                for p in peaks
                if 0 <= int(p["y"]) < mask.shape[0]
                and 0 <= int(p["x"]) < mask.shape[1]
                and mask[int(p["y"]), int(p["x"])] in accepted_labels
            ]
        n_foci = len(peaks)
        h, w = vis.shape[:2]
        peaks_to_draw = peaks if peak_cap > 0 else []
        if get_focus_blob_mask is not None:
            from skimage.segmentation import find_boundaries

            fi_f32 = fi_work.astype(np.float32)
            for i, p in enumerate(peaks_to_draw):
                if i >= peak_cap:
                    break
                yi, xi = int(p["y"]), int(p["x"])
                blob = get_focus_blob_mask(fi_f32, yi, xi, threshold_frac=blob_thresh, max_radius=5, labeled_mask=mask)
                if np.any(blob):
                    vis[blob] = draw_color
                    boundaries = find_boundaries(blob, mode="outer")
                    if np.any(boundaries):
                        vis[boundaries, 0] = 0
                        vis[boundaries, 1] = 0
                        vis[boundaries, 2] = 0
        else:
            for i, p in enumerate(peaks_to_draw):
                if i >= peak_cap:
                    break
                yi, xi = int(p["y"]), int(p["x"])
                r = max(2, min(5, int(round((max(1, p.get("size_px", 1)) / np.pi) ** 0.5))))
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dy * dy + dx * dx <= r * r:
                            yk, xk = yi + dy, xi + dx
                            if 0 <= yk < h and 0 <= xk < w:
                                vis[yk, xk] = draw_color
        n_drawn = min(n_foci, max(0, int(peak_cap)))
        for p in peaks_to_draw[:max(0, int(peak_cap))]:
            pd = dict(p)
            if channel_label:
                pd["foci_channel"] = channel_label
            peaks_out.append(pd)
        return (n_foci, n_drawn, peaks_out, raw_median_fi, min_rel, mode_txt)
    except Exception as e:
        if detection_mode == 2:
            raise
        raise RuntimeError(f"Foci preview detection failed: {e}") from e


def build_preview_vis(
    mask: np.ndarray,
    accepted_labels: set[int],
    rejected_labels: set[int],
    fi_image: Optional[np.ndarray],
    do_fi_contour: bool,
    fi_contour_thickness_val: int,
    run_foci: bool,
    foci_snapshot: Optional[dict[str, Any]],
    progress_callback: Optional[Any] = None,
    multi_foci_fi_2d: Optional[list[np.ndarray]] = None,
    foci_channel_labels: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[dict], int, int, Optional[float], Optional[float], str]:
    """
    Build RGB overlay (vis) and foci hover list. Safe to call from a worker thread.
    foci_snapshot: dict with mode, min_rel, min_sep, min_sz, max_sz, min_peak_raw, min_focus_mean,
    blob_threshold, exclude_outside, cell_factor.
    multi_foci_fi_2d: optional list of 2D FI arrays (same shape as mask) for multi-color foci preview; when set with len>0, foci are drawn for each.
    foci_channel_labels: short names (e.g. TIFF stem) aligned with multi_foci_fi_2d for hover/title.
    """
    height, width = mask.shape
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    peaks_for_hover: list[dict] = []
    n_foci_total = 0
    n_foci_drawn = 0
    raw_median_fi: Optional[float] = None
    min_rel: Optional[float] = None
    mode_txt = ""

    if accepted_labels:
        accepted_array = np.array(list(accepted_labels))
        accepted_mask = np.isin(mask, accepted_array)
        vis[accepted_mask, 1] = 255
    if rejected_labels:
        rejected_array = np.array(list(rejected_labels))
        rejected_mask = np.isin(mask, rejected_array)
        vis[rejected_mask, 0] = 255

    if fi_image is not None and (do_fi_contour or run_foci):
        try:
            if progress_callback:
                progress_callback(86, "Overlay · FI background", "")
            from skimage.segmentation import find_boundaries
            from skimage.morphology import binary_dilation
            fi_flat = np.asarray(fi_image, dtype=np.float32)
            if fi_flat.ndim > 2:
                fi_flat = fi_flat[:, :, 0] if fi_flat.shape[2] < 10 else fi_flat[0, :, :]
            if fi_flat.shape[:2] != mask.shape[:2]:
                raise ValueError("FI shape mismatch")
            accepted_mask_full = np.isin(mask, np.array(list(accepted_labels))) if accepted_labels else np.zeros_like(mask, dtype=bool)
            if fi_flat.size > 0:
                p_low, p_high = float(np.min(fi_flat)), float(np.max(fi_flat))
                if p_high > p_low:
                    fi_norm = (fi_flat - p_low) / (p_high - p_low) * 255
                    fi_norm = np.clip(fi_norm, 0, 255).astype(np.float32)
                else:
                    fi_norm = np.zeros_like(fi_flat)
            else:
                fi_norm = np.zeros_like(fi_flat)
            vis_fi = np.stack([fi_norm, fi_norm, fi_norm], axis=-1).astype(np.uint8)
            if np.any(accepted_mask_full):
                alpha_g = 0.55
                vis_fi[accepted_mask_full, 0] = (vis_fi[accepted_mask_full, 0].astype(np.float32) * (1 - alpha_g)).astype(np.uint8)
                vis_fi[accepted_mask_full, 1] = (vis_fi[accepted_mask_full, 1].astype(np.float32) * (1 - alpha_g) + 255 * alpha_g).astype(np.uint8)
                vis_fi[accepted_mask_full, 2] = (vis_fi[accepted_mask_full, 2].astype(np.float32) * (1 - alpha_g)).astype(np.uint8)
            if rejected_labels:
                rej_arr = np.array(list(rejected_labels))
                rej_mask = np.isin(mask, rej_arr)
                if np.any(rej_mask):
                    alpha_r = 0.55
                    vis_fi[rej_mask, 0] = (vis_fi[rej_mask, 0].astype(np.float32) * (1 - alpha_r) + 255 * alpha_r).astype(np.uint8)
                    vis_fi[rej_mask, 1] = (vis_fi[rej_mask, 1].astype(np.float32) * (1 - alpha_r)).astype(np.uint8)
                    vis_fi[rej_mask, 2] = (vis_fi[rej_mask, 2].astype(np.float32) * (1 - alpha_r)).astype(np.uint8)
            if accepted_labels and np.any(accepted_mask_full):
                if progress_callback:
                    progress_callback(87, "Overlay · Cell boundaries", "")
                mask_accepted = np.where(np.isin(mask, np.array(list(accepted_labels))), mask, 0).astype(np.int32)
                boundaries = find_boundaries(mask_accepted, mode="outer")
                if np.any(boundaries):
                    vis_fi[boundaries, 0] = 0
                    vis_fi[boundaries, 1] = 0
                    vis_fi[boundaries, 2] = 0
            if do_fi_contour and np.any(accepted_mask_full):
                paint_contour_fi_ring_on_display_rgb(
                    vis_fi,
                    mask,
                    set(accepted_labels),
                    thickness_px=int(fi_contour_thickness_val),
                    blend_alpha=0.6,
                )
            vis[:] = vis_fi[:]
        except Exception as e:
            raise RuntimeError(f"FI background preview failed: {e}") from e

    if run_foci and (fi_image is not None or (multi_foci_fi_2d and len(multi_foci_fi_2d) > 0)):
        if progress_callback:
            progress_callback(88, "Overlay · Drawing foci", "")
        render_cap = FOCI_CAP
        sf = (foci_snapshot or {}).get("spotiflow") if isinstance(foci_snapshot, dict) else None
        if isinstance(sf, dict):
            try:
                render_cap = max(0, int(sf.get("preview_render_cap", FOCI_CAP)))
            except (TypeError, ValueError):
                render_cap = FOCI_CAP
        sources: list[tuple[np.ndarray, Optional[str]]] = []
        if multi_foci_fi_2d and len(multi_foci_fi_2d) > 0:
            labels = foci_channel_labels or []
            for i, arr in enumerate(multi_foci_fi_2d):
                lbl = labels[i] if i < len(labels) else None
                w = _fi_to_2d(np.asarray(arr))
                sources.append((w, lbl))
        elif fi_image is not None:
            sources.append((_fi_to_2d(np.asarray(fi_image)), None))
        for ch_i, (fi_work, ch_lbl) in enumerate(sources):
            color = FOCI_MULTI_COLORS[ch_i % len(FOCI_MULTI_COLORS)]
            remaining = max(0, render_cap - n_foci_drawn)
            n_p, n_d, ph, rmed, mr, mt = _foci_detect_draw_channel(
                vis, fi_work, mask, accepted_labels, foci_snapshot, color, remaining, ch_lbl
            )
            n_foci_total += n_p
            n_foci_drawn += n_d
            peaks_for_hover.extend(ph)
            if ch_i == 0:
                raw_median_fi = rmed
                min_rel = mr
                mode_txt = mt

    return (vis, peaks_for_hover, n_foci_total, n_foci_drawn, raw_median_fi, min_rel, mode_txt)
