"""
Pure (no Qt) helper to build the Analyze Masks filter-preview overlay in a background thread.
Returns (vis, peaks_for_hover, n_foci_total, raw_median_fi, min_rel, mode_txt).
"""
from __future__ import annotations

import numpy as np
from typing import Any, Optional

FOCI_CAP = 500


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
) -> tuple[np.ndarray, list[dict], int, Optional[float], Optional[float], str]:
    """
    Build RGB overlay (vis) and foci hover list. Safe to call from a worker thread.
    foci_snapshot: dict with mode, min_rel, min_sep, min_sz, max_sz, min_peak_raw, min_focus_mean, exclude_outside, cell_factor.
    progress_callback(percent, message) optional; called during long steps so the UI can update.
    """
    height, width = mask.shape
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    peaks_for_hover: list[dict] = []
    n_foci_total = 0
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
                progress_callback(86, "FI background... (86%)")
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
                    progress_callback(87, "Cell boundaries... (87%)")
                # Single call: boundaries between all accepted cells (and vs background)
                mask_accepted = np.where(np.isin(mask, np.array(list(accepted_labels))), mask, 0).astype(np.int32)
                boundaries = find_boundaries(mask_accepted, mode="outer")
                if np.any(boundaries):
                    vis_fi[boundaries, 0] = 0
                    vis_fi[boundaries, 1] = 0
                    vis_fi[boundaries, 2] = 0
            if do_fi_contour and np.any(accepted_mask_full):
                temp_mask = np.zeros_like(mask)
                temp_mask[accepted_mask_full] = mask[accepted_mask_full]
                boundaries = find_boundaries(temp_mask, mode="inner")
                contour_mask = boundaries & accepted_mask_full
                if fi_contour_thickness_val > 1:
                    for _ in range(fi_contour_thickness_val - 1):
                        contour_mask = binary_dilation(contour_mask) & accepted_mask_full
                alpha = 0.6
                vis_fi[contour_mask, 0] = (vis_fi[contour_mask, 0].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                vis_fi[contour_mask, 1] = (vis_fi[contour_mask, 1].astype(np.float32) * (1 - alpha) + 0 * alpha).astype(np.uint8)
                vis_fi[contour_mask, 2] = (vis_fi[contour_mask, 2].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
            vis[:] = vis_fi[:]
        except Exception:
            pass

    if run_foci and fi_image is not None:
        try:
            if progress_callback:
                progress_callback(88, "Detecting foci... (88%)")
            from bactimate.modules.foci_utils import (
                detect_peaks_skimage as _detect_foci,
                detect_peaks_skimage_auto as _detect_foci_auto,
                detect_peaks_cell_first_region_grow as _detect_foci_cell_first,
                detect_peaks_spotiflow as _detect_foci_spotiflow,
                get_focus_blob_mask,
            )
        except ImportError:
            _detect_foci = None
            _detect_foci_auto = None
            _detect_foci_cell_first = None
            _detect_foci_spotiflow = None
            get_focus_blob_mask = None
        if _detect_foci is not None or _detect_foci_auto is not None:
            try:
                fi_work = np.asarray(fi_image, dtype=np.float64)
                if fi_work.ndim > 2:
                    fi_work = fi_work[:, :, 0] if fi_work.shape[2] < 10 else fi_work[0, :, :]
                if fi_work.shape[:2] == vis.shape[:2]:
                    raw_median_fi = float(np.median(fi_work))
                    sn = foci_snapshot or {}
                    detection_mode = int(sn.get("detection_mode", 0))
                    min_sep = sn.get("min_sep", 2)
                    peaks = None
                    if detection_mode == 2 and _detect_foci_spotiflow is not None:
                        peaks = _detect_foci_spotiflow(fi_work, model_name="general")
                    if peaks is None and detection_mode in (1, 2):
                        if _detect_foci_cell_first is not None:
                            peaks = _detect_foci_cell_first(fi_work, labeled_mask=mask)
                        elif _detect_foci_auto is not None:
                            peaks = _detect_foci_auto(fi_work, min_distance=min_sep, labeled_mask=mask)
                    if peaks is None and _detect_foci is not None:
                        mode_txt = (sn.get("mode") or "per_cell").strip().lower()
                        if mode_txt in ("per_cell", "per cell"):
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
                        peaks = [p for p in peaks if 0 <= int(p["y"]) < mask.shape[0] and 0 <= int(p["x"]) < mask.shape[1] and mask[int(p["y"]), int(p["x"])] in accepted_labels]
                    if mode_txt and "cell" in mode_txt and accepted_labels:
                        cell_factor = sn.get("cell_factor") or sn.get("min_rel") or 2.0
                        try:
                            fi_flat = np.asarray(fi_work, dtype=np.float64)
                            if fi_flat.ndim > 2:
                                fi_flat = fi_flat[:, :, 0] if fi_flat.shape[2] < 10 else fi_flat[0, :, :]
                            label_to_mean_fi = {}
                            for lab in accepted_labels:
                                cell_mask = (mask == lab)
                                if np.any(cell_mask) and cell_mask.shape == fi_flat.shape[:2]:
                                    mu = float(np.mean(fi_flat[cell_mask]))
                                    if np.isfinite(mu) and mu > 0:
                                        label_to_mean_fi[lab] = mu
                            peaks_filtered = []
                            for p in peaks:
                                yi, xi = int(p["y"]), int(p["x"])
                                if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
                                    continue
                                lab = int(mask[yi, xi])
                                mean_fi = label_to_mean_fi.get(lab)
                                if mean_fi is None or mean_fi <= 0:
                                    continue
                                intensity = p.get("intensity")
                                if intensity is None or not np.isfinite(intensity):
                                    continue
                                if float(intensity) / float(mean_fi) >= float(cell_factor):
                                    peaks_filtered.append(p)
                            peaks = peaks_filtered
                        except Exception:
                            pass
                    n_foci_total = len(peaks)
                    orange = np.array([255, 165, 0], dtype=np.uint8)
                    h, w = vis.shape[:2]
                    peaks_to_draw = peaks
                    if get_focus_blob_mask is not None:
                        from skimage.segmentation import find_boundaries
                        fi_f32 = fi_work.astype(np.float32)
                        for i, p in enumerate(peaks_to_draw):
                            if i >= FOCI_CAP:
                                break
                            yi, xi = int(p["y"]), int(p["x"])
                            blob = get_focus_blob_mask(fi_f32, yi, xi, threshold_frac=0.8, max_radius=5, labeled_mask=mask)
                            if np.any(blob):
                                vis[blob] = orange
                                boundaries = find_boundaries(blob, mode="outer")
                                if np.any(boundaries):
                                    vis[boundaries, 0] = 0
                                    vis[boundaries, 1] = 0
                                    vis[boundaries, 2] = 0
                    else:
                        for i, p in enumerate(peaks_to_draw):
                            if i >= FOCI_CAP:
                                break
                            yi, xi = int(p["y"]), int(p["x"])
                            r = max(2, min(5, int(round((max(1, p.get("size_px", 1)) / np.pi) ** 0.5))))
                            for dy in range(-r, r + 1):
                                for dx in range(-r, r + 1):
                                    if dy * dy + dx * dx <= r * r:
                                        yk, xk = yi + dy, xi + dx
                                        if 0 <= yk < h and 0 <= xk < w:
                                            vis[yk, xk] = orange
                    peaks_for_hover.extend(peaks_to_draw[:FOCI_CAP])
            except Exception:
                pass

    return (vis, peaks_for_hover, n_foci_total, raw_median_fi, min_rel, mode_txt)
