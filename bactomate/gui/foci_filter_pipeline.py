"""Shared foci CSV filtering, blob metrics, and export for Analyze Masks filtering."""
from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Callable
from typing import Any, Optional

FociProgressCallback = Callable[[int, int, str], None]

import numpy as np

FOCI_BLOB_SIZE_PRESETS: tuple[tuple[str, float], ...] = (
    ("90%", 0.9),
    ("80%", 0.8),
    ("70%", 0.7),
    ("60%", 0.6),
    ("50%", 0.5),
)

FOCI_TABLE_COL_MIN_WIDTH = 24
FOCI_TABLE_CELL_PADDING_PX = 10
# Pass column only shows Yes/No — keep narrow so other columns get room.
FOCI_TABLE_PASS_COL_MAX_WIDTH = 36
FOCI_TABLE_LOCATION_COL_MAX_WIDTH = 52

from bactomate.gui.filter_preview_utils import (  # noqa: E402 — re-export for pipeline callers
    FOCI_CHANNEL_SETTING_KEYS,
    default_foci_channel_settings,
    ensure_foci_channels_map,
    foci_channel_key_for_path,
    legacy_foci_settings_slice,
    resolve_foci_settings_for_source,
    seed_foci_channel_settings,
)

def foci_blob_threshold_frac(settings: dict[str, Any]) -> float:
    try:
        frac = float(settings.get("foci_blob_threshold_frac", 0.9) or 0.9)
    except (TypeError, ValueError):
        frac = 0.9
    return float(np.clip(frac, 0.05, 1.0))


def foci_blob_max_radius(settings: dict[str, Any]) -> int:
    try:
        r = int(settings.get("foci_blob_max_radius", 5) or 5)
    except (TypeError, ValueError):
        r = 5
    return max(1, min(50, r))


def draw_foci_blob_contours_on_rgb(
    target_rgb: np.ndarray,
    mask_2d: np.ndarray,
    peaks: list[dict[str, Any]],
    plane: np.ndarray,
    settings: dict[str, Any],
    contour_rgb: tuple[int, int, int],
) -> list[dict[str, Any]]:
    """Draw intensity-grown focus boundaries on *target_rgb*; return peaks with no blob."""
    from skimage.segmentation import find_boundaries

    from bactomate.modules.foci_utils import get_focus_blob_mask

    h0, w0 = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    h1, w1 = int(target_rgb.shape[0]), int(target_rgb.shape[1])
    sx = (w1 / w0) if w0 else 1.0
    sy = (h1 / h0) if h0 else 1.0
    frac = foci_blob_threshold_frac(settings)
    max_r = foci_blob_max_radius(settings)
    label = np.asarray(mask_2d, dtype=np.int32)
    work = np.asarray(plane, dtype=np.float64)
    cr, cg, cb = contour_rgb
    missed: list[dict[str, Any]] = []
    if not peaks:
        return missed
    for p in peaks or []:
        try:
            hy = int(float(p.get("y", 0) or 0))
            hx = int(float(p.get("x", 0) or 0))
        except Exception:
            continue
        blob = get_focus_blob_mask(
            work, hy, hx, threshold_frac=frac, max_radius=max_r, labeled_mask=label
        )
        if blob is None or not np.any(blob):
            missed.append(p)
            continue
        ys, xs = np.nonzero(blob)
        y2 = np.clip(np.round(ys.astype(np.float64) * sy).astype(np.int32), 0, h1 - 1)
        x2 = np.clip(np.round(xs.astype(np.float64) * sx).astype(np.int32), 0, w1 - 1)
        disp_mask = np.zeros((h1, w1), dtype=bool)
        disp_mask[y2, x2] = True
        if not np.any(disp_mask):
            missed.append(p)
            continue
        try:
            edge = find_boundaries(disp_mask, mode="outer")
            if np.any(edge):
                target_rgb[edge, 0] = int(cr)
                target_rgb[edge, 1] = int(cg)
                target_rgb[edge, 2] = int(cb)
            else:
                missed.append(p)
        except Exception:
            missed.append(p)
    return missed


def accepted_cell_labels_cache_key(
    accepted_cell_labels: Optional[set[int]],
) -> tuple:
    """Stable cache token for accepted-cell gating (preview layer / hover caches)."""
    if not accepted_cell_labels:
        return ("all",)
    return ("acc", frozenset(int(x) for x in accepted_cell_labels))


def sync_foci_peaks_with_accepted_cells(
    pass_peaks: list[dict[str, Any]],
    fail_peaks: list[dict[str, Any]],
    *,
    mask_2d: Optional[np.ndarray] = None,
    accepted_cell_labels: Optional[set[int]] = None,
    settings: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Move peaks in rejected cells from pass → fail (fixes stale caches / edge peaks).
    """
    if mask_2d is None or not accepted_cell_labels:
        return list(pass_peaks), list(fail_peaks)
    label = np.asarray(mask_2d, dtype=np.int32)
    if label.ndim != 2:
        return list(pass_peaks), list(fail_peaks)
    fs = dict(settings or {})
    passing: list[dict[str, Any]] = []
    failing = [dict(p) for p in fail_peaks]
    fail_keys = {
        (int(p.get("y", 0)), int(p.get("x", 0)))
        for p in failing
    }
    for p in pass_peaks or []:
        try:
            y = int(float(p.get("y", 0) or 0))
            x = int(float(p.get("x", 0) or 0))
        except (TypeError, ValueError):
            continue
        cid = resolve_foci_cell_id_at_peak(label, y, x)
        if cid > 0 and cid not in accepted_cell_labels:
            peak = dict(p)
            peak["fail_reasons"] = foci_filter_failure_reasons(
                p,
                fs,
                mask_2d=mask_2d,
                accepted_cell_labels=accepted_cell_labels,
                metrics=peak,
            )
            key = (y, x)
            if key not in fail_keys:
                failing.append(peak)
                fail_keys.add(key)
        else:
            passing.append(dict(p))
    return passing, failing


def accepted_cell_labels_for_foci(
    accepted: Any,
    *,
    cell_filter_applied: bool,
) -> Optional[set[int]]:
    """Accepted mask labels for gating foci after Apply (peaks in rejected cells fail)."""
    if not cell_filter_applied or accepted is None:
        return None
    try:
        labels = {int(x) for x in accepted if int(x) > 0}
    except (TypeError, ValueError):
        return None
    return labels


def cell_filters_active_for_foci(settings: dict[str, Any]) -> bool:
    """Whether explicit size/border cell filters are enabled in settings.

    Note: foci gating vs rejected cells uses ``accepted_cell_labels_for_foci`` after
    Apply, not this helper (morphology filters always affect accepted/rejected cells).
    """
    if bool(settings.get("exclude_border", False)):
        return True
    for key in (
        "min_area_um2",
        "max_area_um2",
        "min_width_um",
        "max_width_um",
        "min_length_um",
        "max_length_um",
    ):
        val = settings.get(key)
        if val is None:
            continue
        try:
            if float(val) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _outside_exclude_flags(settings: dict[str, Any]) -> tuple[bool, bool]:
    """Return (exclude_clear_outside, exclude_border_touching) with legacy key support."""
    legacy = bool(settings.get("foci_exclude_outside_cells", False))
    ex_clear = bool(settings.get("foci_exclude_outside_clear", legacy))
    ex_touch = bool(settings.get("foci_exclude_outside_touching", legacy))
    return ex_clear, ex_touch


def classify_foci_cell_relation(
    mask_2d: Optional[np.ndarray],
    y: int,
    x: int,
    *,
    touch_radius: int = 2,
) -> str:
    """
    Classify peak position vs segmentation mask.

    Returns one of: ``inside``, ``outside_clear``, ``outside_touching``.
    """
    label = np.asarray(mask_2d, dtype=np.int32) if mask_2d is not None else None
    if label is None or label.ndim != 2:
        return "inside"
    h, w = int(label.shape[0]), int(label.shape[1])
    if not (0 <= int(y) < h and 0 <= int(x) < w):
        return "outside_clear"
    if int(label[int(y), int(x)]) > 0:
        return "inside"
    r = max(1, int(touch_radius))
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    window = label[y0:y1, x0:x1]
    if np.any(window > 0):
        return "outside_touching"
    return "outside_clear"


def resolve_foci_cell_id_at_peak(
    mask_2d: Optional[np.ndarray],
    y: int,
    x: int,
    *,
    touch_radius: int = 2,
) -> int:
    """
    Cell label for a focus peak: mask label at the peak, or nearest cell when on background edge.
    """
    from bactomate.modules.foci_utils import resolve_peak_cell_id

    label = np.asarray(mask_2d, dtype=np.int32) if mask_2d is not None else None
    if label is None or label.ndim != 2:
        return 0
    return resolve_peak_cell_id(label, int(y), int(x), touch_radius=touch_radius)


def foci_outside_relation_label(relation: str) -> str:
    """Short table label for peak vs cell mask (peak center pixel)."""
    rel = str(relation or "").strip().lower()
    if rel == "outside_touching":
        return "edge"
    if rel == "outside_clear":
        return "outside"
    return "inside"


FOCI_TABLE_LOCATION_TOOLTIP = (
    "Peak location vs cell mask (at peak center): "
    "inside = in a cell; outside = background, not touching cells; "
    "edge = background but touching a cell boundary (Cell column shows the adjacent cell)."
)


def filter_foci_csv_rows(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    mask_2d: Optional[np.ndarray] = None,
    accepted_cell_labels: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """Apply CSV-level filters (probability, min intensity, inside accepted cells)."""
    prob_thresh = (
        float(settings.get("foci_prob_thresh", 0.5))
        if bool(settings.get("foci_prob_enabled", False))
        else None
    )
    min_peak_raw = (
        float(settings.get("foci_min_int", 0.0))
        if bool(settings.get("foci_min_int_enabled", False))
        else None
    )
    ex_clear, ex_touch = _outside_exclude_flags(settings)
    label = np.asarray(mask_2d, dtype=np.int32) if mask_2d is not None else None
    h = w = 0
    if label is not None and label.ndim == 2:
        h, w = int(label.shape[0]), int(label.shape[1])

    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            y = int(float(r.get("y", 0) or 0))
            x = int(float(r.get("x", 0) or 0))
        except Exception:
            continue
        try:
            intensity = float(r.get("intensity", 0) or 0.0)
        except Exception:
            intensity = 0.0
        prob_raw = r.get("probability", None)
        prob = None
        if prob_raw not in (None, "", "None"):
            try:
                prob = float(prob_raw)
            except Exception:
                prob = None
        if label is not None and 0 <= y < h and 0 <= x < w:
            rel = classify_foci_cell_relation(label, y, x)
            cell_lbl = resolve_foci_cell_id_at_peak(label, y, x)
        else:
            rel = "outside_clear"
            cell_lbl = 0
        if accepted_cell_labels is not None:
            if cell_lbl > 0 and cell_lbl not in accepted_cell_labels:
                continue
            if rel == "outside_clear" and ex_clear:
                continue
            if rel == "outside_touching" and ex_touch:
                continue
        else:
            if rel == "outside_clear" and ex_clear:
                continue
            if rel == "outside_touching" and ex_touch:
                continue
        if prob_thresh is not None and prob is not None and prob < float(prob_thresh):
            continue
        if min_peak_raw is not None and intensity < float(min_peak_raw):
            continue
        row = dict(r)
        row["y"] = y
        row["x"] = x
        row["intensity"] = intensity
        if prob is not None:
            row["probability"] = prob
        out.append(row)
    return out


def refine_foci_rows_with_blob(
    rows: list[dict[str, Any]],
    foci_plane: np.ndarray,
    *,
    label_plane: Optional[np.ndarray] = None,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add area_px, mean_intensity, rel_intensity, cell_id via intensity-linked blob growth."""
    if not rows:
        return []
    from bactomate.modules.foci_utils import refine_spotiflow_peaks

    per_cell_ratio = (
        float(settings.get("foci_per_cell_ratio", 2.0))
        if bool(settings.get("foci_per_cell_ratio_enabled", False))
        else None
    )
    refined = refine_spotiflow_peaks(
        np.asarray(foci_plane, dtype=np.float64),
        list(rows),
        labeled_mask=label_plane,
        threshold_frac=foci_blob_threshold_frac(settings),
        max_radius=foci_blob_max_radius(settings),
        per_cell_ratio=per_cell_ratio,
        exclude_outside=False,
    )
    label = np.asarray(label_plane, dtype=np.int32) if label_plane is not None else None
    out: list[dict[str, Any]] = []
    for r in refined:
        row = dict(r)
        y = int(row.get("y", 0))
        x = int(row.get("x", 0))
        if label is not None and label.ndim == 2:
            row["cell_id"] = resolve_foci_cell_id_at_peak(label, y, x)
        else:
            row["cell_id"] = int(row.get("cell_id", 0) or 0)
        row.setdefault("area_px", int(row.get("size_px", 1) or 1))
        out.append(row)
    return out


def measure_foci_rows_metrics(
    rows: list[dict[str, Any]],
    foci_plane: np.ndarray,
    *,
    label_plane: Optional[np.ndarray] = None,
    settings: Optional[dict[str, Any]] = None,
    progress_callback: Optional[FociProgressCallback] = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Blob metrics for every peak (display only; never drops rows)."""
    from bactomate.modules.foci_utils import coerce_plane_to_2d, focus_size_and_mean_at_peak

    settings = dict(settings or {})
    raw = np.asarray(coerce_plane_to_2d(foci_plane), dtype=np.float64)
    if raw.ndim != 2 or raw.size == 0:
        return {}
    h, w = int(raw.shape[0]), int(raw.shape[1])
    raw_median = float(np.median(raw))
    if raw_median <= 0:
        raw_median = 1e-10
    label = np.asarray(label_plane, dtype=np.int32) if label_plane is not None else None
    thresh = foci_blob_threshold_frac(settings)
    max_r = foci_blob_max_radius(settings)
    out: dict[tuple[int, int], dict[str, Any]] = {}
    rows_list = list(rows)
    total = len(rows_list)
    report_every = max(1, min(15, total // 80)) if total > 0 else 1
    for idx, row in enumerate(rows_list):
        if progress_callback is not None and (
            idx == 0 or idx == total - 1 or idx % report_every == 0
        ):
            try:
                progress_callback(idx + 1, total, "Measuring foci metrics")
            except Exception:
                pass
        try:
            y = int(float(row.get("y", 0) or 0))
            x = int(float(row.get("x", 0) or 0))
        except Exception:
            continue
        key = (y, x)
        base = dict(row)
        base["y"] = y
        base["x"] = x
        if 0 <= y < h and 0 <= x < w:
            intensity = row.get("intensity")
            if intensity is None or intensity == "":
                intensity = float(raw[y, x])
            else:
                try:
                    intensity = float(intensity)
                except (TypeError, ValueError):
                    intensity = float(raw[y, x])
            size_px, mean_int = focus_size_and_mean_at_peak(
                raw,
                y,
                x,
                threshold_frac=thresh,
                max_radius=max_r,
                labeled_mask=label,
            )
            base["intensity"] = float(intensity)
            base["size_px"] = int(size_px)
            base["area_px"] = int(max(1, int(size_px)))
            base["mean_intensity"] = float(mean_int)
            base["rel_intensity"] = float(intensity) / raw_median
            if label is not None and label.ndim == 2:
                base["cell_id"] = resolve_foci_cell_id_at_peak(label, y, x)
            else:
                base["cell_id"] = int(base.get("cell_id", 0) or 0)
        out[key] = base
    return out


def _merge_foci_row_metrics(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Copy blob metric fields from source into target when missing."""
    out = dict(target)
    for key in (
        "intensity",
        "probability",
        "area_px",
        "size_px",
        "mean_intensity",
        "rel_intensity",
        "cell_id",
        "frame",
        "fail_reasons",
    ):
        if key not in out or out.get(key) in (None, ""):
            if key in source and source.get(key) not in (None, ""):
                out[key] = source[key]
    if "area_px" not in out or out.get("area_px") in (None, ""):
        sp = out.get("size_px", source.get("size_px"))
        if sp not in (None, ""):
            try:
                out["area_px"] = int(max(1, int(sp)))
            except (TypeError, ValueError):
                pass
    return out


def foci_filter_failure_reasons(
    row: dict[str, Any],
    settings: dict[str, Any],
    *,
    mask_2d: Optional[np.ndarray] = None,
    accepted_cell_labels: Optional[set[int]] = None,
    metrics: Optional[dict[str, Any]] = None,
    foci_plane: Optional[np.ndarray] = None,
) -> list[str]:
    """Human-readable reasons this focus did not pass current filters."""
    m = dict(metrics if metrics is not None else row)
    reasons: list[str] = []
    try:
        y = int(float(row.get("y", m.get("y", 0)) or 0))
        x = int(float(row.get("x", m.get("x", 0)) or 0))
    except Exception:
        return reasons

    try:
        intensity = float(row.get("intensity", m.get("intensity", 0)) or 0.0)
    except (TypeError, ValueError):
        intensity = 0.0

    prob_raw = row.get("probability", m.get("probability"))
    prob = None
    if prob_raw not in (None, "", "None"):
        try:
            prob = float(prob_raw)
        except (TypeError, ValueError):
            prob = None

    if bool(settings.get("foci_prob_enabled", False)) and prob is not None:
        try:
            thresh = float(settings.get("foci_prob_thresh", 0.5))
            if prob < thresh:
                reasons.append(f"Probability {prob:.3f} < {thresh:.3f}")
        except (TypeError, ValueError):
            pass

    if bool(settings.get("foci_min_int_enabled", False)):
        try:
            min_int = float(settings.get("foci_min_int", 0.0))
            if intensity < min_int:
                reasons.append(f"Peak intensity {intensity:.0f} < {min_int:.0f}")
        except (TypeError, ValueError):
            pass

    label = np.asarray(mask_2d, dtype=np.int32) if mask_2d is not None else None
    if label is not None and label.ndim == 2:
        cell_id = resolve_foci_cell_id_at_peak(label, y, x)
    else:
        try:
            cell_id = int(m.get("cell_id", 0) or 0)
        except (TypeError, ValueError):
            cell_id = 0

    ex_clear, ex_touch = _outside_exclude_flags(settings)
    rel = classify_foci_cell_relation(label, y, x) if label is not None else "inside"
    if accepted_cell_labels is not None:
        if cell_id > 0 and cell_id not in accepted_cell_labels:
            reasons.append("In a rejected cell")
        elif rel == "outside_clear" and ex_clear:
            reasons.append("Outside cells (clear)")
        elif rel == "outside_touching" and ex_touch:
            reasons.append("Outside cells (touching edge)")
    else:
        if rel == "outside_clear" and ex_clear:
            reasons.append("Outside cells (clear)")
        elif rel == "outside_touching" and ex_touch:
            reasons.append("Outside cells (touching edge)")

    if bool(settings.get("foci_min_area_px_enabled", False)):
        try:
            min_area = int(settings.get("foci_min_area_px", 1) or 1)
            area = int(m.get("area_px", m.get("size_px", 0)) or 0)
            if area < min_area:
                reasons.append(f"Area {area} px < minimum {min_area} px")
        except (TypeError, ValueError):
            pass

    if bool(settings.get("foci_per_cell_ratio_enabled", False)) and foci_plane is not None:
        try:
            ratio_thresh = float(settings.get("foci_per_cell_ratio", 2.0))
            if ratio_thresh > 0 and cell_id > 0 and label is not None:
                from bactomate.modules.foci_utils import coerce_plane_to_2d, get_focus_blob_mask

                raw = np.asarray(coerce_plane_to_2d(foci_plane), dtype=np.float64)
                if raw.ndim == 2 and 0 <= y < raw.shape[0] and 0 <= x < raw.shape[1]:
                    blob_mask = get_focus_blob_mask(
                        raw,
                        y,
                        x,
                        threshold_frac=foci_blob_threshold_frac(settings),
                        max_radius=foci_blob_max_radius(settings),
                        labeled_mask=label,
                    )
                    rest = (label == cell_id) & ~blob_mask
                    n_rest = int(np.sum(rest))
                    if n_rest > 0:
                        rest_mean = float(np.mean(raw[rest]))
                        mean_int = float(m.get("mean_intensity", 0) or 0)
                        if np.isfinite(rest_mean) and rest_mean > 0 and np.isfinite(mean_int):
                            ratio = mean_int / rest_mean
                            if ratio < ratio_thresh:
                                reasons.append(
                                    f"Per-cell ratio {ratio:.2f} < {ratio_thresh:.2f}"
                                )
        except Exception:
            pass

    if not reasons:
        reasons.append("Did not pass active filters")
    return reasons


def filter_foci_rows_by_min_area_px(
    rows: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Drop refined rows whose blob area_px is below the configured minimum."""
    if not bool(settings.get("foci_min_area_px_enabled", False)):
        return list(rows)
    try:
        min_area = int(settings.get("foci_min_area_px", 1) or 1)
    except (TypeError, ValueError):
        min_area = 1
    min_area = max(1, min_area)
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            area = int(r.get("area_px", r.get("size_px", 0)) or 0)
        except (TypeError, ValueError):
            area = 0
        if area >= min_area:
            out.append(r)
    return out


class _MonotonicFociFrameProgress:
    """Map CSV filter → grow → measure → fail into one increasing done/total for the UI."""

    def __init__(
        self,
        callback: FociProgressCallback,
        *,
        n_rows: int,
        n_filtered: int,
    ) -> None:
        self._callback = callback
        nr = max(1, int(n_rows))
        nf = max(1, int(n_filtered))
        self._grow_budget = nf
        self._measure_budget = nr
        self._fail_budget = nr
        self._total = 1 + nf + nr + nr

    @property
    def total_units(self) -> int:
        return self._total

    def emit(self, units_done: int, msg: str) -> None:
        done = max(0, min(int(units_done), self._total))
        try:
            self._callback(done, self._total, str(msg))
        except Exception:
            pass

    def measure_callback(self) -> FociProgressCallback:
        base = 1 + self._grow_budget

        def _cb(row_done: int, _row_total: int, msg: str) -> None:
            self.emit(base + int(row_done), msg)

        return _cb


def compute_foci_pass_fail_for_frame(
    frame_rows: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    mask_2d: Optional[np.ndarray] = None,
    foci_plane: Optional[np.ndarray] = None,
    accepted_cell_labels: Optional[set[int]] = None,
    progress_callback: Optional[FociProgressCallback] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (passing peaks, failing peaks) for one frame after filters + blob refinement."""

    total_rows = len(frame_rows)
    filtered = filter_foci_csv_rows(
        frame_rows,
        settings,
        mask_2d=mask_2d,
        accepted_cell_labels=accepted_cell_labels,
    )
    mono: Optional[_MonotonicFociFrameProgress] = None
    if progress_callback is not None:
        mono = _MonotonicFociFrameProgress(
            progress_callback,
            n_rows=total_rows,
            n_filtered=len(filtered),
        )
        mono.emit(1, "Applying peak filters")
    if foci_plane is not None:
        label_plane = np.asarray(mask_2d, dtype=np.int32) if mask_2d is not None else None
        if mono is not None:
            mono.emit(1, "Growing passing foci blobs")
        passing = refine_foci_rows_with_blob(
            filtered, foci_plane, label_plane=label_plane, settings=settings
        )
        passing = filter_foci_rows_by_min_area_px(passing, settings)
        if mono is not None:
            mono.emit(1 + mono._grow_budget, "Growing passing foci blobs")
    else:
        passing = filter_foci_rows_by_min_area_px([dict(r) for r in filtered], settings)
        if mono is not None:
            mono.emit(1 + mono._grow_budget, "Growing passing foci blobs")
    pass_keys = {(int(p.get("y", 0)), int(p.get("x", 0))) for p in passing}
    metrics_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if foci_plane is not None and frame_rows:
        label_plane = np.asarray(mask_2d, dtype=np.int32) if mask_2d is not None else None
        try:
            metrics_by_key = measure_foci_rows_metrics(
                [dict(row) for row in frame_rows],
                foci_plane,
                label_plane=label_plane,
                settings=settings,
                progress_callback=mono.measure_callback() if mono is not None else progress_callback,
            )
        except Exception:
            metrics_by_key = {}
    failing: list[dict[str, Any]] = []
    fail_report_every = max(1, min(15, total_rows // 60)) if total_rows > 0 else 1
    fail_base = (1 + mono._grow_budget + mono._measure_budget) if mono is not None else 0
    for fi, r in enumerate(frame_rows):
        if mono is not None and (
            fi == 0 or fi == total_rows - 1 or fi % fail_report_every == 0
        ):
            mono.emit(
                fail_base + fi + 1,
                "Classifying failed foci",
            )
        try:
            key = (int(float(r.get("y", 0) or 0)), int(float(r.get("x", 0) or 0)))
        except Exception:
            continue
        if key in pass_keys:
            continue
        peak = _merge_foci_row_metrics(dict(r), metrics_by_key.get(key, {}))
        peak["fail_reasons"] = foci_filter_failure_reasons(
            r,
            settings,
            mask_2d=mask_2d,
            accepted_cell_labels=accepted_cell_labels,
            metrics=peak,
            foci_plane=foci_plane,
        )
        failing.append(peak)
    return passing, failing


def write_filtered_foci_csv(
    out_path: Path,
    rows: list[dict[str, Any]],
    *,
    include_frame: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "y",
        "x",
        "intensity",
        "probability",
        "area_px",
        "mean_intensity",
        "rel_intensity",
        "cell_id",
    ]
    if include_frame:
        fieldnames = ["frame"] + fieldnames
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            prob = r.get("probability", None)
            out_row: dict[str, Any] = {
                "y": int(r.get("y", 0)),
                "x": int(r.get("x", 0)),
                "intensity": float(r.get("intensity", 0.0) or 0.0),
                "probability": ("" if prob is None else float(prob)),
                "area_px": int(r.get("area_px", r.get("size_px", 1)) or 1),
                "mean_intensity": float(r.get("mean_intensity", r.get("intensity", 0)) or 0.0),
                "rel_intensity": (
                    ""
                    if r.get("rel_intensity") is None
                    else float(r.get("rel_intensity"))
                ),
                "cell_id": int(r.get("cell_id", 0) or 0),
            }
            if include_frame:
                out_row["frame"] = int(r.get("frame", 0) or 0)
            w.writerow(out_row)


def read_raw_foci_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                y = int(float(r.get("y", 0) or 0))
                x = int(float(r.get("x", 0) or 0))
            except Exception:
                continue
            try:
                intensity = float(r.get("intensity", 0) or 0.0)
            except Exception:
                intensity = 0.0
            prob_raw = r.get("probability", None)
            prob = None
            if prob_raw not in (None, "", "None"):
                try:
                    prob = float(prob_raw)
                except Exception:
                    prob = None
            frame = 0
            if "frame" in (reader.fieldnames or []):
                try:
                    frame = int(float(r.get("frame", 0) or 0))
                except Exception:
                    frame = 0
            rows.append(
                {
                    "frame": frame,
                    "y": y,
                    "x": x,
                    "intensity": intensity,
                    "probability": prob,
                }
            )
    return rows


def process_foci_rows_for_mask(
    raw_rows: list[dict[str, Any]],
    foci_tyx: np.ndarray,
    mask_stack: np.ndarray,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter and refine all foci CSV rows against a filtered mask stack and foci image stack."""
    from bactomate.modules.image_stack_utils import coerce_plane_to_2d

    if not raw_rows:
        return []
    mask_stack_3d = mask_stack if mask_stack.ndim == 3 else mask_stack[np.newaxis, :, :]
    per_frame_groups: dict[int, list[dict[str, Any]]] = {}
    for r in raw_rows:
        try:
            fi_idx = int(float(r.get("frame", 0) or 0))
        except Exception:
            fi_idx = 0
        per_frame_groups.setdefault(fi_idx, []).append(dict(r))

    out_rows: list[dict[str, Any]] = []
    n_frames = int(foci_tyx.shape[0])
    for fi_idx, group in per_frame_groups.items():
        f_clamped = max(0, min(int(fi_idx), n_frames - 1))
        try:
            plane = coerce_plane_to_2d(foci_tyx[f_clamped])
        except Exception:
            out_rows.extend(group)
            continue
        mask_idx = max(0, min(int(fi_idx), int(mask_stack_3d.shape[0]) - 1))
        label_plane = np.asarray(mask_stack_3d[mask_idx], dtype=np.int32)
        accepted_labels = {
            int(lbl)
            for lbl in np.unique(label_plane)
            if int(lbl) > 0
        }
        filtered = filter_foci_csv_rows(
            group,
            settings,
            mask_2d=label_plane,
            accepted_cell_labels=accepted_labels,
        )
        refined = refine_foci_rows_with_blob(
            filtered,
            np.asarray(plane, dtype=np.float64),
            label_plane=label_plane,
            settings=settings,
        )
        refined = filter_foci_rows_by_min_area_px(refined, settings)
        for r in refined:
            rr = dict(r)
            rr.setdefault("frame", fi_idx)
            out_rows.append(rr)
    return out_rows


def format_foci_cell_id_display(
    row: dict[str, Any],
    mask_2d: Optional[np.ndarray] = None,
) -> str:
    """Table/tooltip cell ID: numeric label, adjacent cell for edge, em dash when outside."""
    if mask_2d is not None:
        try:
            y = int(float(row.get("y", 0) or 0))
            x = int(float(row.get("x", 0) or 0))
            label = np.asarray(mask_2d, dtype=np.int32)
            if label.ndim == 2 and 0 <= y < label.shape[0] and 0 <= x < label.shape[1]:
                rel = classify_foci_cell_relation(label, y, x)
                if rel == "outside_clear":
                    return "—"
                if rel == "inside":
                    at_peak = int(label[y, x])
                    if at_peak > 0:
                        return str(at_peak)
                cid = resolve_foci_cell_id_at_peak(label, y, x)
                return str(cid) if cid > 0 else "—"
        except Exception:
            pass
    try:
        cid = int(row.get("cell_id", -1))
        if cid > 0:
            return str(cid)
    except (TypeError, ValueError):
        pass
    return "—"


def _foci_table_cell_id(row: dict[str, Any], mask_2d: Optional[np.ndarray] = None) -> str:
    return format_foci_cell_id_display(row, mask_2d)


def _foci_table_metric_text(
    row: dict[str, Any],
    *keys,
    as_int: bool = False,
    decimals: int = 0,
) -> str:
    for key in keys:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        if as_int:
            iv = int(round(val))
            return str(max(0, iv))
        if decimals <= 0:
            return f"{val:.0f}"
        return f"{val:.{decimals}f}"
    return "—"


def populate_foci_preview_table(
    table,
    *,
    frame_rows: list[dict[str, Any]],
    pass_peaks: list[dict[str, Any]],
    fail_peaks: list[dict[str, Any]],
    selected_yx: tuple[int, int] | None = None,
    mask_2d: Optional[np.ndarray] = None,
    foci_plane: Optional[np.ndarray] = None,
    foci_settings: Optional[dict[str, Any]] = None,
) -> None:
    """Fill the preview foci table (all CSV rows on frame; Pass from current filters)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QBrush, QColor
    from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

    if not isinstance(table, QTableWidget):
        return
    headers = [
        "Pass",
        "Location",
        "y",
        "x",
        "Int.",
        "Prob.",
        "Area",
        "Mean",
        "Rel.",
        "Cell",
    ]
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    loc_hdr = table.horizontalHeaderItem(1)
    if loc_hdr is not None:
        loc_hdr.setToolTip(FOCI_TABLE_LOCATION_TOOLTIP)
    pass_keys = {(int(p.get("y", 0)), int(p.get("x", 0))) for p in pass_peaks}
    pass_by_key = {(int(p.get("y", 0)), int(p.get("x", 0))): dict(p) for p in pass_peaks}
    fail_by_key = {(int(p.get("y", 0)), int(p.get("x", 0))): dict(p) for p in fail_peaks}
    metrics_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if foci_plane is not None and foci_settings and (pass_peaks or fail_peaks):
        try:
            label = np.asarray(mask_2d, dtype=np.int32) if mask_2d is not None else None
            to_measure: list[dict[str, Any]] = []
            for p in list(pass_peaks) + list(fail_peaks):
                key = (int(p.get("y", 0)), int(p.get("x", 0)))
                if key in pass_by_key or key in fail_by_key:
                    to_measure.append(dict(pass_by_key.get(key) or fail_by_key.get(key) or p))
            metrics_by_key = measure_foci_rows_metrics(
                to_measure,
                np.asarray(foci_plane, dtype=np.float64),
                label_plane=label,
                settings=dict(foci_settings),
            )
        except Exception:
            metrics_by_key = {}
    rows_out: list[tuple[bool, dict[str, Any]]] = []
    seen: set[tuple[int, int]] = set()
    for p in pass_peaks:
        key = (int(p.get("y", 0)), int(p.get("x", 0)))
        if key in seen:
            continue
        seen.add(key)
        rows_out.append((True, dict(p)))
    for r in frame_rows:
        try:
            key = (int(float(r.get("y", 0) or 0)), int(float(r.get("x", 0) or 0)))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        rows_out.append((key in pass_keys, dict(r)))
    rows_out.sort(key=lambda t: (0 if t[0] else 1, int(t[1].get("y", 0)), int(t[1].get("x", 0))))
    pass_brush = QBrush(QColor(32, 72, 48))
    fail_brush = QBrush(QColor(72, 38, 42))
    table.setRowCount(len(rows_out))
    select_row = -1
    for row_i, (passes, r) in enumerate(rows_out):
        key = (int(r.get("y", 0)), int(r.get("x", 0)))
        if selected_yx is not None and key == tuple(selected_yx):
            select_row = row_i
        if passes and key in pass_by_key:
            enriched = _merge_foci_row_metrics(dict(pass_by_key[key]), metrics_by_key.get(key, {}))
        elif key in fail_by_key:
            enriched = _merge_foci_row_metrics(dict(fail_by_key[key]), metrics_by_key.get(key, {}))
        else:
            enriched = _merge_foci_row_metrics(dict(r), metrics_by_key.get(key, {}))
        prob_raw = enriched.get("probability", r.get("probability", None))
        prob_txt = "—"
        if prob_raw not in (None, "", "None"):
            try:
                prob_txt = f"{float(prob_raw):.3f}"
            except (TypeError, ValueError):
                prob_txt = "—"
        if mask_2d is not None:
            rel = classify_foci_cell_relation(np.asarray(mask_2d, dtype=np.int32), key[0], key[1])
        else:
            rel = "inside"
        values = [
            "Yes" if passes else "No",
            foci_outside_relation_label(rel),
            str(key[0]),
            str(key[1]),
            _foci_table_metric_text(enriched, "intensity", decimals=0),
            prob_txt,
            _foci_table_metric_text(enriched, "area_px", "size_px", as_int=True),
            _foci_table_metric_text(enriched, "mean_intensity", decimals=0),
            _foci_table_metric_text(enriched, "rel_intensity", decimals=2),
            _foci_table_cell_id(enriched, mask_2d),
        ]
        brush = pass_brush if passes else fail_brush
        for col_i, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(brush)
            item.setToolTip(str(text))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col_i == 2:
                item.setData(Qt.ItemDataRole.UserRole, (key[0], key[1]))
            table.setItem(row_i, col_i, item)
    if select_row >= 0:
        table.selectRow(select_row)
        id_item = table.item(select_row, 2)
        if id_item is not None:
            table.scrollToItem(id_item)
    fit_foci_table_columns(table)


def _foci_table_column_content_width(table, col_i: int, fm) -> int:
    """Minimum width so header and every cell in this column display without clipping."""
    w = FOCI_TABLE_COL_MIN_WIDTH
    header = table.horizontalHeaderItem(col_i)
    if header is not None:
        w = max(w, int(fm.horizontalAdvance(header.text())) + FOCI_TABLE_CELL_PADDING_PX)
    for row_i in range(int(table.rowCount())):
        item = table.item(row_i, col_i)
        if item is None:
            continue
        w = max(w, int(fm.horizontalAdvance(item.text())) + FOCI_TABLE_CELL_PADDING_PX)
    return int(w)


def _foci_table_column_area_width(table, width_px: int, *, from_viewport: bool = False) -> int:
    """Convert a width to space available for horizontal column sections."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QTableWidget

    if not isinstance(table, QTableWidget):
        return 0
    n_cols = max(1, int(table.columnCount()))
    w = int(width_px)
    if not from_viewport:
        vhdr = table.verticalHeader()
        if vhdr is not None and vhdr.isVisible():
            w = max(0, w - int(vhdr.width()))
        w = max(0, w - 2 * int(table.frameWidth()))
    if table.verticalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff:
        vbar = table.verticalScrollBar()
        vbar_w = 16
        if vbar is not None:
            try:
                vbar_w = max(12, int(vbar.sizeHint().width()))
            except Exception:
                pass
        w = max(0, w - vbar_w)
    return max(0, w - n_cols - 2)


def _foci_table_viewport_width(table) -> int:
    """Width available for columns (never wider than the visible column area)."""
    from PyQt6.QtWidgets import QTableWidget

    if not isinstance(table, QTableWidget):
        return 0

    vp_w = int(table.viewport().width())
    if vp_w > 8:
        return _foci_table_column_area_width(table, vp_w, from_viewport=True)

    tw = int(table.width())
    if tw > 8:
        return _foci_table_column_area_width(table, tw)

    parent = table.parentWidget()
    depth = 0
    while parent is not None and depth < 6:
        pw = int(parent.width())
        if pw > 8:
            return _foci_table_column_area_width(table, pw)
        parent = parent.parentWidget()
        depth += 1
    return 0


def _distribute_foci_column_widths(weights: list[int], avail: int) -> list[int]:
    """Assign column widths that sum to *avail* (never exceed it)."""
    n = len(weights)
    if n <= 0 or avail <= 0:
        return list(weights)
    wts = [max(1, int(w)) for w in weights]
    total = int(sum(wts))
    if total <= avail:
        widths = list(wts)
        for i in range(avail - total):
            widths[i % n] += 1
        return widths
    widths: list[int] = []
    assigned = 0
    for i, wt in enumerate(wts):
        if i == n - 1:
            w = max(1, avail - assigned)
        else:
            w = max(1, int(avail * wt / total))
            assigned += w
        widths.append(int(w))
    while sum(widths) > avail:
        j = max(range(n), key=lambda k: widths[k])
        widths[j] = max(1, widths[j] - 1)
    while sum(widths) < avail:
        widths[0] += 1
    return widths


def fit_foci_table_columns(table) -> None:
    """Fit all columns inside the viewport; no horizontal scroll."""
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import QAbstractScrollArea, QHeaderView, QTableWidget

    if not isinstance(table, QTableWidget):
        return
    hdr = table.horizontalHeader()
    n_cols = int(table.columnCount())
    if hdr is None or n_cols <= 0:
        return

    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
    hdr.setStretchLastSection(True)
    hdr.setMinimumSectionSize(1)
    hdr.setCascadingSectionResizes(False)
    hdr.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

    avail = _foci_table_viewport_width(table)
    if avail <= 8:
        for col_i in range(n_cols):
            hdr.setSectionResizeMode(col_i, QHeaderView.ResizeMode.Stretch)
        if int(table.width()) > 20 and not getattr(table, "_foci_col_fit_pending", False):
            table._foci_col_fit_pending = True

            def _retry() -> None:
                table._foci_col_fit_pending = False
                fit_foci_table_columns(table)

            QTimer.singleShot(0, _retry)
        return

    fm = table.fontMetrics()
    pass_w = min(
        FOCI_TABLE_PASS_COL_MAX_WIDTH,
        _foci_table_column_content_width(table, 0, fm),
    )
    pass_w = max(1, int(pass_w))

    loc_w = 0
    stretch_start = 1
    if n_cols > 1:
        loc_w = min(
            FOCI_TABLE_LOCATION_COL_MAX_WIDTH,
            _foci_table_column_content_width(table, 1, fm),
        )
        loc_w = max(1, int(loc_w))
        stretch_start = 2

    fixed_total = pass_w + loc_w
    if fixed_total >= avail:
        for col_i in range(n_cols):
            hdr.setSectionResizeMode(col_i, QHeaderView.ResizeMode.Stretch)
        return

    hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
    table.setColumnWidth(0, pass_w)
    if n_cols > 1:
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(1, loc_w)
    for col_i in range(stretch_start, n_cols):
        hdr.setSectionResizeMode(col_i, QHeaderView.ResizeMode.Stretch)

    try:
        hdr.resizeSections(QHeaderView.ResizeMode.Stretch)
    except Exception:
        pass
