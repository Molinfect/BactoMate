from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import tifffile

from bactomate.modules.image_stack_utils import coerce_plane_to_2d, coerce_stack_to_tyx, percentile_normalize_2d


@dataclass(frozen=True)
class FociOverlayStyle:
    """Rendering knobs for foci overlay TIFF artifacts."""

    pmin: float = 5.0
    pmax: float = 99.0
    marker_radius_px: int = 1
    marker_rgb: tuple[int, int, int] = (255, 30, 30)  # Spotiflow peak pixel (red)
    outline_rgb: tuple[int, int, int] = (255, 200, 120)  # exclusive blob outline
    boundary_rgb: tuple[int, int, int] = (0, 255, 255)  # cyan cell boundaries
    draw_boundaries: bool = True
    draw_blob_regions: bool = True
    draw_blob_fill: bool = False
    blob_alpha: float = 0.35

    @classmethod
    def segmentation(cls) -> FociOverlayStyle:
        """Same language as filtering: light outline + red peak."""
        return cls()

    @classmethod
    def filtered(cls) -> FociOverlayStyle:
        """Same language as segmentation (kept as a named style for callers)."""
        return cls()


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def read_foci_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    """
    Read foci CSV rows produced by segmentation/filtering steps.

    Expected columns:
    - snapshot: y, x, intensity, probability
    - time sequence: frame, y, x, intensity, probability

    Returns rows with keys: frame (int, default 0), y (int), x (int),
    intensity (float), probability (float|None).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or not csv_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            y = _safe_int(r.get("y", 0), 0)
            x = _safe_int(r.get("x", 0), 0)
            frame = 0
            if "frame" in (reader.fieldnames or []):
                frame = _safe_int(r.get("frame", 0), 0)
            intensity = _safe_float(r.get("intensity", 0.0), 0.0)
            prob_raw = r.get("probability", None)
            prob: float | None = None
            if prob_raw not in (None, "", "None"):
                try:
                    prob = float(prob_raw)
                except Exception:
                    prob = None
            try:
                size_px = int(float(r.get("area_px", r.get("size_px", 1)) or 1))
            except (TypeError, ValueError):
                size_px = 1
            rows.append(
                {
                    "frame": frame,
                    "y": y,
                    "x": x,
                    "intensity": intensity,
                    "probability": prob,
                    "size_px": size_px,
                    "area_px": size_px,
                }
            )
    return rows


def foci_overlay_path_for_csv(csv_path: Path) -> Path:
    """
    Segmentation-stage raw foci overlay artifact path.

    Mirrors foci CSV naming:
      <stem>_foci.csv -> <stem>_foci_overlay.tif
      <stem>_foci_<ch>.csv -> <stem>_foci_overlay_<ch>.tif
    """
    csv_path = Path(csv_path)
    stem = csv_path.stem
    # Use conservative replacements so we don't mangle unrelated names.
    if "_foci_filtered_" in stem:
        stem = stem.replace("_foci_filtered_", "_foci_overlay_", 1)
    elif stem.endswith("_foci_filtered"):
        stem = stem[: -len("_foci_filtered")] + "_foci_overlay"
    elif "_foci_" in stem:
        stem = stem.replace("_foci_", "_foci_overlay_", 1)
    elif stem.endswith("_foci"):
        stem = stem + "_overlay"
    else:
        stem = stem + "_foci_overlay"
    return csv_path.with_name(stem + ".tif")


def filtered_overlay_path_for_filtered_csv(filtered_csv_path: Path) -> Path:
    """
    Filtering-stage foci overlay artifact path.

    Mirrors filtered foci CSV naming:
      <stem>_foci_filtered.csv -> <stem>_filtered_overlay.tif
      <stem>_foci_filtered_<ch>.csv -> <stem>_filtered_overlay_<ch>.tif
    """
    filtered_csv_path = Path(filtered_csv_path)
    stem = filtered_csv_path.stem
    if "_foci_filtered_" in stem:
        stem = stem.replace("_foci_filtered_", "_filtered_overlay_", 1)
    elif stem.endswith("_foci_filtered"):
        stem = stem[: -len("_foci_filtered")] + "_filtered_overlay"
    else:
        stem = stem + "_filtered_overlay"
    return filtered_csv_path.with_name(stem + ".tif")


def _draw_circles_rgb(frame_rgb: np.ndarray, *, ys: Iterable[int], xs: Iterable[int], radius: int, rgb: tuple[int, int, int]) -> None:
    h, w = frame_rgb.shape[:2]
    r = max(1, min(int(radius), 8))
    rr = r * r
    for y, x in zip(ys, xs):
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        for yy in range(y0, y1):
            dy2 = (yy - y) ** 2
            for xx in range(x0, x1):
                if dy2 + (xx - x) ** 2 <= rr:
                    frame_rgb[yy, xx, 0] = rgb[0]
                    frame_rgb[yy, xx, 1] = rgb[1]
                    frame_rgb[yy, xx, 2] = rgb[2]


def _compute_boundaries(mask_2d: np.ndarray) -> np.ndarray:
    try:
        from skimage.segmentation import find_boundaries

        return np.asarray(find_boundaries(mask_2d, mode="inner"), dtype=bool)
    except Exception:
        # Fallback: simple 4-neighborhood edge detection.
        m = np.asarray(mask_2d, dtype=np.int32)
        if m.ndim != 2 or m.size == 0:
            return np.zeros((0, 0), dtype=bool)
        h, w = m.shape
        out = np.zeros((h, w), dtype=bool)
        out[1:, :] |= m[1:, :] != m[:-1, :]
        out[:-1, :] |= m[:-1, :] != m[1:, :]
        out[:, 1:] |= m[:, 1:] != m[:, :-1]
        out[:, :-1] |= m[:, :-1] != m[:, 1:]
        return out & (m > 0)


def _mask_stack_to_tyx(mask_stack: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(mask_stack)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    # if it is TYXC-like, coerce_stack_to_tyx will attempt to pick channel 0; good enough for boundaries.
    try:
        return coerce_stack_to_tyx(arr)
    except Exception:
        return None


def _coerce_foci_stack_to_tyx_for_overlay(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Overlay-specific coercion to (T, Y, X).

    Unlike the general-purpose `coerce_stack_to_tyx`, we treat 3D arrays as TYX,
    even when T is small (<10). This avoids misclassifying short time sequences
    as channels-first CYX.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    try:
        return coerce_stack_to_tyx(arr)
    except Exception:
        return None


def _paint_blob_on_rgb(
    rgb: np.ndarray,
    plane: np.ndarray,
    mask_2d: np.ndarray | None,
    row: dict[str, Any],
    *,
    threshold_frac: float,
    max_radius: int,
    fill_rgb: tuple[int, int, int],
    alpha: float,
    blob_mask: np.ndarray | None = None,
) -> None:
    from bactomate.modules.foci_utils import get_focus_blob_mask

    y = _safe_int(row.get("y", 0), 0)
    x = _safe_int(row.get("x", 0), 0)
    h, w = rgb.shape[:2]
    if y < 0 or y >= h or x < 0 or x >= w:
        return
    label = np.asarray(mask_2d, dtype=np.int32) if mask_2d is not None else None
    if blob_mask is not None:
        blob = np.asarray(blob_mask, dtype=bool)
    else:
        blob = get_focus_blob_mask(
            np.asarray(plane, dtype=np.float64),
            y,
            x,
            threshold_frac=float(threshold_frac),
            max_radius=int(max_radius),
            labeled_mask=label,
        )
    if blob is None or not np.any(blob):
        return
    if blob.shape != (h, w):
        return
    a = float(np.clip(alpha, 0.05, 1.0))
    r, g, b = fill_rgb
    mask_bool = np.asarray(blob, dtype=bool)
    rgb[mask_bool, 0] = np.clip(rgb[mask_bool, 0].astype(np.float32) * (1 - a) + r * a, 0, 255).astype(
        np.uint8
    )
    rgb[mask_bool, 1] = np.clip(rgb[mask_bool, 1].astype(np.float32) * (1 - a) + g * a, 0, 255).astype(
        np.uint8
    )
    rgb[mask_bool, 2] = np.clip(rgb[mask_bool, 2].astype(np.float32) * (1 - a) + b * a, 0, 255).astype(
        np.uint8
    )


def write_foci_overlay_tif(
    *,
    foci_image_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    mask_path: Path | None = None,
    style: FociOverlayStyle | None = None,
    blob_settings: dict[str, Any] | None = None,
) -> bool:
    """
    Write a saved overlay TIFF:
    - background: foci_image_path, normalized per frame using percentile normalization (pmin/pmax)
    - markers: `rows` with 0-based frame indexing
    - optional: mask boundary overlay (if mask_path provided and readable)
    """
    style = style or FociOverlayStyle()
    foci_image_path = Path(foci_image_path)
    output_path = Path(output_path)

    try:
        raw = tifffile.imread(str(foci_image_path))
    except Exception:
        return False
    foci_tyx = _coerce_foci_stack_to_tyx_for_overlay(np.asarray(raw))
    if foci_tyx is None or foci_tyx.size == 0:
        return False

    t = int(foci_tyx.shape[0])
    h, w = int(foci_tyx.shape[1]), int(foci_tyx.shape[2])
    out = np.zeros((t, h, w, 3), dtype=np.uint8)

    # Optional mask boundaries (frame-aligned best-effort).
    boundaries_by_frame: dict[int, np.ndarray] = {}
    mask_tyx = None
    if style.draw_boundaries and mask_path is not None:
        try:
            mask_raw = tifffile.imread(str(mask_path))
            mask_tyx = _mask_stack_to_tyx(np.asarray(mask_raw))
        except Exception:
            mask_tyx = None
        if mask_tyx is not None and mask_tyx.size > 0:
            # Align by truncation/repeat; we only need boundaries for frames we render.
            if mask_tyx.shape[0] == 1 and t > 1:
                mask_tyx = np.repeat(mask_tyx, t, axis=0)
            n = min(int(mask_tyx.shape[0]), t)
            for fi in range(n):
                boundaries_by_frame[fi] = _compute_boundaries(np.asarray(mask_tyx[fi], dtype=np.int32))

    # Pre-group rows by frame
    rows_by_frame: dict[int, list[dict[str, Any]]] = {}
    for r in rows or []:
        fi = _safe_int(r.get("frame", 0), 0)
        if fi < 0 or fi >= t:
            continue
        rows_by_frame.setdefault(fi, []).append(r)

    for fi in range(t):
        plane = np.asarray(coerce_plane_to_2d(foci_tyx[fi]), dtype=np.float32)
        if plane.shape != (h, w):
            continue
        norm = percentile_normalize_2d(plane, pmin=style.pmin, pmax=style.pmax)
        gray = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

        # Boundaries first (so markers stay visible on top)
        b = boundaries_by_frame.get(fi)
        if b is not None and b.shape == (h, w) and np.any(b):
            rgb[b, 0] = style.boundary_rgb[0]
            rgb[b, 1] = style.boundary_rgb[1]
            rgb[b, 2] = style.boundary_rgb[2]

        rr = rows_by_frame.get(fi) or []
        mask_2d = None
        if mask_tyx is not None and fi < int(mask_tyx.shape[0]):
            mask_2d = np.asarray(mask_tyx[fi], dtype=np.int32)
        bs = dict(blob_settings or {})
        if style.draw_blob_regions and rr:
            from bactomate.gui.foci_filter_pipeline import foci_blob_threshold_frac

            frac = foci_blob_threshold_frac(bs)
            try:
                max_r = int(bs.get("foci_blob_max_radius", 5) or 5)
            except (TypeError, ValueError):
                max_r = 5
            try:
                from skimage.segmentation import find_boundaries

                from bactomate.modules.foci_utils import exclusive_focus_blob_metrics

                measured = exclusive_focus_blob_metrics(
                    plane,
                    rr,
                    threshold_frac=frac,
                    max_radius=max_r,
                    labeled_mask=mask_2d,
                )
                labeled = np.zeros((h, w), dtype=np.int32)
                for i, (peak, blob, _size, _mean) in enumerate(measured):
                    if blob is None or not np.any(blob):
                        continue
                    if bool(style.draw_blob_fill):
                        try:
                            _paint_blob_on_rgb(
                                rgb,
                                plane,
                                mask_2d,
                                peak,
                                threshold_frac=frac,
                                max_radius=max_r,
                                fill_rgb=style.outline_rgb,
                                alpha=float(style.blob_alpha),
                                blob_mask=blob,
                            )
                        except Exception:
                            pass
                    labeled[np.asarray(blob, dtype=bool)] = i + 1
                if np.any(labeled):
                    try:
                        edge = find_boundaries(labeled, mode="outer")
                        if np.any(edge):
                            rgb[edge, 0] = style.outline_rgb[0]
                            rgb[edge, 1] = style.outline_rgb[1]
                            rgb[edge, 2] = style.outline_rgb[2]
                    except Exception:
                        pass
            except Exception:
                for r in rr:
                    try:
                        _paint_blob_on_rgb(
                            rgb,
                            plane,
                            mask_2d,
                            r,
                            threshold_frac=frac,
                            max_radius=max_r,
                            fill_rgb=style.outline_rgb,
                            alpha=float(style.blob_alpha),
                        )
                    except Exception:
                        continue
            # Spotiflow peak pixels on top (red).
            for r in rr:
                y = _safe_int(r.get("y", 0), 0)
                x = _safe_int(r.get("x", 0), 0)
                if 0 <= y < h and 0 <= x < w:
                    rgb[y, x, 0] = style.marker_rgb[0]
                    rgb[y, x, 1] = style.marker_rgb[1]
                    rgb[y, x, 2] = style.marker_rgb[2]
        elif rr:
            for r in rr:
                y = _safe_int(r.get("y", 0), 0)
                x = _safe_int(r.get("x", 0), 0)
                if 0 <= y < h and 0 <= x < w:
                    rgb[y, x, 0] = style.marker_rgb[0]
                    rgb[y, x, 1] = style.marker_rgb[1]
                    rgb[y, x, 2] = style.marker_rgb[2]

        out[fi] = rgb

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(output_path), out, photometric="rgb")
        try:
            from bactomate.modules.pipeline_outputs import write_png_preview_from_tiff

            write_png_preview_from_tiff(output_path)
        except Exception:
            pass
        return True
    except Exception:
        return False


def write_filtered_foci_overlay_for_csv(
    *,
    foci_image_path: Path,
    filtered_csv_path: Path,
    rows: list[dict[str, Any]],
    mask_path: Path | None = None,
    filter_settings: dict[str, Any] | None = None,
) -> tuple[Path, bool]:
    """
    Write ``*_filtered_overlay.tif`` next to a filtered foci CSV (distinct from segmentation ``*_foci_overlay.tif``).

    Returns (output_path, success).
    """
    output_path = filtered_overlay_path_for_filtered_csv(Path(filtered_csv_path))
    ok = write_foci_overlay_tif(
        foci_image_path=Path(foci_image_path),
        output_path=output_path,
        rows=list(rows or []),
        mask_path=mask_path,
        style=FociOverlayStyle.filtered(),
        blob_settings=dict(filter_settings or {}),
    )
    return output_path, bool(ok)


def refresh_filtered_foci_overlays_after_filtering(
    tab,
    mask_paths: list[Path],
    *,
    filter_settings: dict[str, Any] | None = None,
) -> list[str]:
    """
    Rebuild filtered foci overlay TIFFs from ``foci_filtered/*.csv``.

    Segmentation overlays (``*_foci_overlay.tif`` in ``foci_detection/``) are not modified.
    """
    from bactomate.gui.artifact_paths import (
        filtered_foci_csv_path,
        infer_pipeline_dir_from_mask,
        resolve_filtered_foci_csv,
    )
    from bactomate.gui.artifact_paths import filtered_mask_path_for_mask

    fs = dict(filter_settings or {})
    if hasattr(tab, "get_filter_settings"):
        try:
            fs = dict(tab.get_filter_settings() or fs)
        except Exception:
            pass

    log_lines: list[str] = []
    for mask_path in mask_paths:
        mask_path = Path(mask_path)
        pipeline_dir = infer_pipeline_dir_from_mask(mask_path)
        mask_for_overlay = mask_path
        try:
            filt_mask = filtered_mask_path_for_mask(mask_path)
            if filt_mask.is_file():
                mask_for_overlay = filt_mask
        except Exception:
            pass
        try:
            foci_sources = tab._resolve_foci_paths_for_mask(mask_path)
        except Exception:
            foci_sources = []
        if not foci_sources:
            continue
        for src in foci_sources:
            try:
                out_csv = resolve_filtered_foci_csv(
                    pipeline_dir, mask_path=mask_path, foci_source_path=src
                )
                if out_csv is None or not out_csv.is_file():
                    out_csv = filtered_foci_csv_path(
                        pipeline_dir, mask_path=mask_path, foci_source_path=src
                    )
                if not out_csv.is_file():
                    log_lines.append(
                        f"[Filtering] No filtered foci CSV for overlay ({Path(src).name}); skipped.\n"
                    )
                    continue
                rows = read_foci_csv_rows(out_csv)
                overlay_path, ok = write_filtered_foci_overlay_for_csv(
                    foci_image_path=Path(src),
                    filtered_csv_path=out_csv,
                    rows=rows,
                    mask_path=mask_for_overlay if Path(mask_for_overlay).exists() else None,
                    filter_settings=fs,
                )
                if ok:
                    log_lines.append(f"[Filtering] Wrote filtered overlay: {overlay_path}\n")
                else:
                    log_lines.append(
                        f"[Filtering] Failed to write filtered overlay for {Path(src).name}.\n"
                    )
            except Exception as exc:
                log_lines.append(
                    f"[Filtering] Filtered overlay failed for {Path(src).name}: {exc}\n"
                )
    return log_lines

