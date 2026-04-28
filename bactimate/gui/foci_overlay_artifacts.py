from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import tifffile

from bactimate.modules.image_stack_utils import coerce_plane_to_2d, coerce_stack_to_tyx, percentile_normalize_2d


@dataclass(frozen=True)
class FociOverlayStyle:
    """Rendering knobs for foci overlay TIFF artifacts."""

    pmin: float = 5.0
    pmax: float = 99.0
    marker_radius_px: int = 2
    marker_rgb: tuple[int, int, int] = (255, 255, 0)  # yellow
    boundary_rgb: tuple[int, int, int] = (0, 255, 255)  # cyan
    draw_boundaries: bool = True


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
            rows.append({"frame": frame, "y": y, "x": x, "intensity": intensity, "probability": prob})
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
    for y, x in zip(ys, xs, strict=False):
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


def write_foci_overlay_tif(
    *,
    foci_image_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    mask_path: Path | None = None,
    style: FociOverlayStyle | None = None,
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

        # Markers
        rr = rows_by_frame.get(fi) or []
        if rr:
            ys = [_safe_int(r.get("y", 0), 0) for r in rr]
            xs = [_safe_int(r.get("x", 0), 0) for r in rr]
            _draw_circles_rgb(rgb, ys=ys, xs=xs, radius=int(style.marker_radius_px), rgb=style.marker_rgb)

        out[fi] = rgb

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(output_path), out, photometric="rgb")
        return True
    except Exception:
        return False

