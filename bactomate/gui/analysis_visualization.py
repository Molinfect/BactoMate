"""
Combined analysis QC visualization: horizontal multi-panel TIFF (cells | foci channels…).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile

from bactomate.gui.analysis_paths import base_dir_for_mask
from bactomate.gui.foci_overlay_artifacts import (
    read_foci_csv_rows,
    _coerce_foci_stack_to_tyx_for_overlay,
    _compute_boundaries,
    _draw_circles_rgb,
    _mask_stack_to_tyx,
)
from bactomate.modules.image_stack_utils import coerce_plane_to_2d, percentile_normalize_2d
from bactomate.modules.label_render import draw_hq_labels


PANEL_DIVIDER_RGB = (220, 220, 220)
CELL_BOUNDARY_RGB = (0, 255, 0)
FOCI_BOUNDARY_RGB = (0, 255, 255)
FOCI_MARKER_RGB = (255, 255, 0)


def _phase_stem_from_mask(mask_path: Path) -> str:
    """Phase TIFF stem paired with this mask (must match exactly, not a shared prefix)."""
    stem = Path(mask_path).stem
    if stem.lower().endswith("_mask_filtered"):
        stem = stem[: -len("_mask_filtered")]
    elif stem.lower().endswith("_masks"):
        stem = stem[: -len("_masks")]
    return stem


def resolve_phase_image_for_mask(mask_path: Path) -> Path | None:
    """
    Phase / bright-field stack for this mask only.

    Uses an exact stem match (e.g. ``ssb-mNG_cl1_001_Phase60X.tif`` for the 001 mask)
    so similar sample names (``ssb-mNG_cl1__Phase60X`` vs ``ssb-mNG_cl1_001_…``) are not mixed up.
    """
    mask_path = Path(mask_path)
    phase_stem = _phase_stem_from_mask(mask_path)
    if not phase_stem:
        return None

    search_dirs: list[Path] = []
    try:
        search_dirs.append(base_dir_for_mask(mask_path))
    except Exception:
        pass
    for d in (mask_path.parent, mask_path.parent.parent):
        if d.is_dir() and d not in search_dirs:
            search_dirs.append(d)

    for directory in search_dirs:
        for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
            candidate = directory / f"{phase_stem}{ext}"
            if candidate.is_file():
                return candidate
    return None


def _align_stack_tyx(stack: np.ndarray, *, num_frames: int, height: int, width: int) -> np.ndarray:
    """Crop/pad/repeat stack to (num_frames, height, width)."""
    arr = np.asarray(stack)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    t = int(arr.shape[0])
    if t == 1 and num_frames > 1:
        arr = np.repeat(arr, num_frames, axis=0)
    elif t > num_frames:
        arr = arr[:num_frames]
    out = np.zeros((num_frames, height, width), dtype=arr.dtype if arr.dtype != np.bool_ else np.uint8)
    n = min(t, num_frames)
    for fi in range(n):
        plane = np.asarray(coerce_plane_to_2d(arr[fi]), dtype=np.float32)
        if plane.shape != (height, width):
            try:
                from skimage.transform import resize

                plane = resize(plane, (height, width), order=1, preserve_range=True, anti_aliasing=True)
            except Exception:
                ph, pw = plane.shape[:2]
                plane = plane[: min(ph, height), : min(pw, width)]
                pad = np.zeros((height, width), dtype=np.float32)
                pad[: plane.shape[0], : plane.shape[1]] = plane
                plane = pad
        out[fi] = plane
    return out


def _normalize_plane_u8(
    plane: np.ndarray,
    *,
    pmin: float = 0.5,
    pmax: float = 99.5,
    gamma: float = 1.0,
) -> np.ndarray:
    """Per-frame auto contrast (percentile LUT); gamma < 1 brightens mid-tones."""
    norm = percentile_normalize_2d(np.asarray(plane, dtype=np.float32), pmin=pmin, pmax=pmax)
    norm = np.clip(norm, 0.0, 1.0)
    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        norm = np.power(norm, float(gamma))
    return (norm * 255.0).astype(np.uint8)


def _normalize_fluorescence_u8(plane: np.ndarray) -> np.ndarray:
    """
    Brighter display LUT for dim foci/FI channels.

    Uses log1p compression and percentiles from signal-bearing pixels so a dark
  background does not dominate the stretch.
    """
    arr = np.clip(np.asarray(plane, dtype=np.float32), 0.0, None)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    hi_raw = float(np.percentile(arr, 99.9)) if np.isfinite(arr).any() else 0.0
    work = np.log1p(arr) if hi_raw > 0 else arr
    signal = work[work > 0]
    if signal.size < 64:
        signal = work.ravel()
    lo = float(np.percentile(signal, 0.2))
    hi = float(np.percentile(signal, 99.2))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(work))
        hi = float(np.max(work))
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
    norm = np.clip((work - lo) / (hi - lo), 0.0, 1.0)
    norm = np.power(norm, 0.5)
    return (norm * 255.0).astype(np.uint8)


def _separate_label_positions(
    labels: list[tuple[str, int, int]],
    *,
    min_sep_px: int = 14,
) -> list[tuple[str, int, int]]:
    """Nudge crowded text apart with a small offset search."""
    if len(labels) <= 1:
        return labels
    offsets = (
        (0, 0),
        (0, -12),
        (12, 0),
        (-12, 0),
        (0, 12),
        (10, -10),
        (-10, -10),
        (10, 10),
        (-10, 10),
        (0, -20),
        (18, 0),
        (-18, 0),
    )
    min_sep2 = int(min_sep_px) ** 2
    placed: list[tuple[str, int, int]] = []
    occupied: list[tuple[int, int]] = []
    for text, x, y in labels:
        best = (int(x), int(y))
        for dx, dy in offsets:
            tx, ty = int(x) + dx, int(y) + dy
            if all((tx - ox) ** 2 + (ty - oy) ** 2 >= min_sep2 for ox, oy in occupied):
                best = (tx, ty)
                break
        placed.append((text, best[0], best[1]))
        occupied.append(best)
    return placed


def _cell_label_font_size(num_cells: int) -> int:
    if num_cells > 140:
        return 8
    if num_cells > 80:
        return 9
    if num_cells > 45:
        return 10
    return 11


def _cell_id_labels_from_mask(
    mask_2d: np.ndarray,
    *,
    only_cell_ids: set[int] | None = None,
) -> list[tuple[str, int, int]]:
    """
    Place Cell_ID labels from region centroids (fast) and nudge apart when crowded.

    Avoids per-cell distance transforms, which are very slow on large fields.
    """
    labels: list[tuple[str, int, int]] = []
    mask_i = np.asarray(mask_2d, dtype=np.int32)
    if not np.any(mask_i > 0):
        return labels
    try:
        from skimage.measure import regionprops

        for region in regionprops(mask_i):
            cid = int(region.label)
            if cid <= 0:
                continue
            if only_cell_ids is not None and cid not in only_cell_ids:
                continue
            cy, cx = region.centroid
            minr, _, maxr, _ = region.bbox
            cy = int(min(cy, (minr + maxr) // 2))
            labels.append((str(cid), int(cx), int(cy) - 2))
    except Exception:
        pass
    if len(labels) > 1:
        return _separate_label_positions(labels)
    return labels


def _draw_labels_on_rgb(
    rgb: np.ndarray,
    labels: list[tuple[str, int, int]],
    *,
    font_size: int = 11,
    fill_rgb: tuple[int, int, int] = (255, 255, 255),
    stroke_width: int = 2,
) -> np.ndarray:
    if not labels:
        return rgb
    try:
        from PIL import Image

        pil = Image.fromarray(rgb)
        pil = draw_hq_labels(
            pil,
            labels,
            font_size=font_size,
            scale=1,
            stroke_width=stroke_width,
            fill_rgb=fill_rgb,
        )
        # PIL-backed arrays can be read-only; later drawing must mutate in place.
        return np.asarray(pil, dtype=np.uint8).copy()
    except Exception:
        return np.asarray(rgb, dtype=np.uint8).copy()


def _draw_panel_title(rgb: np.ndarray, title: str) -> np.ndarray:
    try:
        from PIL import Image

        pil = Image.fromarray(rgb)
        pil = draw_hq_labels(pil, [(title, 8, 8)], font_size=11, scale=1, stroke_width=2)
        return np.asarray(pil, dtype=np.uint8).copy()
    except Exception:
        return np.asarray(rgb, dtype=np.uint8).copy()


def _render_cells_panel_frame(
    mask_2d: np.ndarray,
    phase_2d: np.ndarray | None,
    *,
    title: str = "Cells",
) -> np.ndarray:
    h, w = mask_2d.shape[:2]
    if phase_2d is not None and phase_2d.shape == (h, w):
        gray = _normalize_plane_u8(phase_2d, pmin=0.5, pmax=99.5, gamma=0.82)
    else:
        gray = np.zeros((h, w), dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    mask_i = np.asarray(mask_2d, dtype=np.int32)
    if np.any(mask_i > 0):
        b = _compute_boundaries(mask_i)
        if np.any(b):
            rgb[b, 0] = CELL_BOUNDARY_RGB[0]
            rgb[b, 1] = CELL_BOUNDARY_RGB[1]
            rgb[b, 2] = CELL_BOUNDARY_RGB[2]

    cell_labels = _cell_id_labels_from_mask(mask_2d)
    rgb = _draw_labels_on_rgb(
        rgb,
        cell_labels,
        font_size=_cell_label_font_size(len(cell_labels)),
        fill_rgb=(255, 255, 255),
        stroke_width=3,
    )
    return _draw_panel_title(rgb, title)


def _render_foci_panel_frame(
    foci_2d: np.ndarray | None,
    mask_2d: np.ndarray,
    foci_rows: list[dict[str, Any]],
    *,
    title: str,
) -> np.ndarray:
    h, w = mask_2d.shape[:2]
    if foci_2d is not None and np.asarray(foci_2d).shape == (h, w):
        gray = _normalize_fluorescence_u8(foci_2d)
    else:
        gray = np.zeros((h, w), dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    mask_i = np.asarray(mask_2d, dtype=np.int32)
    if np.any(mask_i > 0):
        b = _compute_boundaries(mask_i)
        if np.any(b):
            rgb[b, 0] = FOCI_BOUNDARY_RGB[0]
            rgb[b, 1] = FOCI_BOUNDARY_RGB[1]
            rgb[b, 2] = FOCI_BOUNDARY_RGB[2]

    ys: list[int] = []
    xs: list[int] = []
    if foci_rows:
        rgb = np.asarray(rgb, dtype=np.uint8).copy()
        ys = [int(r.get("y", 0)) for r in foci_rows]
        xs = [int(r.get("x", 0)) for r in foci_rows]
        foci_labels: list[tuple[str, int, int]] = []
        for r in foci_rows:
            fid = r.get("foci_id")
            if fid in (None, "", "None"):
                continue
            try:
                fid_i = int(fid)
            except Exception:
                continue
            y, x = int(r.get("y", 0)), int(r.get("x", 0))
            foci_labels.append((str(fid_i), x + 4, y - 4))
        if len(foci_labels) > 1:
            foci_labels = _separate_label_positions(foci_labels, min_sep_px=10)
        foci_font = 9 if len(foci_labels) > 200 else 10
        rgb = _draw_labels_on_rgb(
            rgb,
            foci_labels,
            font_size=foci_font,
            fill_rgb=(255, 230, 80),
            stroke_width=2,
        )

    rgb = _draw_panel_title(rgb, title)
    if ys and xs:
        rgb = np.asarray(rgb, dtype=np.uint8).copy()
        _draw_circles_rgb(rgb, ys=ys, xs=xs, radius=2, rgb=FOCI_MARKER_RGB)
    return rgb


def _concat_panels_horiz(panels: list[np.ndarray], *, divider_px: int = 2) -> np.ndarray:
    if not panels:
        raise ValueError("no panels")
    if len(panels) == 1:
        return panels[0]
    h = panels[0].shape[0]
    div = np.zeros((h, divider_px, 3), dtype=np.uint8)
    div[:] = np.array(PANEL_DIVIDER_RGB, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if i > 0:
            parts.append(div)
        parts.append(p)
    return np.concatenate(parts, axis=1)


def load_foci_table_from_workbook(analysis_path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        df = pd.read_excel(analysis_path, sheet_name="Foci")
    except Exception:
        return []
    if df is None or df.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            out.append(
                {
                    "foci_id": row.get("foci_id"),
                    "cell_id": row.get("cell_id"),
                    "frame": int(row.get("frame", 0) or 0),
                    "channel_name": str(row.get("channel_name") or ""),
                    "y": int(row.get("y", 0)),
                    "x": int(row.get("x", 0)),
                }
            )
        except Exception:
            continue
    return out


def write_combined_analysis_visualization(
    tab,
    *,
    mask_path: Path,
    output_path: Path,
    analysis_workbook: Path | None = None,
    prefer_filtered: bool = True,
) -> bool:
    """
    Write one multi-panel RGB TIFF: [Cells (phase)] | [Foci ch1] | [Foci ch2] | …

    Foci IDs are read from the Foci sheet when present; otherwise peaks are drawn without IDs.
    """
    from bactomate.gui.analysis_workbook import discover_foci_csv_artifacts_for_mask

    mask_path = Path(mask_path)
    output_path = Path(output_path)
    try:
        mask_raw = tifffile.imread(str(mask_path))
        mask_tyx = _mask_stack_to_tyx(np.asarray(mask_raw))
    except Exception:
        return False
    if mask_tyx is None or mask_tyx.size == 0:
        return False

    t, h, w = int(mask_tyx.shape[0]), int(mask_tyx.shape[1]), int(mask_tyx.shape[2])

    phase_tyx = None
    phase_path = resolve_phase_image_for_mask(mask_path)
    if phase_path is not None:
        try:
            phase_tyx = _align_stack_tyx(tifffile.imread(str(phase_path)), num_frames=t, height=h, width=w)
        except Exception:
            phase_tyx = None

    foci_table: list[dict[str, Any]] = []
    if analysis_workbook is not None and Path(analysis_workbook).is_file():
        foci_table = load_foci_table_from_workbook(Path(analysis_workbook))

    artifacts = discover_foci_csv_artifacts_for_mask(tab, mask_path, prefer_filtered=prefer_filtered)

    artifact_labels = [a.channel_label for a in artifacts]
    artifact_label_set = set(artifact_labels)

    def _display_channel_for_row(raw_channel: Any) -> str:
        """Map legacy generic workbook rows onto the resolved source channel."""
        ch = str(raw_channel or "foci").strip() or "foci"
        if ch in artifact_label_set:
            return ch
        if len(artifact_labels) == 1 and ch.lower() in {"", "foci"}:
            return artifact_labels[0]
        return ch

    # Group foci rows by (frame, channel) for panels.
    foci_by_channel_frame: dict[str, dict[int, list[dict[str, Any]]]] = {}
    if foci_table:
        for r in foci_table:
            ch = _display_channel_for_row(r.get("channel_name"))
            try:
                fr = int(r.get("frame", 0) or 0)
            except Exception:
                fr = 0
            foci_by_channel_frame.setdefault(ch, {}).setdefault(fr, []).append(r)
    else:
        for art in artifacts:
            ch = art.channel_label
            rows = read_foci_csv_rows(art.csv_path)
            for r in rows:
                fr = int(r.get("frame", 0) or 0)
                foci_by_channel_frame.setdefault(ch, {}).setdefault(fr, []).append(dict(r))

    foci_stacks: dict[str, np.ndarray | None] = {}
    for art in artifacts:
        try:
            raw = tifffile.imread(str(art.foci_stack))
            coerced = _coerce_foci_stack_to_tyx_for_overlay(np.asarray(raw))
            if coerced is None:
                coerced = raw
            foci_stacks[art.channel_label] = _align_stack_tyx(
                coerced,
                num_frames=t,
                height=h,
                width=w,
            )
        except Exception:
            foci_stacks[art.channel_label] = None

    channel_order = list(artifact_labels)
    for ch in sorted(foci_by_channel_frame.keys(), key=str.lower):
        if ch not in channel_order:
            channel_order.append(ch)

    combined_frames: list[np.ndarray] = []
    try:
        for fi in range(t):
            mask_2d = np.asarray(mask_tyx[fi], dtype=np.int32)
            phase_2d = np.asarray(phase_tyx[fi], dtype=np.float32) if phase_tyx is not None else None
            panels = [_render_cells_panel_frame(mask_2d, phase_2d, title="Cells")]
            for ch in channel_order:
                rows = (foci_by_channel_frame.get(ch) or {}).get(fi, [])
                foci_2d = None
                stack = foci_stacks.get(ch)
                if stack is not None and fi < stack.shape[0]:
                    foci_2d = stack[fi]
                panels.append(_render_foci_panel_frame(foci_2d, mask_2d, rows, title=ch))
            combined_frames.append(_concat_panels_horiz(panels))
    except Exception:
        from bactomate.logging_config import get_logger

        get_logger().exception("Combined analysis visualization render failed")
        return False

    out = np.stack(combined_frames, axis=0).astype(np.uint8)
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
        from bactomate.logging_config import get_logger

        get_logger().exception("Combined analysis visualization write failed")
        return False
