"""
Growth-curve video assembly, preview loading, and export worker.
"""
import io
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from bactomate.gui.log_config import get_logger, log_exception

DEFAULT_GROWTH_GRAPH_STYLE: dict[str, Any] = {
    "x_label": "Time [min]",
    "y_label": "rel. cell area",
    "axis_label_font_size": 36,
    "tick_font_size": 24,
    "font_weight": "bold",
    "line_width": 8.0,
    "marker_size": 200.0,
    "marker_edge_width": 2.0,
    "axis_stroke_width": 3.5,
    "tick_stroke_width": 2.0,
    "show_grid": True,
    "grid_line_width": 1.5,
    "grid_alpha": 0.4,
}

DEFAULT_GROWTH_Y_COLUMN = "Total Cell Area (Relative)"
VIDEO_FI_LAYOUT_MERGED = "merged"
VIDEO_FI_LAYOUT_SEPARATE = "separate_tiles"
DEFAULT_VIDEO_FI_LAYOUT = VIDEO_FI_LAYOUT_MERGED
VIDEO_FI_OVERLAY_TILE_LABELS = frozenset({"Phase", "Merged"})
MAX_VIDEO_FI_COLOR_CHANNELS = 6
VIDEO_FI_AUTO_LUT_NAMES: tuple[str, ...] = ("Green", "Magenta", "Cyan", "Yellow", "Red", "Orange")
VIDEO_OVERLAY_COLOR_PRESETS: dict[str, tuple[float, float, float]] = {
    "Green": (0.0, 1.0, 0.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
    "Red": (1.0, 0.2, 0.2),
    "Orange": (1.0, 0.55, 0.0),
    "Gray": (0.85, 0.85, 0.85),
    "White": (1.0, 1.0, 1.0),
}
DEFAULT_VIDEO_MASK_CONTOUR_RGB = (255, 215, 0)
DEFAULT_VIDEO_FOCI_MARKER_RGB = (255, 80, 255)
DEFAULT_VIDEO_OVERLAY_COLORS: dict[str, Any] = {
    "fi_channels": ["auto"] * MAX_VIDEO_FI_COLOR_CHANNELS,
    "mask_contour": "auto",
    "foci_marker": "auto",
}
PREFERRED_GROWTH_Y_COLUMNS: tuple[str, ...] = (
    "Total Cell Area (Relative)",
    "Total Cell Area (µm²)",
    "Number of Cells",
    "Total Foci",
    "Mean Foci per Cell",
)

BUILTIN_GROWTH_GRAPH_PRESETS: dict[str, dict[str, Any]] = {
    "Paper": {
        "axis_label_font_size": 28,
        "tick_font_size": 18,
        "line_width": 4.0,
        "marker_size": 120.0,
        "marker_edge_width": 1.5,
        "axis_stroke_width": 2.5,
        "tick_stroke_width": 1.5,
        "show_grid": True,
        "grid_line_width": 1.0,
        "grid_alpha": 0.25,
    },
    "Presentation": {
        "axis_label_font_size": 44,
        "tick_font_size": 30,
        "line_width": 10.0,
        "marker_size": 260.0,
        "marker_edge_width": 2.5,
        "axis_stroke_width": 4.0,
        "tick_stroke_width": 2.5,
        "show_grid": True,
        "grid_line_width": 2.0,
        "grid_alpha": 0.35,
    },
    "Minimal": {
        "show_grid": False,
        "line_width": 5.0,
        "marker_size": 140.0,
        "axis_stroke_width": 2.0,
        "tick_stroke_width": 1.5,
    },
}
def _read_tiff_series_array_and_axes(path: Path) -> tuple[np.ndarray, str]:
    """Load first series from a TIFF; return (array, axes string)."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        s0 = tif.series[0]
        arr = s0.asarray()
        axes = getattr(s0, "axes", None) or ""
    return arr, axes


def _read_video_channel_names_from_metadata(path: Path, count: int) -> list[str]:
    """Best-effort channel names from TIFF/ND2 metadata."""
    try:
        from bactomate.modules.tiff_metadata import read_metadata_from_path

        meta = read_metadata_from_path(path)
        names = [str(n).strip() for n in (meta.get("channel_names") or []) if str(n).strip()]
        if len(names) >= count:
            return names[:count]
    except Exception:
        pass
    return [f"Ch{i + 1}" for i in range(count)]


def _video_channel_name_is_phase(name: str) -> bool:
    text = str(name or "").lower()
    return any(token in text for token in ("phase", "brightfield", "transmission", "dic", " bf"))


def _display_video_channel_name(name: str) -> str:
    text = str(name or "").strip()
    lower = text.lower()
    if "foci" in lower or "focus" in lower:
        label = text
        for token in ("_fi_foci_", "_fi_focus_", "_foci_", "_focus_", "foci_", "focus_"):
            if token in lower:
                pos = lower.find(token)
                label = text[pos + len(token):] or text
                break
        label = label.replace("_", " ").strip()
        return f"Foci {label}".strip() if label else "Foci"
    return text.replace("_", " ").strip() or name


def _align_fi_color_presets_to_channels(channel_names: Sequence[str], fi_presets: Sequence[Any]) -> list[Any]:
    """Map FI-only presets onto all image channels, leaving phase-like channels on auto."""
    aligned: list[Any] = []
    fi_idx = 0
    for name in channel_names:
        if _video_channel_name_is_phase(str(name)):
            aligned.append("auto")
            continue
        aligned.append(fi_presets[fi_idx] if fi_idx < len(fi_presets) else "auto")
        fi_idx += 1
    return aligned


def _transpose_to_tcyx(arr: np.ndarray, axes: str) -> Optional[np.ndarray]:
    """If axes is exactly T,C,Y,X (no Z) return array shaped (T, C, Y, X)."""
    if arr.ndim != 4 or not axes or len(axes) != len(arr.shape):
        return None
    if "Z" in axes or set(axes) != {"T", "C", "Y", "X"}:
        return None
    d = {a: i for i, a in enumerate(axes)}
    return np.transpose(arr, (d["T"], d["C"], d["Y"], d["X"]))


def _heuristic_tchw(arr: np.ndarray) -> Optional[np.ndarray]:
    """Guess (T, C, H, W) multi-channel stack from shape when axes are missing."""
    if arr.ndim != 4:
        return None
    t, a, b, c = arr.shape
    if 2 <= a <= 8 and a < b and a < c:
        return arr
    if 2 <= c <= 8 and c < b and c < t and c != 3:
        return np.transpose(arr, (0, 3, 1, 2))
    return None


def normalize_video_overlay_colors(data: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Merge saved overlay color settings with defaults."""
    merged: dict[str, Any] = {
        "fi_channels": list(DEFAULT_VIDEO_OVERLAY_COLORS["fi_channels"]),
        "mask_contour": "auto",
        "foci_marker": "auto",
    }
    if not data:
        return merged
    raw_fi = data.get("fi_channels")
    if isinstance(raw_fi, list):
        for idx in range(MAX_VIDEO_FI_COLOR_CHANNELS):
            if idx < len(raw_fi):
                merged["fi_channels"][idx] = raw_fi[idx]
    for key in ("mask_contour", "foci_marker"):
        if key in data and data[key] is not None:
            merged[key] = data[key]
    return merged


def _overlay_color_entry_to_rgb01(entry: Any, *, lut_index: int) -> tuple[float, float, float]:
    """Resolve a preset name, auto, or [r,g,b] list to 0-1 RGB."""
    if isinstance(entry, (list, tuple)) and len(entry) == 3:
        vals = [float(v) for v in entry]
        if max(vals) > 1.0:
            vals = [v / 255.0 for v in vals]
        return (vals[0], vals[1], vals[2])
    text = str(entry or "auto").strip()
    if text.lower() in {"", "auto", "default"}:
        text = VIDEO_FI_AUTO_LUT_NAMES[lut_index % len(VIDEO_FI_AUTO_LUT_NAMES)]
    preset = VIDEO_OVERLAY_COLOR_PRESETS.get(text)
    if preset is not None:
        return preset
    return VIDEO_OVERLAY_COLOR_PRESETS[VIDEO_FI_AUTO_LUT_NAMES[lut_index % len(VIDEO_FI_AUTO_LUT_NAMES)]]


def _overlay_marker_color_to_uint8(entry: Any, *, kind: str) -> tuple[int, int, int]:
    """Resolve mask/foci overlay color presets to uint8 RGB."""
    default = DEFAULT_VIDEO_MASK_CONTOUR_RGB if kind == "mask" else DEFAULT_VIDEO_FOCI_MARKER_RGB
    if isinstance(entry, (list, tuple)) and len(entry) == 3:
        return (int(entry[0]), int(entry[1]), int(entry[2]))
    text = str(entry or "auto").strip()
    if text.lower() in {"", "auto", "default"}:
        return default
    preset = VIDEO_OVERLAY_COLOR_PRESETS.get(text)
    if preset is None:
        return default
    return (int(preset[0] * 255), int(preset[1] * 255), int(preset[2] * 255))


def _merge_tcyx_to_rgb_uint8(
    stack_tcyx: np.ndarray,
    fi_channel_colors: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """
    Composite C fluorescence channels into RGB using black-to-color LUTs (additive blend).
    stack_tcyx: (T, C, Y, X) numeric array.
    """
    from matplotlib.colors import LinearSegmentedColormap

    t, nc, h, w = stack_tcyx.shape
    out = np.zeros((t, h, w, 3), dtype=np.float32)
    scale = 1.0 / float(max(nc, 1))
    for ti in range(t):
        acc = np.zeros((h, w, 3), dtype=np.float32)
        for ci in range(nc):
            ch = stack_tcyx[ti, ci].astype(np.float32, copy=False)
            p_lo, p_hi = np.percentile(ch, (1.5, 99.5))
            if p_hi <= p_lo:
                p_hi = p_lo + 1e-6
            norm = np.clip((ch - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            preset = None
            if fi_channel_colors is not None and ci < len(fi_channel_colors):
                preset = fi_channel_colors[ci]
            r01, g01, b01 = _overlay_color_entry_to_rgb01(preset, lut_index=ci)
            cmap = LinearSegmentedColormap.from_list(
                f"video_fi_{ci}",
                [(0.0, 0.0, 0.0), (r01, g01, b01)],
            )
            rgb = cmap(norm)[..., :3]
            acc += rgb * scale
        acc = np.clip(acc, 0.0, 1.0)
        out[ti] = (acc * 255.0).astype(np.uint8)
    return out


def _grayscale_stack_to_rgb_uint8(frames_thw: np.ndarray) -> np.ndarray:
    """(T, H, W) -> (T, H, W, 3) uint8."""
    if frames_thw.ndim != 3:
        raise ValueError(f"Expected grayscale stack (T,H,W), got shape {frames_thw.shape}")
    t, h, w = frames_thw.shape
    out = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        f = frames_thw[i]
        if f.dtype != np.uint8:
            f = ((f - f.min()) / (f.max() - f.min() + 1e-10) * 255.0).astype(np.uint8)
        out[i] = np.stack([f, f, f], axis=-1)
    return out


def _normalize_frame_to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    f = frame.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.zeros(frame.shape[:2] + (3,), dtype=np.uint8)
    g = ((f - lo) / (hi - lo) * 255.0).astype(np.uint8)
    if g.ndim == 2:
        return np.stack([g, g, g], axis=-1)
    return g


def _growth_curve_tiff_files_in_folder(folder: Path) -> list[Path]:
    """Sorted TIFF files directly inside a folder."""
    return (
        sorted(folder.glob("*.tif"))
        + sorted(folder.glob("*.tiff"))
        + sorted(folder.glob("*.TIF"))
        + sorted(folder.glob("*.TIFF"))
    )


def _load_growth_curve_folder_thw(folder: Path) -> np.ndarray:
    """Sequence of TIFF files in a folder -> (T, H, W) single-channel stack."""
    import tifffile

    tiff_files = _growth_curve_tiff_files_in_folder(folder)
    if not tiff_files:
        raise ValueError(f"No TIFF files found in folder: {folder}")
    video_frames = []
    for tiff_file in tiff_files:
        frame = tifffile.imread(str(tiff_file))
        if frame.ndim == 2:
            video_frames.append(frame)
        elif frame.ndim == 3:
            video_frames.append(frame[0])
        else:
            video_frames.append(frame.reshape(-1, frame.shape[-2], frame.shape[-1])[0])
    return np.array(video_frames)


def _load_growth_curve_folder_rgb(folder: Path) -> np.ndarray:
    """Sequence of TIFF files in a folder -> (T, H, W, 3) uint8."""
    return _grayscale_stack_to_rgb_uint8(_load_growth_curve_folder_thw(folder))


def _to_thw_single_channel_stack(arr: np.ndarray) -> np.ndarray:
    """Normalize a single-channel TIFF stack to (T, H, W)."""
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        t, a, b, c = arr.shape
        if a <= 8 and a < b and a == 1:
            return arr[:, 0, :, :]
        if c <= 4 and c < b and c == 1:
            return arr[:, :, :, 0]
        raise ValueError(
            "Each channel file must be a single-channel stack (T,H,W). "
            "Use one multi-channel TIFF or split channels into separate files."
        )
    raise ValueError(f"Unsupported TIFF shape for channel file: {arr.shape}")


def _load_growth_curve_multi_file_rgb(
    paths: Sequence[Path],
    overlay_colors: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Several TIFF stacks (one per channel), same T/H/W -> LUT merge -> (T, H, W, 3) uint8."""
    stacks = _load_growth_curve_multi_file_thw_stacks(paths)
    tcyx = np.stack(stacks, axis=1)
    colors = normalize_video_overlay_colors(overlay_colors)
    roles = _assign_growth_video_channel_roles(paths)
    channel_names = [label for _, _, label in roles] if roles else [p.stem for p in paths]
    aligned_presets = _align_fi_color_presets_to_channels(channel_names, colors.get("fi_channels") or [])
    return _merge_tcyx_to_rgb_uint8(tcyx, fi_channel_colors=aligned_presets)


def _load_growth_curve_multi_file_thw_stacks(paths: Sequence[Path]) -> list[np.ndarray]:
    """Load several single-channel TIFF stacks as a list of (T, H, W) arrays."""
    stacks: list[np.ndarray] = []
    for p in paths:
        if p.is_dir():
            stacks.append(_load_growth_curve_folder_thw(p))
            continue
        if not p.is_file():
            raise ValueError(f"Not a file: {p}")
        arr, _ = _read_tiff_series_array_and_axes(p)
        stacks.append(_to_thw_single_channel_stack(arr))
    if not stacks:
        raise ValueError("No channel TIFF stacks loaded.")
    t0 = stacks[0].shape[0]
    h0, w0 = stacks[0].shape[1], stacks[0].shape[2]
    for i, s in enumerate(stacks):
        if s.shape[0] != t0 or s.shape[1] != h0 or s.shape[2] != w0:
            raise ValueError(
                "All channel TIFFs must have the same number of frames and image size.\n"
                f"Expected ({t0}, {h0}, {w0}), got ({s.shape[0]}, {s.shape[1]}, {s.shape[2]}) for {paths[i]}."
            )
    return stacks


_GROWTH_VIDEO_CHANNEL_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_phase60x", "Phase"),
    ("_phase40x", "Phase"),
    ("_phase100x", "Phase"),
    ("_phase", "Phase"),
    ("_Phase60X", "Phase"),
    ("_Phase", "Phase"),
    ("_brightfield", "Phase"),
    ("_bf", "Phase"),
    ("_dic", "Phase"),
    ("_gfp", "GFP"),
    ("_GFP", "GFP"),
    ("_egfp", "GFP"),
    ("_mscarlet", "mScarlet"),
    ("_mScarlet", "mScarlet"),
    ("_dsred", "dsRED"),
    ("_dsRed", "dsRED"),
    ("_mcherry", "mCherry"),
    ("_rfp", "RFP"),
    ("_561", "561nm"),
    ("_orange", "Orange"),
    ("_yfp", "YFP"),
    ("_YFP", "YFP"),
    ("_venus", "Venus"),
    ("_cfp", "CFP"),
    ("_cyan", "Cyan"),
    ("_fi", "FI"),
    ("_FI", "FI"),
    ("_foci", "Foci"),
    ("_focus", "Foci"),
)


def _infer_growth_video_channel_label(path: Path) -> str:
    """Human-readable channel label from a TIFF filename."""
    stem = path.stem
    stem_lower = stem.lower()
    if "foci" in stem_lower or "focus" in stem_lower:
        for token in ("_fi_foci_", "_fi_focus_", "_foci_", "_focus_"):
            if token in stem_lower:
                pos = stem_lower.find(token)
                channel = stem[pos + len(token):].strip("_")
                return f"Foci {channel.replace('_', ' ')}".strip() if channel else "Foci"
        return "Foci"
    if "_fi_" in stem:
        return stem.partition("_fi_")[2].strip("_") or "FI"
    for suffix, label in _GROWTH_VIDEO_CHANNEL_SUFFIXES:
        if stem_lower.endswith(suffix.lower()):
            return label
    if "phase" in stem_lower or stem_lower.endswith("_bf"):
        return "Phase"
    for token in ("gfp", "rfp", "mcherry", "mscarlet", "cy5", "yfp", "cfp"):
        if token in stem_lower:
            return token.upper() if len(token) <= 3 else token.replace("mscarlet", "mScarlet").replace("mcherry", "mCherry")
    return stem


def _infer_growth_video_channel_role(path: Path) -> str:
    """Return ``phase``, ``fi``, or ``unknown`` based on filename tokens."""
    label = _infer_growth_video_channel_label(path)
    label_lower = label.lower()
    if label_lower in {"phase", "bf", "brightfield", "dic", "transmission"}:
        return "phase"
    if "phase" in path.stem.lower() or path.stem.lower().endswith("_bf"):
        return "phase"
    if label_lower in {"fi", "foci"} or label_lower.startswith("foci "):
        return "fi"
    for token in (
        "gfp", "egfp", "rfp", "mcherry", "mscarlet", "dsred", "cy5", "yfp", "cfp",
        "561", "orange", "venus", "cyan", "foci", "_fi",
    ):
        if token in path.stem.lower():
            return "fi"
    return "unknown"


def _assign_growth_video_channel_roles(paths: Sequence[Path]) -> list[tuple[Path, str, str]]:
    """Assign phase/FI roles; fall back to first file as phase when inference is unclear."""
    items: list[tuple[Path, str, str]] = []
    for path in paths:
        role = _infer_growth_video_channel_role(path)
        label = _infer_growth_video_channel_label(path)
        items.append((path, role, label))

    if not items:
        return items

    has_phase = any(role == "phase" for _, role, _ in items)
    if not has_phase:
        path, _, label = items[0]
        phase_label = "Phase" if label.lower() in {"fi", "foci", "unknown"} or label == path.stem else label
        items[0] = (path, "phase", phase_label if phase_label != path.stem else "Phase")

    for idx, (path, role, label) in enumerate(items):
        if role == "unknown":
            if idx == 0 and not has_phase:
                continue
            items[idx] = (path, "fi", label if label != path.stem else f"Ch{idx + 1}")
    return items


def _thw_stack_to_lut_rgb_uint8(
    stack_thw: np.ndarray,
    *,
    lut_index: int = 0,
    color_preset: Any = None,
) -> np.ndarray:
    """Colorize one (T, H, W) stack using the growth-video LUT palette."""
    t, h, w = stack_thw.shape
    lut_index = max(0, int(lut_index))
    if lut_index == 0:
        tcyx = stack_thw[:, np.newaxis, :, :]
    else:
        tcyx = np.zeros((t, lut_index + 1, h, w), dtype=stack_thw.dtype)
        tcyx[:, lut_index, :, :] = stack_thw
    channel_colors = [None] * tcyx.shape[1]
    channel_colors[lut_index] = color_preset
    return _merge_tcyx_to_rgb_uint8(tcyx, fi_channel_colors=channel_colors)


def _order_separate_video_tiles(tiles: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Order tiles as Phase, FI channels, then Merged."""
    phase: list[tuple[str, np.ndarray]] = []
    fi: list[tuple[str, np.ndarray]] = []
    merged: list[tuple[str, np.ndarray]] = []
    for label, frames in tiles:
        label_lower = label.lower()
        if label_lower == "merged":
            merged.append((label, frames))
        elif "phase" in label_lower or label_lower in {"bf", "brightfield"}:
            phase.append((label, frames))
        else:
            fi.append((label, frames))
    if not phase and fi:
        phase.append(fi.pop(0))
        if phase[0][0].lower() not in {"phase", "bf"}:
            phase = [("Phase", phase[0][1])]
    return phase + fi + merged


def _load_growth_curve_separate_tiles(
    paths: Sequence[Path],
    overlay_colors: Optional[dict[str, Any]] = None,
) -> list[tuple[str, np.ndarray]]:
    """Build named RGB tile stacks for separate video-tile layout."""
    paths = [Path(p) for p in paths if str(p).strip()]
    if not paths:
        raise ValueError("No video path(s) provided.")
    colors = normalize_video_overlay_colors(overlay_colors)
    fi_presets = colors.get("fi_channels") or []

    if len(paths) == 1:
        path = paths[0]
        if path.is_dir():
            arr = _load_growth_curve_folder_rgb(path)
            return [("Phase", arr), ("Merged", arr)]
        if not path.is_file():
            raise ValueError(f"Path not found: {path}")
        raw, axes = _read_tiff_series_array_and_axes(path)
        tcyx = _transpose_to_tcyx(raw, axes)
        if tcyx is None:
            tcyx = _heuristic_tchw(raw)
        if tcyx is not None and tcyx.shape[1] >= 2:
            tiles: list[tuple[str, np.ndarray]] = []
            merge_stack = tcyx
            channel_names = _read_video_channel_names_from_metadata(path, int(tcyx.shape[1]))
            aligned_presets = _align_fi_color_presets_to_channels(channel_names, fi_presets)
            for ci in range(tcyx.shape[1]):
                default_label = "Phase" if ci == 0 else f"Ch{ci + 1}"
                label = _display_video_channel_name(channel_names[ci]) if ci < len(channel_names) else default_label
                if ci == 0:
                    rgb = _grayscale_stack_to_rgb_uint8(tcyx[:, ci, :, :])
                else:
                    fi_idx = ci - 1
                    preset = aligned_presets[ci] if ci < len(aligned_presets) else "auto"
                    rgb = _thw_stack_to_lut_rgb_uint8(
                        tcyx[:, ci, :, :],
                        lut_index=fi_idx,
                        color_preset=preset,
                    )
                tiles.append((label, rgb))
            tiles.append(("Merged", _merge_tcyx_to_rgb_uint8(merge_stack, fi_channel_colors=aligned_presets)))
            return _order_separate_video_tiles(tiles)
        rgb = _load_growth_curve_single_file_rgb(path)
        return [("Phase", rgb), ("Merged", rgb)]

    roles = _assign_growth_video_channel_roles(paths)
    stacks_thw = _load_growth_curve_multi_file_thw_stacks([p for p, _, _ in roles])
    tiles = []
    merge_stacks: list[np.ndarray] = []
    fi_index = 0
    for (path, role, label), thw in zip(roles, stacks_thw):
        merge_stacks.append(thw)
        if role == "phase":
            display = "Phase" if label.lower() in {"fi", "foci"} else label
            tiles.append((display, _grayscale_stack_to_rgb_uint8(thw)))
        else:
            display = label if label.lower() not in {"fi", "foci"} else f"Ch{fi_index + 1}"
            preset = fi_presets[fi_index] if fi_index < len(fi_presets) else "auto"
            tiles.append(
                (
                    display,
                    _thw_stack_to_lut_rgb_uint8(thw, lut_index=fi_index, color_preset=preset),
                )
            )
            fi_index += 1
    tcyx = np.stack(merge_stacks, axis=1)
    merge_presets = ["auto"] + list(fi_presets[: max(0, tcyx.shape[1] - 1)])
    tiles.append(("Merged", _merge_tcyx_to_rgb_uint8(tcyx, fi_channel_colors=merge_presets)))
    return _order_separate_video_tiles(tiles)


@dataclass
class GrowthCurveVideoSource:
    """Loaded microscopy input for growth-curve preview/export."""

    fi_layout: str
    num_frames: int
    reference_width: int
    reference_height: int
    merged_frames: Optional[np.ndarray] = None
    tiles: Optional[list[tuple[str, np.ndarray]]] = None


def load_growth_curve_video_source(
    video_paths: Sequence[Path],
    fi_layout: str = DEFAULT_VIDEO_FI_LAYOUT,
    overlay_colors: Optional[dict[str, Any]] = None,
) -> GrowthCurveVideoSource:
    """Load merged video or separate named channel tiles depending on layout mode."""
    layout = fi_layout if fi_layout in (VIDEO_FI_LAYOUT_MERGED, VIDEO_FI_LAYOUT_SEPARATE) else DEFAULT_VIDEO_FI_LAYOUT
    paths = [Path(p) for p in video_paths if str(p).strip()]
    if not paths:
        raise ValueError("No video path(s) provided.")

    if layout == VIDEO_FI_LAYOUT_SEPARATE:
        tiles = _load_growth_curve_separate_tiles(paths, overlay_colors=overlay_colors)
        num_frames = int(tiles[0][1].shape[0])
        ref_h, ref_w = int(tiles[0][1].shape[1]), int(tiles[0][1].shape[2])
        return GrowthCurveVideoSource(
            fi_layout=layout,
            num_frames=num_frames,
            reference_width=ref_w,
            reference_height=ref_h,
            merged_frames=None,
            tiles=tiles,
        )

    merged = load_growth_curve_video_frames_rgb(paths, overlay_colors=overlay_colors)
    _, ref_h, ref_w, _ = merged.shape
    return GrowthCurveVideoSource(
        fi_layout=layout,
        num_frames=int(merged.shape[0]),
        reference_width=int(ref_w),
        reference_height=int(ref_h),
        merged_frames=merged,
        tiles=None,
    )


def _load_growth_curve_single_file_rgb(
    path: Path,
    overlay_colors: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """One TIFF: grayscale time series, RGB (T,H,W,3), or multi-channel (T,C,H,W) with merge."""
    arr, axes = _read_tiff_series_array_and_axes(path)
    tcyx = _transpose_to_tcyx(arr, axes)
    if tcyx is None:
        tcyx = _heuristic_tchw(arr)
    if tcyx is not None and tcyx.shape[1] >= 2:
        colors = normalize_video_overlay_colors(overlay_colors)
        channel_names = _read_video_channel_names_from_metadata(path, int(tcyx.shape[1]))
        aligned_presets = _align_fi_color_presets_to_channels(channel_names, colors.get("fi_channels") or [])
        return _merge_tcyx_to_rgb_uint8(tcyx, fi_channel_colors=aligned_presets)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        t, h, w, _ = arr.shape
        out = np.zeros((t, h, w, 3), dtype=np.uint8)
        for i in range(t):
            out[i] = _normalize_frame_to_uint8_rgb(arr[i])
        return out
    if arr.ndim == 2:
        g = arr
        if g.dtype != np.uint8:
            g = ((g - g.min()) / (g.max() - g.min() + 1e-10) * 255.0).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
        return rgb[np.newaxis, ...]
    if arr.ndim == 3:
        return _grayscale_stack_to_rgb_uint8(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return _grayscale_stack_to_rgb_uint8(arr[:, :, :, 0])
    raise ValueError(f"Unsupported TIFF shape for growth curve video: {arr.shape} (axes={axes!r})")


def load_growth_curve_video_frames_rgb(
    video_paths: Sequence[Path],
    overlay_colors: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """
    Load TIFF input(s) as (T, H, W, 3) uint8 for the growth-curve composer.

    - One folder: sorted *.tif sequence (grayscale -> RGB).
    - One file: grayscale stack, RGB (T,H,W,3), or multi-channel (T,C,H,W) merged with LUTs.
    - Multiple files: one single-channel stack per file, same dimensions; merged with LUTs.
    """
    paths = [Path(p) for p in video_paths if str(p).strip()]
    if not paths:
        raise ValueError("No video path(s) provided.")
    if len(paths) == 1:
        p = paths[0]
        if p.is_dir():
            return _load_growth_curve_folder_rgb(p)
        if p.is_file():
            return _load_growth_curve_single_file_rgb(p, overlay_colors=overlay_colors)
        raise ValueError(f"Path not found: {p}")
    return _load_growth_curve_multi_file_rgb(paths, overlay_colors=overlay_colors)


def list_growth_curve_excel_columns(excel_path: Path) -> list[str]:
    """Numeric Summary-sheet columns usable as the growth-curve Y axis (excluding Frame)."""
    import pandas as pd

    df = pd.read_excel(excel_path, sheet_name="Summary")
    if "Frame" not in df.columns:
        return []
    numeric_cols: list[str] = []
    for col in df.columns:
        if str(col) == "Frame":
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(str(col))
    ordered: list[str] = []
    for pref in PREFERRED_GROWTH_Y_COLUMNS:
        if pref in numeric_cols:
            ordered.append(pref)
    for col in numeric_cols:
        if col not in ordered:
            ordered.append(col)
    return ordered


def load_growth_curve_summary_data(
    excel_path: Path,
    time_interval: float,
    y_column: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Summary sheet columns for growth-curve graph (frames, Y values, time in minutes)."""
    import pandas as pd

    df = pd.read_excel(excel_path, sheet_name="Summary")
    if "Frame" not in df.columns:
        raise ValueError(
            "Excel Summary sheet must contain a 'Frame' column.\n"
            f"Found columns: {', '.join(map(str, df.columns))}"
        )
    available = list_growth_curve_excel_columns(excel_path)
    if not available:
        raise ValueError(
            "Excel Summary sheet has no numeric columns for the growth graph.\n"
            f"Found columns: {', '.join(map(str, df.columns))}"
        )
    y_col = str(y_column or DEFAULT_GROWTH_Y_COLUMN).strip()
    if y_col not in available:
        y_col = DEFAULT_GROWTH_Y_COLUMN if DEFAULT_GROWTH_Y_COLUMN in available else available[0]
    frames = pd.to_numeric(df["Frame"], errors="coerce").values
    y_values = pd.to_numeric(df[y_col], errors="coerce").values
    if np.isnan(frames).all() or np.isnan(y_values).all():
        raise ValueError(f"Column '{y_col}' has no usable numeric values.")
    time_minutes = (frames - 1) * float(time_interval)
    return frames, y_values, time_minutes


def _ensure_growth_curve_video_frame_rgb(video_frame: np.ndarray) -> np.ndarray:
    """Normalize one video frame to (H, W, 3) uint8."""
    import numpy as np

    if video_frame.dtype != np.uint8:
        video_frame = (
            (video_frame - video_frame.min())
            / (video_frame.max() - video_frame.min() + 1e-10)
            * 255
        ).astype(np.uint8)
    if len(video_frame.shape) == 2:
        return np.stack([video_frame] * 3, axis=-1)
    if video_frame.shape[2] == 1:
        return np.repeat(video_frame, 3, axis=2)
    return video_frame


def _load_video_mask_frames(mask_path: Optional[Path]) -> Optional[np.ndarray]:
    """Load a label-mask TIFF as (T, H, W), preserving integer labels."""
    if mask_path is None or not Path(mask_path).exists():
        return None
    import tifffile

    arr = np.asarray(tifffile.imread(str(mask_path)))
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim >= 4:
        return arr.reshape((-1,) + arr.shape[-2:])
    return None


def _mask_contour(mask_2d: np.ndarray) -> np.ndarray:
    """Return a boolean contour mask for labeled cells."""
    labels = np.asarray(mask_2d)
    if labels.size == 0:
        return np.zeros(labels.shape, dtype=bool)
    try:
        from skimage.segmentation import find_boundaries

        return find_boundaries(labels, mode="inner") & (labels > 0)
    except Exception:
        try:
            from scipy.ndimage import binary_erosion

            occupied = labels > 0
            return occupied & ~binary_erosion(occupied)
        except Exception:
            occupied = labels > 0
            edge = np.zeros_like(occupied, dtype=bool)
            edge[1:, :] |= occupied[1:, :] != occupied[:-1, :]
            edge[:, 1:] |= occupied[:, 1:] != occupied[:, :-1]
            return edge & occupied


def _load_foci_points_by_frame(foci_path: Optional[Path]) -> dict[int, list[tuple[int, int]]]:
    """Load simple foci coordinates grouped by zero-based frame index from CSV/XLSX."""
    if foci_path is None or not Path(foci_path).exists():
        return {}
    import pandas as pd

    path = Path(foci_path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        return {}

    lower = {str(c).strip().lower(): c for c in df.columns}

    def _first(names: Sequence[str]) -> Optional[Any]:
        for name in names:
            if name in lower:
                return lower[name]
        return None

    frame_col = _first(("frame", "t", "timepoint", "time point", "frame_idx", "frame_index"))
    x_col = _first(("x", "x_px", "x [px]", "centroid_x", "center_x", "peak_x"))
    y_col = _first(("y", "y_px", "y [px]", "centroid_y", "center_y", "peak_y"))
    if x_col is None or y_col is None:
        return {}

    frame_values = df[frame_col].to_numpy() if frame_col is not None else np.ones(len(df), dtype=int)
    try:
        one_based = bool(len(frame_values) and np.nanmin(frame_values.astype(float)) >= 1)
    except Exception:
        one_based = True
    grouped: dict[int, list[tuple[int, int]]] = {}
    for idx, row in df.iterrows():
        try:
            frame_value = int(row[frame_col]) if frame_col is not None else 1
            frame_idx = frame_value - 1 if one_based else frame_value
            x = int(round(float(row[x_col])))
            y = int(round(float(row[y_col])))
        except Exception:
            continue
        grouped.setdefault(frame_idx, []).append((x, y))
    return grouped


def _draw_video_overlays(
    video_frame: np.ndarray,
    frame_idx: int,
    *,
    mask_frames: Optional[np.ndarray] = None,
    foci_points: Optional[dict[int, list[tuple[int, int]]]] = None,
    show_mask_contours: bool = False,
    show_foci: bool = False,
    mask_color_rgb: Optional[tuple[int, int, int]] = None,
    foci_color_rgb: Optional[tuple[int, int, int]] = None,
) -> np.ndarray:
    """Draw contour-only masks and simple foci rings over real microscopy data."""
    rgb = _ensure_growth_curve_video_frame_rgb(video_frame).copy()
    h, w = rgb.shape[:2]
    if show_mask_contours and mask_frames is not None and len(mask_frames):
        mask = np.asarray(mask_frames[min(frame_idx, len(mask_frames) - 1)])
        if mask.shape[:2] != (h, w):
            try:
                import cv2

                mask = cv2.resize(mask.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                mask = None
        if mask is not None:
            contour = _mask_contour(mask)
            rgb[contour] = np.array(mask_color_rgb or DEFAULT_VIDEO_MASK_CONTOUR_RGB, dtype=np.uint8)

    if show_foci and foci_points:
        points = foci_points.get(frame_idx, [])
        for x, y in points:
            if not (0 <= x < w and 0 <= y < h):
                continue
            yy, xx = np.ogrid[max(0, y - 5) : min(h, y + 6), max(0, x - 5) : min(w, x + 6)]
            dist = (yy - y) ** 2 + (xx - x) ** 2
            ring = (dist >= 9) & (dist <= 25)
            sub = rgb[max(0, y - 5) : min(h, y + 6), max(0, x - 5) : min(w, x + 6)]
            sub[ring] = np.array(foci_color_rgb or DEFAULT_VIDEO_FOCI_MARKER_RGB, dtype=np.uint8)
    return rgb


def apply_video_montage_overlays(
    video_frames: np.ndarray,
    *,
    mask_path: Optional[Path] = None,
    foci_path: Optional[Path] = None,
    show_mask_contours: bool = False,
    show_foci: bool = False,
    overlay_colors: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Return RGB video frames with optional contour/foci overlays drawn on real data."""
    if not show_mask_contours and not show_foci:
        return video_frames
    colors = normalize_video_overlay_colors(overlay_colors)
    mask_rgb = _overlay_marker_color_to_uint8(colors.get("mask_contour"), kind="mask")
    foci_rgb = _overlay_marker_color_to_uint8(colors.get("foci_marker"), kind="foci")
    mask_frames = _load_video_mask_frames(mask_path) if show_mask_contours else None
    foci_points = _load_foci_points_by_frame(foci_path) if show_foci else {}
    return np.stack(
        [
            _draw_video_overlays(
                frame,
                idx,
                mask_frames=mask_frames,
                foci_points=foci_points,
                show_mask_contours=show_mask_contours,
                show_foci=show_foci,
                mask_color_rgb=mask_rgb,
                foci_color_rgb=foci_rgb,
            )
            for idx, frame in enumerate(video_frames)
        ],
        axis=0,
    )


def apply_video_montage_overlays_to_tiles(
    tiles: Sequence[tuple[str, np.ndarray]],
    *,
    overlay_labels: Sequence[str] = VIDEO_FI_OVERLAY_TILE_LABELS,
    mask_path: Optional[Path] = None,
    foci_path: Optional[Path] = None,
    show_mask_contours: bool = False,
    show_foci: bool = False,
    overlay_colors: Optional[dict[str, Any]] = None,
) -> list[tuple[str, np.ndarray]]:
    """Apply mask/foci overlays only to selected named tiles (Phase, Merged by default)."""
    labels = set(overlay_labels)
    if not show_mask_contours and not show_foci:
        return [(label, frames) for label, frames in tiles]
    result: list[tuple[str, np.ndarray]] = []
    for label, frames in tiles:
        if label in labels:
            frames = apply_video_montage_overlays(
                frames,
                mask_path=mask_path,
                foci_path=foci_path,
                show_mask_contours=show_mask_contours,
                show_foci=show_foci,
                overlay_colors=overlay_colors,
            )
        result.append((label, frames))
    return result


def _compose_separate_tiles_frame(
    frame_idx: int,
    tiles: Sequence[tuple[str, np.ndarray]],
    *,
    reference_width: int,
) -> np.ndarray:
    """Horizontally stack named tiles for one frame without changing tile aspect ratios."""
    import cv2

    frame_list = [_ensure_growth_curve_video_frame_rgb(frames[frame_idx]) for _, frames in tiles]
    if not frame_list:
        raise ValueError("No video tiles available for separate-tile layout.")
    target_h = max(1, int(frame_list[0].shape[0]))
    resized: list[np.ndarray] = []
    for frame in frame_list:
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if h <= 0 or w <= 0:
            continue
        if h == target_h:
            resized.append(frame)
            continue
        scale = target_h / float(h)
        target_w = max(1, int(round(w * scale)))
        resized.append(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR))
    if not resized:
        raise ValueError("No video tiles available for separate-tile layout.")
    return np.hstack(resized)


def _render_growth_curve_graph_image(
    frame_idx: int,
    frames: np.ndarray,
    relative_areas: np.ndarray,
    time_minutes: np.ndarray,
    *,
    graph_width: int,
    frame_height: int,
    yaxis_min: Optional[float],
    yaxis_max: Optional[float],
    graph_style: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Render the growth-curve plot for one frame index to (H, W, 3) uint8."""
    import numpy as np
    import cv2
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    y_min = yaxis_min if yaxis_min is not None else max(0.0, float(relative_areas.min()) * 0.9)
    y_max = yaxis_max if yaxis_max is not None else float(relative_areas.max()) * 1.1
    style = dict(DEFAULT_GROWTH_GRAPH_STYLE)
    if graph_style:
        style.update({k: v for k, v in graph_style.items() if v is not None})
    font_weight = str(style.get("font_weight") or "bold")
    show_grid_value = style.get("show_grid", True)
    if isinstance(show_grid_value, str):
        show_grid = show_grid_value.strip().lower() not in {"", "0", "false", "no", "off"}
    else:
        show_grid = bool(show_grid_value)

    current_times = time_minutes[: frame_idx + 1]
    current_areas = relative_areas[: frame_idx + 1]

    fig = Figure(figsize=(graph_width / 100, frame_height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(current_times, current_areas, "k-", linewidth=float(style.get("line_width", 8.0)))
    ax.scatter(
        current_times[-1],
        current_areas[-1],
        color="black",
        s=float(style.get("marker_size", 200.0)),
        zorder=5,
        edgecolors="black",
        linewidths=float(style.get("marker_edge_width", 2.0)),
    )
    ax.set_xlabel(
        str(style.get("x_label") or "Time [min]"),
        fontsize=int(style.get("axis_label_font_size", 36)),
        fontweight=font_weight,
    )
    ax.set_ylabel(
        str(style.get("y_label") or "rel. cell area"),
        fontsize=int(style.get("axis_label_font_size", 36)),
        fontweight=font_weight,
    )
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, float(time_minutes.max()) * 1.05)
    if show_grid:
        ax.grid(
            True,
            alpha=float(style.get("grid_alpha", 0.4)),
            linewidth=float(style.get("grid_line_width", 1.5)),
        )
    else:
        ax.grid(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=int(style.get("tick_font_size", 24)),
        width=float(style.get("tick_stroke_width", 2.0)),
        colors="black",
    )
    ax.tick_params(axis="both", which="minor", labelsize=int(style.get("tick_font_size", 24)), colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(font_weight)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(float(style.get("axis_stroke_width", 3.5)))

    canvas.draw()
    try:
        buf = canvas.buffer_rgba()
        graph_image = np.asarray(buf)
        if graph_image.shape[2] == 4:
            graph_image = graph_image[:, :, :3]
    except AttributeError:
        try:
            graph_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            graph_image = graph_image.reshape(canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            buf = np.array(canvas.renderer.buffer_rgba())
            graph_image = buf[:, :, :3]

    if graph_image.shape[0] != frame_height or graph_image.shape[1] != graph_width:
        graph_image = cv2.resize(graph_image, (graph_width, frame_height), interpolation=cv2.INTER_LINEAR)
    return graph_image


def compose_growth_curve_combined_frame(
    frame_idx: int,
    video_frames: Optional[np.ndarray] = None,
    frames: Optional[np.ndarray] = None,
    relative_areas: Optional[np.ndarray] = None,
    time_minutes: Optional[np.ndarray] = None,
    *,
    tiles: Optional[Sequence[tuple[str, np.ndarray]]] = None,
    fi_layout: str = DEFAULT_VIDEO_FI_LAYOUT,
    reference_width: Optional[int] = None,
    yaxis_min: Optional[float] = None,
    yaxis_max: Optional[float] = None,
    graph_style: Optional[dict[str, Any]] = None,
    graph_width_ratio: float = 0.5,
) -> np.ndarray:
    """Compose one frame: graph + video when growth data exists, otherwise video only."""
    import numpy as np

    if fi_layout == VIDEO_FI_LAYOUT_SEPARATE and tiles:
        video_frame = _compose_separate_tiles_frame(
            frame_idx,
            tiles,
            reference_width=int(reference_width or tiles[0][1].shape[2]),
        )
    else:
        if video_frames is None:
            raise ValueError("Merged video frames are required for merged video layout.")
        video_frame = _ensure_growth_curve_video_frame_rgb(video_frames[frame_idx])
    if frames is None or relative_areas is None or time_minutes is None:
        return video_frame
    frame_height, frame_width = int(video_frame.shape[0]), int(video_frame.shape[1])
    ratio = max(0.2, min(float(graph_width_ratio), 0.8))
    graph_width = max(64, int(round(frame_width * ratio / max(1e-6, 1.0 - ratio))))
    graph_image = _render_growth_curve_graph_image(
        frame_idx,
        frames,
        relative_areas,
        time_minutes,
        graph_width=graph_width,
        frame_height=frame_height,
        yaxis_min=yaxis_min,
        yaxis_max=yaxis_max,
        graph_style=graph_style,
    )
    return np.hstack([graph_image, video_frame])


def build_growth_curve_output_frames(
    video_paths: Sequence[Path],
    excel_path: Optional[Path],
    time_interval: float,
    yaxis_min: Optional[float],
    yaxis_max: Optional[float],
    *,
    mask_path: Optional[Path] = None,
    foci_path: Optional[Path] = None,
    show_mask_contours: bool = False,
    show_foci: bool = False,
    graph_style: Optional[dict[str, Any]] = None,
    graph_width_ratio: float = 0.5,
    y_column: Optional[str] = None,
    fi_layout: str = DEFAULT_VIDEO_FI_LAYOUT,
    overlay_colors: Optional[dict[str, Any]] = None,
    tile_visibility: Optional[dict[str, bool]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Load inputs and build all combined output frames."""
    source = load_growth_curve_video_source(
        video_paths,
        fi_layout=fi_layout,
        overlay_colors=overlay_colors,
    )
    video_frames: Optional[np.ndarray] = None
    tiles: Optional[list[tuple[str, np.ndarray]]] = None

    if source.fi_layout == VIDEO_FI_LAYOUT_SEPARATE and source.tiles:
        tiles = apply_video_montage_overlays_to_tiles(
            source.tiles,
            mask_path=mask_path,
            foci_path=foci_path,
            show_mask_contours=show_mask_contours,
            show_foci=show_foci,
            overlay_colors=overlay_colors,
        )
        if tile_visibility:
            visible_tiles = [
                (label, frames)
                for label, frames in tiles
                if bool(tile_visibility.get(str(label).strip().lower(), True))
            ]
            if visible_tiles:
                tiles = visible_tiles
    else:
        video_frames = apply_video_montage_overlays(
            source.merged_frames,
            mask_path=mask_path,
            foci_path=foci_path,
            show_mask_contours=show_mask_contours,
            show_foci=show_foci,
            overlay_colors=overlay_colors,
        )
    frames: Optional[np.ndarray] = None
    relative_areas: Optional[np.ndarray] = None
    time_minutes: Optional[np.ndarray] = None
    if excel_path is not None and Path(excel_path).exists():
        frames, relative_areas, time_minutes = load_growth_curve_summary_data(
            excel_path,
            time_interval,
            y_column=y_column,
        )
        num_frames = min(source.num_frames, len(frames))
        frames = frames[:num_frames]
        relative_areas = relative_areas[:num_frames]
        time_minutes = time_minutes[:num_frames]
    else:
        num_frames = source.num_frames
    if video_frames is not None:
        video_frames = video_frames[:num_frames]
    if tiles is not None:
        tiles = [(label, stack[:num_frames]) for label, stack in tiles]

    frame_height = int(source.reference_height)
    reference_width = int(source.reference_width)
    if tiles is not None and tiles:
        sample = _compose_separate_tiles_frame(0, tiles, reference_width=reference_width)
        frame_height = int(sample.shape[0])
        frame_width = int(sample.shape[1])
    elif video_frames is not None:
        frame_height = int(video_frames.shape[1])
        frame_width = int(video_frames.shape[2])
    else:
        frame_width = reference_width
    if frames is not None:
        ratio = max(0.2, min(float(graph_width_ratio), 0.8))
        graph_width = max(64, int(round(frame_width * ratio / max(1e-6, 1.0 - ratio))))
        total_width = graph_width + frame_width
    else:
        total_width = frame_width
    output_frames: list[np.ndarray] = []

    for frame_idx in range(num_frames):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled by user")
        if progress_callback is not None and frame_idx % 10 == 0:
            progress = 20 + int((frame_idx / max(num_frames, 1)) * 70)
            progress_callback(progress, f"Processing frame {frame_idx + 1}/{num_frames}...")

        output_frames.append(
            compose_growth_curve_combined_frame(
                frame_idx,
                video_frames=video_frames,
                frames=frames,
                relative_areas=relative_areas,
                time_minutes=time_minutes,
                tiles=tiles,
                fi_layout=source.fi_layout,
                reference_width=reference_width,
                yaxis_min=yaxis_min,
                yaxis_max=yaxis_max,
                graph_style=graph_style,
                graph_width_ratio=graph_width_ratio,
            )
        )

    meta = {
        "num_frames": num_frames,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "total_width": total_width,
        "has_graph": int(frames is not None),
        "fi_layout": source.fi_layout,
    }
    return output_frames, meta


class GrowthCurveVideoWorker(QThread):
    """Worker thread for creating growth curve videos"""
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(int, str)  # value, message
    
    def __init__(
        self,
        video_paths,
        excel_path,
        output_dir,
        time_interval,
        yaxis_min,
        yaxis_max,
        fps,
        formats,
        parent=None,
        mask_path=None,
        foci_path=None,
        show_mask_contours: bool = False,
        show_foci: bool = False,
        graph_style: Optional[dict[str, Any]] = None,
        graph_width_ratio: float = 0.5,
        y_column: Optional[str] = None,
        fi_layout: str = DEFAULT_VIDEO_FI_LAYOUT,
        overlay_colors: Optional[dict[str, Any]] = None,
        tile_visibility: Optional[dict[str, bool]] = None,
    ):
        super().__init__(parent)
        self.setObjectName(type(self).__name__)
        self.video_paths = list(video_paths)
        self.excel_path = excel_path
        self.output_dir = output_dir
        self.time_interval = time_interval
        self.yaxis_min = yaxis_min
        self.yaxis_max = yaxis_max
        self.fps = fps  # Frames per second for video playback
        self.formats = formats  # List of formats to save: ['tif', 'avi', 'gif', 'mp4']
        self.mask_path = mask_path
        self.foci_path = foci_path
        self.show_mask_contours = bool(show_mask_contours)
        self.show_foci = bool(show_foci)
        self.graph_style = dict(graph_style or {})
        self.graph_width_ratio = float(graph_width_ratio)
        self.y_column = y_column
        self.fi_layout = fi_layout
        self.overlay_colors = normalize_video_overlay_colors(overlay_colors)
        self.tile_visibility = dict(tile_visibility or {})
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True
        try:
            self.requestInterruption()
        except Exception:
            pass

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested) or bool(self.isInterruptionRequested())

    def run(self) -> None:
        try:
            import numpy as np
            import tifffile
            import cv2

            self.progress.emit(5, "Loading Excel data...")
            self.progress.emit(10, "Loading video frames...")

            def _progress(value: int, message: str) -> None:
                self.progress.emit(value, message)

            try:
                output_frames, meta = build_growth_curve_output_frames(
                    self.video_paths,
                    Path(self.excel_path) if self.excel_path else None,
                    self.time_interval,
                    self.yaxis_min,
                    self.yaxis_max,
                    mask_path=Path(self.mask_path) if self.mask_path else None,
                    foci_path=Path(self.foci_path) if self.foci_path else None,
                    show_mask_contours=self.show_mask_contours,
                    show_foci=self.show_foci,
                    graph_style=self.graph_style,
                    graph_width_ratio=self.graph_width_ratio,
                    y_column=self.y_column,
                    fi_layout=self.fi_layout,
                    overlay_colors=self.overlay_colors,
                    tile_visibility=self.tile_visibility,
                    progress_callback=_progress,
                    cancel_check=self._is_cancelled,
                )
            except ValueError as e:
                self.finished.emit(False, str(e))
                return
            except RuntimeError as e:
                if "cancel" in str(e).lower():
                    self.finished.emit(False, "Video montage export cancelled.")
                    return
                self.finished.emit(False, str(e))
                return

            num_frames = int(meta["num_frames"])
            frame_height = int(meta["frame_height"])
            total_width = int(meta["total_width"])
            self.progress.emit(20, f"Processing {num_frames} frames...")
            self.progress.emit(95, "Saving output files...")
            
            output_files = []
            output_frames_array = np.array(output_frames)
            
            # Save as multi-page TIFF
            if 'tif' in self.formats:
                try:
                    output_tiff = self.output_dir / f"{self.output_dir.name}.tif"
                    # Ensure uint8 for smaller file size
                    if output_frames_array.dtype != np.uint8:
                        output_frames_array = ((output_frames_array - output_frames_array.min()) / 
                                             (output_frames_array.max() - output_frames_array.min() + 1e-10) * 255).astype(np.uint8)
                    # Use LZW compression for better compression ratio
                    tifffile.imwrite(str(output_tiff), output_frames_array, compression='lzw', photometric='rgb')
                    output_files.append(f"TIFF: {output_tiff.name}")
                except Exception as e:
                    output_files.append(f"TIFF: Error - {str(e)}")
            
            # Save as AVI
            if 'avi' in self.formats:
                try:
                    output_avi = self.output_dir / f"{self.output_dir.name}.avi"
                    # Try multiple codecs in order of preference
                    codecs = ['MJPG', 'XVID', 'H264', 'mp4v']
                    success = False
                    for codec_name in codecs:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec_name)
                            out = cv2.VideoWriter(str(output_avi), fourcc, self.fps, (total_width, frame_height))
                            if out.isOpened():
                                for frame in output_frames:
                                    out.write(frame)
                                out.release()
                                output_files.append(f"AVI: {output_avi.name} (codec: {codec_name})")
                                success = True
                                break
                        except Exception:
                            continue
                    if not success:
                        output_files.append(f"AVI: Error - No compatible codec found")
                except Exception as e:
                    output_files.append(f"AVI: Error - {str(e)}")
            
            # Save as GIF
            if 'gif' in self.formats:
                try:
                    output_gif = self.output_dir / f"{self.output_dir.name}.gif"
                    import imageio
                    from PIL import Image
                    # Convert frames to uint8 if needed and optimize
                    gif_frames = []
                    for frame in output_frames:
                        if frame.dtype != np.uint8:
                            frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255).astype(np.uint8)
                        # Convert to PIL Image for better GIF optimization
                        if len(frame.shape) == 3:
                            pil_frame = Image.fromarray(frame)
                            # Quantize to reduce colors (256 colors for better compression)
                            pil_frame = pil_frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                            gif_frames.append(pil_frame)
                        else:
                            gif_frames.append(frame)
                    # Save with optimization
                    imageio.mimsave(str(output_gif), gif_frames, fps=self.fps, loop=0, subrectangles=True)
                    output_files.append(f"GIF: {output_gif.name}")
                except Exception as e:
                    # Fallback to simple save if PIL optimization fails
                    try:
                        gif_frames = []
                        for frame in output_frames:
                            if frame.dtype != np.uint8:
                                frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255).astype(np.uint8)
                            gif_frames.append(frame)
                        imageio.mimsave(str(output_gif), gif_frames, fps=self.fps, loop=0)
                        output_files.append(f"GIF: {output_gif.name}")
                    except Exception as e2:
                        output_files.append(f"GIF: Error - {str(e2)}")
            
            # Save as MP4
            if 'mp4' in self.formats:
                try:
                    output_mp4 = self.output_dir / f"{self.output_dir.name}.mp4"
                    import imageio
                    # Convert frames to uint8 if needed
                    mp4_frames = []
                    for frame in output_frames:
                        if frame.dtype != np.uint8:
                            frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255).astype(np.uint8)
                        mp4_frames.append(frame)
                    # Try imageio first (uses ffmpeg if available)
                    try:
                        imageio.mimsave(str(output_mp4), mp4_frames, fps=self.fps, codec='libx264', quality=8)
                        output_files.append(f"MP4: {output_mp4.name}")
                    except Exception:
                        # Fallback to opencv
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(output_mp4), fourcc, self.fps, (total_width, frame_height))
                        if out.isOpened():
                            for frame in mp4_frames:
                                out.write(frame)
                            out.release()
                            output_files.append(f"MP4: {output_mp4.name}")
                        else:
                            output_files.append(f"MP4: Error - Could not create MP4 file")
                except Exception as e:
                    output_files.append(f"MP4: Error - {str(e)}")
            
            self.progress.emit(100, "Complete!")
            output_summary = "\n".join(output_files)
            self.finished.emit(True, f"Growth curve video created successfully!\n\n"
                                   f"Output directory: {self.output_dir}\n\n"
                                   f"Files created:\n{output_summary}")
        
        except Exception as e:
            log_exception(get_logger(), "GrowthCurveVideoWorker.run failed")
            import traceback
            error_msg = f"Error creating growth curve video:\n{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)
