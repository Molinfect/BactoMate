from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, QThread, QTimer
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox, QTableWidget, QVBoxLayout, QWidget
from bactomate.gui.analyze_masks_preview_worker import AnalyzeMasksPreviewWorker
from bactomate.gui.filter_apply_progress import FilterApplyProgressDialog
from bactomate.gui.image_utils import scale_for_preview, to_qpixmap
from bactomate.gui.log_config import write_gui_log_line
from bactomate.gui.utils import safe_imread
from bactomate.gui.layout_helpers import PREVIEW_SETTINGS_SCROLL_MIN_HEIGHT, build_settings_card, build_preview_settings_scroll, checkbox_with_control, flex_control, setting_cell, setting_field, style_module_primary_button
from bactomate.gui.widgets import create_info_icon, show_message_dialog
from bactomate.gui.styles import filter_preview_card_stylesheet, preview_table_stylesheet
from bactomate.gui.foci_filter_pipeline import FOCI_BLOB_SIZE_PRESETS, FociPassFailCache, accepted_cell_labels_cache_key, accepted_cell_labels_for_foci, sync_foci_peaks_with_accepted_cells, compute_foci_pass_fail_for_frame, draw_foci_blob_contours_on_rgb, filter_foci_csv_rows, fit_foci_table_columns, foci_blob_max_radius, foci_blob_threshold_frac, foci_pass_fail_cache_key, format_foci_cell_id_display, populate_foci_preview_table
from bactomate.gui.foci_overlay_draw import DEFAULT_FOCI_PREVIEW_RENDER_CAP, draw_peak_rings_on_rgb
from bactomate.gui.filter_preview_utils import cell_filter_settings_signature, foci_channel_key_for_path, foci_channel_settings_signature, resolve_foci_settings_for_source, seed_foci_channel_settings
from bactomate.gui.preview.preview_window import CollapsibleSection, PreviewDialogBase, WorkerGuard, RenderResult
from bactomate.modules.tiff_metadata import resolve_calibration_for_mask
_FOCI_PASS_RGB = (255, 165, 0)
_FOCI_RAW_PEAK_RGB = (255, 200, 80)
_FOCI_FAIL_RGB = (255, 55, 55)
_FOCI_HIGHLIGHT_RGB = (255, 35, 35)
_FOCI_HIGHLIGHT_OUTLINE_RGB = (255, 255, 255)
_FOCI_HIGHLIGHT_RING_RADIUS = 8
_FOCI_HIGHLIGHT_OUTLINE_RADIUS = 11
_FOCI_CHANNEL_RGB: tuple[tuple[int, int, int], ...] = ((255, 165, 0), (0, 220, 255), (255, 80, 220), (120, 255, 120), (255, 230, 80))

def _composite_foci_layer(base_rgb: np.ndarray, foci_layer: np.ndarray, foci_mask: np.ndarray) -> np.ndarray:
    """Paste cached foci pixels onto a freshly LUT-rendered base (cells + background)."""
    out = np.asarray(base_rgb, dtype=np.uint8).copy()
    layer = np.asarray(foci_layer, dtype=np.uint8)
    m = np.asarray(foci_mask, dtype=bool)
    if layer.shape[:2] != out.shape[:2] or m.shape != out.shape[:2]:
        return out
    if np.any(m):
        out[m] = layer[m]
    return out

def _foci_layer_cache_key(frame_idx: int, settings_sig: str, *, draw_blobs: bool, mask_path_key: str | None=None) -> tuple:
    """Foci marker layer cache key (frame, settings, blob mode — not visibility)."""
    return (int(frame_idx), str(mask_path_key or ''), str(settings_sig), bool(draw_blobs))

def _foci_channels_layer_parent_key(frame_idx: int, settings_sig: str, *, draw_blobs: bool, mask_path_key: str | None=None, accepted_cell_labels: set[int] | None=None) -> tuple:
    """Per-channel foci raster cache (visibility toggles merge cached channels only)."""
    return (*_foci_layer_cache_key(frame_idx, settings_sig, draw_blobs=draw_blobs, mask_path_key=mask_path_key), accepted_cell_labels_cache_key(accepted_cell_labels))

def _foci_channel_overlay_vis_key(dlg) -> tuple[bool, ...]:
    """Per-channel overlay checkbox state for cache invalidation."""
    nav = getattr(dlg, '_nav_ctrl', None)
    if nav is not None:
        vis = nav.foci_channel_overlay_visible()
        if vis:
            return tuple((bool(x) for x in vis))
    return ()

def _foci_hover_cache_key(frame_idx: int, settings_sig: str, *, draw_blobs: bool, mask_shape: tuple[int, int], display_shape: tuple[int, int], mask_path_key: str | None=None, channel_overlay_vis: tuple[bool, ...] | None=None, accepted_cell_labels: set[int] | None=None) -> tuple:
    return (int(frame_idx), str(mask_path_key or ''), str(settings_sig), bool(draw_blobs), int(mask_shape[0]), int(mask_shape[1]), int(display_shape[0]), int(display_shape[1]), tuple(channel_overlay_vis) if channel_overlay_vis is not None else None, accepted_cell_labels_cache_key(accepted_cell_labels))

def _cached_foci_hover_data(dlg, *, cache_key: tuple, pass_peaks: list[dict], fail_peaks: list[dict], mask_shape: tuple[int, int], display_shape: tuple[int, int], settings: dict[str, Any], mask_2d: Optional[np.ndarray]=None) -> list[tuple[float, float, str, float]] | None:
    """Return cached hover tooltips for the current frame, or build and store them."""
    if not pass_peaks and (not fail_peaks):
        return None
    cache = getattr(dlg, '_foci_hover_cache', None)
    if isinstance(cache, dict) and cache.get('key') == cache_key:
        data = cache.get('data')
        if data is not None:
            return data
    try:
        hover_data = build_foci_hover_data(pass_peaks, fail_peaks, mask_shape, display_shape, dict(settings or {}), mask_2d=mask_2d)
    except Exception:
        return None
    try:
        dlg._foci_hover_cache = {'key': cache_key, 'data': hover_data}
    except Exception:
        pass
    return hover_data

def _label_set_key(cells: set[int] | frozenset[int]) -> frozenset[int]:
    try:
        return frozenset((int(x) for x in cells))
    except Exception:
        return frozenset()

def _base_layer_cache_key(frame_idx: int, source_key: tuple, mask_path_key: str | None, vmin: float | None, vmax: float | None, accepted: set[int], rejected: set[int]) -> tuple:
    """Cache key for background ± cell overlay (independent of foci overlay visibility)."""
    return (int(frame_idx), tuple(source_key), str(mask_path_key or ''), float(vmin) if vmin is not None else None, float(vmax) if vmax is not None else None, _label_set_key(accepted), _label_set_key(rejected))

def _format_foci_metric_value(peak: dict, key: str, *, alt_key: str | None=None) -> str:
    val = peak.get(key)
    if (val is None or val == '') and alt_key:
        val = peak.get(alt_key)
    if val is None or val == '':
        return '—'
    if key == 'probability':
        try:
            return f'{float(val):.3f}'
        except (TypeError, ValueError):
            return str(val)
    if key == 'cell_id':
        try:
            cid = int(float(val))
            return str(cid) if cid > 0 else '—'
        except (TypeError, ValueError):
            return str(val)
    if key == 'area_px':
        try:
            return str(int(float(val)))
        except (TypeError, ValueError):
            return str(val)
    if key in ('intensity', 'mean_intensity', 'rel_intensity'):
        try:
            return f'{float(val):.2f}'
        except (TypeError, ValueError):
            return str(val)
    return str(val)

def _format_foci_peak_tooltip(peak: dict, *, status: str, mask_2d: Optional[np.ndarray]=None) -> str:
    """Multi-line tooltip for hover on a focus in the filter preview."""
    try:
        y = int(float(peak.get('y', 0) or 0))
        x = int(float(peak.get('x', 0) or 0))
    except (TypeError, ValueError):
        (y, x) = (0, 0)
    is_fail = str(status).lower() == 'fail'
    ch = str(peak.get('_foci_channel') or '').strip()
    ch_prefix = f'{ch} — ' if ch else ''
    title = f'{ch_prefix}Focus — FAIL' if is_fail else f'{ch_prefix}Focus — pass'
    metric_lines = [('Intensity', 'intensity', None), ('Probability', 'probability', None), ('Area (px)', 'area_px', 'size_px'), ('Mean intens.', 'mean_intensity', None), ('Rel. intens.', 'rel_intensity', None), ('Cell ID', 'cell_id', None)]
    reasons = peak.get('fail_reasons')
    if not isinstance(reasons, list):
        reasons = []
    if is_fail and reasons:
        parts = [f'<b>{title}</b>', f'Position: y={y}, x={x}']
        for (label, key, alt) in metric_lines:
            if key == 'cell_id':
                val_txt = format_foci_cell_id_display(peak, mask_2d)
            else:
                val_txt = _format_foci_metric_value(peak, key, alt_key=alt)
            parts.append(f'{label}: {val_txt}')
        parts.append('<br/><b>Failed filters:</b>')
        for reason in reasons:
            text = str(reason).replace('&', '&amp;').replace('<', '&lt;')
            parts.append(f'<span style="color:#ff6666">• {text}</span>')
        return '<br/>'.join(parts)
    lines = [title, f'Position: y={y}, x={x}']
    for (label, key, alt) in metric_lines:
        if key == 'cell_id':
            val_txt = format_foci_cell_id_display(peak, mask_2d)
        else:
            val_txt = _format_foci_metric_value(peak, key, alt_key=alt)
        lines.append(f'{label}: {val_txt}')
    return '\n'.join(lines)

def _merge_foci_peak_lists(peak_lists: list[list[dict]]) -> list[dict]:
    """Merge peak dicts from multiple channels; first occurrence wins per (y, x)."""
    out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for peaks in peak_lists:
        for peak in peaks or []:
            try:
                key = (int(float(peak.get('y', 0) or 0)), int(float(peak.get('x', 0) or 0)))
            except (TypeError, ValueError):
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(peak))
    return out

def collect_visible_foci_peaks_for_hover(*, overlay_sources: list[Path], channel_visible: Callable[[int, Path], bool], channel_peaks: Callable[[str, Path], tuple[list[dict], list[dict]]]) -> tuple[list[dict], list[dict]]:
    """Collect pass/fail peaks for hover (one entry per channel; no cross-channel dedupe)."""
    hover_pass: list[dict] = []
    hover_fail: list[dict] = []
    for (si, fp) in enumerate(overlay_sources):
        if not channel_visible(si, fp):
            continue
        ch_key = foci_channel_key_for_path(fp)
        (ch_pass, ch_fail) = channel_peaks(ch_key, fp)
        for peak in ch_pass or []:
            row = dict(peak)
            row['_foci_channel'] = ch_key
            hover_pass.append(row)
        for peak in ch_fail or []:
            row = dict(peak)
            row['_foci_channel'] = ch_key
            hover_fail.append(row)
    return (hover_pass, hover_fail)

def build_foci_hover_data(pass_peaks: list[dict], fail_peaks: list[dict], mask_shape: tuple[int, int], display_shape: tuple[int, int], settings: dict[str, Any], *, mask_2d: Optional[np.ndarray]=None) -> list[tuple[float, float, str, float]]:
    """Build ImageCanvas hover entries in scaled preview coordinates."""
    (h0, w0) = (int(mask_shape[0]), int(mask_shape[1]))
    (h1, w1) = (int(display_shape[0]), int(display_shape[1]))
    sx = w1 / w0 if w0 else 1.0
    sy = h1 / h0 if h0 else 1.0
    scale = max(sx, sy, 1e-06)
    max_r_setting = float(foci_blob_max_radius(settings))
    grouped: dict[tuple[int, int], list[tuple[dict[str, Any], str]]] = {}
    for (peaks, status) in ((pass_peaks, 'pass'), (fail_peaks, 'fail')):
        for peak in peaks or []:
            try:
                yk = int(float(peak.get('y', 0) or 0))
                xk = int(float(peak.get('x', 0) or 0))
            except (TypeError, ValueError):
                continue
            grouped.setdefault((yk, xk), []).append((dict(peak), status))
    out: list[tuple[float, float, str, float]] = []
    for ((yk, xk), items) in grouped.items():
        dx = float(xk) * sx
        dy = float(yk) * sy
        r_blob = 8.0
        for (peak, _status) in items:
            try:
                area = float(peak.get('area_px', peak.get('size_px', 4)) or 4)
                area = max(1.0, area)
                r_blob = max(r_blob, max(8.0, min(max_r_setting * scale * 1.25, (area / 3.14159265) ** 0.5 * scale * 1.35)))
            except (TypeError, ValueError):
                r_blob = max(r_blob, max(8.0, max_r_setting * scale))
        tips = [_format_foci_peak_tooltip(p, status=s, mask_2d=mask_2d) for (p, s) in items]
        tip = tips[0] if len(tips) == 1 else '<br/><br/>'.join(tips)
        out.append((dx, dy, tip, float(r_blob)))
    return out

def _apply_foci_highlight_contour(rgb: np.ndarray, mask_2d: np.ndarray, plane: Optional[np.ndarray], yx: tuple[int, int], settings: dict[str, Any]) -> None:
    """Draw a prominent red ring on one focus (table double-click / selection)."""
    from bactomate.modules.foci_utils import get_focus_blob_mask
    from skimage.segmentation import find_boundaries
    try:
        (hy, hx) = (int(yx[0]), int(yx[1]))
    except (TypeError, ValueError):
        return
    mask_shape = tuple(np.asarray(mask_2d).shape[:2])
    peak = [{'y': hy, 'x': hx}]
    draw_peak_rings_on_rgb(rgb, peak, _FOCI_HIGHLIGHT_OUTLINE_RGB, mask_shape=mask_shape, radius=_FOCI_HIGHLIGHT_OUTLINE_RADIUS, inner_gap=1)
    draw_peak_rings_on_rgb(rgb, peak, _FOCI_HIGHLIGHT_RGB, mask_shape=mask_shape, radius=_FOCI_HIGHLIGHT_RING_RADIUS, inner_gap=2)
    if plane is None or not np.asarray(plane).size:
        return
    (h0, w0) = (int(mask_2d.shape[0]), int(mask_2d.shape[1]))
    (h1, w1) = (int(rgb.shape[0]), int(rgb.shape[1]))
    sx = w1 / w0 if w0 else 1.0
    sy = h1 / h0 if h0 else 1.0
    blob = get_focus_blob_mask(np.asarray(plane, dtype=np.float64), hy, hx, threshold_frac=foci_blob_threshold_frac(settings), max_radius=foci_blob_max_radius(settings), labeled_mask=np.asarray(mask_2d, dtype=np.int32))
    if blob is None or not np.any(blob):
        return
    (ys, xs) = np.nonzero(blob)
    y2 = np.clip(np.round(ys.astype(np.float64) * sy).astype(np.int32), 0, h1 - 1)
    x2 = np.clip(np.round(xs.astype(np.float64) * sx).astype(np.int32), 0, w1 - 1)
    disp_mask = np.zeros((h1, w1), dtype=bool)
    disp_mask[y2, x2] = True
    if not np.any(disp_mask):
        return
    try:
        edge = find_boundaries(disp_mask, mode='outer')
        if np.any(edge):
            rgb[edge, 0] = int(_FOCI_HIGHLIGHT_RGB[0])
            rgb[edge, 1] = int(_FOCI_HIGHLIGHT_RGB[1])
            rgb[edge, 2] = int(_FOCI_HIGHLIGHT_RGB[2])
    except Exception:
        pass

def _load_preview_plane(file_path: Path, frame_idx: int, cache: dict) -> Optional[np.ndarray]:
    """Load one 2D preview plane; prefer single-frame TIFF read over full-stack load."""
    from bactomate.modules.image_stack_utils import coerce_plane_to_2d, coerce_stack_to_tyx
    try:
        rp = str(file_path.resolve())
    except Exception:
        rp = str(file_path)
    plane_key = (rp, int(frame_idx))
    if plane_key in cache:
        return cache.get(plane_key)
    fi = max(0, int(frame_idx))
    if file_path.suffix.lower() in ('.tif', '.tiff'):
        try:
            import tifffile
            raw = tifffile.imread(str(file_path), key=fi)
            plane = coerce_plane_to_2d(np.asarray(raw))
            if plane is not None and getattr(plane, 'size', 0) > 0:
                cache[plane_key] = np.asarray(plane)
                return cache[plane_key]
        except Exception:
            pass
    stack = cache.get(rp)
    if stack is None:
        try:
            loaded = safe_imread(str(file_path))
            stack = coerce_stack_to_tyx(np.asarray(loaded))
        except Exception:
            stack = None
        if stack is not None and getattr(stack, 'size', 0) > 0:
            cache[rp] = stack
    if stack is None or getattr(stack, 'size', 0) == 0:
        return None
    fi = max(0, min(fi, int(stack.shape[0]) - 1))
    try:
        plane = coerce_plane_to_2d(stack[fi])
    except Exception:
        return None
    if plane is None or getattr(plane, 'size', 0) == 0:
        return None
    cache[plane_key] = np.asarray(plane)
    return cache[plane_key]

def is_filter_apply_cache_hit(last: Optional[dict], frame_idx: int, *, mask_path_key: Optional[str]=None) -> bool:
    """Whether ``_last_preview`` holds an Apply result for this mask frame.

    Independent of the background View (phase vs foci channel): changing only
    the displayed image must not revert cells/foci to the unfiltered baseline.
    """
    if not last:
        return False
    if 'mask_2d' not in last:
        return False
    if mask_path_key is not None:
        cached_mp = last.get('mask_path_key')
        if cached_mp and str(cached_mp) != str(mask_path_key):
            return False
    try:
        return int(last.get('frame_idx', -1)) == int(frame_idx)
    except Exception:
        return False

def is_preview_cache_hit(last: Optional[dict], frame_idx: int, source_key, *, mask_path_key: Optional[str]=None) -> bool:
    """Apply cache hit *and* the cached background View matches (legacy helper)."""
    if not is_filter_apply_cache_hit(last, frame_idx, mask_path_key=mask_path_key):
        return False
    try:
        cached_src = tuple(last.get('source_key', ('none', -1)))
        return cached_src == tuple(source_key)
    except Exception:
        return False

def _preview_mask_path_key(p: Path) -> str:
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)

def _invalidate_filter_preview_overlay_cache(dlg) -> None:
    try:
        dlg._last_preview = {}
        dlg._foci_pass_fail_cache = {}
        dlg._foci_layer_cache = {}
        dlg._foci_channels_layer_cache = {}
        dlg._preview_base_cache = {}
        dlg._preview_overlay_base = None
        dlg._foci_hover_cache = {}
        dlg._foci_composited_cache = {}
    except Exception:
        pass

def _clear_preview_base_caches(dlg) -> None:
    """Drop background/cell compositing only (View change); keep foci layer cache."""
    try:
        dlg._preview_base_cache = {}
        dlg._preview_overlay_base = None
        dlg._foci_composited_cache = {}
    except Exception:
        pass

def _clear_preview_layer_caches(dlg) -> None:
    """Drop all composited layer caches (frame nav / Apply); keep CSV and pass-fail caches."""
    try:
        dlg._foci_layer_cache = {}
        dlg._foci_channels_layer_cache = {}
        dlg._preview_base_cache = {}
        dlg._preview_overlay_base = None
        dlg._foci_hover_cache = {}
        dlg._foci_composited_cache = {}
    except Exception:
        pass

def _effective_show_cells(dlg, state) -> bool:
    """Read cell-overlay visibility from nav controls (source of truth) with state fallback."""
    nav = getattr(dlg, '_nav_ctrl', None)
    if nav is not None:
        try:
            return bool(nav.cell_overlay_visible())
        except Exception:
            pass
    return bool(getattr(state, 'extra', {}).get('show_cells', True))

def _effective_show_foci(dlg, state, *, foci_overlay_available: bool) -> bool:
    """Read foci-overlay visibility from nav controls with state fallback."""
    nav = getattr(dlg, '_nav_ctrl', None)
    if nav is not None:
        try:
            return bool(nav.foci_overlay_visible())
        except Exception:
            pass
    return bool(getattr(state, 'extra', {}).get('show_foci', bool(foci_overlay_available)))

def _source_key_from_state(state, fallback) -> tuple:
    sk = getattr(state, 'source_key', None)
    if isinstance(sk, tuple) and len(sk) >= 2:
        return tuple(sk)
    try:
        return tuple(fallback())
    except Exception:
        return ('none', -1)

def _peaks_from_csv_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows or []:
        try:
            peak = dict(r)
            peak['y'] = float(r.get('y', 0) or 0)
            peak['x'] = float(r.get('x', 0) or 0)
            out.append(peak)
        except Exception:
            continue
    return out

def _safe_release_preview_worker(owner, attr: str='_preview_worker', wait_ms: int=10000) -> None:
    w = getattr(owner, attr, None)
    thread = getattr(owner, '_preview_thread', None)
    try:
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(wait_ms):
                return
    except Exception:
        return
    try:
        if w is not None:
            w.deleteLater()
    except Exception:
        pass
    try:
        if thread is not None:
            thread.deleteLater()
    except Exception:
        pass
    try:
        setattr(owner, attr, None)
    except Exception:
        pass
    try:
        owner._preview_thread = None
    except Exception:
        pass

def _segmentation_tab_host(tab) -> Optional[object]:
    host = tab
    while host is not None:
        if host.__class__.__name__ == 'SegmentationTab':
            return host
        try:
            host = host.parent()
        except Exception:
            return None
    return None

def show_filter_preview(tab, path_text: str, *, masks_preloaded: Optional[np.ndarray]=None, mask_path: Optional[Path]=None, foci_stacks_preloaded: Optional[dict]=None) -> None:
    """
    Section 2 preview (Preview filters): show accepted vs rejected cells for one mask file.

    In the Segmentation module shell, embeds in the right-hand preview pane (no popup).
    This preview is intentionally filtering-only (no re-detection):
    - foci markers come from existing foci CSVs (if present)
    """
    raw = (path_text or '').strip()
    if not raw:
        return
    p = Path(raw)
    if not p.exists():
        show_message_dialog(tab, 'Not Found', 'The selected file or folder does not exist.', kind='warning')
        return
    if mask_path is None:
        from bactomate.gui.filter_preview_utils import pick_mask_path_for_preview
        mask_path = pick_mask_path_for_preview(tab, raw)
    else:
        mask_path = Path(mask_path)
    try:
        tab._filter_preview_mask_path = Path(mask_path) if mask_path is not None else None
    except Exception:
        pass
    if mask_path is None or not mask_path.exists():
        show_message_dialog(tab, 'No Masks Found', 'No mask files were found in the selected path.', kind='warning')
        return
    if masks_preloaded is not None:
        try:
            masks = np.asarray(masks_preloaded)
            if masks.ndim == 2:
                masks = masks[np.newaxis, :, :]
        except Exception as e:
            show_message_dialog(tab, 'Load Error', f'Could not load mask:\n{e}', kind='warning')
            return
    else:
        try:
            masks = np.asarray(safe_imread(str(mask_path)))
            if masks.ndim == 2:
                masks = masks[np.newaxis, :, :]
        except Exception as e:
            show_message_dialog(tab, 'Load Error', f'Could not load mask:\n{e}', kind='warning')
            return
    seg_dir = mask_path.parent
    if not str(seg_dir.name).startswith('_segmentation'):
        cand = mask_path.parent / '_segmentation'
        if cand.exists() and cand.is_dir():
            seg_dir = cand
    num_frames = int(masks.shape[0])

    def _list_background_paths() -> list[Path]:
        """
        Prefer original FI/foci stacks that correspond to this mask (often produced by File Conversion),
        falling back to TIFFs in the segmentation folder if needed.
        """
        from bactomate.gui.filter_preview_utils import file_identity_key
        out: list[Path] = []
        seen: set[str] = set()

        def _append(fp: Path) -> None:
            if not fp.is_file() or fp.suffix.lower() not in ('.tif', '.tiff'):
                return
            key = file_identity_key(fp)
            if key in seen:
                return
            seen.add(key)
            out.append(fp)
        try:
            cand = tab._preview_fi_path_candidates(mask_path)
        except Exception:
            cand = []
        for fp in cand or []:
            _append(Path(fp))
        if not out:
            folder = seg_dir
            for pat in ('*.tif', '*.tiff'):
                for fp in folder.glob(pat):
                    name = fp.name.lower()
                    if '_analysis' in name or '_filter_visualization' in name:
                        continue
                    if name.endswith(('_masks.tif', '_masks.tiff', '_mask_filtered.tif', '_mask_filtered.tiff')):
                        continue
                    if '_masks' in name or '_mask_filtered' in name:
                        continue
                    _append(fp)
        out.sort(key=lambda x: x.name.lower())
        return out
    from bactomate.gui.filter_preview_utils import collect_background_paths_for_mask, default_background_index, enrich_background_paths_for_mask, filter_paths_for_mask, preview_view_label
    bg_paths = collect_background_paths_for_mask(tab, mask_path)
    if not bg_paths:
        bg_paths = filter_paths_for_mask(tab, mask_path, _list_background_paths())
    bg_paths = enrich_background_paths_for_mask(tab, mask_path, bg_paths)
    from bactomate.gui.filter_preview_utils import short_foci_stack_label
    foci_sources = [Path(fp) for fp in bg_paths if '_foci' in fp.name.lower() or 'fi_foci' in fp.name.lower()]
    has_foci_image_files = len(foci_sources) > 0
    try:
        if hasattr(tab, 'ensure_foci_channel_settings_for_sources'):
            tab.ensure_foci_channel_settings_for_sources(foci_sources)
        else:
            fs_seed = tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}
            seed_foci_channel_settings(dict(fs_seed), foci_sources)
            if hasattr(tab, 'set_filter_settings'):
                tab.set_filter_settings(fs_seed)
    except Exception:
        pass
    fs0 = tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}
    if foci_sources:
        try:
            fs_foci0 = tab.get_foci_channel_settings(foci_channel_key_for_path(foci_sources[0]))
        except Exception:
            fs_foci0 = resolve_foci_settings_for_source(dict(fs0), foci_sources[0])
    else:
        fs_foci0 = resolve_foci_settings_for_source(dict(fs0), None)
    settings_host = QWidget()
    settings_outer = QVBoxLayout(settings_host)
    settings_outer.setContentsMargins(0, 0, 0, 0)
    settings_outer.setSpacing(6)

    def _mk_spin(*, decimals: int, mn: float, mx: float, step: float, value: float, width: int=90) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setDecimals(int(decimals))
        w.setRange(float(mn), float(mx))
        w.setSingleStep(float(step))
        w.setValue(float(value))
        flex_control(w, int(width), combo_style=False)
        w.setKeyboardTracking(False)
        return w
    w_min_area = _mk_spin(decimals=2, mn=0.0, mx=10000.0, step=0.1, value=float(fs0.get('min_area_um2') or 0.0))
    w_max_area = _mk_spin(decimals=2, mn=0.0, mx=10000.0, step=0.1, value=float(fs0.get('max_area_um2') or 0.0))
    w_min_w = _mk_spin(decimals=2, mn=0.0, mx=100.0, step=0.1, value=float(fs0.get('min_width_um') or 0.0))
    w_max_w = _mk_spin(decimals=2, mn=0.0, mx=100.0, step=0.1, value=float(fs0.get('max_width_um') or 0.0))
    w_min_len = _mk_spin(decimals=2, mn=0.0, mx=500.0, step=0.1, value=float(fs0.get('min_length_um') or 0.0))
    w_max_len = _mk_spin(decimals=2, mn=0.0, mx=500.0, step=0.1, value=float(fs0.get('max_length_um') or 0.0))
    w_pixel = _mk_spin(decimals=3, mn=0.001, mx=10.0, step=0.001, value=float(fs0.get('pixel_size', 0.108) or 0.108))
    w_min_ar = _mk_spin(decimals=2, mn=1.0, mx=20.0, step=0.1, value=float(fs0.get('min_aspect_ratio', 1.5) or 1.5))
    w_max_ar = _mk_spin(decimals=2, mn=1.0, mx=50.0, step=0.5, value=float(fs0.get('max_aspect_ratio', 10.0) or 10.0))
    w_min_ecc = _mk_spin(decimals=2, mn=0.0, mx=1.0, step=0.05, value=float(fs0.get('min_eccentricity', 0.5) or 0.5))
    w_min_sol = _mk_spin(decimals=2, mn=0.0, mx=1.0, step=0.05, value=float(fs0.get('min_solidity', 0.7) or 0.7))
    w_excl_border = QCheckBox('Exclude cells at borders')
    w_excl_border.setChecked(bool(fs0.get('exclude_border', False)))
    cell_group = build_settings_card('Cell filtering', object_name='FilterCellGroup', parent=settings_host, min_cell_width=220, label_min_width=118, control_max_width=92, max_columns=4, h_spacing=18, stylesheet=filter_preview_card_stylesheet('FilterCellGroup'), fields=[setting_field('Min area (µm²)', w_min_area, min_width=72, max_width=92), setting_field('Max area (µm²)', w_max_area, min_width=72, max_width=92), setting_field('Pixel size (µm)', w_pixel, min_width=72, max_width=92), setting_cell(w_excl_border), setting_field('Min width (µm)', w_min_w, min_width=72, max_width=92), setting_field('Max width (µm)', w_max_w, min_width=72, max_width=92), setting_field('Min aspect', w_min_ar, min_width=72, max_width=92), setting_field('Max aspect', w_max_ar, min_width=72, max_width=92), setting_field('Min length (µm)', w_min_len, min_width=72, max_width=92), setting_field('Max length (µm)', w_max_len, min_width=72, max_width=92), setting_field('Min eccent.', w_min_ecc, min_width=72, max_width=92), setting_field('Min solid.', w_min_sol, min_width=72, max_width=92)])
    cell_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    w_prob_en = QCheckBox('Probability ≥')
    w_prob_en.setToolTip('Minimum Spotiflow probability.')
    w_prob_en.setChecked(bool(fs_foci0.get('foci_prob_enabled', False)))
    w_prob_thr = _mk_spin(decimals=3, mn=0.0, mx=1.0, step=0.05, value=float(fs_foci0.get('foci_prob_thresh', 0.5) or 0.5))
    w_minint_en = QCheckBox('Min intensity ≥')
    w_minint_en.setToolTip('Minimum focus intensity.')
    w_minint_en.setChecked(bool(fs_foci0.get('foci_min_int_enabled', False)))
    w_minint = _mk_spin(decimals=0, mn=0.0, mx=100000.0, step=100.0, value=float(fs_foci0.get('foci_min_int', 0.0) or 0.0), width=90)
    w_minarea_en = QCheckBox('Min area px ≥')
    w_minarea_en.setToolTip('Minimum grown focus area in pixels.')
    w_minarea_en.setChecked(bool(fs_foci0.get('foci_min_area_px_enabled', False)))
    w_minarea = QSpinBox()
    w_minarea.setRange(1, 100000)
    w_minarea.setValue(max(1, int(fs_foci0.get('foci_min_area_px', 1) or 1)))
    flex_control(w_minarea, 64, combo_style=False, max_width=96)
    w_ratio_en = QCheckBox('Per-cell ratio >')
    w_ratio_en.setChecked(bool(fs_foci0.get('foci_per_cell_ratio_enabled', False)))
    w_ratio = _mk_spin(decimals=2, mn=0.5, mx=100.0, step=0.1, value=float(fs_foci0.get('foci_per_cell_ratio', 2.0) or 2.0))
    w_ratio_en.setToolTip('Focus mean / rest-of-cell mean')
    _leg_out = bool(fs_foci0.get('foci_exclude_outside_cells', False))
    w_foci_excl_outside_clear = QCheckBox('Exclude fully outside')
    w_foci_excl_outside_clear.setChecked(bool(fs_foci0.get('foci_exclude_outside_clear', _leg_out)))
    w_foci_excl_outside_clear.setToolTip('Exclude foci whose peak and grown region are fully outside cells.')
    w_foci_excl_outside_touch = QCheckBox('Exclude edge/outside')
    w_foci_excl_outside_touch.setChecked(bool(fs_foci0.get('foci_exclude_outside_touching', _leg_out)))
    w_foci_excl_outside_touch.setToolTip('Peak center is in background but within a few pixels of a cell boundary.')
    w_show_foci_table = QCheckBox('Show foci table')
    w_show_foci_table.setChecked(False)
    w_blob_size = QComboBox()
    blob_frac0 = foci_blob_threshold_frac(fs_foci0)
    blob_sel = 0
    for (i, (lbl, frac)) in enumerate(FOCI_BLOB_SIZE_PRESETS):
        w_blob_size.addItem(lbl, float(frac))
        if abs(float(frac) - blob_frac0) < 1e-06:
            blob_sel = i
    w_blob_size.setCurrentIndex(blob_sel)
    flex_control(w_blob_size, 72, max_width=140)
    w_blob_radius = QSpinBox()
    w_blob_radius.setRange(1, 50)
    w_blob_radius.setValue(int(fs_foci0.get('foci_blob_max_radius', 5) or 5))
    flex_control(w_blob_radius, 52, combo_style=False, max_width=84)
    _foci_size_info = 'Foci size sets the intensity threshold used to grow each peak into a region.\n\nPixels are included if intensity ≥ (selected % of peak intensity at the focus center). The connected region is used for area, mean intensity, and per-cell ratio filters.\n\n• Higher % (e.g. 90%) — smaller, tighter focus\n• Lower % (e.g. 50%) — larger region, includes dimmer pixels near the peak'
    _foci_radius_info = 'Maximum search radius limits how far from each peak the focus region can grow.\n\nGrowth is computed only within ±N pixels around the peak (square window). Bright pixels beyond this distance are ignored even if they would otherwise pass the foci size threshold.\n\nIncrease for large or diffuse foci; decrease to keep measurements local to the peak.'
    w_blob_size.setToolTip(_foci_size_info)
    w_blob_radius.setToolTip(_foci_radius_info)
    foci_channel_badge = QLabel('')
    foci_channel_badge.setObjectName('FociChannelBadge')
    foci_channel_badge.setStyleSheet('#FociChannelBadge { color: #8fbfff; background-color: #202733; border: 1px solid #303844; border-radius: 6px; padding: 4px 8px; font-weight: 600;}')
    foci_channel_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    foci_channel_badge.setVisible(False)
    foci_group = build_settings_card('Foci filtering', object_name='FilterFociGroup', parent=settings_host, min_cell_width=160, label_min_width=84, control_max_width=104, max_columns=5, h_spacing=18, stylesheet=filter_preview_card_stylesheet('FilterFociGroup'), fields=[setting_cell(foci_channel_badge), setting_cell(checkbox_with_control(w_prob_en, w_prob_thr, parent=settings_host)), setting_cell(checkbox_with_control(w_minint_en, w_minint, parent=settings_host)), setting_cell(w_foci_excl_outside_clear), setting_cell(w_foci_excl_outside_touch), setting_field('Foci size', w_blob_size, min_width=72, max_width=112), setting_cell(checkbox_with_control(w_minarea_en, w_minarea, parent=settings_host)), setting_cell(checkbox_with_control(w_ratio_en, w_ratio, parent=settings_host)), setting_cell(w_show_foci_table), setting_field('Max radius (px)', w_blob_radius, min_width=52, max_width=76)])
    foci_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    settings_outer.addWidget(cell_group)
    if has_foci_image_files:
        settings_outer.addWidget(foci_group)
    else:
        foci_group.setVisible(False)
    foci_table_host = QWidget()
    foci_table_lay = QVBoxLayout(foci_table_host)
    foci_table_lay.setContentsMargins(0, 0, 0, 0)
    foci_table_lay.setSpacing(4)
    foci_table = QTableWidget()
    foci_table.setAlternatingRowColors(False)
    foci_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    foci_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    foci_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    try:
        vhdr = foci_table.verticalHeader()
        vhdr.setVisible(True)
        vhdr.setDefaultSectionSize(20)
        vhdr.setMinimumSectionSize(18)
        vhdr.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
    except Exception:
        pass
    foci_table.setStyleSheet(preview_table_stylesheet())
    foci_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    foci_table.setMinimumHeight(72)
    foci_table.setMaximumHeight(140)
    foci_table.setMinimumWidth(0)
    foci_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    foci_table_lay.addWidget(foci_table, 1)
    foci_table_section = CollapsibleSection('Foci table', foci_table_host, parent=settings_host, expanded=True, compact=True)
    foci_table_section.setVisible(False)
    settings_outer.addWidget(foci_table_section)

    class _FociTableColumnFitFilter(QObject):

        def __init__(self, table: QTableWidget, host: QWidget) -> None:
            super().__init__(host)
            self._table = table

        def eventFilter(self, obj, event) -> bool:
            if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest):
                QTimer.singleShot(0, lambda : fit_foci_table_columns(self._table))
            return False
    _foci_col_fit = _FociTableColumnFitFilter(foci_table, foci_table_host)
    foci_table.installEventFilter(_foci_col_fit)
    foci_table_host.installEventFilter(_foci_col_fit)
    foci_table_section.installEventFilter(_foci_col_fit)
    settings_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    settings_host.setMinimumWidth(0)
    settings_scroll = build_preview_settings_scroll(settings_host)
    settings_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _FILTER_SETTINGS_VIEWPORT_MAX_PX = 340

    def _settings_viewport_cap_px() -> int:
        """Cap visible settings height so the image keeps room when many sections are open."""
        ref: Optional[QWidget] = settings_scroll
        host_h = 0
        for _ in range(16):
            if ref is None:
                break
            try:
                host_h = max(host_h, int(ref.height()))
            except Exception:
                pass
            ref = ref.parentWidget()
        if host_h < 280:
            return _FILTER_SETTINGS_VIEWPORT_MAX_PX
        return max(160, min(_FILTER_SETTINGS_VIEWPORT_MAX_PX, int(host_h * 0.42)))

    def _sync_filter_settings_panel_height() -> None:
        try:
            cap = _settings_viewport_cap_px()
            frame_extra = max(0, int(settings_scroll.frameWidth()) * 2)
            content_hint = max(settings_host.sizeHint().height(), settings_host.minimumSizeHint().height())
            viewport_h = max(PREVIEW_SETTINGS_SCROLL_MIN_HEIGHT, min(cap, content_hint))
            settings_scroll.setFixedHeight(viewport_h + frame_extra)
            fit_foci_table_columns(foci_table)
        except Exception:
            pass

    def _scroll_settings_to_top() -> None:
        try:
            sb = settings_scroll.verticalScrollBar()
            if sb is not None:
                sb.setValue(int(sb.minimum()))
        except Exception:
            pass

    def _scroll_filter_section_into_view(section: CollapsibleSection) -> None:
        try:
            _sync_filter_settings_panel_height()
            settings_scroll.ensureWidgetVisible(section, 36, 24)
            content = section.content()
            if content is not None and section.is_expanded():
                settings_scroll.ensureWidgetVisible(content, 24, 24)
        except Exception:
            pass

    def _on_foci_table_section_toggled(expanded: bool, section: CollapsibleSection) -> None:
        _sync_filter_settings_panel_height()
        if not expanded:
            return
        QTimer.singleShot(0, lambda s=section: _scroll_filter_section_into_view(s))
    try:
        foci_table_section.toggled.connect(lambda expanded, s=foci_table_section: _on_foci_table_section_toggled(expanded, s))
    except Exception:
        pass

    class _FilterSettingsViewportResizeFilter(QObject):

        def eventFilter(self, obj, event) -> bool:
            if event.type() == QEvent.Type.Resize:
                _sync_filter_settings_panel_height()
            return False
    try:
        vp = settings_scroll.viewport()
        if vp is not None:
            _settings_vp_filter = _FilterSettingsViewportResizeFilter(settings_scroll)
            vp.installEventFilter(_settings_vp_filter)
    except Exception:
        pass
    _sync_filter_settings_panel_height()
    QTimer.singleShot(0, _scroll_settings_to_top)
    image_combo = QComboBox()
    frame_spin = QSpinBox()
    frame_spin.setRange(1, max(1, num_frames))
    frame_spin.setValue(1)
    try:
        image_combo.setVisible(False)
        frame_spin.setVisible(False)
    except Exception:
        pass
    from bactomate.gui.preview.navigation_controls import PreviewNavSpec, PreviewNavState, PreviewSource, PreviewNavigationControls
    reapply_btn = QPushButton('Apply current filters')
    reapply_btn.setMinimumHeight(32)
    style_module_primary_button(reapply_btn)
    nav_ctrl = PreviewNavigationControls(spec=PreviewNavSpec(frame_count=int(max(1, num_frames)), image_sources=[], show_image=True, show_channel=False, show_z=False, debounce_ms=400), inline_trailer=reapply_btn)
    dlg = None
    from bactomate.modules.image_stack_utils import coerce_stack_to_tyx
    sources: list[PreviewSource] = []
    if not bg_paths:
        image_combo.addItem('Mask only', ('none', -1))
        sources.append(PreviewSource(key=('none', -1), label='Mask only', kind='image', resets_lut=True))
    for (i, fp) in enumerate(bg_paths):
        view_lbl = preview_view_label(Path(fp))
        image_combo.addItem(view_lbl, ('file', i))
        sources.append(PreviewSource(key=('file', i), label=view_lbl, kind='image', resets_lut=True))
    default_idx = default_background_index(tab, mask_path, bg_paths)
    if bg_paths:
        image_combo.setCurrentIndex(default_idx)
    nav_ctrl.set_spec(PreviewNavSpec(frame_count=int(max(1, num_frames)), image_sources=sources, show_image=True, show_channel=False, show_z=False, show_cell_overlay_toggle=True, show_foci_overlay_toggle=True, cell_overlay_default=True, debounce_ms=400), emit_init=False)
    try:
        nav_ctrl.set_state(PreviewNavState(source_key=image_combo.currentData(), frame_idx=int(frame_spin.value()) - 1), emit=False)
    except Exception:
        pass

    def _selected_source() -> tuple[str, int]:
        sel = image_combo.currentData()
        if not isinstance(sel, tuple) or len(sel) < 2:
            return ('none', -1)
        try:
            return (str(sel[0]), int(sel[1]))
        except Exception:
            return ('none', -1)

    def _image_path_for_source_key(source_key) -> Optional[Path]:
        try:
            (kind, idx) = (str(source_key[0]), int(source_key[1]))
        except Exception:
            return None
        if kind != 'file' or idx < 0:
            return None
        if idx >= len(bg_paths):
            return None
        return Path(bg_paths[idx])

    def _selected_image_path() -> Optional[Path]:
        return _image_path_for_source_key(_selected_source())

    def _calibration_for_preview_source(source_path: Optional[Path]) -> tuple[float, Optional[Path], str]:
        """Read pixel size from active view metadata, falling back to the mask/shared metadata."""
        candidates: list[Path] = []
        if source_path is not None:
            candidates.append(Path(source_path))
        candidates.append(Path(mask_path))
        for cand in candidates:
            try:
                (ps, _tp, src, kind) = resolve_calibration_for_mask(Path(cand))
            except Exception:
                continue
            if ps > 0:
                return (float(ps), src or Path(cand), str(kind or 'file'))
        parent = getattr(tab, 'parent_window', None)
        sm = getattr(parent, 'shared_metadata', None) if parent is not None else None
        if isinstance(sm, dict) and sm.get('pixel_size') is not None:
            try:
                ps = float(sm['pixel_size'])
            except (TypeError, ValueError):
                ps = 0.0
            if ps > 0:
                return (ps, None, 'metadata')
        return (0.0, None, 'none')

    def _sync_preview_pixel_size_from_metadata(source_path: Optional[Path]=None) -> None:
        (ps, src, kind) = _calibration_for_preview_source(source_path or _selected_image_path())
        if ps <= 0:
            w_pixel.setToolTip('Pixel size for physical cell filters. No metadata calibration was found.')
            return
        try:
            w_pixel.blockSignals(True)
            w_pixel.setValue(float(ps))
        finally:
            try:
                w_pixel.blockSignals(False)
            except Exception:
                pass
        src_txt = f' from {src.name}' if src is not None else ''
        w_pixel.setToolTip(f'Auto-filled{src_txt} ({kind}).')
        if hasattr(tab, 'set_filter_settings'):
            try:
                tab.set_filter_settings({'pixel_size': float(ps)})
            except Exception:
                pass
        if hasattr(tab, '_apply_pixel_size_from_calibration'):
            try:
                tab._apply_pixel_size_from_calibration(float(ps), indicator='(from preview metadata)')
            except Exception:
                pass

    def _view_path_is_foci(path: Optional[Path]) -> bool:
        if path is None:
            return False
        low = path.name.lower()
        return '_foci' in low or 'fi_foci' in low

    def _foci_index_for_path(target: Optional[Path]) -> int:
        if target is None or not foci_sources:
            return 0
        from bactomate.gui.filter_preview_utils import file_identity_key
        try:
            tkey = file_identity_key(Path(target))
        except Exception:
            tkey = str(target)
        for (i, fp) in enumerate(foci_sources):
            try:
                if file_identity_key(Path(fp)) == tkey:
                    return int(i)
            except Exception:
                if Path(fp) == Path(target):
                    return int(i)
        return 0

    def _active_foci_channel_index() -> int:
        if not foci_sources:
            return 0
        view_p = _selected_image_path()
        if _view_path_is_foci(view_p):
            return _foci_index_for_path(view_p)
        return 0

    def _active_foci_channel_key() -> str:
        if not foci_sources:
            return '_default'
        view_p = _selected_image_path()
        if _view_path_is_foci(view_p):
            return foci_channel_key_for_path(view_p)
        return foci_channel_key_for_path(foci_sources[_active_foci_channel_index()])

    def _foci_channel_settings_for_key(ch_key: str) -> dict[str, Any]:
        try:
            if hasattr(tab, 'get_foci_channel_settings'):
                return dict(tab.get_foci_channel_settings(ch_key))
        except Exception:
            pass
        try:
            fs = tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}
            channels = fs.get('foci_channels') if isinstance(fs, dict) else None
            if isinstance(channels, dict) and ch_key in channels:
                return dict(channels[ch_key])
        except Exception:
            pass
        return {}

    def _foci_channel_sigs_for_sources() -> dict[str, str]:
        sigs: dict[str, str] = {}
        for fp in foci_sources:
            ch_key = foci_channel_key_for_path(fp)
            sigs[ch_key] = foci_channel_settings_signature(_foci_channel_settings_for_key(ch_key))
        return sigs

    def _plan_incremental_apply(frame_idx: int, fs: dict[str, Any]) -> dict[str, Any]:
        """Decide whether to reuse cell filter and which foci channels need rescoring."""
        last = getattr(dlg, '_last_preview', None) or {}
        mpk = getattr(dlg, '_preview_mask_path_key', None)
        cell_sig = cell_filter_settings_signature(fs)
        last_cell_sig = last.get('cell_settings_sig')
        cache_hit = is_filter_apply_cache_hit(last, frame_idx, mask_path_key=mpk)
        reuse_cells = bool(cache_hit and last_cell_sig == cell_sig and (last.get('accepted') is not None))
        cell_changed = not reuse_cells
        last_foci = dict(last.get('foci_channels') or {}) if cache_hit else {}
        last_foci_sigs = dict(last.get('foci_channel_sigs') or {}) if cache_hit else {}
        prefilled: dict[str, dict[str, list]] = {}
        jobs: list[dict] = []
        channels_to_score: list[str] = []
        for fp in foci_sources:
            ch_key = foci_channel_key_for_path(fp)
            ch_only = _foci_channel_settings_for_key(ch_key)
            ch_sig = foci_channel_settings_signature(ch_only)
            must_rescore = not reuse_cells
            if reuse_cells:
                if cell_changed:
                    must_rescore = True
                elif ch_sig != last_foci_sigs.get(ch_key):
                    must_rescore = True
                elif ch_key not in last_foci:
                    must_rescore = True
                else:
                    prefilled[ch_key] = {'pass_peaks': list(last_foci[ch_key].get('pass_peaks') or []), 'fail_peaks': list(last_foci[ch_key].get('fail_peaks') or [])}
            if must_rescore:
                channels_to_score.append(ch_key)
                ch_rows = _rows_for_frame(_load_rows_for_source(fp), int(frame_idx))
                ch_plane = _get_foci_plane(int(frame_idx), fp)
                if not ch_rows or ch_plane is None:
                    prefilled[ch_key] = {'pass_peaks': [], 'fail_peaks': []}
                    continue
                jobs.append({'channel_key': ch_key, 'frame_rows': [dict(r) for r in ch_rows], 'foci_plane': np.asarray(ch_plane).copy(), 'settings': dict(_foci_settings_for_source(fp))})
        return {'reuse_cells': reuse_cells, 'cell_sig': cell_sig, 'cell_changed': cell_changed, 'prefilled': prefilled, 'jobs': jobs, 'channels_to_score': channels_to_score}

    def _invalidate_foci_pass_fail_cache(frame_idx: int, channel_keys: Optional[set[str]]=None) -> None:
        cache = getattr(dlg, '_foci_pass_fail_cache', None)
        if not isinstance(cache, dict):
            return
        mpk = str(getattr(dlg, '_preview_mask_path_key', None) or '')
        fid = int(frame_idx)
        if not channel_keys:
            keys_del = [k for k in list(cache.keys()) if k[0] == fid and str(k[1]) == mpk]
        else:
            keys_del = [k for k in list(cache.keys()) if k[0] == fid and str(k[1]) == mpk and (len(k) >= 5) and (k[4] in channel_keys)]
        for k in keys_del:
            cache.pop(k, None)

    def _store_foci_pass_fail_cache(frame_idx: int, foci_channels_last: dict[str, dict[str, list]], *, accepted: set[int], request_settings: dict[str, Any]) -> None:
        cache = getattr(dlg, '_foci_pass_fail_cache', None)
        if not isinstance(cache, dict):
            cache = {}
            dlg._foci_pass_fail_cache = cache
        mpk_pf = getattr(dlg, '_preview_mask_path_key', None)
        acc_for_foci = accepted_cell_labels_for_foci(accepted, cell_filter_applied=True)
        acc_key = frozenset((int(x) for x in acc_for_foci)) if acc_for_foci else None
        for fp in foci_sources:
            ch_key = foci_channel_key_for_path(fp)
            payload = foci_channels_last.get(ch_key)
            if payload is None:
                continue
            ch_sig = foci_channel_settings_signature(_foci_channel_settings_for_key(ch_key))
            cache[int(frame_idx), str(mpk_pf or ''), ch_sig, acc_key, ch_key] = (list(payload.get('pass_peaks') or []), list(payload.get('fail_peaks') or []))

    def _commit_filter_apply_result(*, frame_idx: int, accepted: set[int], rejected: set[int], n_obj: Optional[int], foci_channels_last: dict[str, dict[str, list]], request_settings_sig: str, cell_settings_sig: str, foci_channel_sigs: dict[str, str], request_source_key: tuple, foci_txt: str='') -> None:
        source_key = tuple(request_source_key)
        active_ch = _active_foci_channel_key()
        pass_peaks = list(foci_channels_last.get(active_ch, {}).get('pass_peaks') or [])
        fail_peaks = list(foci_channels_last.get(active_ch, {}).get('fail_peaks') or [])
        foci_src = _foci_source_for_source_key(source_key)
        frame_rows = _rows_for_frame(_load_rows_for_source(foci_src), int(frame_idx))
        raw_n = len(frame_rows)
        dlg._last_preview = {'mask_path_key': getattr(dlg, '_preview_mask_path_key', None), 'source_key': source_key, 'frame_idx': int(frame_idx), 'settings_sig': request_settings_sig, 'cell_settings_sig': cell_settings_sig, 'foci_channel_sigs': dict(foci_channel_sigs), 'mask_2d': np.asarray(_preview_mask_stack()[int(frame_idx)], dtype=np.int32), 'accepted': accepted, 'rejected': rejected, 'n_obj': int(n_obj) if n_obj is not None else None, 'foci_txt': foci_txt, 'pass_peaks': pass_peaks, 'fail_peaks': fail_peaks, 'foci_channels': foci_channels_last, 'bg': _get_bg_frame(int(frame_idx), source_key), 'foci_plane': _get_foci_plane(int(frame_idx), foci_src), 'frame_rows': frame_rows, 'info_text': f'Accepted: {len(accepted)}  Rejected: {len(rejected)}  Total: {int(n_obj or 0)}  |  Foci(frame): raw {raw_n} / pass {len(pass_peaks)} / fail {len(fail_peaks)}{foci_txt}'}
        _store_foci_pass_fail_cache(int(frame_idx), foci_channels_last, accepted=accepted, request_settings=dict(tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}))
        _clear_preview_layer_caches(dlg)
        try:
            _update_foci_table_live()
        except Exception:
            pass
        dlg.refresh()
        try:
            _prewarm_after_apply(int(frame_idx))
        except Exception:
            pass

    def _prewarm_after_apply(frame_idx: int) -> None:
        """Warm caches after Apply without changing the visible frame."""
        if dlg is None:
            return
        try:
            num_frames0 = int(np.asarray(_preview_mask_stack()).shape[0])
        except Exception:
            return
        if num_frames0 <= 1:
            return
        fid0 = max(0, min(int(frame_idx), num_frames0 - 1))
        mpk = getattr(dlg, '_preview_mask_path_key', None)
        last = getattr(dlg, '_last_preview', {}) or {}
        if not is_filter_apply_cache_hit(last, fid0, mask_path_key=mpk):
            return

        def _warm_foci_for_frame(fid: int) -> None:
            try:
                src_key = _source_key_from_state(dlg.state(), _selected_source)
            except Exception:
                src_key = ('none', -1)
            foci_src = _foci_source_for_source_key(src_key)
            if foci_src is None:
                return
            try:
                mask_2d = np.asarray(_preview_mask_stack()[int(fid)], dtype=np.int32)
            except Exception:
                return
            rows = _rows_for_frame(_load_rows_for_source(foci_src), int(fid))
            plane = _get_foci_plane(int(fid), foci_src)
            if plane is None or not rows:
                return
            accepted_cells = accepted_cell_labels_for_foci(last.get('accepted'), cell_filter_applied=True)
            _foci_pass_fail_cached(int(fid), src_key, _foci_settings_for_source(foci_src), mask_2d=mask_2d, foci_plane=np.asarray(plane), frame_rows=rows, accepted_cell_labels=accepted_cells)
        for (off, delay) in ((-1, 15), (1, 30)):
            f2 = fid0 + int(off)
            if 0 <= f2 < num_frames0:
                QTimer.singleShot(int(delay), lambda _fid=f2: _warm_foci_for_frame(int(_fid)))

        def _warm_overlay_rasters_current_frame() -> None:
            try:
                dlg.set_extra('overlay_only', True)
                dlg.set_extra('nav_only', True)
                dlg.refresh()
            finally:
                try:
                    dlg.set_extra('overlay_only', False)
                    dlg.set_extra('nav_only', False)
                except Exception:
                    pass
        QTimer.singleShot(0, _warm_overlay_rasters_current_frame)

    def _foci_settings_for_source(fp: Optional[Path]) -> dict[str, Any]:
        try:
            base = dict(tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {})
        except Exception:
            base = {}
        if fp is None:
            return resolve_foci_settings_for_source(base, None)
        if hasattr(tab, 'get_settings_for_foci_source'):
            try:
                return dict(tab.get_settings_for_foci_source(Path(fp)))
            except Exception:
                pass
        return resolve_foci_settings_for_source(base, Path(fp))

    def _sync_filter_panels_for_view() -> None:
        """View switches background; cell vs foci filter panels swap."""
        view_p = _selected_image_path()
        is_foci_view = _view_path_is_foci(view_p)
        try:
            cell_group.setVisible(not is_foci_view)
            foci_group.setVisible(bool(is_foci_view and has_foci_image_files))
            if has_foci_image_files:
                ch_key = _active_foci_channel_key()
                foci_group.setTitle('Foci filtering')
                if ch_key and ch_key != '_default':
                    foci_channel_badge.setText(f'Channel: {ch_key}')
                    foci_channel_badge.setVisible(True)
                else:
                    foci_channel_badge.setText('')
                    foci_channel_badge.setVisible(False)
        except Exception:
            pass
        _sync_filter_settings_panel_height()

    def _foci_source_for_channel_index(ch_idx: int) -> Optional[Path]:
        if not foci_sources:
            return None
        ch_idx = max(0, min(int(ch_idx), len(foci_sources) - 1))
        return Path(foci_sources[ch_idx])

    def _active_foci_source() -> Optional[Path]:
        return _foci_source_for_channel_index(_active_foci_channel_index())

    def _bg_index_for_path(target: Optional[Path]) -> Optional[int]:
        if target is None:
            return None
        from bactomate.gui.filter_preview_utils import file_identity_key
        try:
            tkey = file_identity_key(Path(target))
        except Exception:
            tkey = str(target)
        for (i, fp) in enumerate(bg_paths):
            try:
                if file_identity_key(Path(fp)) == tkey:
                    return int(i)
            except Exception:
                if Path(fp) == Path(target):
                    return int(i)
        return None

    def _foci_source_for_source_key(source_key) -> Optional[Path]:
        img_path = _image_path_for_source_key(source_key)
        if img_path is not None:
            img_name = img_path.name.lower()
            if '_foci' in img_name:
                for fp in foci_sources:
                    try:
                        if Path(fp).resolve() == img_path.resolve():
                            return Path(fp)
                    except Exception:
                        if Path(fp) == img_path:
                            return Path(fp)
            try:
                mapped = tab._foci_path_for_fi_path(mask_path, img_path, [Path(x) for x in foci_sources])
                if mapped is not None:
                    return Path(mapped)
            except Exception:
                pass
        if len(foci_sources) == 1:
            return Path(foci_sources[0])
        if len(foci_sources) > 1:
            return _active_foci_source()
        return None

    def _foci_source_for_selected_image() -> Optional[Path]:
        src = _foci_source_for_source_key(_selected_source())
        if src is not None:
            return src
        return _active_foci_source()
    from bactomate.gui.artifact_paths import infer_pipeline_dir_from_mask, resolve_filtered_foci_csv, resolve_segmentation_foci_csv
    _pipeline_dir = infer_pipeline_dir_from_mask(mask_path)

    def _csv_path_for_foci_source(src: Optional[Path]) -> Optional[Path]:
        if src is None:
            return None
        try:
            filtered = resolve_filtered_foci_csv(_pipeline_dir, mask_path=mask_path, foci_source_path=src)
            if filtered is not None and filtered.is_file():
                return filtered
            return resolve_segmentation_foci_csv(_pipeline_dir, mask_path=mask_path, foci_source_path=src)
        except Exception:
            return None

    def _rows_cache_key(p: Optional[Path]) -> str:
        if p is None:
            return '__none__'
        try:
            return str(p.resolve())
        except Exception:
            return str(p)

    def _load_rows_for_source(src: Optional[Path]) -> list[dict]:
        key = _rows_cache_key(src)
        cache = getattr(dlg, '_foci_rows_cache', None) if dlg is not None else None
        if isinstance(cache, dict) and key in cache:
            return list(cache.get(key) or [])
        csv_path = _csv_path_for_foci_source(src)
        rows: list[dict] = []
        if csv_path is not None and csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows.append(r)
            except Exception:
                rows = []
        if isinstance(cache, dict):
            cache[key] = list(rows)
        return rows
    foci_overlay_sources: list[Path] = [Path(fp) for fp in foci_sources if _csv_path_for_foci_source(fp) is not None]
    foci_overlay_available = len(foci_overlay_sources) > 0
    _foci_ch_overlay_labels: tuple[str, ...] = ()
    if foci_overlay_available:
        _foci_ch_overlay_labels = tuple((f'Foci {short_foci_stack_label(fp)}' for fp in foci_overlay_sources))
    nav_ctrl.set_spec(PreviewNavSpec(frame_count=int(max(1, num_frames)), image_sources=sources, show_image=True, show_channel=False, show_z=False, show_cell_overlay_toggle=True, show_foci_overlay_toggle=bool(foci_overlay_available and (not _foci_ch_overlay_labels)), cell_overlay_default=True, foci_overlay_available=bool(foci_overlay_available), foci_overlay_default=bool(foci_overlay_available), foci_channel_overlay_labels=_foci_ch_overlay_labels, foci_channel_overlay_defaults=tuple((True for _ in _foci_ch_overlay_labels)), debounce_ms=400), emit_init=False)
    try:
        nav_ctrl.set_state(PreviewNavState(source_key=image_combo.currentData(), frame_idx=int(frame_spin.value()) - 1), emit=False)
    except Exception:
        pass

    def _rows_for_frame(rows: list[dict], frame_idx: int) -> list[dict]:
        out: list[dict] = []
        for r in rows or []:
            try:
                fr = int(float(r.get('frame', 0) or 0)) if 'frame' in r else 0
            except Exception:
                fr = 0
            if fr == int(frame_idx):
                out.append(r)
        return out

    def _apply_lut_to_bg(bg_2d: Optional[np.ndarray], vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
        if bg_2d is None:
            return np.zeros(masks.shape[1:3], dtype=np.uint8)
        bg_f = np.asarray(bg_2d, dtype=np.float64)
        if bg_f.ndim != 2:
            bg_f = np.asarray(bg_f.squeeze())
            if bg_f.ndim != 2:
                return np.zeros(masks.shape[1:3], dtype=np.uint8)
        mn = float(np.nanmin(bg_f)) if bg_f.size else 0.0
        mx = float(np.nanmax(bg_f)) if bg_f.size else mn + 1.0
        if vmin is None or vmax is None or (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            vmin = mn
            vmax = mx if mx > mn else mn + 1.0
        vrange = max(float(vmax) - float(vmin), 1e-06)
        return ((bg_f - float(vmin)) / vrange * 255.0).clip(0, 255).astype(np.uint8)

    def _get_bg_frame(frame_idx: int, source_key=None) -> Optional[np.ndarray]:
        sel = source_key if source_key is not None else image_combo.currentData()
        if not isinstance(sel, tuple) or len(sel) < 2:
            return None
        (kind, idx) = (sel[0], int(sel[1]))
        if kind != 'file' or idx < 0:
            return None
        paths = getattr(dlg, '_bg_paths', []) or []
        if idx >= len(paths):
            return None
        cache = getattr(dlg, '_bg_cache', None)
        if not isinstance(cache, dict):
            return None
        return _load_preview_plane(Path(paths[idx]), int(frame_idx), cache)

    def _get_foci_plane(frame_idx: int, foci_src: Optional[Path]=None) -> Optional[np.ndarray]:
        src = foci_src
        if src is None:
            src = _foci_source_for_selected_image()
        if src is None:
            return None
        cache = getattr(dlg, '_bg_cache', None)
        if not isinstance(cache, dict):
            return None
        return _load_preview_plane(Path(src), int(frame_idx), cache)

    def _render_overlay(mask_2d: np.ndarray, accepted: set[int], rejected: set[int], bg_u8: np.ndarray, pass_peaks: list[dict], fail_peaks: list[dict], show_cells: bool=True, show_foci: bool=True, foci_plane: Optional[np.ndarray]=None, foci_settings: Optional[dict]=None, draw_pass_foci_as_blobs: bool=False, foci_filters_applied: bool=False, skip_foci: bool=False) -> np.ndarray:
        from skimage.segmentation import find_boundaries
        (scaled_img, scaled_mask) = scale_for_preview(bg_u8, mask_2d)
        rgb = np.stack([scaled_img, scaled_img, scaled_img], axis=-1).astype(np.uint8)
        if show_cells and accepted:
            am = np.isin(scaled_mask, np.array(list(accepted), dtype=np.int32))
            if np.any(am):
                alpha = 0.55
                rgb[am, 0] = (rgb[am, 0].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                rgb[am, 1] = (rgb[am, 1].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                rgb[am, 2] = (rgb[am, 2].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                mask_acc = np.where(am, scaled_mask, 0).astype(np.int32)
                b = find_boundaries(mask_acc, mode='outer')
                rgb[b] = 0
        if show_cells and rejected:
            rm = np.isin(scaled_mask, np.array(list(rejected), dtype=np.int32))
            if np.any(rm):
                alpha = 0.55
                rgb[rm, 0] = (rgb[rm, 0].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                rgb[rm, 1] = (rgb[rm, 1].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                rgb[rm, 2] = (rgb[rm, 2].astype(np.float32) * (1 - alpha)).astype(np.uint8)

        def _draw_peak_rings(peaks: list[dict], color: tuple[int, int, int], *, radius: int=4) -> None:
            """Hollow ring markers for raw (pre-apply) foci only — not used after Apply."""
            (h0, w0) = mask_2d.shape[:2]
            (hs, ws) = rgb.shape[:2]
            sx = ws / w0 if w0 else 1.0
            sy = hs / h0 if h0 else 1.0
            r = max(2, int(radius))
            inner = max(1, r - 2)
            outer = r * r
            inner_sq = inner * inner
            for p in peaks or []:
                yy = int(round(float(p.get('y', 0)) * sy))
                xx = int(round(float(p.get('x', 0)) * sx))
                if 0 <= yy < hs and 0 <= xx < ws:
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            d2 = dx * dx + dy * dy
                            if d2 > outer or d2 < inner_sq:
                                continue
                            (y2, x2) = (yy + dy, xx + dx)
                            if 0 <= y2 < hs and 0 <= x2 < ws:
                                rgb[y2, x2, 0] = int(color[0])
                                rgb[y2, x2, 1] = int(color[1])
                                rgb[y2, x2, 2] = int(color[2])

        def _draw_peak_marker_rings(peaks: list[dict], color: tuple[int, int, int], *, radius: int=3) -> None:
            """Thin circular outline when blob growth fails (not a filled square)."""
            (h0, w0) = mask_2d.shape[:2]
            (hs, ws) = rgb.shape[:2]
            sx = ws / w0 if w0 else 1.0
            sy = hs / h0 if h0 else 1.0
            r = max(2, int(radius))
            inner_sq = max(0, (r - 1) * (r - 1))
            outer_sq = (r + 1) * (r + 1)
            for p in peaks or []:
                yy = int(round(float(p.get('y', 0)) * sy))
                xx = int(round(float(p.get('x', 0)) * sx))
                if 0 <= yy < hs and 0 <= xx < ws:
                    for dy in range(-r - 1, r + 2):
                        for dx in range(-r - 1, r + 2):
                            d2 = dx * dx + dy * dy
                            if d2 > outer_sq or d2 < inner_sq:
                                continue
                            (y2, x2) = (yy + dy, xx + dx)
                            if 0 <= y2 < hs and 0 <= x2 < ws:
                                rgb[y2, x2, 0] = int(color[0])
                                rgb[y2, x2, 1] = int(color[1])
                                rgb[y2, x2, 2] = int(color[2])

        def _draw_foci_contours(peaks: list[dict], plane: np.ndarray, settings: dict, contour_rgb: tuple[int, int, int]) -> list[dict]:
            return draw_foci_blob_contours_on_rgb(rgb, mask_2d, peaks, np.asarray(plane), settings, contour_rgb)
        if show_foci and (not skip_foci):
            try:
                fs_draw = dict(foci_settings or {})
                plane_draw = foci_plane
                foci_after_apply = bool(draw_pass_foci_as_blobs)
                if foci_after_apply and plane_draw is not None:
                    if pass_peaks:
                        missed = _draw_foci_contours(pass_peaks, np.asarray(plane_draw), fs_draw, _FOCI_PASS_RGB)
                        if missed:
                            _draw_peak_marker_rings(missed, _FOCI_PASS_RGB, radius=3)
                    if fail_peaks:
                        missed = _draw_foci_contours(fail_peaks, np.asarray(plane_draw), fs_draw, _FOCI_FAIL_RGB)
                        if missed:
                            _draw_peak_marker_rings(missed, _FOCI_FAIL_RGB, radius=3)
                elif pass_peaks:
                    _draw_peak_rings(pass_peaks, _FOCI_RAW_PEAK_RGB, radius=4)
            except Exception:
                pass
        return rgb

    def get_current_parameters(fs: Optional[dict]=None) -> dict:
        if fs is None:
            fs = tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}
        return {'pixel_size': float(fs.get('pixel_size', 0.108) or 0.108), 'exclude_border': bool(fs.get('exclude_border', False)), 'min_aspect_ratio': float(fs.get('min_aspect_ratio', 1.5) or 1.5), 'max_aspect_ratio': float(fs.get('max_aspect_ratio', 10.0) or 10.0), 'min_eccentricity': float(fs.get('min_eccentricity', 0.5) or 0.5), 'min_solidity': float(fs.get('min_solidity', 0.7) or 0.7), 'min_area': fs.get('min_area_um2', None), 'max_area': fs.get('max_area_um2', None), 'min_width': fs.get('min_width_um', None), 'max_width': fs.get('max_width_um', None), 'min_length': fs.get('min_length_um', None), 'max_length': fs.get('max_length_um', None)}

    def _settings_signature_for(fs: Optional[dict]) -> str:
        """Best-effort stable signature for a filter settings snapshot."""
        fs = fs or {}
        norm: dict[str, object] = {}
        for k in sorted((fs or {}).keys()):
            v = (fs or {}).get(k)
            if v is None:
                norm[k] = None
                continue
            if isinstance(v, bool):
                norm[k] = bool(v)
                continue
            try:
                norm[k] = round(float(v), 6)
                continue
            except Exception:
                pass
            try:
                norm[k] = str(v)
            except Exception:
                norm[k] = None
        try:
            import json as _json
            return _json.dumps(norm, sort_keys=True, separators=(',', ':'))
        except Exception:
            return str(norm)

    def _settings_signature() -> str:
        """Best-effort stable signature for current filter settings (for cache invalidation)."""
        try:
            fs = tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}
        except Exception:
            fs = {}
        return _settings_signature_for(fs)

    def _build_foci_apply_bundle(frame_idx: int, request_settings: dict[str, Any], *, incremental_plan: Optional[dict]=None, request_source_key: Optional[tuple]=None) -> dict[str, Any]:
        if incremental_plan is None:
            incremental_plan = _plan_incremental_apply(frame_idx, request_settings)
        src_key = tuple(request_source_key if request_source_key is not None else _selected_source())
        foci_src = _foci_source_for_source_key(src_key)
        raw_rows_all = _load_rows_for_source(foci_src)
        frame_rows_apply = _rows_for_frame(raw_rows_all, int(frame_idx))
        foci_plane_apply = _get_foci_plane(int(frame_idx), foci_src)
        foci_plane_work = np.asarray(foci_plane_apply).copy() if foci_plane_apply is not None else None
        reuse_cell_filter = None
        if incremental_plan.get('reuse_cells'):
            last = getattr(dlg, '_last_preview', None) or {}
            all_labels = set((int(x) for x in last.get('accepted') or [])) | set((int(x) for x in last.get('rejected') or []))
            reuse_cell_filter = {'accepted': list(last.get('accepted') or []), 'rejected': list(last.get('rejected') or []), 'all': list(all_labels), 'n_obj': last.get('n_obj')}
        from bactomate.gui.filter_preview_utils import format_incremental_apply_plan
        return {'channels': list(incremental_plan.get('jobs') or []), 'prefilled_channels': dict(incremental_plan.get('prefilled') or {}), 'reuse_cell_filter': reuse_cell_filter, 'active_channel_key': _active_foci_channel_key(), 'frame_rows': [dict(r) for r in frame_rows_apply], 'foci_plane': foci_plane_work, 'settings': dict(request_settings), 'cell_settings_sig': incremental_plan.get('cell_sig'), 'apply_plan_summary': format_incremental_apply_plan(incremental_plan), 'cell_changed': bool(incremental_plan.get('cell_changed', True))}

    def _start_worker(frame_idx: int, *, settings_snapshot: Optional[dict]=None, settings_sig: Optional[str]=None, source_key=None, foci_apply_bundle: Optional[dict]=None) -> None:
        if frame_idx < 0 or frame_idx >= num_frames:
            return
        if settings_snapshot is None:
            try:
                settings_snapshot = tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}
            except Exception:
                settings_snapshot = {}
        settings_snapshot = dict(settings_snapshot or {})
        if settings_sig is None:
            settings_sig = _settings_signature_for(settings_snapshot)
        source_key = tuple(source_key if source_key is not None else _selected_source())
        try:
            prev_prog = getattr(dlg, '_progress_dialog', None)
            if prev_prog is not None:
                try:
                    prev_prog.canceled.disconnect()
                except Exception:
                    pass
                prev_prog.close()
                dlg._progress_dialog = None
        except Exception:
            pass
        existing_thread = getattr(dlg, '_preview_thread', None)
        if existing_thread is not None and existing_thread.isRunning():
            try:
                dlg._preview_request_id += 1
                dlg._pending_apply = True
                dlg._pending_apply_frame_idx = int(frame_idx)
                dlg._pending_apply_settings = dict(settings_snapshot)
                dlg._pending_apply_settings_sig = str(settings_sig)
                dlg._pending_apply_source_key = tuple(source_key)
            except Exception:
                pass
            try:
                dlg.set_primary_action(None, label='Queued…', enabled=False, visible=False)
                reapply_btn.setEnabled(False)
                reapply_btn.setText('Queued…')
            except Exception:
                pass
            return
        dlg._preview_request_id += 1
        rid = int(dlg._preview_request_id)
        _stack = getattr(dlg, '_preview_masks', masks)
        mask_2d = np.asarray(_stack[frame_idx], dtype=np.int32)
        request_settings = dict(settings_snapshot)
        request_settings_sig = str(settings_sig)
        request_source_key = tuple(source_key)
        params = get_current_parameters(request_settings)
        display_params = {'do_fi_contour': False, 'fi_contour_thickness_val': 1, 'run_foci': False, 'foci_snapshot': None}
        foci_src = _foci_source_for_source_key(request_source_key)
        raw_rows_all = _load_rows_for_source(foci_src)
        frame_rows_apply = _rows_for_frame(raw_rows_all, int(frame_idx))
        display_params['raw_foci_rows'] = frame_rows_apply
        foci_plane_apply = _get_foci_plane(int(frame_idx), foci_src)
        foci_plane_work = np.asarray(foci_plane_apply).copy() if foci_plane_apply is not None else None
        if foci_apply_bundle is None:
            plan = _plan_incremental_apply(frame_idx, request_settings)
            foci_apply_bundle = _build_foci_apply_bundle(frame_idx, request_settings, incremental_plan=plan, request_source_key=request_source_key)
        else:
            plan = _plan_incremental_apply(frame_idx, request_settings)
        try:
            fs = request_settings
            display_params['filter_peaks'] = {'prob_thresh': float(fs.get('foci_prob_thresh', 0.5)) if bool(fs.get('foci_prob_enabled', False)) else None, 'min_peak_raw': float(fs.get('foci_min_int', 0.0)) if bool(fs.get('foci_min_int_enabled', False)) else None, 'per_cell_ratio': float(fs.get('foci_per_cell_ratio', 2.0)) if bool(fs.get('foci_per_cell_ratio_enabled', False)) else None}
        except Exception:
            display_params['filter_peaks'] = {}
        request_foci_apply_bundle = dict(foci_apply_bundle or {})
        plan_line = str(request_foci_apply_bundle.get('apply_plan_summary') or '').strip()
        try:
            dlg._filter_apply_plan_line = plan_line
            dlg.set_extra('filter_cell_recomputing', bool(request_foci_apply_bundle.get('cell_changed', plan.get('cell_changed', True))))
        except Exception:
            pass
        prog = FilterApplyProgressDialog(dlg)
        prog.set_progress(0, 'Applying filters', plan_line or 'Starting…')
        prog.show()
        try:
            dlg._progress_dialog = prog
        except Exception:
            pass
        worker = AnalyzeMasksPreviewWorker(rid, frame_idx, mask_2d, params, fi_image=None, display_params=display_params, skip_vis=True, latest_request_id=lambda : int(getattr(dlg, '_preview_request_id', rid)), foci_apply=request_foci_apply_bundle)
        thread = QThread(dlg)
        thread.setObjectName('AnalyzeMasksPreviewWorkerThread')
        worker.moveToThread(thread)
        dlg._preview_worker = worker
        dlg._preview_thread = thread

        def on_progress(pct: int, step: str, detail: str='') -> None:
            try:
                base = str(getattr(dlg, '_filter_apply_plan_line', '') or '').strip()
                sub = str(detail or '').strip()
                combined = '\n'.join((p for p in (base, sub) if p))
                prog.set_progress(int(pct), str(step), combined)
            except Exception:
                pass
            if int(pct) >= 100:
                try:
                    prog.set_progress(100, str(step), str(detail or ''))
                except Exception:
                    pass
        worker.progress.connect(on_progress, Qt.ConnectionType.QueuedConnection)

        def _set_primary_idle() -> None:
            try:
                dlg.set_primary_action(apply_current, label='Apply current filters', enabled=True, visible=False)
                reapply_btn.setEnabled(True)
                reapply_btn.setText('Apply current filters')
            except Exception:
                pass

        def _cancel_current() -> None:
            try:
                dlg.set_extra('filter_apply_busy', False)
            except Exception:
                pass
            try:
                dlg._preview_request_id += 1
            except Exception:
                pass
            try:
                t0 = getattr(dlg, '_preview_thread', None)
                if t0 is not None:
                    t0.requestInterruption()
                    t0.quit()
            except Exception:
                pass
            try:
                dlg.set_primary_action(None, label='Cancelling…', enabled=False, visible=False)
                reapply_btn.setEnabled(False)
            except Exception:
                pass
            try:
                prog.set_progress(0, 'Cancelling', 'Stopping current apply…')
            except Exception:
                pass
        try:
            reapply_btn.setEnabled(False)
            dlg.set_primary_action(apply_current, label='Apply current filters', enabled=False, visible=False)
            dlg.set_extra('filter_apply_busy', True)
        except Exception:
            pass
        try:
            prog.canceled.connect(_cancel_current)
        except Exception:
            pass

        def _safely_dismiss_progress() -> None:
            """Dismiss the progress dialog without treating completion as cancel."""
            try:
                prog.canceled.disconnect(_cancel_current)
            except Exception:
                pass
            try:
                prog.set_progress(100, 'Step 4/4 · Complete', '')
            except Exception:
                pass
            try:
                prog.close()
            except Exception:
                pass

        def on_result_ready(rid_out, fid, acc_list, rej_list, _all_list, n_obj, err, vis, _peaks, _n_foci_total, _n_foci_drawn, _raw_median_fi, _min_rel, _mode_txt, pass_peaks_out, fail_peaks_out, foci_channels_out):
            _safely_dismiss_progress()
            try:
                if getattr(dlg, '_progress_dialog', None) is prog:
                    dlg._progress_dialog = None
            except Exception:
                pass
            try:
                if int(rid_out) != int(getattr(dlg, '_preview_request_id', rid_out)):
                    return
            except Exception:
                pass
            if getattr(dlg, '_close_pending', False):
                return
            try:
                dlg.set_extra('filter_apply_busy', False)
                dlg.set_extra('filter_cell_recomputing', False)
                dlg._filter_apply_plan_line = ''
            except Exception:
                pass
            if err:
                try:
                    write_gui_log_line('Preview', f'Filter preview worker error: {str(err)[:200]}')
                except Exception:
                    pass
                dlg._last_preview = {}
                dlg.refresh()
                return
            accepted = set(acc_list or [])
            rejected = set(rej_list or [])
            foci_txt = ''
            try:
                if int(_n_foci_total) > 0:
                    foci_txt = f'  |  Foci CSV: {int(_n_foci_total)} raw → {int(_n_foci_drawn)} pass'
            except Exception:
                foci_txt = ''
            bg = _get_bg_frame(int(fid), request_source_key)
            source_key = tuple(request_source_key)
            frame_rows = list(frame_rows_apply)
            foci_plane = foci_plane_work
            pass_peaks = list(pass_peaks_out or [])
            fail_peaks = list(fail_peaks_out or [])
            foci_channels_last: dict[str, dict[str, list]] = {}
            if isinstance(foci_channels_out, dict):
                for (ch_key, payload) in foci_channels_out.items():
                    if not isinstance(payload, dict):
                        continue
                    foci_channels_last[str(ch_key)] = {'pass_peaks': list(payload.get('pass_peaks') or []), 'fail_peaks': list(payload.get('fail_peaks') or [])}
            cell_sig = str(request_foci_apply_bundle.get('cell_settings_sig') or cell_filter_settings_signature(dict(request_settings or {})))
            _commit_filter_apply_result(frame_idx=int(fid), accepted=accepted, rejected=rejected, n_obj=int(n_obj) if n_obj is not None else None, foci_channels_last=foci_channels_last, request_settings_sig=request_settings_sig, cell_settings_sig=cell_sig, foci_channel_sigs=_foci_channel_sigs_for_sources(), request_source_key=source_key, foci_txt=foci_txt)

        def on_thread_finished() -> None:
            try:
                if getattr(dlg, '_progress_dialog', None) is prog:
                    _safely_dismiss_progress()
                    dlg._progress_dialog = None
                elif getattr(dlg, '_progress_dialog', None) is not None:
                    try:
                        dlg._progress_dialog.canceled.disconnect()
                    except Exception:
                        pass
                    dlg._progress_dialog.close()
                    dlg._progress_dialog = None
            except Exception:
                pass
            _safe_release_preview_worker(dlg)
            _set_primary_idle()
            try:
                dlg._preview_worker_slot = None
                dlg._preview_worker_done_slot = None
            except Exception:
                pass
            if getattr(dlg, '_close_pending', False):
                try:
                    dlg.close()
                except Exception:
                    pass
                return
            try:
                if bool(getattr(dlg, '_pending_apply', False)):
                    dlg._pending_apply = False
                    fi2 = int(getattr(dlg, '_pending_apply_frame_idx', 0) or 0)
                    fs2 = dict(getattr(dlg, '_pending_apply_settings', {}) or {})
                    sig2 = str(getattr(dlg, '_pending_apply_settings_sig', '') or _settings_signature_for(fs2))
                    src2 = tuple(getattr(dlg, '_pending_apply_source_key', _selected_source()))
                    _start_worker(fi2, settings_snapshot=fs2, settings_sig=sig2, source_key=src2)
                    return
            except Exception:
                pass
        dlg._preview_worker_slot = on_result_ready
        dlg._preview_worker_done_slot = on_thread_finished
        worker.result_ready.connect(on_result_ready, Qt.ConnectionType.QueuedConnection)
        worker.result_ready.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(on_thread_finished)
        thread.start()

    def _foci_pass_fail_cached(frame_idx: int, source_key: tuple, fs: dict, *, mask_2d: np.ndarray, foci_plane: Optional[np.ndarray], frame_rows: list[dict], accepted_cell_labels: Optional[set[int]]=None) -> tuple[list[dict], list[dict]]:
        cache = getattr(dlg, '_foci_pass_fail_cache', None)
        if not isinstance(cache, FociPassFailCache):
            cache = FociPassFailCache(max_size=48)
            dlg._foci_pass_fail_cache = cache
        foci_src = _foci_source_for_source_key(source_key)
        ch_key = foci_channel_key_for_path(foci_src) if foci_src is not None else '_default'
        ch_sig = foci_channel_settings_signature(_foci_channel_settings_for_key(ch_key))
        mpk = getattr(dlg, '_preview_mask_path_key', None)
        key = foci_pass_fail_cache_key(int(frame_idx), str(mpk or ''), ch_sig, accepted_cell_labels, ch_key)
        hit = cache.lookup(key)
        if hit is not None:
            return hit

        def _compute() -> tuple[list[dict], list[dict]]:
            try:
                return compute_foci_pass_fail_for_frame(frame_rows, dict(fs or {}), mask_2d=mask_2d, foci_plane=foci_plane, accepted_cell_labels=accepted_cell_labels)
            except Exception:
                return ([], [])
        return cache.get_or_compute(key, _compute)

    def _update_foci_table_live() -> None:
        if dlg is None:
            return
        show = bool(w_show_foci_table.isChecked())
        foci_table_section.setVisible(show and bool(foci_sources))
        if not show:
            w_show_foci_table.setVisible(bool(foci_sources))
            return
        w_show_foci_table.setVisible(bool(foci_sources))
        fid = int(getattr(dlg.state(), 'frame_idx', 0) or 0)
        src_key = _source_key_from_state(dlg.state(), _selected_source)
        last = getattr(dlg, '_last_preview', {}) or {}
        foci_src = _foci_source_for_source_key(src_key)
        frame_rows = _rows_for_frame(_load_rows_for_source(foci_src), fid)
        try:
            ui_sync = getattr(dlg, '_pf_ui_to_settings', None)
            if callable(ui_sync):
                ui_sync()
        except Exception:
            pass
        try:
            fs = dict(tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {})
        except Exception:
            fs = {}
        mpk = getattr(dlg, '_preview_mask_path_key', None)
        if is_filter_apply_cache_hit(last, fid, mask_path_key=mpk) and last.get('mask_2d') is not None:
            mask_2d = np.asarray(last.get('mask_2d'), dtype=np.int32)
        else:
            mask_2d = np.asarray(_preview_mask_stack()[fid], dtype=np.int32)
        foci_plane = _get_foci_plane(fid, foci_src)
        accepted_cells = accepted_cell_labels_for_foci(last.get('accepted') if is_filter_apply_cache_hit(last, fid, mask_path_key=mpk) else None, cell_filter_applied=is_filter_apply_cache_hit(last, fid, mask_path_key=mpk))
        ch_fs = _foci_settings_for_source(foci_src)
        if is_filter_apply_cache_hit(last, fid, mask_path_key=mpk):
            ch_map = last.get('foci_channels')
            ch_key = foci_channel_key_for_path(foci_src) if foci_src is not None else ''
            if isinstance(ch_map, dict) and ch_key and (ch_key in ch_map):
                pass_peaks = list(ch_map[ch_key].get('pass_peaks') or [])
                fail_peaks = list(ch_map[ch_key].get('fail_peaks') or [])
            elif last.get('pass_peaks') is not None:
                pass_peaks = list(last.get('pass_peaks') or [])
                fail_peaks = list(last.get('fail_peaks') or [])
            else:
                (pass_peaks, fail_peaks) = ([], [])
            if accepted_cells:
                (pass_peaks, fail_peaks) = sync_foci_peaks_with_accepted_cells(pass_peaks, fail_peaks, mask_2d=mask_2d, accepted_cell_labels=accepted_cells, settings=ch_fs)
        elif foci_plane is not None and frame_rows:
            (pass_peaks, fail_peaks) = _foci_pass_fail_cached(fid, src_key, ch_fs, mask_2d=mask_2d, foci_plane=foci_plane, frame_rows=frame_rows, accepted_cell_labels=accepted_cells)
        else:
            try:
                filtered = filter_foci_csv_rows(frame_rows, ch_fs, mask_2d=mask_2d)
            except Exception:
                filtered = []
            pass_peaks = list(filtered)
            fail_keys = {(int(p.get('y', 0)), int(p.get('x', 0))) for p in pass_peaks}
            fail_peaks = []
            for r in frame_rows:
                try:
                    key = (int(float(r.get('y', 0) or 0)), int(float(r.get('x', 0) or 0)))
                except Exception:
                    continue
                if key not in fail_keys:
                    fail_peaks.append(dict(r))
        sel = getattr(dlg, '_highlight_foci_yx', None)
        try:
            cap = 2500
            if len(frame_rows) > cap:
                frame_rows = list(frame_rows)[:cap]
        except Exception:
            pass
        populate_foci_preview_table(foci_table, frame_rows=frame_rows, pass_peaks=pass_peaks, fail_peaks=fail_peaks, selected_yx=sel if isinstance(sel, tuple) else None, mask_2d=mask_2d, foci_plane=foci_plane, foci_settings=ch_fs)
        try:
            if not bool(getattr(dlg, '_foci_table_fitted_once', False)):
                dlg._foci_table_fitted_once = True
                QTimer.singleShot(0, lambda : fit_foci_table_columns(foci_table))
        except Exception:
            QTimer.singleShot(0, lambda : fit_foci_table_columns(foci_table))

    def _preview_display_xy(y: float, x: float, frame_shape: tuple[int, ...]) -> tuple[float, float]:
        """Map mask (y, x) to coordinates of the scaled preview pixmap on the canvas."""
        from bactomate.gui.image_utils import PREVIEW_MAX_SIZE
        (h, w) = (int(frame_shape[0]), int(frame_shape[1]))
        if max(h, w) <= PREVIEW_MAX_SIZE:
            return (float(x), float(y))
        scale = PREVIEW_MAX_SIZE / max(h, w)
        return (float(x) * scale, float(y) * scale)

    def _refresh_foci_highlight_only() -> None:
        """Redraw only the selected-focus highlight (table double-click); avoid full preview render."""
        if dlg is None:
            return
        base = getattr(dlg, '_preview_overlay_base', None)
        if base is None:
            dlg.refresh()
            return
        highlight_yx = getattr(dlg, '_highlight_foci_yx', None)
        if not isinstance(highlight_yx, tuple):
            return
        state = dlg.state()
        fid = int(getattr(state, 'frame_idx', 0) or 0)
        src_key = _source_key_from_state(state, _selected_source)
        foci_src = _foci_source_for_source_key(src_key)
        mpk = getattr(dlg, '_preview_mask_path_key', None)
        last = getattr(dlg, '_last_preview', {}) or {}
        if is_filter_apply_cache_hit(last, fid, mask_path_key=mpk) and last.get('mask_2d') is not None:
            mask_2d = np.asarray(last.get('mask_2d'), dtype=np.int32)
        else:
            try:
                mask_2d = np.asarray(_preview_mask_stack()[fid], dtype=np.int32)
            except Exception:
                dlg.refresh()
                return
        foci_plane = _get_foci_plane(fid, foci_src)
        if foci_plane is None and is_filter_apply_cache_hit(last, fid, mask_path_key=mpk):
            foci_plane = last.get('foci_plane')
        overlay = np.asarray(base).copy()
        try:
            ch_fs = _foci_settings_for_source(foci_src) if foci_src is not None else dict(tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {})
            _apply_foci_highlight_contour(overlay, mask_2d, np.asarray(foci_plane) if foci_plane is not None else None, highlight_yx, ch_fs)
        except Exception:
            pass
        from bactomate.gui.image_utils import to_qpixmap
        pm = to_qpixmap(overlay)
        dlg.canvas.set_pixmap(pm, (int(overlay.shape[1]), int(overlay.shape[0])), reset_view=False)

    def _focus_foci_from_table(row: int, _col: int) -> None:
        if row < 0 or dlg is None:
            return
        y_item = foci_table.item(row, 2)
        if y_item is None:
            return
        (y, x) = (None, None)
        data = y_item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            try:
                (y, x) = (int(data[0]), int(data[1]))
            except (TypeError, ValueError):
                pass
        if y is None or x is None:
            x_item = foci_table.item(row, 3)
            if x_item is None:
                return
            try:
                y = int(y_item.text())
                x = int(x_item.text())
            except (TypeError, ValueError):
                return
        dlg._highlight_foci_yx = (int(y), int(x))
        try:
            fid = int(getattr(dlg.state(), 'frame_idx', 0) or 0)
            shape = np.asarray(_preview_mask_stack()[fid]).shape
            (dx, dy) = _preview_display_xy(float(y), float(x), shape)
            dlg.canvas.center_on_image_point(dx, dy)
        except Exception:
            pass
        _refresh_foci_highlight_only()
        try:
            foci_table.selectRow(int(row))
            id_item = foci_table.item(int(row), 2)
            if id_item is not None:
                foci_table.scrollToItem(id_item)
        except Exception:
            pass

    def apply_current() -> None:
        try:
            _ui_to_settings()
        except Exception:
            pass
        idx = int(frame_spin.value()) - 1
        try:
            fs = tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}
        except Exception:
            fs = {}
        sig = _settings_signature_for(fs)
        src = tuple(_selected_source())
        plan = _plan_incremental_apply(idx, dict(fs))
        try:
            if plan.get('cell_changed'):
                _invalidate_foci_pass_fail_cache(idx, None)
            else:
                _invalidate_foci_pass_fail_cache(idx, set(plan.get('channels_to_score') or []))
            _clear_preview_layer_caches(dlg)
            dlg.set_frame_idx(idx)
            dlg.set_source_key(src)
        except Exception:
            pass
        if plan.get('reuse_cells') and (not plan.get('jobs')):
            last = getattr(dlg, '_last_preview', None) or {}
            accepted = set((int(x) for x in last.get('accepted') or []))
            rejected = set((int(x) for x in last.get('rejected') or []))
            n_obj = last.get('n_obj')
            from bactomate.gui.filter_preview_utils import format_incremental_apply_plan
            prog = FilterApplyProgressDialog(dlg)
            prog.set_progress(100, 'Complete', format_incremental_apply_plan(plan))
            prog.show()
            try:
                dlg._progress_dialog = prog
            except Exception:
                pass
            _commit_filter_apply_result(frame_idx=idx, accepted=accepted, rejected=rejected, n_obj=n_obj, foci_channels_last=dict(plan.get('prefilled') or {}), request_settings_sig=sig, cell_settings_sig=str(plan.get('cell_sig') or ''), foci_channel_sigs=_foci_channel_sigs_for_sources(), request_source_key=src)
            try:
                dlg.set_extra('filter_cell_recomputing', False)
            except Exception:
                pass
            QTimer.singleShot(250, prog.close)
            try:
                dlg._progress_dialog = None
            except Exception:
                pass
            return
        bundle = _build_foci_apply_bundle(idx, dict(fs), incremental_plan=plan, request_source_key=src)
        _start_worker(idx, settings_snapshot=dict(fs), settings_sig=sig, source_key=src, foci_apply_bundle=bundle)

    def _preview_mask_stack() -> np.ndarray:
        stack = getattr(dlg, '_preview_masks', None)
        if stack is None:
            return masks
        return np.asarray(stack)

    def _render(state) -> RenderResult:
        last = getattr(dlg, '_last_preview', None) or {}
        mask_stack = _preview_mask_stack()
        mask_path_key = getattr(dlg, '_preview_mask_path_key', None)
        current_source_key = _source_key_from_state(state, _selected_source)
        bg = _get_bg_frame(state.frame_idx, current_source_key)
        lut_frame = np.asarray(bg) if bg is not None else np.zeros(mask_stack.shape[1:3], dtype=np.float32)
        bg_u8 = _apply_lut_to_bg(bg, state.vmin, state.vmax)
        apply_busy = bool(state.extra.get('filter_apply_busy'))
        cell_filter_applied = is_filter_apply_cache_hit(last, int(state.frame_idx), mask_path_key=mask_path_key)
        preserve_foci_while_busy = bool(apply_busy and cell_filter_applied and (not bool(state.extra.get('filter_cell_recomputing'))))
        if cell_filter_applied:
            mask_2d = np.asarray(last.get('mask_2d'), dtype=np.int32)
            accepted = set(last.get('accepted') or [])
            rejected = set(last.get('rejected') or [])
            info_text = str(last.get('info_text') or '')
        else:
            mask_2d = np.asarray(mask_stack[int(state.frame_idx)], dtype=np.int32)
            labels = mask_2d[mask_2d > 0]
            accepted = set(np.unique(labels).astype(np.int32).tolist()) if labels.size else set()
            rejected = set()
            info_text = f'Unfiltered cells: {len(accepted)} shown. Click Apply current filters to apply cell and foci filters.'
        try:
            foci_fs = tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}
        except Exception:
            foci_fs = {}
        accepted_for_foci_gating = accepted_cell_labels_for_foci(accepted, cell_filter_applied=cell_filter_applied)
        overlay_only = bool(state.extra.get('overlay_only'))
        nav_only = bool(state.extra.get('nav_only'))
        lut_only = bool(state.extra.get('lut_only'))
        reuse_layers = bool(lut_only or overlay_only or nav_only)
        show_cells = _effective_show_cells(dlg, state)
        show_foci = _effective_show_foci(dlg, state, foci_overlay_available=bool(foci_overlay_available))
        state.extra['show_cells'] = bool(show_cells)
        state.extra['show_foci'] = bool(show_foci)
        base_cache_key = _base_layer_cache_key(int(state.frame_idx), current_source_key, mask_path_key, state.vmin, state.vmax, accepted, rejected)
        pass_peaks: list[dict] = []
        fail_peaks: list[dict] = []
        foci_plane = None
        foci_src = _foci_source_for_source_key(current_source_key)
        frame_rows_foci: list[dict] = []
        base_cache = getattr(dlg, '_preview_base_cache', None)
        overlay_fast = bool(overlay_only and isinstance(base_cache, dict) and (base_cache.get('key') == base_cache_key) and (base_cache.get('with_cells' if show_cells else 'no_cells') is not None))
        light_foci_load = bool((overlay_fast or nav_only) and show_foci and (foci_src is not None))
        skip_foci_stats_recompute = bool((overlay_only or nav_only) and (overlay_fast or (cell_filter_applied and (last.get('foci_channels') or last.get('pass_peaks') is not None))))
        if skip_foci_stats_recompute and cell_filter_applied:
            pass_peaks = list(last.get('pass_peaks') or [])
            fail_peaks = list(last.get('fail_peaks') or [])
            foci_plane = last.get('foci_plane')
        elif light_foci_load or (show_foci and foci_src is not None):
            if not frame_rows_foci:
                frame_rows_foci = _rows_for_frame(_load_rows_for_source(foci_src), int(state.frame_idx))
            if foci_plane is None:
                foci_plane = _get_foci_plane(int(state.frame_idx), foci_src)
                if foci_plane is None and cell_filter_applied:
                    foci_plane = last.get('foci_plane')
            raw_n = len(frame_rows_foci)
            if preserve_foci_while_busy and last.get('pass_peaks') is not None:
                pass_peaks = list(last.get('pass_peaks') or [])
                fail_peaks = list(last.get('fail_peaks') or [])
                info_text = f'{info_text}  |  Foci(frame): raw {raw_n} / pass {len(pass_peaks)} / fail {len(fail_peaks)}'.strip()
            elif apply_busy:
                pass_peaks = []
                fail_peaks = []
                if bool(state.extra.get('filter_cell_recomputing')):
                    info_text = f'{info_text}  |  Updating cells — foci will rescore when cells finish…'.strip()
                else:
                    info_text = f'{info_text}  |  Applying filters…'.strip()
            elif cell_filter_applied and last.get('pass_peaks') is not None:
                pass_peaks = list(last.get('pass_peaks') or [])
                fail_peaks = list(last.get('fail_peaks') or [])
                info_text = f'{info_text}  |  Foci(frame): raw {raw_n} / pass {len(pass_peaks)} / fail {len(fail_peaks)}'.strip()
            elif foci_plane is not None and frame_rows_foci:
                (pass_peaks, fail_peaks) = _foci_pass_fail_cached(int(state.frame_idx), current_source_key, dict(foci_fs or {}), mask_2d=mask_2d, foci_plane=foci_plane, frame_rows=frame_rows_foci, accepted_cell_labels=accepted_for_foci_gating)
                info_text = f'{info_text}  |  Foci(frame): raw {raw_n} / pass {len(pass_peaks)} / fail {len(fail_peaks)}'.strip()
            else:
                pass_peaks = _peaks_from_csv_rows(frame_rows_foci)
                fail_peaks = []
                info_text = f'{info_text}  |  Foci(frame): {raw_n} detected (apply filters for pass/fail)'.strip()
        if cell_filter_applied and accepted_for_foci_gating and (pass_peaks or fail_peaks):
            (pass_peaks, fail_peaks) = sync_foci_peaks_with_accepted_cells(pass_peaks, fail_peaks, mask_2d=mask_2d, accepted_cell_labels=accepted_for_foci_gating, settings=dict(foci_fs or {}))
        if show_foci and (pass_peaks or fail_peaks):
            preview_total = len(pass_peaks) + len(fail_peaks)
            preview_cap = max(0, int(DEFAULT_FOCI_PREVIEW_RENDER_CAP))
            if preview_cap > 0 and preview_total > preview_cap:
                info_text = f'{info_text}  |  Foci preview: {preview_cap} / {preview_total} drawn'.strip()
        highlight_yx = getattr(dlg, '_highlight_foci_yx', None)
        foci_after_apply = bool(cell_filter_applied)
        foci_sig = _settings_signature_for(dict(foci_fs or {}))
        foci_channels_parent_key = _foci_channels_layer_parent_key(int(state.frame_idx), foci_sig, draw_blobs=bool(foci_after_apply), mask_path_key=mask_path_key, accepted_cell_labels=accepted_for_foci_gating)

        def _draw_peaks_on_rgb(target_rgb: np.ndarray, peaks: list[dict], color: tuple[int, int, int], *, radius: int=4, cap: Optional[int]=None) -> None:
            draw_peak_rings_on_rgb(target_rgb, peaks, color, mask_shape=tuple(mask_2d.shape[:2]), radius=radius, cap=cap)

        def _foci_preview_draw_cap() -> int:
            return max(0, int(DEFAULT_FOCI_PREVIEW_RENDER_CAP))

        def _cap_peaks_for_preview(peaks: list[dict]) -> tuple[list[dict], int, int]:
            total = len(peaks or [])
            cap = _foci_preview_draw_cap()
            if cap <= 0 or total <= cap:
                return (list(peaks or []), total, total)
            return (list(peaks[:cap]), cap, total)

        def _foci_channel_should_draw(si: int, fp: Path) -> bool:
            nav = getattr(dlg, '_nav_ctrl', None)
            if nav is not None and nav.foci_channel_overlay_visible():
                vis = nav.foci_channel_overlay_visible()
                return si < len(vis) and bool(vis[si])
            if not show_foci:
                return False
            vp = _image_path_for_source_key(current_source_key)
            if not _view_path_is_foci(vp):
                return True
            return _foci_index_for_path(vp) == int(si)

        def _build_foci_layer_only() -> tuple[np.ndarray, np.ndarray]:
            """Merge per-channel foci rasters; channel blobs are cached across visibility toggles."""
            (_, scaled_mask) = scale_for_preview(bg_u8, mask_2d)
            (hs, ws) = (int(scaled_mask.shape[0]), int(scaled_mask.shape[1]))
            layer_rgb = np.zeros((hs, ws, 3), dtype=np.uint8)
            layer_mask = np.zeros((hs, ws), dtype=bool)
            fid = int(state.frame_idx)
            last_ch_map = last.get('foci_channels') if cell_filter_applied else None
            overlay_sources = foci_overlay_sources if foci_overlay_sources else list(foci_sources)
            ch_layers_cache = getattr(dlg, '_foci_channels_layer_cache', None)
            if not isinstance(ch_layers_cache, dict) or ch_layers_cache.get('key') != foci_channels_parent_key:
                ch_layers_cache = {'key': foci_channels_parent_key, 'channels': {}}
                dlg._foci_channels_layer_cache = ch_layers_cache
            channels_store: dict = ch_layers_cache.setdefault('channels', {})
            for (si, fp) in enumerate(overlay_sources):
                ch_key = foci_channel_key_for_path(fp)
                if ch_key in channels_store:
                    continue
                ch_rows = _rows_for_frame(_load_rows_for_source(fp), fid)
                if not ch_rows:
                    continue
                ch_color = _FOCI_CHANNEL_RGB[si % len(_FOCI_CHANNEL_RGB)]
                ch_fs = _foci_settings_for_source(fp)
                ch_layer_rgb = np.zeros((hs, ws, 3), dtype=np.uint8)
                if foci_after_apply:
                    ch_plane = _get_foci_plane(fid, fp)
                    if ch_plane is None:
                        continue
                    if isinstance(last_ch_map, dict) and ch_key in last_ch_map:
                        ch_pass = list(last_ch_map[ch_key].get('pass_peaks') or [])
                        ch_fail = list(last_ch_map[ch_key].get('fail_peaks') or [])
                        if accepted_for_foci_gating:
                            (ch_pass, ch_fail) = sync_foci_peaks_with_accepted_cells(ch_pass, ch_fail, mask_2d=mask_2d, accepted_cell_labels=accepted_for_foci_gating, settings=dict(ch_fs))
                    elif isinstance(last_ch_map, dict):
                        (ch_pass, ch_fail) = ([], [])
                    else:
                        (ch_pass, ch_fail) = _foci_pass_fail_cached(fid, ('file', -1), dict(ch_fs), mask_2d=mask_2d, foci_plane=ch_plane, frame_rows=ch_rows, accepted_cell_labels=accepted_cell_labels_for_foci(accepted, cell_filter_applied=cell_filter_applied))
                    ch_plane_arr = np.asarray(ch_plane)
                    (pass_draw, pass_drawn, pass_total) = _cap_peaks_for_preview(ch_pass)
                    (fail_draw, fail_drawn, fail_total) = _cap_peaks_for_preview(ch_fail)
                    missed_pass = draw_foci_blob_contours_on_rgb(ch_layer_rgb, mask_2d, pass_draw, ch_plane_arr, dict(ch_fs), ch_color, cap=_foci_preview_draw_cap())
                    if missed_pass:
                        _draw_peaks_on_rgb(ch_layer_rgb, missed_pass, ch_color, radius=3)
                    missed_fail = draw_foci_blob_contours_on_rgb(ch_layer_rgb, mask_2d, fail_draw, ch_plane_arr, dict(ch_fs), _FOCI_FAIL_RGB, cap=_foci_preview_draw_cap())
                    if missed_fail:
                        _draw_peaks_on_rgb(ch_layer_rgb, missed_fail, _FOCI_FAIL_RGB, radius=3)
                    ch_mask = np.any(np.asarray(ch_layer_rgb) > 12, axis=-1)
                    channels_store[ch_key] = {'layer': ch_layer_rgb, 'mask': ch_mask, 'drawn': int(pass_drawn + fail_drawn), 'total': int(pass_total + fail_total)}
                    continue
                raw_peaks = _peaks_from_csv_rows(ch_rows)
                (draw_peaks, drawn_n, total_n) = _cap_peaks_for_preview(raw_peaks)
                _draw_peaks_on_rgb(ch_layer_rgb, draw_peaks, ch_color, radius=4, cap=_foci_preview_draw_cap())
                ch_mask = np.any(np.asarray(ch_layer_rgb) > 12, axis=-1)
                channels_store[ch_key] = {'layer': ch_layer_rgb, 'mask': ch_mask, 'drawn': int(drawn_n), 'total': int(total_n)}
            for (si, fp) in enumerate(overlay_sources):
                if not _foci_channel_should_draw(si, fp):
                    continue
                ch_key = foci_channel_key_for_path(fp)
                entry = channels_store.get(ch_key)
                if not isinstance(entry, dict):
                    continue
                ch_mask = np.asarray(entry.get('mask'), dtype=bool)
                if not np.any(ch_mask):
                    continue
                ch_layer = np.asarray(entry.get('layer'), dtype=np.uint8)
                layer_rgb[ch_mask] = ch_layer[ch_mask]
                layer_mask[ch_mask] = True
            return (layer_rgb, layer_mask)

        def _composite_cached_foci(base_rgb: np.ndarray) -> np.ndarray:
            if not show_foci:
                return np.asarray(base_rgb, dtype=np.uint8)
            comp_key = (
                base_cache_key,
                foci_channels_parent_key,
                _foci_channel_overlay_vis_key(dlg),
                bool(show_cells),
                bool(show_foci),
            )
            comp_cache = getattr(dlg, '_foci_composited_cache', None)
            if isinstance(comp_cache, dict) and comp_cache.get('key') == comp_key:
                cached_rgb = comp_cache.get('rgb')
                if isinstance(cached_rgb, np.ndarray) and cached_rgb.shape == np.asarray(base_rgb).shape:
                    return np.asarray(cached_rgb, dtype=np.uint8)
            (layer_rgb, layer_mask) = _build_foci_layer_only()
            result = _composite_foci_layer(base_rgb, layer_rgb, layer_mask)
            try:
                dlg._foci_composited_cache = {'key': comp_key, 'rgb': np.asarray(result).copy()}
            except Exception:
                dlg._foci_composited_cache = None
            return result

        def _finalize_overlay(overlay: np.ndarray) -> RenderResult:
            try:
                dlg._preview_overlay_base = np.asarray(overlay).copy()
            except Exception:
                dlg._preview_overlay_base = None
            contour_info = ''
            try:
                from bactomate.gui.fi_contour_filter_preview import paint_analyze_fi_contour_if_active
                suffix = paint_analyze_fi_contour_if_active(overlay, mask_2d, set(accepted) if accepted else set(), dlg, frame_idx=int(state.frame_idx))
                if suffix:
                    contour_info = str(suffix)
            except Exception:
                from bactomate.logging_config import get_logger
                get_logger().exception('Contour FI preview overlay failed')
            if isinstance(highlight_yx, tuple):
                try:
                    _apply_foci_highlight_contour(overlay, mask_2d, np.asarray(foci_plane) if foci_plane is not None else None, highlight_yx, dict(foci_fs or {}))
                except Exception:
                    pass
            pm = to_qpixmap(overlay)
            hover_data = None
            _hover_overlay_sources = foci_overlay_sources if foci_overlay_sources else list(foci_sources)
            _hover_last_ch_map = last.get('foci_channels') if cell_filter_applied else None

            def _hover_peaks_for_channel(ch_key: str, fp: Path) -> tuple[list[dict], list[dict]]:
                if isinstance(_hover_last_ch_map, dict) and ch_key in _hover_last_ch_map:
                    payload = _hover_last_ch_map[ch_key]
                    if isinstance(payload, dict):
                        ch_pass = list(payload.get('pass_peaks') or [])
                        ch_fail = list(payload.get('fail_peaks') or [])
                        if accepted_for_foci_gating:
                            ch_fs_h = _foci_settings_for_source(fp)
                            (ch_pass, ch_fail) = sync_foci_peaks_with_accepted_cells(ch_pass, ch_fail, mask_2d=mask_2d, accepted_cell_labels=accepted_for_foci_gating, settings=dict(ch_fs_h))
                        return (ch_pass, ch_fail)
                if reuse_layers and isinstance(_hover_last_ch_map, dict):
                    return ([], [])
                ch_rows = _rows_for_frame(_load_rows_for_source(fp), int(state.frame_idx))
                if not ch_rows:
                    return ([], [])
                if reuse_layers or not foci_after_apply:
                    return (_peaks_from_csv_rows(ch_rows), [])
                ch_plane = _get_foci_plane(int(state.frame_idx), fp)
                if ch_plane is None:
                    return (_peaks_from_csv_rows(ch_rows), [])
                ch_fs = _foci_settings_for_source(fp)
                return _foci_pass_fail_cached(int(state.frame_idx), ('file', -1), dict(ch_fs), mask_2d=mask_2d, foci_plane=ch_plane, frame_rows=ch_rows, accepted_cell_labels=accepted_cell_labels_for_foci(accepted, cell_filter_applied=cell_filter_applied))
            (hover_pass, hover_fail) = collect_visible_foci_peaks_for_hover(overlay_sources=_hover_overlay_sources, channel_visible=_foci_channel_should_draw, channel_peaks=_hover_peaks_for_channel)
            if not hover_pass and (not hover_fail):
                (hover_pass, hover_fail) = (list(pass_peaks), list(fail_peaks))
            hover_builder = None
            if show_foci and (hover_pass or hover_fail):
                hover_key = _foci_hover_cache_key(int(state.frame_idx), foci_sig, draw_blobs=bool(foci_after_apply), mask_shape=mask_2d.shape[:2], display_shape=overlay.shape[:2], mask_path_key=mask_path_key, channel_overlay_vis=_foci_channel_overlay_vis_key(dlg), accepted_cell_labels=accepted_for_foci_gating)

                def _builder(_dlg=dlg, _key=hover_key, _pass=list(hover_pass), _fail=list(hover_fail), _mask_shape=tuple(mask_2d.shape[:2]), _disp_shape=tuple(overlay.shape[:2]), _fs=dict(foci_fs or {}), _mask=np.asarray(mask_2d)):
                    return _cached_foci_hover_data(_dlg, cache_key=_key, pass_peaks=_pass, fail_peaks=_fail, mask_shape=_mask_shape, display_shape=_disp_shape, settings=_fs, mask_2d=_mask)
                hover_builder = _builder
            info_out = info_text
            if contour_info:
                info_out = f'{info_text}  |  {contour_info}'.strip()
            return RenderResult(pixmap=pm, image_size=(overlay.shape[1], overlay.shape[0]), lut_frame=lut_frame, title_text=f'Filter Effects Preview — {mask_path.name}', info_text=info_out, foci_hover_data=None, foci_hover_builder=hover_builder)
        if reuse_layers and overlay_fast:
            cached_base = base_cache.get('with_cells' if show_cells else 'no_cells')
            return _finalize_overlay(_composite_cached_foci(np.asarray(cached_base)))
        if reuse_layers and isinstance(base_cache, dict) and (base_cache.get('key') == base_cache_key):
            cached_base = base_cache.get('with_cells' if show_cells else 'no_cells')
            if cached_base is not None:
                return _finalize_overlay(_composite_cached_foci(np.asarray(cached_base)))
        no_cells = _render_overlay(mask_2d, accepted, rejected, bg_u8, pass_peaks, fail_peaks, show_cells=False, show_foci=False, foci_filters_applied=bool(cell_filter_applied), skip_foci=True)
        with_cells = _render_overlay(mask_2d, accepted, rejected, bg_u8, pass_peaks, fail_peaks, show_cells=True, show_foci=False, foci_filters_applied=bool(cell_filter_applied), skip_foci=True)
        dlg._preview_base_cache = {'key': base_cache_key, 'no_cells': no_cells, 'with_cells': with_cells}
        base = with_cells if show_cells else no_cells
        return _finalize_overlay(_composite_cached_foci(base))

    def _disconnect_worker_finished() -> None:
        w = getattr(dlg, '_preview_worker', None)
        thread = getattr(dlg, '_preview_thread', None)
        s = getattr(dlg, '_preview_worker_slot', None)
        if w is not None and s is not None:
            try:
                w.result_ready.disconnect(s)
            except Exception:
                pass
            try:
                if thread is not None:
                    w.result_ready.disconnect(thread.quit)
            except Exception:
                pass
        ds = getattr(dlg, '_preview_worker_done_slot', None)
        if thread is not None and ds is not None:
            try:
                thread.finished.disconnect(ds)
            except Exception:
                pass
        _safe_release_preview_worker(dlg)

    def _is_running() -> bool:
        thread = getattr(dlg, '_preview_thread', None)
        return bool(thread is not None and thread.isRunning())

    def _wait(ms: int) -> bool:
        thread = getattr(dlg, '_preview_thread', None)
        if thread is None:
            return True
        try:
            return bool(thread.wait(int(ms)))
        except Exception:
            return True

    def _on_close_defer() -> None:
        write_gui_log_line('Preview', 'Filter preview: worker running; deferring close until finished.')
        dlg._close_pending = True

    def _request_stop() -> None:
        try:
            dlg._preview_request_id += 1
        except Exception:
            pass
        thread = getattr(dlg, '_preview_thread', None)
        if thread is not None:
            try:
                thread.requestInterruption()
                thread.quit()
            except Exception:
                pass
        try:
            if getattr(dlg, '_progress_dialog', None) is not None:
                try:
                    dlg._progress_dialog.canceled.disconnect()
                except Exception:
                    pass
                dlg._progress_dialog.close()
                dlg._progress_dialog = None
        except Exception:
            pass
        _on_close_defer()

    def _disconnect_and_restore() -> None:
        _disconnect_worker_finished()
    mount = getattr(tab, '_module_preview_mount', None)
    embed = mount is not None
    preview_surface = getattr(tab, '_preview_surface', None)
    seg_tab = _segmentation_tab_host(tab)
    if embed and seg_tab is not None:
        try:
            if preview_surface is not None:
                preview_surface.set_preview_context('filtering')
            seg_tab._dismiss_embedded_previews()
            QTimer.singleShot(0, seg_tab.refresh_filtering_candidate_list)
        except Exception:
            pass
    try:
        dlg_parent = mount if embed else tab.window() if hasattr(tab, 'window') and callable(tab.window) else tab
        if embed and preview_surface is not None:
            preview_surface.set_toolbar_widget(None)
        dlg = PreviewDialogBase(dlg_parent, title=f'Filter Effects Preview — {mask_path.name}', header_controls=nav_ctrl, navigation_controls=None, top_controls=settings_scroll, lut_expanded=False, primary_action_label='Apply current filters', primary_action_callback=apply_current, render_callback=_render, worker_guard=WorkerGuard(is_running=_is_running, wait=_wait, request_stop=_request_stop, disconnect=_disconnect_and_restore, on_close_while_running='defer_hide', warning_title='Filtering in progress', warning_text='Please wait for filtering to finish before closing the preview.'), minimum_size=(200, 200) if embed else (1140, 760), embedded=embed, show_close_button=not embed, hide_title_and_builtin_info=embed)
        dlg.set_primary_action(apply_current, label='Apply current filters', visible=False)
        reapply_btn.clicked.connect(apply_current)
        try:
            tab._filter_preview_dialog = dlg
        except Exception:
            pass
        try:
            seg_host = _segmentation_tab_host(tab)
            if seg_host is not None:
                at = getattr(seg_host, 'analyze_tab', None)
                if at is not None and hasattr(at, '_update_fi_contour_preview_button_state'):
                    at._update_fi_contour_preview_button_state()
        except Exception:
            pass
        dlg._close_pending = False
        dlg._preview_request_id = 0

        class _FilterPreviewResizeFilter(QObject):

            def eventFilter(self, obj, event) -> bool:
                if event.type() == QEvent.Type.Resize:
                    _sync_filter_settings_panel_height()
                return False
        try:
            dlg._filter_settings_resize_filter = _FilterPreviewResizeFilter(dlg)
            dlg.installEventFilter(dlg._filter_settings_resize_filter)
        except Exception:
            pass
        QTimer.singleShot(0, _sync_filter_settings_panel_height)
        dlg._bg_paths = bg_paths
        dlg._bg_cache = {}
        if isinstance(foci_stacks_preloaded, dict):
            for (k, stack) in foci_stacks_preloaded.items():
                try:
                    dlg._bg_cache[str(k)] = np.asarray(stack)
                except Exception:
                    pass
        dlg._foci_rows_cache = {}
        dlg._foci_pass_fail_cache = {}
        dlg._foci_layer_cache = {}
        dlg._foci_channels_layer_cache = {}
        dlg._preview_base_cache = {}
        dlg._foci_hover_cache = {}
        dlg._foci_composited_cache = {}
        dlg._last_preview = {}
        dlg._preview_overlay_base = None
        dlg._highlight_foci_yx = None
        dlg._preview_masks = masks
        dlg._preview_mask_path_key = _preview_mask_path_key(mask_path)
        _lut_debounce = QTimer(dlg)
        _lut_debounce.setSingleShot(True)
        _lut_pending: dict[str, float] = {'vmin': float('nan'), 'vmax': float('nan')}

        def _commit_lut_refresh() -> None:
            try:
                dlg.state().vmin = float(_lut_pending.get('vmin'))
                dlg.state().vmax = float(_lut_pending.get('vmax'))
            except Exception:
                return
            dlg.set_extra('lut_only', True)
            try:
                dlg.refresh()
            finally:
                dlg.set_extra('lut_only', False)
        _lut_debounce.timeout.connect(_commit_lut_refresh)

        def _on_filter_lut_changed(vmin: float, vmax: float) -> None:
            _lut_pending['vmin'] = float(vmin)
            _lut_pending['vmax'] = float(vmax)
            _lut_debounce.start(80)
        try:
            dlg.lut_widget.lut_changed.disconnect(dlg._on_lut_changed)
        except Exception:
            pass
        dlg.lut_widget.lut_changed.connect(_on_filter_lut_changed)

        def _sync_nav_state_only(*, reset_lut: bool=False) -> None:
            dlg.set_frame_idx(int(frame_spin.value()) - 1)
            dlg.set_source_key(_selected_source())
            if reset_lut:
                try:
                    dlg.state().vmin = None
                    dlg.state().vmax = None
                    dlg.lut_widget.set_lut(None, None)
                except Exception:
                    pass
            _clear_preview_base_caches(dlg)
            dlg.set_extra('nav_only', True)
            try:
                dlg.refresh()
            finally:
                dlg.set_extra('nav_only', False)
        frame_spin.valueChanged.connect(lambda _v: _sync_nav_state_only(reset_lut=False))

        def _on_image_combo_changed(*_) -> None:
            _sync_preview_pixel_size_from_metadata(_selected_image_path())
            _sync_nav_state_only(reset_lut=True)
        image_combo.currentIndexChanged.connect(_on_image_combo_changed)

        def _on_nav_changed(st: PreviewNavState, reason: str) -> None:
            if reason == 'source':
                try:
                    _ui_to_settings()
                except Exception:
                    pass
            try:
                if reason in ('frame', 'init'):
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(int(st.frame_idx) + 1)
                    frame_spin.blockSignals(False)
                if reason in ('source', 'init'):
                    key = st.source_key
                    idx = 0
                    for ii in range(int(image_combo.count())):
                        if image_combo.itemData(ii) == key:
                            idx = int(ii)
                            break
                    image_combo.blockSignals(True)
                    image_combo.setCurrentIndex(idx)
                    image_combo.blockSignals(False)
            except Exception:
                return
            try:
                dlg.set_frame_idx(int(st.frame_idx))
                if st.source_key is not None:
                    dlg.set_source_key(tuple(st.source_key))
                else:
                    dlg.set_source_key(_selected_source())
                if reason == 'frame':
                    _clear_preview_layer_caches(dlg)
                elif reason == 'source':
                    _clear_preview_base_caches(dlg)
                if reason == 'source':
                    try:
                        dlg.state().vmin = None
                        dlg.state().vmax = None
                        dlg.lut_widget.set_lut(None, None)
                    except Exception:
                        pass
                    try:
                        _settings_to_ui()
                        _sync_preview_pixel_size_from_metadata(_selected_image_path())
                        _sync_filter_panels_for_view()
                    except Exception:
                        pass
                dlg.set_extra('nav_only', True)
                try:
                    dlg.refresh()
                finally:
                    dlg.set_extra('nav_only', False)
                if bool(w_show_foci_table.isChecked()):
                    try:
                        _schedule_foci_table_refresh()
                    except Exception:
                        pass
            except Exception:
                pass
        nav_ctrl.nav_changed.connect(_on_nav_changed)
        try:
            dlg._nav_ctrl = nav_ctrl
        except Exception:
            pass

        def _refresh_foci_overlay_only() -> None:
            dlg.set_extra('overlay_only', True)
            try:
                dlg.refresh()
            finally:
                dlg.set_extra('overlay_only', False)

        def _sync_overlay_extras_from_nav() -> None:
            dlg.set_extra('show_cells', nav_ctrl.cell_overlay_visible())
            dlg.set_extra('show_foci', nav_ctrl.foci_overlay_visible())

        def _on_overlay_changed(key: str, checked: bool) -> None:
            dlg.set_extra(str(key), bool(checked))
            if str(key) == 'show_cells':
                _sync_overlay_extras_from_nav()
                _clear_preview_layer_caches(dlg)
                dlg.set_extra('overlay_only', False)
                dlg.set_extra('nav_only', False)
                dlg.refresh()
                return
            _sync_overlay_extras_from_nav()
            _clear_preview_base_caches(dlg)
            _refresh_foci_overlay_only()

        def _on_foci_channel_overlay_changed(_idx: int, _checked: bool) -> None:
            _sync_overlay_extras_from_nav()
            _clear_preview_base_caches(dlg)
            _refresh_foci_overlay_only()
        nav_ctrl.overlay_changed.connect(_on_overlay_changed)
        try:
            nav_ctrl.foci_channel_overlay_changed.connect(_on_foci_channel_overlay_changed)
        except Exception:
            pass
        _sync_overlay_extras_from_nav()
        dlg.set_frame_idx(0)
        dlg.set_source_key(_selected_source())
        _foci_table_debounce = QTimer(dlg)
        _foci_table_debounce.setSingleShot(True)

        def _schedule_foci_table_refresh() -> None:
            if not bool(w_show_foci_table.isChecked()):
                return
            _foci_table_debounce.start(120)
        _foci_table_debounce.timeout.connect(_update_foci_table_live)

        def _deferred_initial_refresh() -> None:
            try:
                _sync_filter_settings_panel_height()
                _sync_filter_panels_for_view()
                _scroll_settings_to_top()
                dlg.refresh()
            except Exception:
                pass
        QTimer.singleShot(0, _deferred_initial_refresh)

        def _ui_to_settings() -> None:
            if not hasattr(tab, 'set_filter_settings'):
                return
            for w in (w_pixel, w_min_ar, w_max_ar, w_min_ecc, w_min_sol, w_min_area, w_max_area, w_min_w, w_max_w, w_min_len, w_max_len, w_prob_thr, w_minint, w_ratio, w_minarea):
                try:
                    w.interpretText()
                except Exception:
                    pass

            def _opt0(w: QDoubleSpinBox) -> Optional[float]:
                try:
                    v = float(w.value())
                    return v if v > 0 else None
                except Exception:
                    return None
            tab.set_filter_settings({'pixel_size': float(w_pixel.value()), 'min_aspect_ratio': float(w_min_ar.value()), 'max_aspect_ratio': float(w_max_ar.value()), 'min_eccentricity': float(w_min_ecc.value()), 'min_solidity': float(w_min_sol.value()), 'min_area_um2': _opt0(w_min_area), 'max_area_um2': _opt0(w_max_area), 'min_width_um': _opt0(w_min_w), 'max_width_um': _opt0(w_max_w), 'min_length_um': _opt0(w_min_len), 'max_length_um': _opt0(w_max_len), 'exclude_border': bool(w_excl_border.isChecked())})
            ch_key = _active_foci_channel_key()
            ch_payload = {'foci_prob_enabled': bool(w_prob_en.isChecked()), 'foci_prob_thresh': float(w_prob_thr.value()), 'foci_min_int_enabled': bool(w_minint_en.isChecked()), 'foci_min_int': float(w_minint.value()), 'foci_per_cell_ratio_enabled': bool(w_ratio_en.isChecked()), 'foci_per_cell_ratio': float(w_ratio.value()), 'foci_exclude_outside_clear': bool(w_foci_excl_outside_clear.isChecked()), 'foci_exclude_outside_touching': bool(w_foci_excl_outside_touch.isChecked()), 'foci_exclude_outside_cells': bool(w_foci_excl_outside_clear.isChecked() and w_foci_excl_outside_touch.isChecked()), 'foci_blob_threshold_frac': float(w_blob_size.currentData() or 0.9), 'foci_blob_max_radius': int(w_blob_radius.value()), 'foci_min_area_px_enabled': bool(w_minarea_en.isChecked()), 'foci_min_area_px': int(w_minarea.value())}
            if hasattr(tab, 'set_foci_channel_settings'):
                tab.set_foci_channel_settings(ch_key, ch_payload)
            else:
                tab.set_filter_settings(ch_payload)

        def _settings_to_ui() -> None:
            fs = tab.get_filter_settings() if hasattr(tab, 'get_filter_settings') else {}
            ch_key = _active_foci_channel_key()
            try:
                if hasattr(tab, 'get_foci_channel_settings'):
                    fs_ch = tab.get_foci_channel_settings(ch_key)
                else:
                    fs_ch = resolve_foci_settings_for_source(dict(fs), None)
            except Exception:
                fs_ch = resolve_foci_settings_for_source(dict(fs), None)
            for w in (w_pixel, w_min_ar, w_max_ar, w_min_ecc, w_min_sol, w_min_area, w_max_area, w_min_w, w_max_w, w_min_len, w_max_len, w_prob_thr, w_minint, w_ratio, w_minarea, w_blob_radius):
                try:
                    w.blockSignals(True)
                except Exception:
                    pass
            for cb in (w_excl_border, w_prob_en, w_minint_en, w_ratio_en, w_foci_excl_outside_clear, w_foci_excl_outside_touch, w_minarea_en, w_show_foci_table):
                try:
                    cb.blockSignals(True)
                except Exception:
                    pass
            try:
                w_blob_size.blockSignals(True)
            except Exception:
                pass
            try:
                w_pixel.setValue(float(fs.get('pixel_size', 0.108) or 0.108))
                w_min_ar.setValue(float(fs.get('min_aspect_ratio', 1.5) or 1.5))
                w_max_ar.setValue(float(fs.get('max_aspect_ratio', 10.0) or 10.0))
                w_min_ecc.setValue(float(fs.get('min_eccentricity', 0.5) or 0.5))
                w_min_sol.setValue(float(fs.get('min_solidity', 0.7) or 0.7))
                w_min_area.setValue(float(fs.get('min_area_um2') or 0.0))
                w_max_area.setValue(float(fs.get('max_area_um2') or 0.0))
                w_min_w.setValue(float(fs.get('min_width_um') or 0.0))
                w_max_w.setValue(float(fs.get('max_width_um') or 0.0))
                w_min_len.setValue(float(fs.get('min_length_um') or 0.0))
                w_max_len.setValue(float(fs.get('max_length_um') or 0.0))
                w_excl_border.setChecked(bool(fs.get('exclude_border', False)))
                w_prob_en.setChecked(bool(fs_ch.get('foci_prob_enabled', False)))
                w_prob_thr.setValue(float(fs_ch.get('foci_prob_thresh', 0.5) or 0.5))
                w_minint_en.setChecked(bool(fs_ch.get('foci_min_int_enabled', False)))
                w_minint.setValue(float(fs_ch.get('foci_min_int', 0.0) or 0.0))
                w_ratio_en.setChecked(bool(fs_ch.get('foci_per_cell_ratio_enabled', False)))
                w_ratio.setValue(float(fs_ch.get('foci_per_cell_ratio', 2.0) or 2.0))
                _leg = bool(fs_ch.get('foci_exclude_outside_cells', False))
                w_foci_excl_outside_clear.setChecked(bool(fs_ch.get('foci_exclude_outside_clear', _leg)))
                w_foci_excl_outside_touch.setChecked(bool(fs_ch.get('foci_exclude_outside_touching', _leg)))
                w_minarea_en.setChecked(bool(fs_ch.get('foci_min_area_px_enabled', False)))
                w_minarea.setValue(max(1, int(fs_ch.get('foci_min_area_px', 1) or 1)))
                blob_frac = foci_blob_threshold_frac(fs_ch)
                blob_sel = 0
                for (i, (_lbl, frac)) in enumerate(FOCI_BLOB_SIZE_PRESETS):
                    if abs(float(frac) - blob_frac) < 1e-06:
                        blob_sel = i
                        break
                w_blob_size.setCurrentIndex(blob_sel)
                w_blob_radius.setValue(int(fs_ch.get('foci_blob_max_radius', 5) or 5))
            finally:
                for w in (w_pixel, w_min_ar, w_max_ar, w_min_ecc, w_min_sol, w_min_area, w_max_area, w_min_w, w_max_w, w_min_len, w_max_len, w_prob_thr, w_minint, w_ratio, w_minarea, w_blob_radius):
                    try:
                        w.blockSignals(False)
                    except Exception:
                        pass
                try:
                    w_blob_size.blockSignals(False)
                except Exception:
                    pass
                for cb in (w_excl_border, w_prob_en, w_minint_en, w_ratio_en, w_foci_excl_outside_clear, w_foci_excl_outside_touch, w_minarea_en, w_show_foci_table):
                    try:
                        cb.blockSignals(False)
                    except Exception:
                        pass
        from bactomate.gui.layout_helpers import wire_spin_activates_checkbox
        wire_spin_activates_checkbox(w_prob_thr, w_prob_en)
        wire_spin_activates_checkbox(w_minint, w_minint_en)
        wire_spin_activates_checkbox(w_ratio, w_ratio_en)
        wire_spin_activates_checkbox(w_minarea, w_minarea_en)
        try:

            def _on_show_foci_table_toggled(_checked: bool) -> None:
                try:
                    dlg._foci_table_fitted_once = False
                except Exception:
                    pass
                _update_foci_table_live()
                _sync_filter_settings_panel_height()
                if _checked:
                    QTimer.singleShot(0, lambda : _scroll_filter_section_into_view(foci_table_section))
            w_show_foci_table.toggled.connect(_on_show_foci_table_toggled)
        except Exception:
            pass
        try:
            foci_table.cellDoubleClicked.connect(_focus_foci_from_table)
        except Exception:
            pass
        try:

            def _on_foci_table_section_fit(expanded: bool) -> None:
                if expanded:
                    fit_foci_table_columns(foci_table)
            foci_table_section.toggled.connect(_on_foci_table_section_fit)
        except Exception:
            pass
        _settings_to_ui()
        _sync_preview_pixel_size_from_metadata(_selected_image_path())
        try:
            dlg._pf_apply_current = apply_current
            dlg._pf_ui_to_settings = _ui_to_settings
            dlg._pf_settings_to_ui = _settings_to_ui
            dlg._pf_spinboxes = {'pixel_size': w_pixel, 'min_aspect_ratio': w_min_ar, 'max_aspect_ratio': w_max_ar, 'min_eccentricity': w_min_ecc, 'min_solidity': w_min_sol, 'min_area_um2': w_min_area, 'max_area_um2': w_max_area, 'min_width_um': w_min_w, 'max_width_um': w_max_w, 'min_length_um': w_min_len, 'max_length_um': w_max_len, 'foci_prob_thresh': w_prob_thr, 'foci_min_int': w_minint, 'foci_per_cell_ratio': w_ratio, 'foci_min_area_px': w_minarea}
            dlg._pf_checkboxes = {'exclude_border': w_excl_border, 'foci_prob_enabled': w_prob_en, 'foci_min_int_enabled': w_minint_en, 'foci_per_cell_ratio_enabled': w_ratio_en, 'foci_exclude_outside_clear': w_foci_excl_outside_clear, 'foci_exclude_outside_touching': w_foci_excl_outside_touch, 'foci_min_area_px_enabled': w_minarea_en}
            dlg._pf_blob_combo = w_blob_size
            dlg._pf_blob_radius = w_blob_radius
            dlg._pf_show_foci_table = w_show_foci_table
            dlg._pf_foci_table = foci_table
            dlg._pf_update_foci_table = _update_foci_table_live
            dlg._pf_sync_settings_panel_height = _sync_filter_settings_panel_height
        except Exception:
            pass
        try:
            dlg.state().vmin = None
            dlg.state().vmax = None
            dlg.lut_widget.set_lut(None, None)
        except Exception:
            pass

        def _on_embedded_filter_preview_closed(_result: int=0) -> None:
            try:
                from bactomate.gui.fi_contour_filter_preview import clear_analyze_fi_contour_preview
                clear_analyze_fi_contour_preview(dlg)
            except Exception:
                pass
            try:
                tab._filter_preview_dialog = None
                tab._filter_preview_mask_path = None
            except Exception:
                pass
            if embed and preview_surface is not None:
                try:
                    preview_surface.clear_mount()
                    preview_surface.set_preview_context('segmentation')
                    preview_surface.set_toolbar_widget(None)
                except Exception:
                    pass
        if embed:
            dlg.finished.connect(_on_embedded_filter_preview_closed)
            prev = getattr(tab, '_filter_preview_dialog', None)
            if prev is not None and prev is not dlg:
                try:
                    prev.close()
                    prev.deleteLater()
                except Exception:
                    pass
            tab._filter_preview_dialog = dlg
            from bactomate.gui.module_shell import mount_embedded_preview
            mount_embedded_preview(mount, dlg, preview_surface=preview_surface)
            dlg.show()
        else:
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
    except Exception as e:
        _disconnect_and_restore()
        show_message_dialog(tab, 'Preview Error', f'Could not open filtering preview:\n{e}', kind='warning')
