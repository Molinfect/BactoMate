"""Contour-FI preview: dedicated mask picker + channel/LUT preview (Analyze tab)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import numpy as np
from bactomate.gui.widgets import show_message_dialog
ANALYZE_FI_CONTOUR_PREVIEW_ATTR = '_analyze_fi_contour_preview'

def auto_display_lut_range(plane: np.ndarray) -> tuple[float, float]:
    """
    Conservative auto-LUT for microscopy previews (avoids 0.01–99.99 wash-out on phase).

    Uses a modest percentile band so bright cells stay visible and contours remain readable.
    """
    flat = np.asarray(plane, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return (0.0, 1.0)
    (fmin, fmax) = (float(flat.min()), float(flat.max()))
    if fmax <= fmin:
        return (fmin, fmax + 1.0)
    try:
        lo = float(np.percentile(flat, 0.5))
        hi = float(np.percentile(flat, 99.5))
    except Exception:
        (lo, hi) = (fmin, fmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        (lo, hi) = (fmin, fmax)
    return (float(lo), float(hi))

def clear_analyze_fi_contour_preview(dlg: Any) -> None:
    try:
        setattr(dlg, ANALYZE_FI_CONTOUR_PREVIEW_ATTR, None)
    except Exception:
        pass

def accepted_cell_labels_from_mask(mask_2d: np.ndarray) -> set[int]:
    """All positive segmentation labels on the current frame."""
    m = np.asarray(mask_2d, dtype=np.int32)
    if m.size == 0:
        return set()
    labels = np.unique(m)
    return {int(x) for x in labels if int(x) > 0}

def _list_masks_for_contour_picker(analyze_tab: Any) -> list[Path]:
    raw = (getattr(analyze_tab, 'file_path', None) and analyze_tab.file_path.text() or '').strip()
    if not raw:
        return []
    p = Path(raw)
    if p.is_file():
        return [p] if p.exists() else []
    try:
        from bactomate.gui.artifact_paths import dedupe_mask_paths_prefer_filtered
        paths = [Path(m) for m in analyze_tab._get_mask_files_for_feedback() or [] if Path(m).is_file()]
        return dedupe_mask_paths_prefer_filtered(paths)
    except Exception:
        return []

def _pick_mask_for_contour_preview(analyze_tab: Any) -> Path | None:
    from bactomate.gui.filter_preview_utils import pick_mask_path_for_preview
    raw = (getattr(analyze_tab, 'file_path', None) and analyze_tab.file_path.text() or '').strip()
    if raw:
        picked = pick_mask_path_for_preview(analyze_tab, raw)
        if picked is not None and Path(picked).is_file():
            return Path(picked)
    masks = _list_masks_for_contour_picker(analyze_tab)
    if not masks:
        return None
    if len(masks) == 1:
        return Path(masks[0])
    from bactomate.gui.widgets import show_file_selection_dialog
    result = show_file_selection_dialog(analyze_tab, masks, title='Select a mask', message='Select a mask file to preview Contour FI on the input and FI channels.')
    if result is None:
        return None
    selected = result[0] if isinstance(result, tuple) else result
    if not selected:
        return None
    return Path(selected)

def discover_background_paths_for_mask(tab: Any, mask_path: Path) -> list[Path]:
    """Segmentation input (phase), FI, and foci stacks — same discovery as filter preview."""
    from bactomate.gui.filter_preview_utils import collect_background_paths_for_mask, file_identity_key
    mask_path = Path(mask_path)
    paths = collect_background_paths_for_mask(mask_path)
    phase_first: list[Path] = []
    rest: list[Path] = []
    phase_keys: set[str] = set()
    try:
        from bactomate.gui.filter_preview_utils import phase_stack_candidates_for_mask
        for p in phase_stack_candidates_for_mask(mask_path):
            phase_keys.add(file_identity_key(p))
    except Exception:
        pass
    for p in paths:
        if file_identity_key(p) in phase_keys:
            phase_first.append(p)
        else:
            rest.append(p)
    phase_first = sorted({file_identity_key(p): p for p in phase_first}.values(), key=lambda x: x.name.lower())
    rest = sorted(rest, key=lambda x: x.name.lower())
    return phase_first + rest

def _load_mask_stack_for_contour(mask_path: Path) -> tuple[np.ndarray, Path]:
    """Return (TYX int32 stack, path used for contours — filtered mask when present)."""
    from bactomate.gui.artifact_paths import filtered_mask_path_for_mask
    from bactomate.gui.utils import safe_imread
    contour_path = Path(mask_path)
    try:
        filt = filtered_mask_path_for_mask(mask_path)
        if filt.is_file():
            contour_path = Path(filt)
    except Exception:
        pass
    stack = np.asarray(safe_imread(str(contour_path)))
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    return (np.asarray(stack, dtype=np.int32), contour_path)

def _contour_thickness_px(analyze_tab: Any) -> int:
    try:
        thick = int(getattr(analyze_tab, 'fi_contour_thickness').value())
    except Exception:
        thick = 1
    return max(1, min(10, thick))

def _preview_host_widgets(analyze_tab: Any) -> tuple[Any | None, Any | None]:
    """Resolve the module-shell right-hand preview mount (Analyze → Filtering → Segmentation)."""
    hosts: list[Any] = [analyze_tab]
    try:
        ft = analyze_tab._filtering_tab()
    except Exception:
        ft = None
    if ft is not None:
        hosts.append(ft)
    seg = _segmentation_tab_host(analyze_tab)
    if seg is not None and seg not in hosts:
        hosts.append(seg)
    for host in hosts:
        mount = getattr(host, '_module_preview_mount', None)
        surf = getattr(host, '_preview_surface', None)
        if mount is None:
            continue
        if surf is None:
            try:
                parent = mount.parent()
                while parent is not None:
                    if parent.__class__.__name__ == 'PreviewSurface':
                        surf = parent
                        break
                    parent = parent.parent()
            except Exception:
                surf = None
        if surf is not None:
            return (mount, surf)
    return (None, None)

def _segmentation_tab_host(tab: Any) -> Any | None:
    host = tab
    while host is not None:
        if host.__class__.__name__ == 'SegmentationTab':
            return host
        try:
            host = host.parent()
        except Exception:
            return None
    return None

def fi_contour_preview_button_enabled(analyze_tab: Any) -> tuple[bool, str]:
    """Return (enabled, tooltip) for the Analyze-tab contour preview button."""
    if not bool(getattr(analyze_tab, 'analyze_fi', None) and analyze_tab.analyze_fi.isChecked()):
        return (False, 'Enable Analyze FI and Contour FI first.')
    if not bool(getattr(analyze_tab, 'analyze_fi_contour', None) and analyze_tab.analyze_fi_contour.isChecked()):
        return (False, 'Enable Contour FI first.')
    if not _list_masks_for_contour_picker(analyze_tab):
        return (False, 'Set a mask file or folder on Analyze or Filtering first.')
    return (True, 'Select a mask, then preview Contour FI on the input image and each FI/foci channel (View, LUT, overlay toggle).')

def show_fi_contour_preview(analyze_tab: Any) -> str:
    """
    Open the Contour-FI preview: mask picker, then the module-shell preview pane (right side).

    Returns an empty string on success, otherwise a user-facing error message.
    """
    from PyQt6.QtWidgets import QApplication, QCheckBox, QHBoxLayout, QLabel, QSpinBox, QWidget
    from bactomate.gui.preview.navigation_controls import PreviewNavSpec, PreviewNavState, PreviewNavigationControls, PreviewSource
    from bactomate.gui.preview.preview_window import PreviewDialogBase, PreviewState, RenderResult
    from bactomate.gui.tabs._analyze_masks_preview_filtering import _load_preview_plane
    mask_path = _pick_mask_for_contour_preview(analyze_tab)
    if mask_path is None or not mask_path.is_file():
        return 'No mask selected. Set a mask path or choose a mask in the dialog.'
    try:
        (masks, contour_mask_path) = _load_mask_stack_for_contour(mask_path)
    except Exception as exc:
        return f'Could not load mask:\n{exc}'
    if masks.size == 0:
        return 'Mask stack is empty.'
    bg_paths = discover_background_paths_for_mask(analyze_tab, mask_path)
    if not bg_paths:
        show_message_dialog(analyze_tab, 'No channel images', 'No input/FI/foci TIFFs were found for this mask. The preview will show the mask shape only.', kind='warning')
    num_frames = int(masks.shape[0])
    (mount, preview_surface) = _preview_host_widgets(analyze_tab)
    if mount is None or preview_surface is None:
        return 'The preview area is not available. Open the Segmentation module from the dashboard (Analyze lives in that pipeline) and try again.'
    seg_tab = _segmentation_tab_host(analyze_tab)
    if seg_tab is not None:
        try:
            if preview_surface is not None:
                preview_surface.set_preview_context('filtering')
            dismiss = getattr(seg_tab, '_dismiss_embedded_previews', None)
            if callable(dismiss):
                dismiss()
        except Exception:
            pass
    ft = getattr(analyze_tab, '_filtering_tab', None)
    if ft is not None:
        try:
            prev_f = getattr(ft, '_filter_preview_dialog', None)
            if prev_f is not None and prev_f is not getattr(analyze_tab, '_fi_contour_preview_dialog', None):
                prev_f.close()
                ft._filter_preview_dialog = None
        except Exception:
            pass
    prev = getattr(analyze_tab, '_fi_contour_preview_dialog', None)
    if prev is not None:
        try:
            prev.close()
            prev.deleteLater()
        except Exception:
            pass
    thick_default = _contour_thickness_px(analyze_tab)

    def _sync_thickness_to_analyze(px: int) -> None:
        try:
            spin = getattr(analyze_tab, 'fi_contour_thickness', None)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(max(1, min(10, int(px))))
                spin.blockSignals(False)
        except Exception:
            pass
    chrome = QWidget()
    chrome_lay = QHBoxLayout(chrome)
    chrome_lay.setContentsMargins(0, 0, 0, 4)
    chrome_lay.setSpacing(10)
    contour_cb = QCheckBox('Overlay Contour FI')
    contour_cb.setChecked(True)
    contour_cb.setToolTip('Draw the FI contour ring on each cell in the mask for the current frame.')
    thick_lbl = QLabel('Thickness')
    thick_spin = QSpinBox()
    thick_spin.setRange(1, 10)
    thick_spin.setValue(int(thick_default))
    thick_spin.setSuffix(' px')
    thick_spin.setToolTip('Contour ring width in pixels (used for Analyze when Contour FI is enabled).')
    sources: list[PreviewSource] = []
    for (i, fp) in enumerate(bg_paths):
        from bactomate.gui.filter_preview_utils import preview_view_label
        sources.append(PreviewSource(key=('file', i), label=preview_view_label(Path(fp)), kind='image', resets_lut=True))
    if not sources:
        sources.append(PreviewSource(key=('none', -1), label='No background', kind='image', resets_lut=True))
    nav_ctrl = PreviewNavigationControls(spec=PreviewNavSpec(frame_count=num_frames, image_sources=sources, show_image=True, show_channel=False, show_z=False, debounce_ms=200))
    from bactomate.gui.filter_preview_utils import default_background_index
    init_idx = default_background_index(analyze_tab, mask_path, bg_paths) if bg_paths else 0
    init_key = ('file', init_idx) if bg_paths else ('none', -1)
    nav_ctrl.set_state(PreviewNavState(source_key=init_key, frame_idx=0), emit=False)
    chrome_lay.addWidget(nav_ctrl, 1)
    chrome_lay.addWidget(contour_cb)
    chrome_lay.addWidget(thick_lbl)
    chrome_lay.addWidget(thick_spin)
    preview_surface.set_toolbar_widget(None)
    dlg = PreviewDialogBase(mount, title=f'Contour FI preview — {mask_path.name}', header_controls=chrome, top_controls=None, lut_expanded=False, render_callback=None, minimum_size=(200, 200), embedded=True, show_close_button=False, hide_title_and_builtin_info=True)
    dlg.set_primary_action(None, visible=False)
    try:
        dlg.lut_widget._auto_lo_pct = 0.5
        dlg.lut_widget._auto_hi_pct = 99.5
    except Exception:
        pass
    analyze_tab._fi_contour_preview_dialog = dlg
    dlg._contour_mask_path = Path(mask_path)
    dlg._contour_mask_stack = masks
    dlg._contour_mask_label_path = Path(contour_mask_path)
    dlg._bg_paths = [Path(p) for p in bg_paths]
    dlg._bg_cache: dict = {}

    def _plane_to_rgb_u8(plane: np.ndarray, vmin: float | None, vmax: float | None) -> np.ndarray:
        bg_f = np.asarray(plane, dtype=np.float64)
        if bg_f.ndim != 2:
            bg_f = np.squeeze(bg_f)
        if bg_f.ndim != 2:
            (h, w) = (int(masks.shape[1]), int(masks.shape[2]))
            return np.zeros((h, w), dtype=np.uint8)
        mn = float(np.nanmin(bg_f)) if bg_f.size else 0.0
        mx = float(np.nanmax(bg_f)) if bg_f.size else mn + 1.0
        if vmin is None or vmax is None or (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            vmin = mn
            vmax = mx if mx > mn else mn + 1.0
        vrange = max(float(vmax) - float(vmin), 1e-06)
        gray = ((bg_f - float(vmin)) / vrange * 255.0).clip(0, 255).astype(np.uint8)
        return gray

    def _resolve_bg_path(state: PreviewState) -> Path | None:
        key = state.source_key
        if not isinstance(key, tuple) or len(key) < 2 or key[0] != 'file':
            return None
        idx = int(key[1])
        paths = getattr(dlg, '_bg_paths', []) or []
        if idx < 0 or idx >= len(paths):
            return None
        return Path(paths[idx])

    def _render(state: PreviewState) -> RenderResult:
        from bactomate.gui.image_utils import to_qpixmap
        fi = max(0, min(int(state.frame_idx), num_frames - 1))
        mask_2d = np.asarray(masks[fi], dtype=np.int32)
        labels = accepted_cell_labels_from_mask(mask_2d)
        bg_path = _resolve_bg_path(state)
        lut_frame = np.zeros(mask_2d.shape, dtype=np.float32)
        if bg_path is not None:
            plane = _load_preview_plane(bg_path, fi, dlg._bg_cache)
            if plane is not None:
                lut_frame = np.asarray(plane, dtype=np.float32)
                if state.vmin is None or state.vmax is None or (not np.isfinite(state.vmin)) or (not np.isfinite(state.vmax)) or (float(state.vmax) <= float(state.vmin)):
                    (lo, hi) = auto_display_lut_range(lut_frame)
                    state.vmin = lo
                    state.vmax = hi
                gray = _plane_to_rgb_u8(lut_frame, state.vmin, state.vmax)
            else:
                gray = np.zeros(mask_2d.shape, dtype=np.uint8)
        else:
            gray = (mask_2d > 0).astype(np.uint8) * 40
        rgb = np.stack([gray, gray, gray], axis=-1)
        info_parts = [f'Frame {fi + 1}/{num_frames}', f'{len(labels)} cells']
        if bg_path is not None:
            info_parts.append(bg_path.name)
        if contour_cb.isChecked() and labels:
            from bactomate.gui.analyze_masks_preview_vis import paint_contour_fi_ring_on_display_rgb
            paint_contour_fi_ring_on_display_rgb(rgb, mask_2d, labels, thickness_px=int(thick_spin.value()), blend_alpha=0.95, high_contrast_outline=True)
            info_parts.append(f'Contour FI {int(thick_spin.value())} px')
        pm = to_qpixmap(rgb)
        return RenderResult(pixmap=pm, image_size=(int(rgb.shape[1]), int(rgb.shape[0])), lut_frame=lut_frame if lut_frame.size else None, info_text='  |  '.join(info_parts))
    dlg._render_callback = _render
    dlg.set_render_callback(_render)

    def _on_nav(_reason: str) -> None:
        st = dlg.state()
        st.vmin = None
        st.vmax = None
        dlg.set_source_key(nav_ctrl.state().source_key)
        dlg.set_frame_idx(int(nav_ctrl.state().frame_idx))
        dlg.refresh()

    def _on_contour_ui(*_a) -> None:
        _sync_thickness_to_analyze(int(thick_spin.value()))
        dlg.refresh()
    nav_ctrl.nav_changed.connect(_on_nav)
    contour_cb.toggled.connect(_on_contour_ui)
    thick_spin.valueChanged.connect(_on_contour_ui)
    dlg.set_frame_idx(0)
    dlg.set_source_key(init_key)
    dlg.refresh()

    def _on_closed(*_a) -> None:
        try:
            analyze_tab._fi_contour_preview_dialog = None
        except Exception:
            pass
        try:
            preview_surface.clear_mount()
            preview_surface.set_preview_context('segmentation')
        except Exception:
            pass
    dlg.finished.connect(_on_closed)
    from bactomate.gui.module_shell import mount_embedded_preview
    mount_embedded_preview(mount, dlg, preview_surface=preview_surface)
    dlg.show()
    try:
        QApplication.processEvents()
    except Exception:
        pass
    return ''

def apply_analyze_fi_contour_to_filter_preview(analyze_tab: Any) -> str:
    """Legacy: open dedicated Contour-FI preview (replaces filter-preview overlay flow)."""
    return show_fi_contour_preview(analyze_tab)

def paint_analyze_fi_contour_if_active(rgb: np.ndarray, mask_2d: np.ndarray, accepted_labels: set[int], dlg: Any, *, frame_idx: Optional[int]=None) -> Optional[str]:
    """Paint Contour-FI ring when legacy filter-preview overlay state is set."""
    st = getattr(dlg, ANALYZE_FI_CONTOUR_PREVIEW_ATTR, None)
    if not isinstance(st, dict) or not st.get('active'):
        return None
    if frame_idx is not None:
        try:
            want_fr = int(st.get('frame_idx', frame_idx))
            if int(frame_idx) != want_fr:
                return None
        except Exception:
            pass
    labels = set(accepted_labels)
    stored = st.get('accepted_ids')
    if stored:
        try:
            labels = {int(x) for x in stored}
        except Exception:
            labels = set(accepted_labels)
    if not labels:
        return None
    from bactomate.gui.analyze_masks_preview_vis import paint_contour_fi_ring_on_display_rgb
    paint_contour_fi_ring_on_display_rgb(rgb, mask_2d, labels, thickness_px=int(st.get('thickness_px', 1) or 1), blend_alpha=0.95)
    thick = int(st.get('thickness_px', 1) or 1)
    fi_name = Path(str(st.get('fi_path') or '')).name
    return f'Contour FI preview ({thick} px, {fi_name})'
