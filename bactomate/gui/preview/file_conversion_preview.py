from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import QEventLoop, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from bactomate.modules.conversion_crop import (
    apply_yx_crop,
    format_roi_status,
    normalize_roi,
    roi_display_to_native,
    roi_native_to_display,
)

from bactomate.gui.image_utils import scale_for_preview, to_qpixmap
from bactomate.gui.layout_helpers import style_compact_action_button
from bactomate.gui.preview.navigation_controls import PreviewNavSpec, PreviewNavState, PreviewNavigationControls
from bactomate.gui.preview.preview_window import PreviewDialogBase, RenderResult
from bactomate.gui.preview.render_utils import apply_lut_to_uint8
from bactomate.gui.widgets import show_message_dialog


class _PreviewComputeWorker(QThread):
    finished_result = pyqtSignal(object, object)

    def __init__(self, compute: Callable[[], Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._compute = compute

    def run(self) -> None:
        try:
            self.finished_result.emit(self._compute(), None)
        except BaseException as exc:
            self.finished_result.emit(None, exc)


@dataclass
class FileConversionPreviewModel:
    raw_frames: np.ndarray
    num_time: int
    num_channels: int
    num_z: int
    file_path: str
    nd2_axis_keys: Optional[tuple[str, ...]] = None

    # Correction preview (optional)
    corrections_enabled: bool = False  # whether checkboxes exist (num_time > 1)
    get_bleach_kwargs: Optional[Callable[[], dict]] = None
    drift_corrected_frames: Optional[np.ndarray] = None
    corrected_frames: Optional[np.ndarray] = None
    showing_drift: bool = False
    showing_bleach: bool = False
    bleach_display_name: Optional[str] = None
    bleach_method_used: Optional[str] = None
    bleach_channels_used: tuple[int, ...] = ()


def _slice_plane(
    frames: np.ndarray,
    *,
    frame_idx: int,
    channel_idx: int,
    z_idx: int,
    num_time: int,
    num_channels: int,
    num_z: int,
    nd2_axis_keys: Optional[tuple[str, ...]] = None,
) -> np.ndarray:
    arr = np.asarray(frames)
    if nd2_axis_keys is not None and len(nd2_axis_keys) == len(arr.shape):
        # ND2: axis order is explicit (often T,Z,C,Y,X).
        idx: list[Any] = []
        keep_axes: list[str] = []
        for ax in nd2_axis_keys:
            if ax == "T":
                idx.append(max(0, min(int(frame_idx), max(0, num_time - 1))))
            elif ax == "C":
                idx.append(max(0, min(int(channel_idx), max(0, num_channels - 1))))
            elif ax == "Z":
                idx.append(max(0, min(int(z_idx), max(0, num_z - 1))))
            elif ax in ("Y", "X"):
                idx.append(slice(None))
                keep_axes.append(ax)
            else:
                idx.append(0)
        plane = np.asarray(arr[tuple(idx)])
        # Reorder to Y,X if needed.
        if plane.ndim == 2 and keep_axes == ["X", "Y"]:
            plane = plane.T
        if plane.ndim != 2:
            plane = np.asarray(plane.squeeze())
            if plane.ndim > 2:
                # best-effort: take first plane
                plane = np.asarray(plane.reshape(plane.shape[0], -1))
        return plane

    # Generic shape handling (mirrors the old preview heuristics, but centralized here).
    shp = arr.shape
    ndim = len(shp)
    fi = max(0, min(int(frame_idx), max(0, num_time - 1)))
    ci = max(0, min(int(channel_idx), max(0, num_channels - 1)))
    zi = max(0, min(int(z_idx), max(0, num_z - 1)))

    if ndim == 5:
        # Historically treated as (T, C, Z, Y, X) or similar.
        try:
            return np.asarray(arr[fi, ci, zi, :, :])
        except Exception:
            return np.asarray(arr[fi, 0, 0, :, :])
    if ndim == 4:
        # (T, C, Y, X) — including single-frame multi-channel (T=1, C>1).
        if num_channels > 1 and shp[1] == num_channels:
            return np.asarray(arr[min(fi, shp[0] - 1), min(ci, shp[1] - 1), :, :])
        if num_time > 1 and num_z > 1:
            return np.asarray(arr[fi, zi, :, :])
        return np.asarray(arr[min(fi, shp[0] - 1), :, :])
    if ndim == 3:
        # Could be TYX or CYX (single frame).
        if shp[0] == num_time and num_time > 1:
            return np.asarray(arr[fi, :, :])
        if num_channels > 1 and shp[0] == num_channels:
            return np.asarray(arr[ci, :, :])
        return np.asarray(arr[min(fi, shp[0] - 1), :, :])
    if ndim == 2:
        return np.asarray(arr)
    # Last resort
    try:
        return np.asarray(arr[0]).squeeze()
    except Exception:
        return np.zeros((1, 1), dtype=np.float32)


def show_file_conversion_preview(
    tab: Any,
    *,
    all_frames: np.ndarray,
    num_time: int,
    num_channels: int,
    num_z: int,
    file_path: str,
    nd2_axis_keys: Optional[tuple[str, ...]] = None,
    embed_parent: Optional[QWidget] = None,
) -> PreviewDialogBase:
    main_win = tab.window() if hasattr(tab, "window") and callable(tab.window) else tab
    parent_for_dlg = embed_parent if embed_parent is not None else main_win
    embedded = embed_parent is not None
    model = FileConversionPreviewModel(
        raw_frames=np.asarray(all_frames),
        num_time=int(max(1, num_time)),
        num_channels=int(max(1, num_channels)),
        num_z=int(max(1, num_z)),
        file_path=str(file_path),
        nd2_axis_keys=tuple(nd2_axis_keys) if nd2_axis_keys is not None else None,
        corrections_enabled=bool(int(max(1, num_time)) > 1),
        get_bleach_kwargs=(lambda: dict(getattr(tab, "_get_bleach_preview_kwargs")())) if hasattr(tab, "_get_bleach_preview_kwargs") else None,
    )

    # ---- top controls: bleach toggle + status -------------------------
    top = QWidget()
    top_layout = QVBoxLayout(top)
    top_layout.setContentsMargins(0, 0, 0, 0)
    top_layout.setSpacing(6)

    bleach_row = QWidget()
    bleach_row_layout = QHBoxLayout(bleach_row)
    bleach_row_layout.setContentsMargins(0, 0, 0, 0)
    bleach_row_layout.setSpacing(8)

    drift_check = None
    bleach_check = None
    compare_drift_btn = None
    if model.corrections_enabled:
        drift_check = QCheckBox("Show X/Y drift corrected")
        drift_check.setChecked(False)
        drift_check.setToolTip(
            "Preview X/Y drift correction. Computed only when checked, then cached for this preview."
        )
        bleach_row_layout.addWidget(drift_check)
        bleach_check = QCheckBox("Show bleach correction")
        bleach_check.setChecked(False)
        bleach_check.setToolTip(
            "Preview the bleach correction effect. Uses the same method and fluorescence channel settings as the conversion tab. "
            "If X/Y drift preview is checked, bleach correction is applied after drift correction."
        )
        bleach_row_layout.addWidget(bleach_check)
        compare_drift_btn = QPushButton("Compare X/Y correction")
        compare_drift_btn.setToolTip(
            "Open a side-by-side raw/corrected movie preview with playback. "
            "Uses the current crop ROI if Extract crop is enabled."
        )
        style_compact_action_button(compare_drift_btn, horizontal_padding=18, dense=True)
        bleach_row_layout.addWidget(compare_drift_btn)
        bleach_row_layout.addStretch(1)
        top_layout.addWidget(bleach_row)

    crop_hint = QLabel(
        "Extract crop: enable on the left, then right-drag a box on the image. "
        "Middle-drag pans; wheel zooms. Drag handles to adjust the crop."
    )
    crop_hint.setWordWrap(True)
    crop_hint.setStyleSheet("color: #9E9E9E; font-size: 10pt;")
    top_layout.addWidget(crop_hint)

    # ---- navigation controls ------------------------------------------
    nav_spec = PreviewNavSpec(
        frame_count=model.num_time,
        image_sources=[],
        channel_count=model.num_channels,
        z_count=model.num_z,
        show_image=False,
        show_channel=bool(model.num_channels > 1),
        show_z=bool(model.nd2_axis_keys is not None and "Z" in model.nd2_axis_keys and model.num_z > 1),
        show_frame_of_total=bool(model.num_time > 1),
        show_frame_slider=bool(model.num_time > 1),
        debounce_ms=400,
    )

    view_lut_label = QLabel("")
    view_lut_label.setStyleSheet("font-weight: bold; color: #2196F3; font-size: 10pt;")
    view_lut_label.setMinimumWidth(80)
    view_lut_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    view_lut_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    nav = PreviewNavigationControls(spec=nav_spec, inline_trailer=view_lut_label)
    # Initial state
    nav.set_state(PreviewNavState(source_key=None, frame_idx=0, channel_idx=0, z_idx=0), emit=False)

    has_z_nav = bool(model.nd2_axis_keys is not None and "Z" in model.nd2_axis_keys and model.num_z > 1)
    roi_state: dict = {"native_hw": None, "display_hw": None, "drag_start": None}

    def _crop_mode_enabled() -> bool:
        fn = getattr(tab, "extract_crop_enabled", None)
        return bool(fn()) if callable(fn) else False

    # ---- renderer ------------------------------------------------------
    def _render(state) -> RenderResult:
        z_idx = 0
        try:
            z_idx = int(state.extra.get("z_idx", 0) or 0)
        except Exception:
            z_idx = 0
        frames = model.raw_frames
        if model.showing_bleach and model.corrected_frames is not None:
            frames = model.corrected_frames
        elif model.showing_drift and model.drift_corrected_frames is not None:
            frames = model.drift_corrected_frames
        plane = _slice_plane(
            frames,
            frame_idx=int(state.frame_idx),
            channel_idx=int(state.channel_idx),
            z_idx=int(z_idx),
            num_time=model.num_time,
            num_channels=model.num_channels,
            num_z=model.num_z,
            nd2_axis_keys=model.nd2_axis_keys,
        )
        plane2d = np.asarray(plane)
        if plane2d.ndim != 2:
            plane2d = np.asarray(plane2d.squeeze())
        if plane2d.ndim != 2:
            plane2d = np.zeros((1, 1), dtype=np.float32)

        u8 = apply_lut_to_uint8(plane2d, state.vmin, state.vmax)
        dummy_mask = np.zeros(u8.shape[:2], dtype=np.int32)
        disp_scaled, _ = scale_for_preview(u8, dummy_mask)
        roi_state["native_hw"] = (int(plane2d.shape[0]), int(plane2d.shape[1]))
        roi_state["display_hw"] = (int(disp_scaled.shape[0]), int(disp_scaled.shape[1]))
        pm = to_qpixmap(disp_scaled)

        view_parts: list[str] = []
        if model.showing_drift:
            view_parts.append("X/Y drift corrected")
        if model.showing_bleach:
            dn = model.bleach_display_name or "Bleach corrected"
            view_parts.append(f"Bleach corrected ({dn})")
        view_txt = " + ".join(view_parts) if view_parts else "Raw"

        ch_txt = f" | Channel {int(state.channel_idx) + 1}" if model.num_channels > 1 else ""
        z_txt = ""
        if has_z_nav:
            if model.showing_drift or model.showing_bleach:
                z_txt = " | Z fixed to 1 (corrected preview)"
            else:
                z_txt = f" | Z {int(z_idx) + 1}"

        title = None
        info = f"View: {view_txt}{ch_txt}{z_txt} | LUT: [{state.vmin:.1f}, {state.vmax:.1f}]" if (state.vmin is not None and state.vmax is not None) else f"View: {view_txt}{ch_txt}{z_txt}"
        view_lut_label.setText(info)

        return RenderResult(
            pixmap=pm,
            image_size=(int(disp_scaled.shape[1]), int(disp_scaled.shape[0])),
            lut_frame=np.asarray(plane2d, dtype=np.float32),
            title_text=title,
            info_text=None,
        )

    dlg = PreviewDialogBase(
        parent_for_dlg,
        title="File Preview",
        top_controls=top,
        navigation_controls=nav,
        primary_action_label="Apply",
        primary_action_callback=None,
        render_callback=_render,
        worker_guard=None,
        minimum_size=(900, 700),
        embedded=embedded,
        show_close_button=not embedded,
        info_row_above_navigation=False,
        hide_title_and_builtin_info=True,
    )
    dlg.set_primary_action(None, visible=False)

    def sync_crop_overlay() -> None:
        canvas = dlg.canvas
        if not _crop_mode_enabled():
            canvas.enable_drag_selection = False
            canvas.enable_roi_editing = False
            canvas.clear_roi_overlays()
            canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
            return
        canvas.enable_drag_selection = True
        canvas.enable_roi_editing = True
        canvas.selection_button = Qt.MouseButton.RightButton
        canvas.click_button = Qt.MouseButton.LeftButton
        canvas.pan_button = Qt.MouseButton.MiddleButton
        canvas.setToolTip(
            "Zoom: mouse wheel. Pan: middle-drag. Extract crop enabled: right-drag a box; drag handles to adjust."
        )
        native = roi_state.get("native_hw")
        display = roi_state.get("display_hw")
        roi = getattr(tab, "_conversion_crop_roi", None)
        if roi and native and display:
            disp_roi = roi_native_to_display(roi, native, display)
            if disp_roi is not None:
                canvas.set_editable_roi(disp_roi, color=QColor(255, 230, 0))
                return
        canvas.set_editable_roi(None)
        canvas.set_pending_roi(None)

    def _display_dims() -> tuple[int, int] | None:
        display = roi_state.get("display_hw")
        if not display:
            return None
        dh, dw = int(display[0]), int(display[1])
        if dh <= 0 or dw <= 0:
            return None
        return dh, dw

    def _commit_display_roi(disp_roi: tuple[int, int, int, int]) -> None:
        native = roi_state.get("native_hw")
        display = roi_state.get("display_hw")
        if not native or not display:
            return
        native_roi = roi_display_to_native(disp_roi, native, display)
        if native_roi is None:
            return
        if hasattr(tab, "set_conversion_crop_roi"):
            tab.set_conversion_crop_roi(native_roi)

    def on_crop_drag_started(x: int, y: int) -> None:
        if not _crop_mode_enabled():
            return
        roi_state["drag_start"] = (int(x), int(y))

    def on_crop_drag_updated(x: int, y: int) -> None:
        if not _crop_mode_enabled() or roi_state.get("drag_start") is None:
            return
        dims = _display_dims()
        if dims is None:
            return
        dh, dw = dims
        x0, y0 = roi_state["drag_start"]
        pending = normalize_roi(int(x0), int(y0), int(x), int(y), dw, dh, min_size=8)
        dlg.canvas.set_pending_roi(pending)

    def on_crop_drag_finished(x: int, y: int) -> None:
        drag_start = roi_state.get("drag_start")
        roi_state["drag_start"] = None
        dlg.canvas.set_pending_roi(None)
        if not _crop_mode_enabled() or drag_start is None:
            return
        dims = _display_dims()
        if dims is None:
            return
        dh, dw = dims
        x0, y0 = drag_start
        disp_roi = normalize_roi(int(x0), int(y0), int(x), int(y), dw, dh)
        if disp_roi is None:
            sync_crop_overlay()
            return
        _commit_display_roi(disp_roi)
        sync_crop_overlay()

    def on_roi_edit_finished(x0: int, y0: int, x1: int, y1: int) -> None:
        if not _crop_mode_enabled():
            return
        _commit_display_roi((int(x0), int(y0), int(x1), int(y1)))
        sync_crop_overlay()

    dlg.sync_crop_overlay = sync_crop_overlay  # type: ignore[attr-defined]
    dlg.canvas.drag_started.connect(on_crop_drag_started)
    dlg.canvas.drag_updated.connect(on_crop_drag_updated)
    dlg.canvas.drag_finished.connect(on_crop_drag_finished)
    dlg.canvas.roi_edit_finished.connect(on_roi_edit_finished)

    if hasattr(tab, "extract_crop_cb"):
        tab.extract_crop_cb.toggled.connect(lambda _checked: sync_crop_overlay())
    if hasattr(tab, "clear_crop_btn"):
        tab.clear_crop_btn.clicked.connect(lambda: sync_crop_overlay())

    _orig_refresh = dlg.refresh

    def _refresh_with_crop() -> None:
        _orig_refresh()
        sync_crop_overlay()

    dlg.refresh = _refresh_with_crop  # type: ignore[method-assign]

    def _reset_lut() -> None:
        try:
            dlg.state().vmin = None
            dlg.state().vmax = None
            dlg.lut_widget.set_lut(None, None)
        except Exception:
            pass

    def _on_nav_changed(st: PreviewNavState, reason: str) -> None:
        dlg.set_frame_idx(int(st.frame_idx))
        dlg.set_channel_idx(int(st.channel_idx))
        dlg.set_source_key(st.source_key)
        dlg.set_extra("z_idx", int(st.z_idx))
        # Match old behavior: switching channels should auto-recompute LUT.
        if reason in ("channel",):
            _reset_lut()
        dlg.refresh()

    nav.nav_changed.connect(_on_nav_changed)

    # ---- correction preview toggles -----------------------------------
    def _prepare_stack_for_correction() -> np.ndarray:
        data = np.asarray(model.raw_frames)
        if data.ndim == 5 and model.nd2_axis_keys is not None and "Z" in model.nd2_axis_keys:
            data = np.take(data, 0, axis=model.nd2_axis_keys.index("Z")).copy()
        elif data.ndim == 5:
            data = np.asarray(data[:, :, 0, :, :]).copy()
        elif data.ndim == 4 and model.nd2_axis_keys is not None and "Z" in model.nd2_axis_keys:
            data = np.take(data, 0, axis=model.nd2_axis_keys.index("Z")).copy()
        elif data.ndim == 3 and model.num_channels == 1:
            data = data[:, np.newaxis, :, :]
        return data

    def _compute_drift_corrected() -> Optional[np.ndarray]:
        try:
            from bactomate.scripts.convert_nd2_to_tif import drift_correct_stack
        except Exception:
            return None
        data = _prepare_stack_for_correction()
        return np.asarray(drift_correct_stack(data, method="frame_by_frame"))

    def _compute_bleach_corrected(source: np.ndarray) -> tuple[Optional[np.ndarray], Optional[str], tuple[int, ...], Optional[str]]:
        if model.get_bleach_kwargs is None:
            return None, None, (), None
        kwargs = model.get_bleach_kwargs() or {}
        method = str(kwargs.get("bleach_method") or "exponential")
        channel_indices = kwargs.get("bleach_channel_indices", None)
        channels_tuple = tuple(sorted(int(x) for x in (channel_indices or []))) if channel_indices is not None else ()
        try:
            from bactomate.scripts.convert_nd2_to_tif import bleach_correct_stack
        except Exception:
            return None, None, (), None
        corrected = bleach_correct_stack(
            np.asarray(source),
            method=method,
            channel_indices=(list(channels_tuple) if channel_indices is not None else None),
        )
        disp = {
            "exponential": "Exponential",
            "linear": "Linear",
            "histogram_matching": "Histogram matching",
            "per_frame_minmax": "Per-frame min-max",
        }.get(method, method)
        return np.asarray(corrected), method, channels_tuple, disp

    def _run_with_progress(step: str, detail: str, compute: Callable[[], Any]) -> Any:
        message = str(step or "Processing preview").strip()
        detail_txt = str(detail or "").strip()
        if detail_txt:
            message = f"{message}\n\n{detail_txt}"
        progress = QProgressDialog(message, None, 0, 0, parent_for_dlg)
        progress.setWindowTitle("Processing")
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        QApplication.processEvents()

        result: dict[str, Any] = {"value": None, "error": None}
        loop = QEventLoop()
        worker = _PreviewComputeWorker(compute, progress)

        def _finish(value: Any, error: Any) -> None:
            result["value"] = value
            result["error"] = error
            loop.quit()

        worker.finished_result.connect(_finish)
        worker.start()
        try:
            loop.exec()
            if result["error"] is not None:
                raise result["error"]
            return result["value"]
        finally:
            if worker.isRunning():
                worker.wait(1000)
            progress.close()
            progress.deleteLater()
            QApplication.processEvents()

    def _set_check_silently(check: Optional[QCheckBox], checked: bool) -> None:
        if check is None:
            return
        check.blockSignals(True)
        try:
            check.setChecked(bool(checked))
        finally:
            check.blockSignals(False)

    def _active_native_crop_for_stack(stack: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        if not _crop_mode_enabled():
            return None
        roi = getattr(tab, "_conversion_crop_roi", None)
        if not roi or np.asarray(stack).ndim < 2:
            return None
        h, w = int(np.asarray(stack).shape[-2]), int(np.asarray(stack).shape[-1])
        try:
            x0, y0, x1, y1 = (int(v) for v in roi)
        except Exception:
            return None
        return normalize_roi(x0, y0, x1, y1, w, h)

    def _plane_from_compare_stack(stack: np.ndarray, frame_idx: int, channel_idx: int) -> np.ndarray:
        arr = np.asarray(stack)
        if arr.ndim == 4:
            return np.asarray(arr[min(frame_idx, arr.shape[0] - 1), min(channel_idx, arr.shape[1] - 1), :, :])
        if arr.ndim == 3:
            return np.asarray(arr[min(frame_idx, arr.shape[0] - 1), :, :])
        if arr.ndim == 2:
            return arr
        return np.asarray(arr.squeeze())

    def _compare_pixmap(raw_plane: np.ndarray, corrected_plane: np.ndarray, which: str) -> Any:
        plane = np.asarray(raw_plane if which == "raw" else corrected_plane)
        state = dlg.state()
        vmin = state.vmin
        vmax = state.vmax
        if vmin is None or vmax is None:
            try:
                vmin = float(min(np.nanmin(raw_plane), np.nanmin(corrected_plane)))
                vmax = float(max(np.nanmax(raw_plane), np.nanmax(corrected_plane)))
            except Exception:
                vmin, vmax = None, None
        u8 = apply_lut_to_uint8(plane, vmin, vmax)
        dummy_mask = np.zeros(u8.shape[:2], dtype=np.int32)
        disp_scaled, _ = scale_for_preview(u8, dummy_mask, max_size=760)
        return to_qpixmap(disp_scaled)

    def _show_drift_compare_dialog() -> None:
        if model.drift_corrected_frames is None:
            model.drift_corrected_frames = _run_with_progress(
                "Computing X/Y drift correction",
                "Preparing corrected preview frames. This can take a moment for larger time lapses.",
                _compute_drift_corrected,
            )
        if model.drift_corrected_frames is None:
            show_message_dialog(parent_for_dlg, "Drift preview unavailable", "Could not compute X/Y drift correction preview.", kind="warning")
            return

        raw_stack_full = np.asarray(_prepare_stack_for_correction())
        corrected_stack_full = np.asarray(model.drift_corrected_frames)
        roi = _active_native_crop_for_stack(raw_stack_full)

        frame_count = int(min(raw_stack_full.shape[0], corrected_stack_full.shape[0])) if raw_stack_full.ndim >= 3 and corrected_stack_full.ndim >= 3 else 1
        if frame_count <= 0:
            show_message_dialog(parent_for_dlg, "Drift preview unavailable", "No frames are available for comparison.", kind="warning")
            return

        old_dialog = getattr(dlg, "_drift_compare_dialog", None)
        try:
            if old_dialog is not None:
                old_dialog.close()
        except Exception:
            pass

        compare_dlg = QDialog(parent_for_dlg)
        compare_dlg.setWindowTitle("X/Y Drift Correction Comparison")
        compare_dlg.setMinimumSize(980, 640)
        compare_dlg.setStyleSheet(
            "QDialog { background-color: #1e1e1e; }"
            "QLabel { color: #e8eef6; }"
        )
        outer = QVBoxLayout(compare_dlg)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        apply_crop_cb = QCheckBox("Apply extract crop to comparison")
        apply_crop_cb.setChecked(bool(roi is not None))
        apply_crop_cb.setEnabled(bool(roi is not None))
        apply_crop_cb.setToolTip(
            "When enabled, both panels show only the current extract-crop ROI from the main preview."
        )
        outer.addWidget(apply_crop_cb)

        info = QLabel(
            "Left: original. Right: X/Y drift corrected. "
            "Correction is computed on the full frame first, then optionally cropped for comparison."
        )
        info.setWordWrap(True)
        outer.addWidget(info)

        title_row = QHBoxLayout()
        raw_title = QLabel("Original")
        corrected_title = QLabel("X/Y drift corrected")
        raw_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        corrected_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_row.addWidget(raw_title, 1)
        title_row.addWidget(corrected_title, 1)
        outer.addLayout(title_row)

        image_row = QHBoxLayout()
        raw_label = QLabel()
        corrected_label = QLabel()
        for image_label in (raw_label, corrected_label):
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setMinimumSize(420, 420)
            image_label.setStyleSheet("background-color: #111; border: 1px solid #333;")
            image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            image_row.addWidget(image_label, 1)
        outer.addLayout(image_row, 1)

        controls = QHBoxLayout()
        play_btn = QPushButton("Play")
        play_btn.setCheckable(True)
        style_compact_action_button(play_btn, horizontal_padding=32)
        play_btn.setFixedWidth(96)
        frame_slider = QSlider(Qt.Orientation.Horizontal)
        frame_slider.setRange(0, max(0, frame_count - 1))
        frame_slider.setValue(max(0, min(int(dlg.state().frame_idx), frame_count - 1)))
        frame_label = QLabel("")
        frame_label.setMinimumWidth(110)
        controls.addWidget(play_btn)
        controls.addWidget(frame_slider, 1)
        controls.addWidget(frame_label)
        close_btn = QPushButton("Close")
        style_compact_action_button(close_btn, horizontal_padding=28)
        close_btn.clicked.connect(compare_dlg.close)
        controls.addWidget(close_btn)
        outer.addLayout(controls)

        timer = QTimer(compare_dlg)
        timer.setInterval(180)

        def _stacks_for_compare() -> tuple[np.ndarray, np.ndarray, str]:
            raw_stack = raw_stack_full
            corrected_stack = corrected_stack_full
            crop_text = "Full image"
            if apply_crop_cb.isChecked() and roi is not None:
                raw_stack = apply_yx_crop(raw_stack_full, roi)
                corrected_stack = apply_yx_crop(corrected_stack_full, roi)
                crop_text = format_roi_status(roi)
            return raw_stack, corrected_stack, crop_text

        def update_compare_frame(frame_idx: int) -> None:
            raw_stack, corrected_stack, crop_text = _stacks_for_compare()
            info.setText(
                f"Left: original. Right: X/Y drift corrected. {crop_text}. "
                "Correction is computed on the full frame first, then optionally cropped for comparison."
            )
            idx = max(0, min(int(frame_idx), frame_count - 1))
            channel_idx = int(dlg.state().channel_idx)
            raw_plane = _plane_from_compare_stack(raw_stack, idx, channel_idx)
            corrected_plane = _plane_from_compare_stack(corrected_stack, idx, channel_idx)
            raw_label.setPixmap(_compare_pixmap(raw_plane, corrected_plane, "raw"))
            corrected_label.setPixmap(_compare_pixmap(raw_plane, corrected_plane, "corrected"))
            frame_label.setText(f"Frame {idx + 1} / {frame_count}")

        def advance_frame() -> None:
            next_idx = int(frame_slider.value()) + 1
            if next_idx >= frame_count:
                next_idx = 0
            frame_slider.setValue(next_idx)

        def toggle_play(checked: bool) -> None:
            if checked:
                play_btn.setText("Pause")
                timer.start()
            else:
                play_btn.setText("Play")
                timer.stop()

        frame_slider.valueChanged.connect(update_compare_frame)
        apply_crop_cb.toggled.connect(lambda _checked: update_compare_frame(int(frame_slider.value())))
        play_btn.toggled.connect(toggle_play)
        timer.timeout.connect(advance_frame)
        compare_dlg.destroyed.connect(lambda *_: timer.stop())

        dlg._drift_compare_dialog = compare_dlg  # type: ignore[attr-defined]
        update_compare_frame(int(frame_slider.value()))
        compare_dlg.show()
        compare_dlg.raise_()
        compare_dlg.activateWindow()

    if drift_check is not None or bleach_check is not None:
        def _set_z_nav_for_mode(*, corrected: bool) -> None:
            if not has_z_nav:
                return
            st = nav.state()
            spec2 = replace(nav.spec(), show_z=(not corrected), z_count=(1 if corrected else model.num_z))
            nav.set_spec(spec2, emit_init=False)
            nav.set_state(
                PreviewNavState(
                    source_key=st.source_key,
                    frame_idx=st.frame_idx,
                    channel_idx=st.channel_idx,
                    z_idx=(0 if corrected else st.z_idx),
                ),
                emit=False,
            )
            dlg.set_extra("z_idx", 0 if corrected else int(st.z_idx))

        def _update_correction_preview() -> None:
            show_drift = bool(drift_check.isChecked()) if drift_check is not None else False
            show_bleach = bool(bleach_check.isChecked()) if bleach_check is not None else False
            model.showing_drift = show_drift
            model.showing_bleach = show_bleach

            if show_drift:
                if model.drift_corrected_frames is None:
                    model.drift_corrected_frames = _run_with_progress(
                        "Computing X/Y drift correction",
                        "Preparing corrected preview frames. This can take a moment for larger time lapses.",
                        _compute_drift_corrected,
                    )
                if model.drift_corrected_frames is None:
                    model.showing_drift = False
                    _set_check_silently(drift_check, False)
            else:
                model.drift_corrected_frames = None

            if show_bleach:
                source = model.drift_corrected_frames if show_drift and model.drift_corrected_frames is not None else _prepare_stack_for_correction()
                corrected, method, channels_tuple, disp = _run_with_progress(
                    "Computing bleach correction",
                    "Preparing corrected preview frames with the current bleach settings.",
                    lambda: _compute_bleach_corrected(source),
                ) or (None, None, (), None)
                if corrected is None:
                    model.showing_bleach = False
                    _set_check_silently(bleach_check, False)
                else:
                    model.corrected_frames = corrected
                    model.bleach_method_used = method
                    model.bleach_channels_used = channels_tuple
                    model.bleach_display_name = disp
            else:
                model.corrected_frames = None

            _set_z_nav_for_mode(corrected=show_drift or show_bleach)
            _reset_lut()
            dlg.refresh()

        if drift_check is not None:
            drift_check.toggled.connect(lambda _checked: _update_correction_preview())
        if bleach_check is not None:
            bleach_check.toggled.connect(lambda _checked: _update_correction_preview())
        if compare_drift_btn is not None:
            compare_drift_btn.clicked.connect(_show_drift_compare_dialog)

    if embed_parent is not None:
        lay = embed_parent.layout()
        if lay is None:
            lay = QVBoxLayout(embed_parent)
            lay.setContentsMargins(0, 0, 0, 0)
            embed_parent.setLayout(lay)
        surface = embed_parent.parent()
        ph = getattr(surface, "_placeholder", None) if surface is not None else None
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is None:
                continue
            if ph is not None and w is ph:
                w.hide()
                w.setParent(None)
                continue
            w.deleteLater()
        lay.addWidget(dlg, 1)
    else:
        dlg.show()

    # Initial draw
    dlg.set_frame_idx(0)
    dlg.set_channel_idx(0)
    dlg.set_extra("z_idx", 0)
    dlg.refresh()
    if embed_parent is None:
        try:
            dlg.raise_()
            dlg.activateWindow()
        except Exception:
            pass
    return dlg
