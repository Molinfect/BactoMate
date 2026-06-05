from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from bactomate.gui.image_utils import scale_for_preview, to_qpixmap
from bactomate.gui.preview.navigation_controls import PreviewNavSpec, PreviewNavState, PreviewNavigationControls
from bactomate.gui.preview.preview_window import PreviewDialogBase, RenderResult
from bactomate.gui.preview.render_utils import apply_lut_to_uint8


@dataclass
class FileConversionPreviewModel:
    raw_frames: np.ndarray
    num_time: int
    num_channels: int
    num_z: int
    file_path: str
    nd2_axis_keys: Optional[tuple[str, ...]] = None

    # Bleach preview (optional)
    bleach_enabled: bool = False  # whether checkbox exists (num_time > 1)
    get_bleach_kwargs: Optional[Callable[[], dict]] = None
    corrected_frames: Optional[np.ndarray] = None
    showing_corrected: bool = False
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
        bleach_enabled=bool(int(max(1, num_time)) > 1),
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

    bleach_check = None
    if model.bleach_enabled:
        bleach_check = QCheckBox("Show bleach correction")
        bleach_check.setChecked(False)
        bleach_check.setToolTip(
            "Preview the bleach correction effect. Uses the same method and fluorescence channel settings as the conversion tab. "
            "Raw vs corrected mode appears in the blue text on the navigation row (next to channel / frame controls)."
        )
        bleach_row_layout.addWidget(bleach_check)
        bleach_row_layout.addStretch(1)
        top_layout.addWidget(bleach_row)

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

    # ---- renderer ------------------------------------------------------
    def _render(state) -> RenderResult:
        z_idx = 0
        try:
            z_idx = int(state.extra.get("z_idx", 0) or 0)
        except Exception:
            z_idx = 0
        frames = model.corrected_frames if model.showing_corrected and model.corrected_frames is not None else model.raw_frames
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
        pm = to_qpixmap(disp_scaled)

        view_txt = "Raw"
        if model.showing_corrected:
            dn = model.bleach_display_name or "Bleach corrected"
            view_txt = f"Bleach corrected ({dn})"

        ch_txt = f" | Channel {int(state.channel_idx) + 1}" if model.num_channels > 1 else ""
        z_txt = ""
        if has_z_nav:
            if model.showing_corrected:
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
        top_controls=(top if model.bleach_enabled else None),
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

    # ---- wiring: nav -> dlg.state -------------------------------------
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

    # ---- bleach toggle + sync -----------------------------------------
    def _compute_corrected_from_tab() -> tuple[Optional[np.ndarray], Optional[str], tuple[int, ...], Optional[str]]:
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

        data = np.asarray(model.raw_frames)
        # Prepare as T,C,Y,X for correction function.
        if data.ndim == 5 and model.nd2_axis_keys is not None and "Z" in model.nd2_axis_keys:
            # Take first Z for correction preview (matches old behavior).
            data = np.take(data, 0, axis=model.nd2_axis_keys.index("Z")).copy()
        elif data.ndim == 5:
            data = np.asarray(data[:, :, 0, :, :]).copy()
        elif data.ndim == 4 and model.nd2_axis_keys is not None and "Z" in model.nd2_axis_keys:
            data = np.take(data, 0, axis=model.nd2_axis_keys.index("Z")).copy()
        elif data.ndim == 3 and model.num_channels == 1:
            data = data[:, np.newaxis, :, :]

        corrected = bleach_correct_stack(data, method=method, channel_indices=(list(channels_tuple) if channel_indices is not None else None))
        disp = {
            "exponential": "Exponential",
            "linear": "Linear",
            "histogram_matching": "Histogram matching",
            "per_frame_minmax": "Per-frame min-max",
        }.get(method, method)
        return np.asarray(corrected), method, channels_tuple, disp

    if bleach_check is not None:
        def _set_z_nav_for_mode(*, corrected: bool) -> None:
            if not has_z_nav:
                return
            st = nav.state()
            # Corrected preview uses first-Z projection; hide Z control to avoid misleading interaction.
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

        def _apply_bleach_view(checked: bool) -> None:
            if checked:
                corrected, method, channels_tuple, disp = _compute_corrected_from_tab()
                if corrected is None:
                    bleach_check.setChecked(False)
                    return
                model.corrected_frames = corrected
                model.showing_corrected = True
                model.bleach_method_used = method
                model.bleach_channels_used = channels_tuple
                model.bleach_display_name = disp
                _set_z_nav_for_mode(corrected=True)
            else:
                model.showing_corrected = False
                _set_z_nav_for_mode(corrected=False)
            _reset_lut()
            dlg.refresh()

        bleach_check.toggled.connect(_apply_bleach_view)

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
