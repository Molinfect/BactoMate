from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class PreviewSource:
    """Stable identity + label for the Image: dropdown."""

    key: Any
    label: str
    kind: str = "image"
    resets_lut: bool = True


@dataclass(frozen=True)
class PreviewNavSpec:
    """Defines which navigation controls are shown and their ranges."""

    frame_count: int
    image_sources: list[PreviewSource] = field(default_factory=list)
    channel_count: int = 1
    z_count: int = 1
    show_image: bool = True
    show_channel: bool = False
    show_z: bool = False
    show_frame_of_total: bool = True
    show_cell_overlay_toggle: bool = False
    show_foci_overlay_toggle: bool = False
    cell_overlay_default: bool = True
    foci_overlay_available: bool = False
    foci_overlay_default: bool = True
    # When non-empty (Detect foci + N stacks), per-channel checkboxes replace the single foci toggle.
    foci_channel_overlay_labels: tuple[str, ...] = ()
    foci_channel_overlay_defaults: tuple[bool, ...] = ()
    debounce_ms: int = 400


@dataclass(frozen=True)
class PreviewNavState:
    """Current selection (0-based indices)."""

    source_key: Any = None
    frame_idx: int = 0
    channel_idx: int = 0
    z_idx: int = 0


class PreviewNavigationControls(QWidget):
    """
    Shared navigation row for previews.

    Emits nav_changed(state, reason) where reason is one of:
    'source' | 'frame' | 'channel' | 'z' | 'init'.

    Optional ``inline_trailer`` is appended before the stretch (e.g. Re-apply).
    Optional ``inline_end`` is appended after overlays (before the stretch), e.g. ``|`` + cache combo.
    Optional ``trailing_end`` is pinned after the stretch (right edge), e.g. Clear Cache aligned with a row below.
    """

    nav_changed = pyqtSignal(object, str)  # (PreviewNavState, reason)
    overlay_changed = pyqtSignal(str, bool)  # ("show_cells"|"show_foci", checked)
    foci_channel_overlay_changed = pyqtSignal(int, bool)  # (channel_index, checked)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        spec: Optional[PreviewNavSpec] = None,
        on_change: Optional[Callable[[PreviewNavState, str], None]] = None,
        inline_trailer: Optional[QWidget] = None,
        inline_end: Optional[QWidget] = None,
        trailing_end: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._spec = spec or PreviewNavSpec(frame_count=1, show_image=False, show_channel=False, show_z=False)
        self._state = PreviewNavState()
        self._on_change = on_change
        self._inline_trailer = inline_trailer
        self._inline_end = inline_end
        self._trailing_end = trailing_end
        self._nav_breakpoint = 700
        self._source_widgets: list[QWidget] = []
        self._overlay_widgets: list[QWidget] = []
        self._action_widgets: list[QWidget] = []
        self._last_nav_narrow: Optional[bool] = None
        self._nav_layout_dirty = True

        self._root = QVBoxLayout(self)
        # Small bottom inset so bordered compact buttons (e.g. Clear Cache) are not clipped by the row.
        self._root.setContentsMargins(0, 0, 0, 8)
        self._root.setSpacing(6)
        self._rows_host = QWidget(self)
        self._rows_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        self._root.addWidget(self._rows_host)

        self._image_label = QLabel("View:")
        self._image_combo = QComboBox()
        self._image_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        try:
            from bactomate.gui.styles import PREVIEW_NAV_COMBO_STYLESHEET

            self._image_combo.setStyleSheet(PREVIEW_NAV_COMBO_STYLESHEET + " QComboBox { font-size: 11pt; }")
        except Exception:
            pass
        self._image_combo.setMinimumWidth(180)
        self._image_combo.setFixedHeight(38)
        self._frame_label = QLabel("Frame:")
        self._frame_spin = QSpinBox()
        # Prevent intermediate jumps while typing (e.g. '55' briefly becoming '5').
        try:
            self._frame_spin.setKeyboardTracking(False)
        except Exception:
            pass
        self._frame_of_label = QLabel("")
        self._show_cells_cb = QCheckBox("Show cell overlay")
        self._show_cells_cb.setToolTip("Show or hide the segmentation/filter mask overlay.")
        self._show_foci_cb = QCheckBox("Show foci overlay")
        self._show_foci_cb.setToolTip("Show or hide foci markers without re-running detection.")
        self._foci_channel_cbs: list[QCheckBox] = []

        self._channel_label = QLabel("Channel:")
        self._channel_spin = QSpinBox()
        try:
            self._channel_spin.setKeyboardTracking(False)
        except Exception:
            pass

        self._z_label = QLabel("Z:")
        self._z_spin = QSpinBox()
        try:
            self._z_spin.setKeyboardTracking(False)
        except Exception:
            pass
        for spin in (self._frame_spin, self._channel_spin, self._z_spin):
            spin.setFixedHeight(38)
            spin.setMinimumWidth(64)
            spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            try:
                from bactomate.gui.styles import PREVIEW_NAV_SPINBOX_STYLESHEET

                spin.setStyleSheet(PREVIEW_NAV_SPINBOX_STYLESHEET)
            except Exception:
                pass
        for lbl in (self._image_label, self._frame_label, self._frame_of_label, self._channel_label, self._z_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            lbl.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        # Match Spotiflow / foci param row labels (e.g. "Peaks min prob"): same size, weight, hue.
        _nav_lbl = "QLabel { color: #c8d0dc; font-size: 10pt; font-weight: 500; }"
        self._image_label.setStyleSheet(_nav_lbl)
        self._frame_label.setStyleSheet(_nav_lbl)
        self._channel_label.setStyleSheet(_nav_lbl)
        self._z_label.setStyleSheet(_nav_lbl)
        self._frame_of_label.setStyleSheet(_nav_lbl)
        _nav_cb = "QCheckBox { font-size: 10pt; color: #dce4ee; spacing: 6px; }"
        self._show_cells_cb.setStyleSheet(_nav_cb)
        self._show_foci_cb.setStyleSheet(_nav_cb)

        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.timeout.connect(lambda: self._commit_spin("frame"))

        self._channel_timer = QTimer(self)
        self._channel_timer.setSingleShot(True)
        self._channel_timer.timeout.connect(lambda: self._commit_spin("channel"))

        self._z_timer = QTimer(self)
        self._z_timer.setSingleShot(True)
        self._z_timer.timeout.connect(lambda: self._commit_spin("z"))

        self._wire_signals()
        self.set_spec(self._spec, emit_init=False)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reflow_nav_rows()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._reflow_nav_rows()

    def spec(self) -> PreviewNavSpec:
        return self._spec

    def state(self) -> PreviewNavState:
        return self._state

    def set_on_change(self, cb: Optional[Callable[[PreviewNavState, str], None]]) -> None:
        self._on_change = cb

    def set_spec(self, spec: PreviewNavSpec, *, emit_init: bool = True) -> None:
        self._spec = spec
        self._rebuild_ui()
        self._apply_spec_ranges()
        self._sync_ui_from_state(clamp=True)
        if emit_init:
            self._emit("init")

    def set_state(self, state: PreviewNavState, *, emit: bool = True, reason: str = "init") -> None:
        self._state = state
        self._sync_ui_from_state(clamp=True)
        if emit:
            self._emit(reason)

    def selected_source(self) -> Optional[PreviewSource]:
        if not self._spec.show_image:
            return None
        idx = int(self._image_combo.currentIndex())
        if idx < 0:
            return None
        try:
            return self._spec.image_sources[idx]
        except Exception:
            return None

    def foci_channel_overlay_visible(self) -> list[bool]:
        """Per-channel foci ring visibility (empty when using the single foci toggle)."""
        if self._foci_channel_cbs:
            return [bool(cb.isChecked()) for cb in self._foci_channel_cbs]
        return []

    def set_foci_channel_overlay_visible(self, states: list[bool]) -> None:
        if not self._foci_channel_cbs:
            return
        for i, cb in enumerate(self._foci_channel_cbs):
            if i >= len(states):
                break
            cb.blockSignals(True)
            try:
                cb.setChecked(bool(states[i]))
            finally:
                cb.blockSignals(False)

    # ---- internals ----------------------------------------------------

    def _wire_signals(self) -> None:
        self._image_combo.currentIndexChanged.connect(lambda *_: self._commit_combo())
        self._show_cells_cb.toggled.connect(lambda checked: self._emit_overlay("show_cells", bool(checked)))
        self._show_foci_cb.toggled.connect(lambda checked: self._emit_overlay("show_foci", bool(checked)))

        def _debounce(timer: QTimer) -> None:
            timer.stop()
            timer.start(max(0, int(self._spec.debounce_ms)))

        self._frame_spin.valueChanged.connect(lambda *_: _debounce(self._frame_timer))
        self._frame_spin.editingFinished.connect(lambda: self._commit_spin("frame", immediate=True))

        self._channel_spin.valueChanged.connect(lambda *_: _debounce(self._channel_timer))
        self._channel_spin.editingFinished.connect(lambda: self._commit_spin("channel", immediate=True))

        self._z_spin.valueChanged.connect(lambda *_: _debounce(self._z_timer))
        self._z_spin.editingFinished.connect(lambda: self._commit_spin("z", immediate=True))

    def _nav_separator(self) -> QFrame:
        line = QFrame()
        line.setFixedWidth(1)
        line.setMaximumWidth(1)
        line.setMinimumHeight(22)
        line.setStyleSheet(
            "QFrame { background-color: #3d4d62; border: none; margin-left: 8px; margin-right: 8px; }"
        )
        return line

    def _new_nav_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        return row

    def _clear_rows_host(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            container = item.widget()
            if container is not None:
                container.deleteLater()

    def _add_nav_row(self, row: QHBoxLayout) -> None:
        wrap = QWidget(self._rows_host)
        wrap.setMinimumHeight(46)
        wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        wrap.setLayout(row)
        self._rows_layout.addWidget(wrap)

    def _reflow_nav_rows(self) -> None:
        if not self._source_widgets and not self._overlay_widgets and not self._action_widgets:
            return
        width = self.width()
        if width <= 0:
            width = self._nav_breakpoint + 1
        narrow = width < self._nav_breakpoint
        if not self._nav_layout_dirty and self._last_nav_narrow == narrow:
            return

        for widget in (*self._source_widgets, *self._overlay_widgets, *self._action_widgets):
            widget.setParent(self._rows_host)

        self._clear_rows_host()

        if not narrow:
            row = self._new_nav_row()
            for widget in self._source_widgets:
                row.addWidget(widget, 0)
            for widget in self._overlay_widgets:
                row.addWidget(widget, 0)
            for widget in self._action_widgets:
                if widget is self._inline_end:
                    row.addWidget(widget, 1)
                else:
                    row.addWidget(widget, 0)
            if self._inline_end is None:
                row.addStretch(1)
            self._add_nav_row(row)
            self._last_nav_narrow = narrow
            self._nav_layout_dirty = False
            return

        source_row = self._new_nav_row()
        for widget in self._source_widgets:
            source_row.addWidget(widget, 0)
        source_row.addStretch(1)
        self._add_nav_row(source_row)

        if self._overlay_widgets:
            overlay_row = self._new_nav_row()
            for widget in self._overlay_widgets:
                overlay_row.addWidget(widget, 0)
            overlay_row.addStretch(1)
            self._add_nav_row(overlay_row)

        if self._action_widgets:
            action_row = self._new_nav_row()
            for widget in self._action_widgets:
                if widget is self._inline_end:
                    action_row.addWidget(widget, 1)
                else:
                    action_row.addWidget(widget, 0)
            if self._inline_end is None:
                action_row.addStretch(1)
            self._add_nav_row(action_row)

        self._last_nav_narrow = narrow
        self._nav_layout_dirty = False
        self.updateGeometry()

    def _rebuild_ui(self) -> None:
        self._source_widgets = []
        self._overlay_widgets = []
        self._action_widgets = []
        self._nav_layout_dirty = True

        if self._spec.show_image:
            self._source_widgets.extend(
                [self._image_label, self._image_combo, self._nav_separator()]
            )
            self._populate_sources()

        self._source_widgets.extend(
            [self._frame_label, self._frame_spin, self._frame_of_label]
        )

        if self._spec.show_channel:
            self._source_widgets.extend([self._channel_label, self._channel_spin])

        if self._spec.show_z:
            self._source_widgets.extend([self._z_label, self._z_spin])

        _foci_ch_labels = tuple(getattr(self._spec, "foci_channel_overlay_labels", ()) or ())
        self._foci_channel_cbs = []
        has_overlays = (
            self._spec.show_cell_overlay_toggle
            or self._spec.show_foci_overlay_toggle
            or bool(_foci_ch_labels)
        )
        if has_overlays:
            self._overlay_widgets.append(self._nav_separator())

        if self._spec.show_cell_overlay_toggle:
            self._overlay_widgets.append(self._show_cells_cb)

        if _foci_ch_labels:
            _nav_cb_style = "QCheckBox { font-size: 10pt; color: #dce4ee; spacing: 6px; }"
            _ring_color_names = ("orange", "cyan", "magenta", "lime", "yellow")
            defaults = tuple(getattr(self._spec, "foci_channel_overlay_defaults", ()) or ())
            for i, lbl in enumerate(_foci_ch_labels):
                cb = QCheckBox(str(lbl))
                cb.setStyleSheet(_nav_cb_style)
                cname = _ring_color_names[i % len(_ring_color_names)]
                cb.setToolTip(f"Show foci rings ({cname}) for {lbl}")

                def _mk_handler(idx: int):
                    return lambda checked: self.foci_channel_overlay_changed.emit(idx, bool(checked))

                cb.toggled.connect(_mk_handler(i))
                self._overlay_widgets.append(cb)
                self._foci_channel_cbs.append(cb)
                if i < len(defaults):
                    cb.setChecked(bool(defaults[i]))
        elif self._spec.show_foci_overlay_toggle:
            self._overlay_widgets.append(self._show_foci_cb)

        if self._inline_trailer is not None:
            self._action_widgets.append(self._inline_trailer)
        if self._inline_end is not None:
            self._action_widgets.append(self._inline_end)
        if self._trailing_end is not None:
            self._action_widgets.append(self._trailing_end)

        self._reflow_nav_rows()

    def _populate_sources(self) -> None:
        self._image_combo.blockSignals(True)
        try:
            self._image_combo.clear()
            for s in self._spec.image_sources:
                self._image_combo.addItem(str(s.label), s.key)
        finally:
            self._image_combo.blockSignals(False)

    def _apply_spec_ranges(self) -> None:
        fc = max(1, int(self._spec.frame_count))
        self._frame_spin.setRange(1, fc)
        if bool(getattr(self._spec, "show_frame_of_total", True)):
            self._frame_of_label.setText(f"of {fc}")
            self._frame_of_label.show()
        else:
            self._frame_of_label.setText("")
            self._frame_of_label.hide()

        cc = max(1, int(self._spec.channel_count))
        self._channel_spin.setRange(1, cc)

        zc = max(1, int(self._spec.z_count))
        self._z_spin.setRange(1, zc)

        self._show_cells_cb.blockSignals(True)
        self._show_foci_cb.blockSignals(True)
        try:
            self._show_cells_cb.setChecked(bool(self._spec.cell_overlay_default))
            foci_available = bool(self._spec.foci_overlay_available)
            self._show_foci_cb.setEnabled(foci_available)
            self._show_foci_cb.setChecked(bool(self._spec.foci_overlay_default) if foci_available else False)
            defaults = tuple(getattr(self._spec, "foci_channel_overlay_defaults", ()) or ())
            for i, cb in enumerate(self._foci_channel_cbs):
                cb.setEnabled(foci_available)
                on = bool(defaults[i]) if i < len(defaults) else bool(self._spec.foci_overlay_default)
                cb.setChecked(on if foci_available else False)
        finally:
            self._show_cells_cb.blockSignals(False)
            self._show_foci_cb.blockSignals(False)

    def _sync_ui_from_state(self, *, clamp: bool) -> None:
        st = self._state
        spec = self._spec

        # Clamp indices.
        fi = int(st.frame_idx)
        ci = int(st.channel_idx)
        zi = int(st.z_idx)
        if clamp:
            fi = max(0, min(fi, max(0, int(spec.frame_count) - 1)))
            ci = max(0, min(ci, max(0, int(spec.channel_count) - 1)))
            zi = max(0, min(zi, max(0, int(spec.z_count) - 1)))

        resolved_key = st.source_key
        # Combo: try to match by key; fallback to first source when key missing (e.g. foci rows removed).
        if spec.show_image:
            idx = 0
            if spec.image_sources:
                if st.source_key is not None:
                    for i, s in enumerate(spec.image_sources):
                        if s.key == st.source_key:
                            idx = i
                            resolved_key = st.source_key
                            break
                    else:
                        resolved_key = spec.image_sources[0].key
                        idx = 0
                else:
                    resolved_key = spec.image_sources[0].key
                    idx = 0
            self._image_combo.blockSignals(True)
            try:
                self._image_combo.setCurrentIndex(max(0, idx))
            finally:
                self._image_combo.blockSignals(False)

        # Spins
        self._frame_spin.blockSignals(True)
        self._channel_spin.blockSignals(True)
        self._z_spin.blockSignals(True)
        try:
            self._frame_spin.setValue(fi + 1)
            self._channel_spin.setValue(ci + 1)
            self._z_spin.setValue(zi + 1)
        finally:
            self._frame_spin.blockSignals(False)
            self._channel_spin.blockSignals(False)
            self._z_spin.blockSignals(False)

        rk = resolved_key
        self._state = PreviewNavState(source_key=rk, frame_idx=fi, channel_idx=ci, z_idx=zi)

    def _commit_combo(self) -> None:
        if not self._spec.show_image:
            return
        idx = int(self._image_combo.currentIndex())
        key = None
        if 0 <= idx < len(self._spec.image_sources):
            key = self._spec.image_sources[idx].key
        self._state = PreviewNavState(
            source_key=key,
            frame_idx=self._state.frame_idx,
            channel_idx=self._state.channel_idx,
            z_idx=self._state.z_idx,
        )
        self._emit("source")

    def _commit_spin(self, which: str, *, immediate: bool = False) -> None:
        # Stop debounce timers so we don't double-emit.
        if which == "frame":
            self._frame_timer.stop()
            val = int(self._frame_spin.value()) - 1
            val = max(0, min(val, max(0, int(self._spec.frame_count) - 1)))
            self._state = PreviewNavState(
                source_key=self._state.source_key,
                frame_idx=val,
                channel_idx=self._state.channel_idx,
                z_idx=self._state.z_idx,
            )
            self._emit("frame")
            return
        if which == "channel":
            self._channel_timer.stop()
            val = int(self._channel_spin.value()) - 1
            val = max(0, min(val, max(0, int(self._spec.channel_count) - 1)))
            self._state = PreviewNavState(
                source_key=self._state.source_key,
                frame_idx=self._state.frame_idx,
                channel_idx=val,
                z_idx=self._state.z_idx,
            )
            self._emit("channel")
            return
        if which == "z":
            self._z_timer.stop()
            val = int(self._z_spin.value()) - 1
            val = max(0, min(val, max(0, int(self._spec.z_count) - 1)))
            self._state = PreviewNavState(
                source_key=self._state.source_key,
                frame_idx=self._state.frame_idx,
                channel_idx=self._state.channel_idx,
                z_idx=val,
            )
            self._emit("z")
            return

    def _emit(self, reason: str) -> None:
        st = self._state
        self.nav_changed.emit(st, str(reason))
        if self._on_change is not None:
            try:
                self._on_change(st, str(reason))
            except Exception:
                pass

    def _emit_overlay(self, key: str, checked: bool) -> None:
        self.overlay_changed.emit(str(key), bool(checked))

