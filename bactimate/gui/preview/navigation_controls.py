from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QSpinBox, QWidget


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
    show_cell_overlay_toggle: bool = False
    show_foci_overlay_toggle: bool = False
    cell_overlay_default: bool = True
    foci_overlay_available: bool = False
    foci_overlay_default: bool = True
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
    """

    nav_changed = pyqtSignal(object, str)  # (PreviewNavState, reason)
    overlay_changed = pyqtSignal(str, bool)  # ("show_cells"|"show_foci", checked)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        spec: Optional[PreviewNavSpec] = None,
        on_change: Optional[Callable[[PreviewNavState, str], None]] = None,
    ) -> None:
        super().__init__(parent)
        self._spec = spec or PreviewNavSpec(frame_count=1, show_image=False, show_channel=False, show_z=False)
        self._state = PreviewNavState()
        self._on_change = on_change

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

        self._image_label = QLabel("Image:")
        self._image_combo = QComboBox()
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

    def _rebuild_ui(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        if self._spec.show_image:
            self._layout.addWidget(self._image_label)
            self._layout.addWidget(self._image_combo)
            self._populate_sources()

        self._layout.addWidget(self._frame_label)
        self._layout.addWidget(self._frame_spin)
        self._layout.addWidget(self._frame_of_label)

        if self._spec.show_cell_overlay_toggle:
            self._layout.addWidget(self._show_cells_cb)

        if self._spec.show_foci_overlay_toggle:
            self._layout.addWidget(self._show_foci_cb)

        if self._spec.show_channel:
            self._layout.addWidget(self._channel_label)
            self._layout.addWidget(self._channel_spin)

        if self._spec.show_z:
            self._layout.addWidget(self._z_label)
            self._layout.addWidget(self._z_spin)

        self._layout.addStretch(1)

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
        self._frame_of_label.setText(f"of {fc}")

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

        # Combo: try to match by key; fallback to 0.
        if spec.show_image:
            idx = 0
            if spec.image_sources and st.source_key is not None:
                for i, s in enumerate(spec.image_sources):
                    if s.key == st.source_key:
                        idx = i
                        break
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

        self._state = PreviewNavState(source_key=st.source_key, frame_idx=fi, channel_idx=ci, z_idx=zi)

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

