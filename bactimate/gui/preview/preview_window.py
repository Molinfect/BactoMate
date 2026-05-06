from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.image_canvas import ImageCanvas


@dataclass
class PreviewState:
    frame_idx: int = 0
    channel_idx: int = 0
    source_key: Any = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderResult:
    pixmap: Any  # QPixmap, but keep Qt import surface small here
    image_size: tuple[int, int]  # (w, h) in image coordinates
    lut_frame: Optional[np.ndarray] = None  # 2D frame used for LUT/histogram
    title_text: Optional[str] = None
    info_text: Optional[str] = None


class WorkerGuard:
    """Stage-provided wrapper around a worker/QThread-like object."""

    def __init__(
        self,
        *,
        is_running: Callable[[], bool],
        wait: Callable[[int], bool],
        request_stop: Optional[Callable[[], None]] = None,
        disconnect: Optional[Callable[[], None]] = None,
        on_close_while_running: str = "block_with_warning",
        warning_title: str = "Work in progress",
        warning_text: str = "Please wait for the current operation to finish before closing the preview.",
    ) -> None:
        self._is_running = is_running
        self._wait = wait
        self._request_stop = request_stop
        self._disconnect = disconnect
        self.on_close_while_running = on_close_while_running
        self.warning_title = warning_title
        self.warning_text = warning_text

    def is_running(self) -> bool:
        try:
            return bool(self._is_running())
        except Exception:
            return False

    def request_stop(self) -> None:
        if self._request_stop is None:
            return
        try:
            self._request_stop()
        except Exception:
            return

    def wait(self, ms: int) -> bool:
        try:
            return bool(self._wait(int(ms)))
        except Exception:
            return True

    def disconnect(self) -> None:
        if self._disconnect is None:
            return
        try:
            self._disconnect()
        except Exception:
            return


def _initial_dialog_dimensions(
    parent: Optional[QWidget],
    min_w: int,
    min_h: int,
    *,
    size_ratio: float = 0.85,
    screen_margin: int = 48,
) -> tuple[int, int, int, int]:
    """
    Return (effective_min_w, effective_min_h, initial_w, initial_h) for a preview dialog.

    Preview dialogs open as a fraction of the available desktop on the active screen,
    while minimums are capped so small displays are still usable.
    """
    screen = None
    try:
        if parent is not None:
            screen = parent.screen()
    except Exception:
        screen = None
    if screen is None:
        screen = QGuiApplication.screenAt(QCursor.pos())
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    if screen is None:
        return (int(min_w), int(min_h), int(min_w), int(min_h))

    avail = screen.availableGeometry()
    uw = max(0, avail.width() - screen_margin)
    uh = max(0, avail.height() - screen_margin)
    eff_min_w = min(int(min_w), uw)
    eff_min_h = min(int(min_h), uh)
    ratio = max(0.1, min(float(size_ratio), 1.0))
    init_w = min(uw, max(eff_min_w, int(round(uw * ratio))))
    init_h = min(uh, max(eff_min_h, int(round(uh * ratio))))
    return (eff_min_w, eff_min_h, init_w, init_h)


class CollapsibleSection(QWidget):
    """A reusable expandable/collapsible section.

    Wraps an existing content widget under a clickable header (QToolButton)
    that toggles the content's visibility. The arrow indicator flips between
    Down (expanded) and Right (collapsed). Layout-friendly: when collapsed,
    the content is removed from the visible layout and the section shrinks
    to just the header height, so any sibling stretchable widgets (like the
    image canvas) reclaim the space.
    """

    toggled = pyqtSignal(bool)  # True when expanded after the toggle

    def __init__(
        self,
        title: str,
        content: QWidget,
        parent: Optional[QWidget] = None,
        *,
        expanded: bool = True,
    ) -> None:
        super().__init__(parent)
        self._content = content
        self._expanded = bool(expanded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._toggle = QToolButton(self)
        self._toggle.setText(str(title))
        self._toggle.setCheckable(True)
        self._toggle.setChecked(self._expanded)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: 600; padding: 4px 6px; text-align: left; }"
            "QToolButton:hover { background-color: rgba(255,255,255,0.06); }"
        )
        self._toggle.clicked.connect(self._on_clicked)
        outer.addWidget(self._toggle)

        outer.addWidget(self._content)
        self._content.setVisible(self._expanded)

    def _on_clicked(self, checked: bool) -> None:
        self._expanded = bool(checked)
        self._content.setVisible(self._expanded)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )
        self.toggled.emit(self._expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, value: bool) -> None:
        if bool(value) == self._expanded:
            return
        self._toggle.setChecked(bool(value))
        self._on_clicked(bool(value))

    def content(self) -> QWidget:
        return self._content


class HistogramLUTWidget(QWidget):
    """Interactive histogram LUT: drag min/max lines to adjust vmin/vmax."""

    lut_changed = pyqtSignal(float, float)  # vmin, vmax

    # Default header used when a wrapping CollapsibleSection is created.
    DEFAULT_TITLE = "LUT (click/drag lines to adjust min/max)"

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        auto_lo_pct: float = 0.01,
        auto_hi_pct: float = 99.99,
        max_bins: int = 256,
    ) -> None:
        super().__init__(parent)
        self._auto_lo_pct = float(auto_lo_pct)
        self._auto_hi_pct = float(auto_hi_pct)
        self._max_bins = int(max_bins)

        self._frame_min: Optional[float] = None
        self._frame_max: Optional[float] = None
        self._vmin: Optional[float] = None
        self._vmax: Optional[float] = None
        self._dragging: Optional[str] = None  # 'min'|'max'|None
        self._is_updating = False

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
        except Exception:
            self._has_hist = False
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel("Histogram LUT unavailable (matplotlib not installed)."))
            return

        self._has_hist = True
        self._FigureCanvas = FigureCanvas
        self._Figure = Figure

        gl = QVBoxLayout(self)
        gl.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(10, 3.0))
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(180)
        self._canvas.setMaximumHeight(180)
        self._ax = self._fig.add_subplot(111)
        self._fig.patch.set_facecolor("#2b2b2b")
        self._ax.set_facecolor("#2b2b2b")
        self._ax.tick_params(colors="white", labelsize=9)
        for spine in self._ax.spines.values():
            spine.set_color("white")
        self._ax.xaxis.label.set_color("white")
        self._ax.yaxis.label.set_color("white")
        self._fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.15)
        self._canvas.setStyleSheet("background-color: #2b2b2b;")
        gl.addWidget(self._canvas)

        self._hist_vmin_line = None
        self._hist_vmax_line = None

        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    def has_histogram(self) -> bool:
        return bool(getattr(self, "_has_hist", False))

    def current_lut(self) -> tuple[Optional[float], Optional[float]]:
        return self._vmin, self._vmax

    def set_lut(self, vmin: Optional[float], vmax: Optional[float]) -> None:
        self._vmin = float(vmin) if vmin is not None else None
        self._vmax = float(vmax) if vmax is not None else None
        if self.has_histogram() and self._frame_min is not None and self._frame_max is not None:
            self._redraw_histogram()

    def set_frame(self, frame_2d: Optional[np.ndarray], *, preserve_user_lut: bool = True) -> None:
        if frame_2d is None or not isinstance(frame_2d, np.ndarray) or frame_2d.size == 0:
            self._frame_min = None
            self._frame_max = None
            if self.has_histogram():
                self._ax.clear()
                self._ax.set_xlabel("Intensity", color="white", fontsize=9)
                self._ax.set_ylabel("Log(Count+1)", color="white", fontsize=9)
                self._canvas.draw()
            return

        flat = np.asarray(frame_2d, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return

        fmin, fmax = float(flat.min()), float(flat.max())
        self._frame_min = fmin
        self._frame_max = fmax

        if not preserve_user_lut or self._vmin is None or self._vmax is None or self._vmax <= self._vmin:
            # Wide percentile band so we use most of the dynamic range.
            try:
                lo = float(np.percentile(flat, self._auto_lo_pct))
                hi = float(np.percentile(flat, self._auto_hi_pct))
            except Exception:
                lo, hi = fmin, fmax
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = fmin, fmax if fmax > fmin else fmin + 1.0
            self._vmin, self._vmax = lo, hi
            self.lut_changed.emit(float(self._vmin), float(self._vmax))

        if self.has_histogram():
            self._redraw_histogram(frame_2d=np.asarray(frame_2d, dtype=np.float64))

    def _redraw_histogram(self, frame_2d: Optional[np.ndarray] = None) -> None:
        if not self.has_histogram():
            return
        if self._frame_min is None or self._frame_max is None or self._vmin is None or self._vmax is None:
            return

        self._ax.clear()
        if frame_2d is None:
            # no raw frame cached; can't draw histogram bars
            self._ax.set_xlabel("Intensity", color="white", fontsize=9)
            self._ax.set_ylabel("Log(Count+1)", color="white", fontsize=9)
            self._canvas.draw_idle()
            return

        flat = np.asarray(frame_2d, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        flat_hist = flat[flat > 0]
        if flat_hist.size == 0:
            flat_hist = flat

        fmin, fmax = float(self._frame_min), float(self._frame_max)
        bins = max(2, min(self._max_bins, int(max(2.0, fmax - fmin + 1.0))))
        hist, bin_edges = np.histogram(flat_hist, bins=bins, range=(fmin, fmax))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self._ax.bar(
            bin_centers,
            np.log1p(hist),
            width=bin_edges[1] - bin_edges[0],
            color="#4A90E2",
            alpha=0.8,
            edgecolor="none",
        )

        vmin = float(self._vmin)
        vmax = float(self._vmax)
        self._hist_vmin_line = self._ax.axvline(vmin, color="#FF6B6B", linewidth=2, linestyle="--", label=f"Min: {vmin:.1f}")
        self._hist_vmax_line = self._ax.axvline(vmax, color="#4ECDC4", linewidth=2, linestyle="--", label=f"Max: {vmax:.1f}")
        self._ax.axvspan(vmin, vmax, alpha=0.15, color="yellow")

        self._ax.set_xlabel("Intensity", color="white", fontsize=9)
        self._ax.set_ylabel("Log(Count+1)", color="white", fontsize=9)
        self._ax.set_title("Click/drag lines to adjust min/max", color="white", fontsize=9, pad=10)
        self._ax.legend(
            loc="upper right",
            facecolor="#2b2b2b",
            edgecolor="white",
            labelcolor="white",
            fontsize=8,
            framealpha=0.9,
            bbox_to_anchor=(0.98, 0.98),
        )
        self._ax.grid(True, alpha=0.25, color="white", linestyle="-", linewidth=0.5)
        self._ax.margins(x=0.02, y=0.08)
        try:
            self._fig.tight_layout(pad=2.0)
        except Exception:
            pass
        self._canvas.draw()

    def _on_press(self, event) -> None:
        if not self.has_histogram():
            return
        if event.inaxes != self._ax or event.xdata is None:
            return
        if self._vmin is None or self._vmax is None or self._frame_min is None or self._frame_max is None:
            return
        x = float(event.xdata)
        vmin, vmax = float(self._vmin), float(self._vmax)
        fmin, fmax = float(self._frame_min), float(self._frame_max)
        threshold = max((fmax - fmin) * 0.05, 1.0)
        if abs(x - vmin) < threshold and abs(x - vmin) <= abs(x - vmax):
            self._dragging = "min"
        elif abs(x - vmax) < threshold:
            self._dragging = "max"
        else:
            self._dragging = "min" if x < (vmin + vmax) / 2 else "max"

    def _on_motion(self, event) -> None:
        if not self.has_histogram():
            return
        if self._dragging is None or event.inaxes != self._ax or event.xdata is None:
            return
        if self._vmin is None or self._vmax is None or self._frame_min is None or self._frame_max is None:
            return
        if self._is_updating:
            return

        self._is_updating = True
        try:
            x = float(event.xdata)
            fmin, fmax = float(self._frame_min), float(self._frame_max)
            vmin, vmax = float(self._vmin), float(self._vmax)
            if self._dragging == "min":
                new_min = max(fmin, min(x, vmax - 1e-6))
                self._vmin = float(new_min)
            else:
                new_max = min(fmax, max(x, vmin + 1e-6))
                self._vmax = float(new_max)

            self.lut_changed.emit(float(self._vmin), float(self._vmax))
            # Update only line positions for responsiveness.
            if self._hist_vmin_line is not None:
                self._hist_vmin_line.set_xdata([float(self._vmin), float(self._vmin)])
            if self._hist_vmax_line is not None:
                self._hist_vmax_line.set_xdata([float(self._vmax), float(self._vmax)])
            self._canvas.draw_idle()
        finally:
            self._is_updating = False

    def _on_release(self, event) -> None:
        self._dragging = None


class PreviewDialogBase(QDialog):
    """Reusable preview shell: top controls, ImageCanvas, histogram LUT, bottom controls, actions."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        title: str,
        top_controls: Optional[QWidget] = None,
        bottom_controls: Optional[QWidget] = None,
        navigation_controls: Optional[QWidget] = None,
        primary_action_label: str = "Apply",
        primary_action_callback: Optional[Callable[[], None]] = None,
        render_callback: Optional[Callable[[PreviewState], RenderResult]] = None,
        worker_guard: Optional[WorkerGuard] = None,
        minimum_size: tuple[int, int] = (900, 700),
        top_controls_title: Optional[str] = None,
        top_controls_expanded: bool = True,
        lut_title: Optional[str] = None,
        lut_expanded: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(str(title))
        self.setModal(False)
        eff_min_w, eff_min_h, initial_w, initial_h = _initial_dialog_dimensions(
            parent,
            int(minimum_size[0]),
            int(minimum_size[1]),
            size_ratio=0.85,
        )
        self.setMinimumSize(eff_min_w, eff_min_h)
        self.resize(initial_w, initial_h)

        self._state = PreviewState()
        self._worker_guard = worker_guard
        self._render_callback = render_callback
        self._primary_action_callback = primary_action_callback
        self._dialog_closed = False

        self._layout = QVBoxLayout(self)

        self._title_label = QLabel("")
        self._title_label.setStyleSheet("font-weight: bold; color: #ddd;")
        self._layout.addWidget(self._title_label)
        # Public handles for stage code that previously owned these labels.
        self.title_label = self._title_label

        # Wrap the parameters/top controls in a collapsible section when the
        # caller provided a title. Without a title we keep the legacy behavior
        # (raw widget added directly) so existing layouts are unchanged.
        self.top_section: Optional[CollapsibleSection] = None
        if top_controls is not None:
            if top_controls_title:
                self.top_section = CollapsibleSection(
                    str(top_controls_title),
                    top_controls,
                    expanded=bool(top_controls_expanded),
                )
                self._layout.addWidget(self.top_section)
            else:
                self._layout.addWidget(top_controls)

        self.canvas = ImageCanvas()
        self.canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
        self.canvas.setMinimumSize(400, 300)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        self._layout.addWidget(self.canvas, 1)

        self.lut_widget = HistogramLUTWidget(self)
        self.lut_widget.lut_changed.connect(self._on_lut_changed)
        # Always wrap the histogram in a collapsible so users can free up
        # vertical space for the image canvas when the LUT is not in active use.
        self.lut_section = CollapsibleSection(
            str(lut_title) if lut_title else HistogramLUTWidget.DEFAULT_TITLE,
            self.lut_widget,
            expanded=bool(lut_expanded),
        )
        self._layout.addWidget(self.lut_section)

        # Back-compat: existing callers use bottom_controls for navigation rows.
        self.bottom_controls = bottom_controls
        self.navigation_controls = navigation_controls if navigation_controls is not None else bottom_controls
        if self.navigation_controls is not None:
            self._layout.addWidget(self.navigation_controls)

        actions = QHBoxLayout()
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        actions.addWidget(self._info_label)
        self.info_label = self._info_label
        actions.addStretch()
        self.primary_button = QPushButton(str(primary_action_label))
        self.primary_button.clicked.connect(self._on_primary_clicked)
        actions.addWidget(self.primary_button)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        actions.addWidget(close_btn)
        self._layout.addLayout(actions)

    def set_navigation_controls(self, w: Optional[QWidget]) -> None:
        """Replace the navigation row widget (below LUT, above action buttons)."""
        old = getattr(self, "navigation_controls", None)
        if old is w:
            return
        try:
            if old is not None:
                self._layout.removeWidget(old)
                old.setParent(None)
        except Exception:
            pass
        self.navigation_controls = w
        if w is not None:
            try:
                # Insert right above the action row (last item).
                idx = max(0, self._layout.count() - 1)
                self._layout.insertWidget(idx, w)
            except Exception:
                self._layout.addWidget(w)

    def state(self) -> PreviewState:
        return self._state

    def set_primary_action(
        self,
        callback: Optional[Callable[[], None]],
        *,
        label: Optional[str] = None,
        enabled: Optional[bool] = None,
        visible: Optional[bool] = None,
    ) -> None:
        self._primary_action_callback = callback
        if label is not None:
            try:
                self.primary_button.setText(str(label))
            except Exception:
                pass
        if enabled is not None:
            try:
                self.primary_button.setEnabled(bool(enabled))
            except Exception:
                pass
        if visible is not None:
            try:
                self.primary_button.setVisible(bool(visible))
            except Exception:
                pass

    def set_render_callback(self, cb: Optional[Callable[[PreviewState], RenderResult]]) -> None:
        """Replace the render callback used by refresh()."""
        self._render_callback = cb

    def set_worker_guard(self, guard: Optional[WorkerGuard]) -> None:
        self._worker_guard = guard

    def set_frame_idx(self, idx: int) -> None:
        self._state.frame_idx = max(0, int(idx))

    def set_channel_idx(self, idx: int) -> None:
        self._state.channel_idx = max(0, int(idx))

    def set_source_key(self, key: Any) -> None:
        self._state.source_key = key

    def set_extra(self, key: str, value: Any) -> None:
        self._state.extra[str(key)] = value

    def refresh(self) -> None:
        cb = self._render_callback
        if cb is None:
            return
        try:
            result = cb(self._state)
        except Exception as e:
            self._info_label.setText(f"Preview error: {str(e)[:80]}")
            return
        if result.title_text is not None:
            self._title_label.setText(str(result.title_text))
        if result.info_text is not None:
            self._info_label.setText(str(result.info_text))
        try:
            self.canvas.set_pixmap(result.pixmap, result.image_size, reset_view=False)
        except Exception:
            pass
        if result.lut_frame is not None:
            # Use this frame for histogram; preserve current LUT so user settings persist.
            self.lut_widget.set_frame(np.asarray(result.lut_frame), preserve_user_lut=True)

    def closeEvent(self, event) -> None:
        self._dialog_closed = True
        g = self._worker_guard
        if g is not None and g.is_running():
            g.request_stop()
            if g.on_close_while_running == "defer_hide":
                self.hide()
                event.ignore()
                return
            ok = g.wait(60000)
            if not ok:
                event.ignore()
                QMessageBox.warning(self, g.warning_title, g.warning_text, QMessageBox.StandardButton.Ok)
                return
        if g is not None:
            g.disconnect()
        super().closeEvent(event)

    def _on_primary_clicked(self) -> None:
        if self._primary_action_callback is None:
            return
        try:
            self._primary_action_callback()
        except Exception as e:
            self._info_label.setText(f"Action error: {str(e)[:80]}")

    def _on_lut_changed(self, vmin: float, vmax: float) -> None:
        self._state.vmin = float(vmin)
        self._state.vmax = float(vmax)
        # Renderers generally need to redraw when LUT changes.
        self.refresh()
