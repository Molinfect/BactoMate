"""
Qt-native image canvas: display a QPixmap with zoom (wheel) and pan (middle-drag).
No matplotlib dependency. Optional signals for click/drag in image coordinates.
Zoom/pan tooltip is shown only until the user has used zoom or pan once (then suppressed).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QPointF, QRect, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QKeyEvent, QWheelEvent, QMouseEvent, QResizeEvent, QPaintEvent
from PyQt6.QtWidgets import QWidget


# Substring that identifies the standard zoom/pan hint so we can suppress it once the user has tried zoom/pan
_ZOOM_PAN_TOOLTIP_MARKER = "Zoom: mouse wheel"


class ImageCanvas(QWidget):
    """Widget that displays a pixmap with zoom and pan. Mouse wheel: zoom; middle button + drag: pan."""

    clicked_image = pyqtSignal(int, int)
    drag_started = pyqtSignal(int, int)
    drag_updated = pyqtSignal(int, int)
    drag_finished = pyqtSignal(int, int)
    view_scale_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._image_size: Optional[Tuple[int, int]] = None
        self._zoom = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_pos = QPointF(0.0, 0.0)
        self._dragging_selection = False
        self._left_potential_click = False
        self._left_press_pos = QPointF(0.0, 0.0)
        self.enable_drag_selection = False
        self.selection_button = Qt.MouseButton.LeftButton
        self.click_button = Qt.MouseButton.LeftButton
        self.pan_button = Qt.MouseButton.MiddleButton
        self._base_tooltip: str = ""
        self._foci_hover_data: Optional[List[Tuple[float, float, str, float]]] = None  # (x, y, tooltip, max_radius) in image coords; radius limits hit area
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # allow + / - keyboard zoom when focused

    def set_foci_hover_data(self, data: Optional[List[Tuple[float, float, str, float]]]) -> None:
        """Set optional list of (x, y, tooltip_str, max_radius) for focus peaks. Tooltip shown only when cursor is within max_radius of peak. None to clear."""
        self._foci_hover_data = data
        if data is None:
            super().setToolTip(self._effective_base_tooltip())

    def setToolTip(self, text: str) -> None:
        """Store as base tooltip; if no foci hover data, set as current tooltip. Suppress zoom/pan hint if already seen."""
        self._base_tooltip = text or ""
        try:
            from bactimate.gui.preferences import get_preview_zoom_pan_tip_seen
            if text and _ZOOM_PAN_TOOLTIP_MARKER in text and get_preview_zoom_pan_tip_seen():
                text = ""
        except Exception:
            pass
        if self._foci_hover_data is None:
            super().setToolTip(text)

    def _effective_base_tooltip(self) -> str:
        """Base tooltip to show on hover; empty if zoom/pan hint and user has already used zoom/pan."""
        text = self._base_tooltip or ""
        try:
            from bactimate.gui.preferences import get_preview_zoom_pan_tip_seen
            if text and _ZOOM_PAN_TOOLTIP_MARKER in text and get_preview_zoom_pan_tip_seen():
                return ""
        except Exception:
            pass
        return text

    def _mark_zoom_pan_used(self) -> None:
        """Remember that the user used zoom/pan and clear the long tooltip so it does not pop up again."""
        try:
            from bactimate.gui.preferences import set_preview_zoom_pan_tip_seen
            set_preview_zoom_pan_tip_seen()
            super().setToolTip("")
        except Exception:
            pass

    def set_pixmap(
        self,
        pixmap: QPixmap,
        image_size: Tuple[int, int],
        reset_view: bool = False,
    ) -> None:
        self._pixmap = pixmap
        self._image_size = image_size
        if reset_view:
            self._zoom = 1.0
            self._pan_offset = QPointF(0.0, 0.0)
        self.update()

    def _base_scale(self) -> float:
        if not self._image_size:
            return 1.0
        image_width, image_height = self._image_size
        if image_width == 0 or image_height == 0:
            return 1.0
        return min(self.width() / image_width, self.height() / image_height)

    def view_scale(self) -> float:
        return self._base_scale() * self._zoom

    def _total_offset(self, scale: float) -> QPointF:
        if not self._image_size:
            return QPointF(0.0, 0.0)
        image_width, image_height = self._image_size
        display_width = image_width * scale
        display_height = image_height * scale
        offset_x = (self.width() - display_width) / 2
        offset_y = (self.height() - display_height) / 2
        return QPointF(offset_x, offset_y) + self._pan_offset

    def _widget_to_image(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        if not self._image_size:
            return None
        scale = self._base_scale() * self._zoom
        offset = self._total_offset(scale)
        x = (pos.x() - offset.x()) / scale
        y = (pos.y() - offset.y()) / scale
        image_width, image_height = self._image_size
        if x < 0 or y < 0 or x >= image_width or y >= image_height:
            return None
        return int(x), int(y)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap is None or not self._image_size:
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.1 if angle > 0 else 1 / 1.1
        new_zoom = max(0.1, min(10.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        cursor_pos = event.position()
        scale_before = self._base_scale() * self._zoom
        scale_after = self._base_scale() * new_zoom
        if scale_before <= 0 or scale_after <= 0:
            return
        image_pos = self._widget_to_image(cursor_pos)
        if image_pos is None:
            self._zoom = new_zoom
            self.update()
            return
        image_x, image_y = image_pos
        offset_after = QPointF(
            cursor_pos.x() - image_x * scale_after,
            cursor_pos.y() - image_y * scale_after,
        )
        center_after = self._total_offset(scale_after) - self._pan_offset
        self._pan_offset = offset_after - center_after
        self._zoom = new_zoom
        self.update()
        self.view_scale_changed.emit(self.view_scale())
        self._mark_zoom_pan_used()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._pixmap is None or not self._image_size:
            super().keyPressEvent(event)
            return
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Asterisk):  # + (numpad), = (US), +/* (e.g. German main keyboard)
            factor = 1.1
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):  # - and _ (same key, with/without Shift)
            factor = 1 / 1.1
        else:
            super().keyPressEvent(event)
            return
        new_zoom = max(0.1, min(10.0, self._zoom * factor))
        if new_zoom != self._zoom:
            self._zoom = new_zoom
            self.update()
            self.view_scale_changed.emit(self.view_scale())
            self._mark_zoom_pan_used()
        event.accept()

    def _is_pan_button(self, button: Qt.MouseButton) -> bool:
        """True if this button should start pan (middle, or left when not in selection mode)."""
        if button == self.pan_button:
            return True
        if button == self.click_button and not self.enable_drag_selection:
            return True
        return False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._is_pan_button(event.button()):
            self._dragging = True
            self._last_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == self.selection_button and self.enable_drag_selection:
            image_pos = self._widget_to_image(event.position())
            if image_pos is not None:
                self._dragging_selection = True
                self.drag_started.emit(image_pos[0], image_pos[1])
        elif event.button() == self.click_button and self.enable_drag_selection:
            if self.click_button != self.selection_button:
                self._left_potential_click = True
                self._left_press_pos = event.position()
            else:
                image_pos = self._widget_to_image(event.position())
                if image_pos is not None:
                    self.clicked_image.emit(image_pos[0], image_pos[1])
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._left_potential_click and (event.position() - self._left_press_pos).manhattanLength() > 5:
            self._left_potential_click = False
            self._dragging = True
            self._last_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        if self._dragging:
            delta = event.position() - self._last_pos
            self._pan_offset += delta
            self._last_pos = event.position()
            self.update()
        if self._dragging_selection:
            image_pos = self._widget_to_image(event.position())
            if image_pos is not None:
                self.drag_updated.emit(image_pos[0], image_pos[1])
        # Foci hover tooltips: show only when cursor is within this focus's radius (so not "closest" when far from all)
        if self._foci_hover_data:
            image_pos = self._widget_to_image(event.position())
            if image_pos is not None:
                ix, iy = float(image_pos[0]), float(image_pos[1])
                best_tooltip: Optional[str] = None
                best_d_sq = float("inf")
                for item in self._foci_hover_data:
                    if len(item) >= 4:
                        xd, yd, tooltip, max_r = item[0], item[1], item[2], item[3]
                    else:
                        xd, yd, tooltip = item[0], item[1], item[2]
                        max_r = 25.0
                    d_sq = (ix - xd) * (ix - xd) + (iy - yd) * (iy - yd)
                    if d_sq <= max_r * max_r and d_sq < best_d_sq:
                        best_d_sq = d_sq
                        best_tooltip = tooltip
                base = self._effective_base_tooltip()
                super().setToolTip(best_tooltip if best_tooltip is not None else base)
            else:
                super().setToolTip(self._effective_base_tooltip())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == self.click_button and self._left_potential_click:
            image_pos = self._widget_to_image(event.position())
            if image_pos is not None:
                self.clicked_image.emit(image_pos[0], image_pos[1])
            self._left_potential_click = False
        elif self._is_pan_button(event.button()) or (event.button() == self.click_button and self._dragging):
            if self._dragging:
                self._mark_zoom_pan_used()
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == self.selection_button and self._dragging_selection:
            image_pos = self._widget_to_image(event.position())
            if image_pos is not None:
                self.drag_finished.emit(image_pos[0], image_pos[1])
            self._dragging_selection = False
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.view_scale_changed.emit(self.view_scale())

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self._pixmap is None or not self._image_size:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            scale = self._base_scale() * self._zoom
            offset = self._total_offset(scale)
            image_width, image_height = self._image_size
            target = QRect(
                int(offset.x()),
                int(offset.y()),
                int(image_width * scale),
                int(image_height * scale),
            )
            source = QRect(0, 0, self._pixmap.width(), self._pixmap.height())
            painter.drawPixmap(target, self._pixmap, source)
        finally:
            painter.end()
