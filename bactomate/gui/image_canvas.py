"""
Qt-native image canvas: display a QPixmap with zoom (wheel) and pan (middle-drag).
No matplotlib dependency. Optional signals for click/drag in image coordinates.
Zoom/pan tooltip is shown only until the user has used zoom or pan once (then suppressed).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QPointF, QRect, QRectF, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QKeyEvent, QWheelEvent, QMouseEvent, QResizeEvent, QPaintEvent, QColor
from PyQt6.QtWidgets import QWidget


# Substring that identifies the standard zoom/pan hint so we can suppress it once the user has tried zoom/pan
_ZOOM_PAN_TOOLTIP_MARKER = "Zoom: mouse wheel"


class ImageCanvas(QWidget):
    """Widget that displays a pixmap with zoom and pan. Mouse wheel: zoom; middle button + drag: pan."""

    clicked_image = pyqtSignal(int, int)
    double_clicked_image = pyqtSignal(int, int)
    drag_started = pyqtSignal(int, int)
    drag_updated = pyqtSignal(int, int)
    drag_finished = pyqtSignal(int, int)
    view_scale_changed = pyqtSignal(float)
    roi_changed = pyqtSignal(int, int, int, int)
    roi_edit_finished = pyqtSignal(int, int, int, int)

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
        self._editable_roi: Optional[Tuple[int, int, int, int]] = None
        self._pending_roi: Optional[Tuple[int, int, int, int]] = None
        self._overlay_color = QColor(255, 230, 0)
        self._overlay_line_width = 2
        self._overlay_show_handles = True
        self.enable_roi_editing = False
        self._roi_edit_mode: Optional[str] = None
        self._roi_edit_anchor: Optional[Tuple[int, int, int, int]] = None
        self._roi_edit_start_image: Optional[Tuple[int, int]] = None
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
            from bactomate.gui.preferences import get_preview_zoom_pan_tip_seen
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
            from bactomate.gui.preferences import get_preview_zoom_pan_tip_seen
            if text and _ZOOM_PAN_TOOLTIP_MARKER in text and get_preview_zoom_pan_tip_seen():
                return ""
        except Exception:
            pass
        return text

    def _mark_zoom_pan_used(self) -> None:
        """Remember that the user used zoom/pan and clear the long tooltip so it does not pop up again."""
        try:
            from bactomate.gui.preferences import set_preview_zoom_pan_tip_seen
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

    def pixmap(self) -> Optional[QPixmap]:
        """Return the currently displayed pixmap (Qt QLabel compatibility)."""
        return self._pixmap

    def set_editable_roi(
        self,
        roi: Optional[Tuple[int, int, int, int]],
        *,
        color: Optional[QColor] = None,
        line_width: int = 2,
        show_handles: bool = True,
    ) -> None:
        """Show an editable ROI overlay in image coordinates."""
        self._editable_roi = roi
        if color is not None:
            self._overlay_color = color
        self._overlay_line_width = max(1, int(line_width))
        self._overlay_show_handles = bool(show_handles)
        self.update()

    def set_pending_roi(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        """Rubber-band overlay while drawing a new ROI."""
        self._pending_roi = roi
        self.update()

    def clear_roi_overlays(self) -> None:
        self._editable_roi = None
        self._pending_roi = None
        self.update()

    def _image_rect_to_widget(self, x0: int, y0: int, x1: int, y1: int) -> QRectF:
        scale = self._base_scale() * self._zoom
        offset = self._total_offset(scale)
        left = min(x0, x1) * scale + offset.x()
        top = min(y0, y1) * scale + offset.y()
        right = max(x0, x1) * scale + offset.x()
        bottom = max(y0, y1) * scale + offset.y()
        return QRectF(left, top, max(1.0, right - left), max(1.0, bottom - top))

    def _hit_tolerance_px(self) -> float:
        scale = max(self.view_scale(), 1e-6)
        return max(8.0, 8.0 / scale)

    def _roi_hit_test(self, x: int, y: int) -> Optional[str]:
        roi = self._editable_roi
        if roi is None:
            return None
        x0, y0, x1, y1 = roi
        left, right = min(x0, x1), max(x0, x1)
        top, bottom = min(y0, y1), max(y0, y1)
        tol = self._hit_tolerance_px()
        on_left = abs(x - left) <= tol
        on_right = abs(x - right) <= tol
        on_top = abs(y - top) <= tol
        on_bottom = abs(y - bottom) <= tol
        inside_x = left + tol < x < right - tol
        inside_y = top + tol < y < bottom - tol
        if on_left and on_top:
            return "resize_tl"
        if on_right and on_top:
            return "resize_tr"
        if on_left and on_bottom:
            return "resize_bl"
        if on_right and on_bottom:
            return "resize_br"
        if on_left and top <= y <= bottom:
            return "resize_l"
        if on_right and top <= y <= bottom:
            return "resize_r"
        if on_top and left <= x <= right:
            return "resize_t"
        if on_bottom and left <= x <= right:
            return "resize_b"
        if inside_x and inside_y:
            return "move"
        return None

    def _apply_roi_edit(self, x: int, y: int) -> None:
        if self._roi_edit_mode is None or self._roi_edit_anchor is None or self._roi_edit_start_image is None:
            return
        ax0, ay0, ax1, ay1 = self._roi_edit_anchor
        sx, sy = self._roi_edit_start_image
        dx, dy = x - sx, y - sy
        mode = self._roi_edit_mode
        if mode == "move":
            nx0, ny0, nx1, ny1 = ax0 + dx, ay0 + dy, ax1 + dx, ay1 + dy
        else:
            nx0, ny0, nx1, ny1 = ax0, ay0, ax1, ay1
            if mode in ("resize_tl", "resize_l", "resize_bl"):
                nx0 = ax0 + dx
            if mode in ("resize_tr", "resize_r", "resize_br"):
                nx1 = ax1 + dx
            if mode in ("resize_tl", "resize_t", "resize_tr"):
                ny0 = ay0 + dy
            if mode in ("resize_bl", "resize_b", "resize_br"):
                ny1 = ay1 + dy
        if not self._image_size:
            return
        iw, ih = self._image_size
        w = abs(nx1 - nx0)
        h = abs(ny1 - ny0)
        if w < 8 or h < 8:
            return
        if mode == "move":
            if nx0 < 0:
                nx1 -= nx0
                nx0 = 0
            if ny0 < 0:
                ny1 -= ny0
                ny0 = 0
            if nx1 > iw:
                shift = nx1 - iw
                nx0 -= shift
                nx1 = iw
            if ny1 > ih:
                shift = ny1 - ih
                ny0 -= shift
                ny1 = ih
        nx0 = max(0, min(nx0, iw - 1))
        ny0 = max(0, min(ny0, ih - 1))
        nx1 = max(0, min(nx1, iw))
        ny1 = max(0, min(ny1, ih))
        if abs(nx1 - nx0) < 8 or abs(ny1 - ny0) < 8:
            return
        self._editable_roi = (int(nx0), int(ny0), int(nx1), int(ny1))
        self.roi_changed.emit(*self._editable_roi)
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

    def center_on_image_point(self, x: float, y: float) -> None:
        """Pan so (x, y) in image coordinates is centered in the widget (keeps current zoom)."""
        if not self._image_size:
            return
        scale = self._base_scale() * self._zoom
        if scale <= 0:
            return
        image_width, image_height = self._image_size
        display_width = image_width * scale
        display_height = image_height * scale
        offset_x = (self.width() - display_width) / 2
        offset_y = (self.height() - display_height) / 2
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        self._pan_offset = QPointF(cx - offset_x - x * scale, cy - offset_y - y * scale)
        self.update()

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
        if (
            event.button() == self.click_button
            and self.enable_roi_editing
            and self._editable_roi is not None
            and self.click_button != self.selection_button
        ):
            image_pos = self._widget_to_image(event.position())
            if image_pos is not None:
                hit = self._roi_hit_test(image_pos[0], image_pos[1])
                if hit is not None:
                    self._roi_edit_mode = hit
                    self._roi_edit_anchor = self._editable_roi
                    self._roi_edit_start_image = image_pos
                    self._left_potential_click = False
                    return
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
        if self._roi_edit_mode is not None:
            image_pos = self._widget_to_image(event.position())
            if image_pos is not None:
                self._apply_roi_edit(image_pos[0], image_pos[1])
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
        elif self._roi_edit_mode is not None and event.button() == self.click_button:
            if self._editable_roi is not None:
                self.roi_edit_finished.emit(*self._editable_roi)
            self._roi_edit_mode = None
            self._roi_edit_anchor = None
            self._roi_edit_start_image = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == self.click_button:
            image_pos = self._widget_to_image(event.position())
            if image_pos is not None:
                self.double_clicked_image.emit(image_pos[0], image_pos[1])
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

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
            for roi in (self._editable_roi, self._pending_roi):
                if roi is None:
                    continue
                rect = self._image_rect_to_widget(*roi)
                pen = QPen(self._overlay_color, max(1.0, self._overlay_line_width * scale))
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)
                if roi is self._editable_roi and self._overlay_show_handles:
                    handle = max(4.0, 6.0 * scale)
                    for hx, hy in (
                        (rect.left(), rect.top()),
                        (rect.right(), rect.top()),
                        (rect.left(), rect.bottom()),
                        (rect.right(), rect.bottom()),
                    ):
                        painter.fillRect(
                            int(hx - handle / 2),
                            int(hy - handle / 2),
                            int(handle),
                            int(handle),
                            self._overlay_color,
                        )
        finally:
            painter.end()
