"""Non-modal progress UI for filter-preview Apply (full-width Cancel)."""
from __future__ import annotations

import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout

from bactomate.gui.filter_batch_progress import (
    batch_filter_percent,
    batch_foci_step_title,
    format_filter_batch_detail,
)
from bactomate.gui.layout_helpers import style_module_primary_button

__all__ = [
    "FilterApplyProgressDialog",
    "batch_filter_percent",
    "batch_foci_step_title",
    "format_filter_batch_detail",
]


def _qt_alive(obj) -> bool:
    """True while the underlying C++ QObject still exists."""
    try:
        from shiboken6 import isValid

        return bool(obj is not None and isValid(obj))
    except Exception:
        return obj is not None


class FilterApplyProgressDialog(QDialog):
    """Progress window with separate step title and detail lines."""

    canceled = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("FilterApplyProgressDialog")
        self.setWindowTitle("Applying filters")
        self.setWindowModality(Qt.WindowModality.NonModal)
        # Do not DeleteOnClose: set_progress() may processEvents() and deliver a
        # queued worker result that closes this dialog mid-update. Immediate C++
        # deletion caused Windows access violations / macOS segfaults in CI.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setMinimumWidth(560)
        self.setMaximumWidth(920)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        label_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        label_policy.setHeightForWidth(True)

        self._step_label = QLabel("Starting…")
        self._step_label.setWordWrap(True)
        self._step_label.setSizePolicy(label_policy)
        self._step_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._step_label.setStyleSheet(
            "color: #e8eef6; font-weight: 600; font-size: 11pt;"
        )

        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        self._detail_label.setSizePolicy(label_policy)
        self._detail_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._detail_label.setStyleSheet("color: #aeb6c2; font-size: 10pt;")

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")

        self._cancel = QPushButton("Cancel")
        style_module_primary_button(self._cancel)
        self._cancel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._cancel.clicked.connect(self.canceled.emit)

        layout.addWidget(self._step_label)
        layout.addWidget(self._detail_label)
        layout.addWidget(self._bar)
        layout.addWidget(self._cancel)
        self._processing_events = False
        self._closing = False
        # When False, skip processEvents (async preview workers). Batch Filter
        # keeps the default True so the bar paints during long sync work.
        self._pump_events = True
        self._last_ui_update_ts = 0.0
        self._min_ui_interval_s = 0.05

    def set_pump_events(self, enabled: bool) -> None:
        """Enable/disable nested processEvents during set_progress."""
        self._pump_events = bool(enabled)

    def _resize_to_content(self) -> None:
        if not _qt_alive(self) or self._closing:
            return
        try:
            self.adjustSize()
            hint = self.sizeHint()
            w = max(int(self.minimumWidth()), min(int(hint.width()) + 8, int(self.maximumWidth())))
            h = max(int(hint.height()) + 4, 150)
            self.resize(w, h)
        except Exception:
            pass

    def set_progress(self, percent: int, step: str, detail: str = "") -> None:
        """Update overall percent, current step title, and optional sub-detail."""
        if not _qt_alive(self) or self._closing or self._processing_events:
            return
        try:
            prev = int(self._bar.value())
        except Exception:
            prev = 0
        pct = max(prev, max(0, min(100, int(percent))))
        step_txt = str(step or "Working…").strip() or "Working…"
        detail_txt = str(detail or "").strip()
        now = time.monotonic()
        ui_changed = (
            pct != prev
            or self._step_label.text() != step_txt
            or self._detail_label.text() != detail_txt
        )
        elapsed = now - self._last_ui_update_ts
        # Never drop a percent increase — early throttle was freezing the bar at 0%
        # while a long step (e.g. focus scoring) ran with no further UI updates.
        if (
            pct <= prev
            and self._last_ui_update_ts > 0
            and elapsed < self._min_ui_interval_s
            and pct < 100
        ):
            return
        if not ui_changed:
            return
        self._step_label.setText(step_txt)
        self._detail_label.setText(detail_txt)
        self._detail_label.setVisible(bool(detail_txt))
        self._bar.setValue(pct)
        self._resize_to_content()
        self._last_ui_update_ts = now
        if not self._pump_events:
            return
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                self._processing_events = True
                try:
                    app.processEvents()
                finally:
                    # Nested close/result delivery must not touch a dying dialog.
                    if _qt_alive(self):
                        self._processing_events = False
        except Exception:
            if _qt_alive(self):
                self._processing_events = False

    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True
        super().closeEvent(event)

    def finish_and_close(self, delay_ms: int = 0) -> None:
        """Show 100% then close automatically (batch Filter File/Folder)."""
        from PyQt6.QtCore import QTimer

        try:
            self._cancel.setEnabled(False)
            self._cancel.setText("Close")
        except Exception:
            pass
        self.set_progress(100, "Complete", "")
        if delay_ms <= 0:
            self._closing = True
            self.close()
            self.deleteLater()
            return
        QTimer.singleShot(int(delay_ms), self, self._finish_close_later)

    def _finish_close_later(self) -> None:
        if not _qt_alive(self):
            return
        self._closing = True
        self.close()
        self.deleteLater()
