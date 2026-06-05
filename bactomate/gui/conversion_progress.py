"""Non-modal progress UI for Convert to TIFF."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QDialog, QLabel, QProgressBar, QSizePolicy, QVBoxLayout


class ConversionProgressDialog(QDialog):
    """Progress window with a step title, detail text, and percent bar."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ConversionProgressDialog")
        self.setWindowTitle("Converting to TIFF")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMinimumWidth(560)
        self.setMaximumWidth(920)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        label_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        label_policy.setHeightForWidth(True)

        self._step_label = QLabel("Starting...")
        self._step_label.setWordWrap(True)
        self._step_label.setSizePolicy(label_policy)
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._step_label.setStyleSheet("color: #e8eef6; font-weight: 600; font-size: 11pt;")

        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        self._detail_label.setSizePolicy(label_policy)
        self._detail_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._detail_label.setStyleSheet("color: #aeb6c2; font-size: 10pt;")

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")

        layout.addWidget(self._step_label)
        layout.addWidget(self._detail_label)
        layout.addWidget(self._bar)

    def _resize_to_content(self) -> None:
        try:
            self.adjustSize()
            hint = self.sizeHint()
            width = max(int(self.minimumWidth()), min(int(hint.width()) + 8, int(self.maximumWidth())))
            height = max(int(hint.height()) + 4, 120)
            self.resize(width, height)
        except Exception:
            pass

    def set_progress(self, percent: int, step: str, detail: str = "") -> None:
        pct = max(0, min(100, int(percent)))
        self._step_label.setText(str(step or "Working...").strip() or "Working...")
        detail_txt = str(detail or "").strip()
        self._detail_label.setText(detail_txt)
        self._detail_label.setVisible(bool(detail_txt))
        self._bar.setValue(pct)
        self._resize_to_content()
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.processEvents()
        except Exception:
            pass

    def finish_and_close(self, delay_ms: int = 0) -> None:
        self.set_progress(100, "Complete", "")
        if delay_ms <= 0:
            self.close()
            return
        QTimer.singleShot(int(delay_ms), self, self.close)
