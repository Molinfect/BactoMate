"""
Main application window and entrypoint. Import tab classes from the main module to avoid circular imports.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Type

from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtGui import QCursor, QFont, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.constants import LOGS_DIR, SESSION_TAB_KEYS, TAB_WIDGET_ATTR
from bactimate.gui.log_config import get_logger, log_exception, write_crash_log
from bactimate.gui.preferences import (
    get_geometry,
    get_last_dir,
    get_state,
    set_geometry,
    set_last_dir,
    set_state,
)
from bactimate.gui.tabs import (
    AnalyzeMasksTab,
    GrowthCurveVideoTab,
    ND2ConversionTab,
    ProcessingTab,
    SAM2SegmentTab,
    SessionLogsTab,
    TrackingTab,
)
from bactimate.gui.paths import get_app_root
from bactimate.gui.styles import get_modern_stylesheet


def _get_tab_classes() -> Tuple[Type[Any], ...]:
    """Return tab classes (all from bactimate.gui.tabs)."""
    return (
        ND2ConversionTab,
        ProcessingTab,
        AnalyzeMasksTab,
        TrackingTab,
        GrowthCurveVideoTab,
        SAM2SegmentTab,
    )


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.shared_metadata = {"pixel_size": None}
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("BactiMate")
        self.setMinimumSize(1200, 700)
        self.resize(1200, 900)
        self._center_window()

        central_widget = QWidget()
        main_layout = QVBoxLayout()

        title_bar = QWidget()
        title_bar.setProperty("class", "title_bar")
        title_bar.setStyleSheet(
            "QWidget[class='title_bar'] {"
            "background: transparent;"
            "padding: 8px;"
            "border-bottom: 1px solid #4a9eff;"
            "}"
        )
        title_bar_layout = QHBoxLayout()
        title_bar_layout.setContentsMargins(12, 6, 12, 6)
        title_bar_layout.addStretch()

        logo_label = QLabel()
        logo_path = get_app_root() / "img" / "logo2.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if pixmap.height() > 60:
                pixmap = pixmap.scaledToHeight(60, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setStyleSheet("background: transparent;")
        else:
            logo_label.setText("BactiMate")
            logo_label.setStyleSheet(
                "font-size: 18pt; font-weight: bold; color: #ECF0F1; background: transparent;"
            )
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        title_bar_layout.addWidget(logo_label)

        title_text = QLabel("BactiMate - your mate for bacteria time-lapse data")
        title_text.setStyleSheet(
            "QLabel {"
            "font-size: 20pt;"
            "font-weight: bold;"
            "color: #ECF0F1;"
            "background: transparent;"
            "padding-left: 12px;"
            "}"
        )
        title_text.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        title_bar_layout.addWidget(title_text)
        title_bar_layout.addStretch()

        title_bar.setLayout(title_bar_layout)
        main_layout.addWidget(title_bar)

        (
            ND2ConversionTab,
            ProcessingTab,
            AnalyzeMasksTab,
            TrackingTab,
            GrowthCurveVideoTab,
            SAM2SegmentTab,
        ) = _get_tab_classes()

        tabs = QTabWidget()
        self.nd2_tab = ND2ConversionTab(self)
        self.processing_tab = ProcessingTab(self)
        self.analyze_tab = AnalyzeMasksTab(self)
        self.tracking_tab = TrackingTab(self)
        self.growth_curve_tab = GrowthCurveVideoTab(self)
        self.sam2_tab = SAM2SegmentTab(self)
        self.logs_tab = SessionLogsTab(self)

        tabs.addTab(self.nd2_tab, "TIFF Conversion")
        tabs.addTab(self.processing_tab, "Omnipose Segment")
        # tabs.addTab(self.sam2_tab, "SAM2 Segment")
        tabs.addTab(self.analyze_tab, "Analyze Masks")
        tabs.addTab(self.tracking_tab, "Tracking")
        tabs.addTab(self.growth_curve_tab, "Video Graph")
        tabs.addTab(self.logs_tab, "Session Logs")

        main_layout.addWidget(tabs)

        log_group = QGroupBox("Status & Log")
        log_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        log_layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        monospace_font = QFont("Monaco", 11)
        if not monospace_font.exactMatch():
            monospace_font = QFont("Courier New", 11)
        self.log_text.setFont(monospace_font)
        self.log_text.setMinimumHeight(100)
        self.log_text.setMaximumHeight(150)
        log_size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.log_text.setSizePolicy(log_size_policy)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        main_layout.setStretchFactor(tabs, 1)
        main_layout.setStretchFactor(log_group, 0)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self._restore_geometry_and_state()

        self.statusBar().showMessage("Ready")
        self.log_output("Omnipose Batch Processing GUI started.\n")
        self.log_output("Select a tab and configure parameters to begin.\n\n")

    def closeEvent(self, event):
        self._save_geometry_and_state()
        self._save_session_log()
        super().closeEvent(event)

    def _restore_geometry_and_state(self) -> None:
        """Restore main window geometry and state from QSettings."""
        geom = get_geometry()
        if geom and not geom.isEmpty():
            self.restoreGeometry(geom)
        state = get_state()
        if state and not state.isEmpty():
            self.restoreState(state)

    def _save_geometry_and_state(self) -> None:
        """Save main window geometry and state to QSettings."""
        set_geometry(self.saveGeometry())
        set_state(self.saveState())

    def get_last_dir_for_tab(self, key: str) -> str:
        """Return last used directory for the given tab (e.g. preferences.LAST_DIR_ND2)."""
        return get_last_dir(key)

    def set_last_dir_for_tab(self, key: str, path: str) -> None:
        """Save last used directory for the given tab after a file/dir selection."""
        set_last_dir(key, path)

    def _save_session_log(self):
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            now = datetime.now()
            timestamp_in_filename = now.strftime("%Y-%m-%d_%H-%M-%S")
            ts_display = now.strftime("%Y-%m-%d %H:%M:%S")
            path = LOGS_DIR / f"{timestamp_in_filename}_session.json"
            data = {
                "schema_version": 1,
                "timestamp": ts_display,
                "tabs": self.gather_session_params(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            log_exception(get_logger(), "Failed to save session log")

    def gather_session_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, _ in SESSION_TAB_KEYS:
            tab = getattr(self, TAB_WIDGET_ATTR[key])
            if hasattr(tab, "get_session_params"):
                out[key] = tab.get_session_params()
        return out

    def apply_session_params(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        for key, _ in SESSION_TAB_KEYS:
            if key not in data:
                continue
            tab = getattr(self, TAB_WIDGET_ATTR[key])
            if hasattr(tab, "set_session_params"):
                tab.set_session_params(data[key])

    def _center_window(self):
        frame_geometry = self.frameGeometry()
        cursor_pos = QCursor.pos()
        screen = QApplication.screenAt(cursor_pos)
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen:
            center_point = screen.geometry().center()
            frame_geometry.moveCenter(center_point)
            self.move(frame_geometry.topLeft())

    def log_output(self, message: str) -> None:
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        message_lower = message.lower()
        if any(
            keyword in message_lower
            for keyword in [
                "applying drift correction",
                "applying bleach correction",
                "processing frame",
                "segmenting",
                "analyzing masks",
                "reading frames",
                "extracting frames",
                "processing file",
            ]
        ):
            if not self.progress_bar.isVisible():
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)

    def update_all_tabs_with_metadata(self):
        pixel_size = self.shared_metadata.get("pixel_size")
        if pixel_size is None:
            return
        if hasattr(self, "processing_tab") and hasattr(self.processing_tab, "pixel_size"):
            self.processing_tab.pixel_size.setValue(pixel_size)
            if hasattr(self.processing_tab, "pixel_size_indicator"):
                self.processing_tab.pixel_size_indicator.setText("(from metadata)")
            self.log_output(f"✓ Updated pixel size in Processing tab: {pixel_size:.4f} µm\n")
        if hasattr(self, "analyze_tab") and hasattr(self.analyze_tab, "pixel_size"):
            self.analyze_tab.pixel_size.setValue(pixel_size)
            if hasattr(self.analyze_tab, "pixel_size_indicator"):
                self.analyze_tab.pixel_size_indicator.setText("(from metadata)")
                self.analyze_tab.pixel_size_indicator.setStyleSheet(
                    "color: #4CAF50; font-size: 9pt;"
                )
                self.analyze_tab.pixel_size_indicator.setFixedWidth(100)
                self.analyze_tab.pixel_size_indicator.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.analyze_tab.pixel_size_indicator.show()
                self.analyze_tab.pixel_size_metadata_loaded = True
            self.log_output(f"✓ Updated pixel size in Analyze Masks tab: {pixel_size:.4f} µm\n")


def main() -> None:
    if sys.platform == "darwin":
        plugins_dir = os.environ.get("QT_PLUGIN_PATH")
        if plugins_dir:
            abs_plugins_dir = os.path.abspath(plugins_dir)
            if abs_plugins_dir.endswith("platforms"):
                abs_plugins_dir = os.path.dirname(abs_plugins_dir)
            os.environ["QT_PLUGIN_PATH"] = abs_plugins_dir
            QCoreApplication.setLibraryPaths([abs_plugins_dir])

    original_excepthook = sys.excepthook

    def pyqt6_exception_handler(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        # Write to project Logs/crash.log so we can read it for debugging
        try:
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            block = "".join(lines)
            write_crash_log(f"\n--- {datetime.now().isoformat()} ---\n{block}")
        except Exception:
            pass

    sys.excepthook = pyqt6_exception_handler

    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setStyleSheet(get_modern_stylesheet())
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        log_exception(get_logger(), "Failed to start GUI")
        print(f"Error starting GUI: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
