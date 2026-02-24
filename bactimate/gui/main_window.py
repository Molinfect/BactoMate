"""
Main application window and entrypoint. Import tab classes from the main module to avoid circular imports.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Type

from PyQt6.QtCore import QEvent, Qt, QCoreApplication, QObject, QTimer, QUrl
from PyQt6.QtGui import QCursor, QFont, QIcon, QPixmap, QAction, QKeySequence, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.constants import LOGS_DIR, MAX_SESSION_LOGS, SESSION_TAB_KEYS, TAB_WIDGET_ATTR
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
    OmniposeSegTab,
    SAM2SegmentTab,
    SessionLogsPanel,
    TrackingTab,
    UNetSegmentTab,
)
from bactimate.gui.paths import get_app_root, get_omnipose_user_models_dir, get_unet_models_dir
from bactimate.gui.styles import get_modern_stylesheet


def _get_app_icon_path() -> Path | None:
    """Path to app icon: BactiMate.icns in app root (bundle Resources) or img/logo2.png."""
    root = get_app_root()
    for candidate in (root / "BactiMate.icns", root / "img" / "logo2.png"):
        if candidate.exists():
            return candidate
    return None


def _set_macos_dock_icon(icon_path: Path) -> None:
    """Set the Dock icon on macOS via Cocoa (Qt setWindowIcon does not update Dock)."""
    try:
        from AppKit import NSApp, NSImage  # type: ignore[import-untyped]
        img = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
        if img is not None:
            NSApp.setApplicationIconImage_(img)
    except Exception:
        pass


def _prune_old_session_logs(logs_dir: Path, keep_count: int) -> None:
    """Keep only the most recent keep_count session log files; delete older ones."""
    if keep_count <= 0 or not logs_dir.is_dir():
        return
    session_files = sorted(
        logs_dir.glob("*_session.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in session_files[keep_count:]:
        try:
            path.unlink()
        except OSError:
            log_exception(get_logger(), f"Failed to remove old session log {path.name}")


def _get_tab_classes() -> Tuple[Type[Any], ...]:
    """Return tab classes (all from bactimate.gui.tabs)."""
    return (
        ND2ConversionTab,
        OmniposeSegTab,
        UNetSegmentTab,
        AnalyzeMasksTab,
        TrackingTab,
        GrowthCurveVideoTab,
        SAM2SegmentTab,
    )


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.shared_metadata = {"pixel_size": None, "time_per_frame_min": None}
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("BactiMate")
        # Ensure user model dirs exist on first launch (dev: repo/omnipose_user_models, unet_user_models; installed: BactiMate/Models/...)
        get_omnipose_user_models_dir()
        get_unet_models_dir()
        self.setMinimumSize(1200, 700)
        self.resize(1200, 900)
        self._center_window()

        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)

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
            OmniposeSegTab,
            UNetSegmentTab,
            AnalyzeMasksTab,
            TrackingTab,
            GrowthCurveVideoTab,
            SAM2SegmentTab,
        ) = _get_tab_classes()

        tabs = QTabWidget()
        self.nd2_tab = ND2ConversionTab(self)
        self.omnipose_seg_tab = OmniposeSegTab(self)
        self.unet_segment_tab = UNetSegmentTab(self)
        self.analyze_tab = AnalyzeMasksTab(self)
        self.tracking_tab = TrackingTab(self)
        self.growth_curve_tab = GrowthCurveVideoTab(self)
        self.sam2_tab = SAM2SegmentTab(self)
        tabs.addTab(self.nd2_tab, "TIFF Conversion")
        tabs.addTab(self.omnipose_seg_tab, "Omnipose Segment")
        tabs.addTab(self.unet_segment_tab, "U-NET Segment")
        # tabs.addTab(self.sam2_tab, "SAM2 Segment")
        tabs.addTab(self.analyze_tab, "Analyze Masks")
        tabs.addTab(self.tracking_tab, "Tracking")
        tabs.addTab(self.growth_curve_tab, "Video Graph")

        main_layout.addWidget(tabs)

        # Use native QGroupBox title (like "Extracted Metadata") so header aligns identically
        log_group = QGroupBox("Status && Session logs  \u25B2")
        log_layout = QVBoxLayout()
        self._log_content = QWidget()
        log_content_layout = QHBoxLayout(self._log_content)
        log_content_layout.setContentsMargins(0, 0, 0, 0)
        # Left half: status / progress / log output
        left_log = QWidget()
        left_log_layout = QVBoxLayout(left_log)
        left_log_layout.setContentsMargins(0, 0, 0, 0)
        status_title = QLabel("Status (all tabs)")
        status_title.setStyleSheet("font-weight: bold; color: #4a9eff;")
        status_title.setToolTip("Activity log from all tabs: TIFF conversion, Omnipose/U-Net segmentation, Analyze Masks, Tracking, Video, etc.")
        left_log_layout.addWidget(status_title)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        left_log_layout.addWidget(self.progress_bar)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        monospace_font = QFont("Monaco", 11)
        if not monospace_font.exactMatch():
            monospace_font = QFont("Courier New", 11)
        self.log_text.setFont(monospace_font)
        self.log_text.setMinimumHeight(70)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        log_size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log_text.setSizePolicy(log_size_policy)
        left_log_layout.addWidget(self.log_text, 1)
        log_content_layout.addWidget(left_log, 1)
        # Right 2/3: session logs (compact list + log content + Load other + Apply)
        self.logs_panel = SessionLogsPanel(self)
        log_content_layout.addWidget(self.logs_panel, 2)
        log_layout.addWidget(self._log_content)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        self._log_group = log_group

        _COLLAPSED_LINE_STYLE = (
            "QGroupBox {"
            " border: none; border-top: 1px solid #2d2d2d; border-radius: 0;"
            " margin-top: 0; padding-top: 6px; padding-bottom: 6px; padding-left: 0;"
            " background: #1a1a1a;"
            "}"
            " QGroupBox::title {"
            " subcontrol-origin: padding; subcontrol-position: top left; left: 12px;"
            " padding: 8px 0 0 0; color: #4a9eff; background: transparent;"
            " }"
        )

        def toggle_log_visibility():
            try:
                if not self._log_content or not log_group:
                    return
                visible = self._log_content.isVisible()
                self._log_content.setVisible(not visible)
                log_group.setTitle("Status && Session logs  \u25B2" if visible else "Status && Session logs  \u25BC")
                if visible:
                    log_group.setFixedHeight(32)
                    log_group.setStyleSheet(_COLLAPSED_LINE_STYLE)
                else:
                    log_group.setMinimumHeight(0)
                    log_group.setMaximumHeight(16777215)
                    log_group.setStyleSheet("")
            except Exception:
                pass

        class _LogTitleClickFilter(QObject):
            def __init__(self, group, content, parent=None):
                super().__init__(parent)
                self._group = group
                self._content = content

            def eventFilter(self, obj, event):
                if obj is self._group and event.type() == QEvent.Type.MouseButtonPress:
                    if event.position().y() < 28:
                        toggle_log_visibility()
                        return True
                return False

        log_group.installEventFilter(_LogTitleClickFilter(log_group, self._log_content, log_group))

        # Start collapsed: thin line at bottom, no tooltip
        self._log_content.setVisible(False)
        log_group.setFixedHeight(32)
        log_group.setStyleSheet(_COLLAPSED_LINE_STYLE)

        main_layout.setStretchFactor(tabs, 1)
        main_layout.setStretchFactor(log_group, 0)

        # No bottom margin so Status & Session logs sits at rock bottom when collapsed
        margins = main_layout.contentsMargins()
        main_layout.setContentsMargins(margins.left(), margins.top(), margins.right(), 0)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self._add_menu_bar()

        self._restore_geometry_and_state()

        self.statusBar().hide()  # no "Ready" bar so no extra space below Status & Session logs
        self.log_output("Omnipose Batch Processing GUI started.\n")
        self.log_output("Select a tab and configure parameters to begin.\n\n")

    def expand_status_log(self) -> None:
        """Expand the Status & Session logs panel if it is collapsed (e.g. when starting long-running tracking)."""
        if not getattr(self, "_log_content", None) or not getattr(self, "_log_group", None):
            return
        if self._log_content.isVisible():
            return
        self._log_content.setVisible(True)
        self._log_group.setTitle("Status && Session logs  \u25B2")
        self._log_group.setMinimumHeight(0)
        self._log_group.setMaximumHeight(16777215)
        self._log_group.setStyleSheet("")

    def closeEvent(self, event):
        # Wait for tracking worker to finish so we never destroy a running QThread
        tracking_tab = getattr(self, "tracking_tab", None)
        if tracking_tab is not None and getattr(tracking_tab, "worker", None) is not None:
            worker = tracking_tab.worker
            if worker.isRunning():
                old_title = self.windowTitle()
                self.setWindowTitle("Waiting for tracking to finish…")
                QApplication.processEvents()
                worker.wait(300000)  # wait up to 5 minutes
                self.setWindowTitle(old_title)
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
            _prune_old_session_logs(LOGS_DIR, MAX_SESSION_LOGS)
            if hasattr(self, "logs_panel") and self.logs_panel is not None:
                self.logs_panel._refresh_list()
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

    def _add_menu_bar(self) -> None:
        """Minimal menu bar: app menu (About, Minimize, Quit), File, Help. On macOS the first menu is the app menu and shows the app name once."""
        menubar = self.menuBar()
        # First menu = application menu. On macOS the system shows the app name for it; use empty title to avoid duplicate "BactiMate"
        app_menu = menubar.addMenu("" if sys.platform == "darwin" else "BactiMate")
        about_action = QAction("About BactiMate", self)
        about_action.triggered.connect(self._show_about)
        app_menu.addAction(about_action)
        app_menu.addSeparator()
        minimize_action = QAction("Minimize", self)
        minimize_action.setShortcut(QKeySequence("Ctrl+M"))  # Cmd+M on macOS (Qt maps Ctrl→Cmd)
        def minimize_active_or_self():
            active = QApplication.activeWindow()
            if active and active is not self and (active.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint):
                active.showMinimized()
            else:
                self.showMinimized()
        minimize_action.triggered.connect(minimize_active_or_self)
        app_menu.addAction(minimize_action)
        quit_action = QAction("Quit BactiMate", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        app_menu.addAction(quit_action)
        file_menu = menubar.addMenu("File")
        file_quit = QAction("Quit", self)
        file_quit.setShortcut(QKeySequence.StandardKey.Quit)
        file_quit.triggered.connect(self.close)
        file_menu.addAction(file_quit)
        help_menu = menubar.addMenu("Help")
        user_manual_action = QAction("User manual", self)
        user_manual_action.triggered.connect(self._open_user_manual)
        help_menu.addAction(user_manual_action)

    def _show_about(self) -> None:
        try:
            from bactimate import __version__
            version = __version__
        except Exception:
            version = "1.0"
        QMessageBox.about(
            self,
            "About BactiMate",
            f"<b>BactiMate</b> – bacteria time-lapse batch processing and analysis.<br><br>Version {version}<br><br>"
            "Documentation: <b>Help → User manual</b>.",
        )

    def _open_user_manual(self) -> None:
        """Open the user manual (docs/BACTIMATE_USER_MANUAL.md) with the system default app."""
        app_root = get_app_root()
        manual_path = app_root / "docs" / "BACTIMATE_USER_MANUAL.md"
        if manual_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(manual_path)))
        else:
            # Fallback: open docs folder or show path
            docs_dir = app_root / "docs"
            if docs_dir.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(docs_dir)))
            else:
                QMessageBox.information(
                    self,
                    "User manual",
                    f"Manual not found at:\n{manual_path}\n\nOpen the docs/ folder from the app or installation.",
                )

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

    @staticmethod
    def _format_status_line(line: str) -> str:
        """Strip log prefixes (timestamp, [INFO]) and condense tqdm progress to one short line for the status window."""
        if not line or not line.strip():
            return line
        s = line.strip()
        # Strip common logging prefix: "2026-02-19 12:49:28,644	[INFO]  " or "[INFO]  " or "bactimate:INFO: "
        s = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+\[(?:INFO|DEBUG|WARNING|ERROR)\]\s*", "", s)
        s = re.sub(r"^\[(?:INFO|DEBUG|WARNING|ERROR)\]\s*", "", s)
        s = re.sub(r"^[\w.]+:(?:INFO|DEBUG|WARNING|ERROR):\s*", "", s)
        s = s.strip()
        # Strip tqdm position prefix "line  46	" or "line 46 "
        s = re.sub(r"^line\s+\d+\s*[\t ]*", "", s)
        s = s.strip()
        # Condense tqdm-style progress to one short line: "28% (9/32) ETA 1:09"
        tqdm_match = re.match(
            r"^(\d+)%\s*\|[#\s]+\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]",
            s,
        )
        if tqdm_match:
            pct, current, total, rest = tqdm_match.groups()
            eta_match = re.search(r"<(\d+):(\d+)", rest)
            if eta_match:
                return f"  {pct}% ({current}/{total})  ETA {eta_match.group(1)}:{eta_match.group(2)}"
            return f"  {pct}% ({current}/{total})"
        # If we stripped everything (line was only timestamp/[INFO]), skip it
        return s if s else ""

    def log_output(self, message: str) -> None:
        # Normalize each line for display: strip log prefixes, condense tqdm
        lines = message.split("\n")
        formatted = [self._format_status_line(line) for line in lines]
        formatted = [f for f in formatted if f]
        text = "\n".join(formatted)
        if not text:
            return
        if not text.endswith("\n") and message.endswith("\n"):
            text += "\n"
        self.log_text.append(text)
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
                "starting training",
                "epoch ",
                "u-net",
                "running u-net",
                "starting tracking",
                "tracking completed",
                "starting nd2 conversion",
                "converted:",
                "starting mask analysis",
                "analysis complete",
                "starting batch",
                "create growth curve",
            ]
        ):
            if not self.progress_bar.isVisible():
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)

    def update_all_tabs_with_metadata(self):
        pixel_size = self.shared_metadata.get("pixel_size")
        time_per_frame_min = self.shared_metadata.get("time_per_frame_min")
        if pixel_size is None and time_per_frame_min is None:
            return
        if pixel_size is not None:
            if hasattr(self, "omnipose_seg_tab") and hasattr(self.omnipose_seg_tab, "pixel_size_um"):
                self.omnipose_seg_tab.pixel_size_um.setValue(pixel_size)
                self.log_output(f"✓ Updated pixel size in Omnipose Segment tab: {pixel_size:.4f} µm\n")
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
            if hasattr(self, "tracking_tab") and hasattr(self.tracking_tab, "filter_pixel_size"):
                self.tracking_tab.filter_pixel_size.setValue(pixel_size)
                self.log_output(f"✓ Updated pixel size in Tracking tab: {pixel_size:.4f} µm\n")
        if time_per_frame_min is not None and time_per_frame_min > 0:
            if hasattr(self, "tracking_tab") and hasattr(self.tracking_tab, "filter_time_per_frame"):
                self.tracking_tab.filter_time_per_frame.setValue(time_per_frame_min)
                if hasattr(self.tracking_tab, "filter_time_per_frame_unit"):
                    self.tracking_tab.filter_time_per_frame_unit.setCurrentText("min")
                self.log_output(f"✓ Updated time per frame in Tracking tab: {time_per_frame_min:.4f} min\n")


def main() -> None:
    log = get_logger()
    log.info("Creating application and main window...")
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
        app.setApplicationName("BactiMate")
        app.setApplicationDisplayName("BactiMate")
        # Icon: Dock and window (macOS Dock needs Cocoa; bundle may have BactiMate.icns or img/logo2.png)
        icon_path = _get_app_icon_path()
        if icon_path is not None:
            app.setWindowIcon(QIcon(str(icon_path)))
            if sys.platform == "darwin":
                _set_macos_dock_icon(icon_path)
        # Re-apply process title after Qt init (Qt can overwrite it on macOS; menu bar shows process name)
        if sys.platform == "darwin":
            try:
                import setproctitle
                setproctitle.setproctitle("BactiMate")
            except Exception:
                pass
        app.setStyle("Fusion")
        app.setStyleSheet(get_modern_stylesheet())
        if sys.platform == "win32":
            try:
                from bactimate.gui.windows_patches import apply_after_ui
                apply_after_ui(app)
            except Exception:
                pass
        window = MainWindow()
        if icon_path is not None:
            window.setWindowIcon(QIcon(str(icon_path)))
        window.show()
        # Re-apply process title once after show (macOS menu bar can be updated late)
        if sys.platform == "darwin":
            def _reapply_title() -> None:
                try:
                    import setproctitle
                    setproctitle.setproctitle("BactiMate")
                except Exception:
                    pass
            QTimer.singleShot(500, _reapply_title)
        log.info("BactiMate window shown.")
        sys.exit(app.exec())
    except Exception as e:
        log_exception(get_logger(), "Failed to start GUI")
        print(f"Error starting GUI: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
