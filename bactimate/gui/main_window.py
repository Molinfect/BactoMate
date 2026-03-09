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

from PyQt6.QtCore import QEvent, Qt, QCoreApplication, QObject, QUrl
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
from bactimate.gui.log_config import get_logger, log_exception, write_crash_log, write_gui_log_line
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
    PostProcessingTab,
    ND2ConversionTab,
    OmniposeSegTab,
    SessionLogsPanel,
    TrackingTab,
    UNetSegmentTab,
)
from bactimate.gui.paths import get_app_root, get_omnipose_user_models_dir, get_unet_models_dir, _running_from_app_bundle
from bactimate.gui.styles import get_modern_stylesheet


def _get_app_icon_path() -> Path | None:
    """App icon: img/logo2.png or BactiMate.icns under get_app_root()."""
    root = get_app_root()
    for p in (root / "img" / "logo2.png", root / "BactiMate.icns"):
        if p.exists():
            return p.resolve()
    return None


def _set_macos_dock_icon(icon_path: Path) -> None:
    """Set Dock icon on macOS (Qt setWindowIcon does not update Dock)."""
    try:
        from AppKit import NSApp, NSImage  # type: ignore[import-untyped]
        img = NSImage.alloc().initWithContentsOfFile_(str(icon_path.resolve()))
        if img is not None:
            NSApp.setApplicationIconImage_(img)
    except Exception:
        pass


def _set_macos_process_name(name: str) -> None:
    """Set process name on macOS so Dock/menu bar show 'BactiMate' not 'Python'."""
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSProcessInfo  # type: ignore[import-untyped]
        NSProcessInfo.processInfo().setProcessName_(name)
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
        PostProcessingTab,
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
        min_width, min_height = 1200, 700
        initial_width, initial_height = 1200, 800
        if sys.platform == "win32":
            # Extra width so 140px button min-width and tab content don't cause horizontal scrollbar
            min_width += 190
            initial_width += 190
        self.setMinimumSize(min_width, min_height)
        self.resize(initial_width, initial_height)
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
            PostProcessingTab,
        ) = _get_tab_classes()

        tabs = QTabWidget()
        self.nd2_tab = ND2ConversionTab(self)
        self.omnipose_seg_tab = OmniposeSegTab(self)
        self.unet_segment_tab = UNetSegmentTab(self)
        self.analyze_tab = AnalyzeMasksTab(self)
        self.tracking_tab = TrackingTab(self)
        self.post_processing_tab = PostProcessingTab(self)
        tabs.addTab(self.nd2_tab, "TIFF Conversion")
        tabs.addTab(self.omnipose_seg_tab, "Omnipose Segment")
        tabs.addTab(self.unet_segment_tab, "U-NET Segment")
        tabs.addTab(self.analyze_tab, "Analyze Masks")
        tabs.addTab(self.tracking_tab, "Tracking")
        tabs.addTab(self.post_processing_tab, "Post-processing")

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
        # Wait for any running workers so we never destroy a QThread while it is running
        tracking_tab = getattr(self, "tracking_tab", None)
        if tracking_tab is not None and getattr(tracking_tab, "worker", None) is not None:
            worker = tracking_tab.worker
            if worker.isRunning():
                old_title = self.windowTitle()
                self.setWindowTitle("Waiting for tracking to finish…")
                QApplication.processEvents()
                if not worker.wait(300000):  # wait up to 5 minutes; if timeout, do not close
                    self.setWindowTitle(old_title)
                    QMessageBox.warning(
                        self,
                        "Please wait",
                        "Tracking is still running. Wait for it to finish or cancel it, then close the application.",
                    )
                    event.ignore()
                    return
                self.setWindowTitle(old_title)
        # Wait for Analyze Masks filter preview worker if dialog is open (never destroy QThread while running)
        analyze_tab = getattr(self, "analyze_tab", None)
        preview_dialog = getattr(analyze_tab, "_filter_preview_dialog", None) if analyze_tab else None
        if preview_dialog is not None:
            w = getattr(preview_dialog, "_preview_worker", None)
            if w is not None and w.isRunning():
                old_title = self.windowTitle()
                self.setWindowTitle("Waiting for filter preview to finish…")
                QApplication.processEvents()
                if not w.wait(120000):  # wait up to 2 minutes; if timeout, do not close
                    self.setWindowTitle(old_title)
                    QMessageBox.warning(
                        self,
                        "Please wait",
                        "Filter Effects Preview is still computing. Close the preview window first or wait for it to finish, then close the application.",
                    )
                    event.ignore()
                    return
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

    # Keys that refer to file/folder paths. When applying a session we do not restore these,
    # so the user's current paths (e.g. new input to re-run with same settings) are kept.
    _SESSION_PATH_KEYS = frozenset({"file_path", "single_fi_path"})

    def apply_session_params(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        for key, _ in SESSION_TAB_KEYS:
            if key not in data:
                continue
            tab = getattr(self, TAB_WIDGET_ATTR[key])
            if hasattr(tab, "set_session_params"):
                tab_data = dict(data[key])
                for path_key in self._SESSION_PATH_KEYS:
                    tab_data.pop(path_key, None)
                tab.set_session_params(tab_data)

    def run_workflow_from_session(self, log_data: dict[str, Any]) -> None:
        """Apply session params from log (FI, z-stack, foci, model, filters, etc.); run full pipeline (conversion → segment → analyze → tracking) on the current TIFF Conversion path only. That path is the new file/folder; downstream steps use outputs from the previous step (Tif_files → _segmentation_omni → analyze → tracking). So you can run the same analysis type from a session on a different file or folder."""
        tabs_data = log_data.get("tabs", log_data)
        if not isinstance(tabs_data, dict):
            QMessageBox.warning(self, "Re-run analysis", "Session log has no tab data.")
            return
        nd2_tab = getattr(self, TAB_WIDGET_ATTR.get("nd2"), None)
        if nd2_tab is None or not hasattr(nd2_tab, "file_path"):
            QMessageBox.warning(self, "Re-run analysis", "TIFF Conversion tab not available.")
            return
        seed_path = (nd2_tab.file_path.text() or "").strip()
        if not seed_path or not Path(seed_path).exists():
            QMessageBox.information(
                self,
                "Re-run analysis",
                "Set the path to your new file or folder in the TIFF Conversion tab. Re-run will apply the session parameters and run conversion → segment → analyze → tracking on that path (outputs go next to the input).",
            )
            return
        # Apply session (all parameters: FI, z-stack, foci, model, filters, etc.; paths from log may overwrite UI)
        self.apply_session_params(tabs_data)
        # Use only the user's chosen path: set it back on nd2 and load file info; we will set other tabs when chaining
        nd2_tab.file_path.setText(seed_path)
        if Path(seed_path).is_file():
            nd2_tab.load_file_info(seed_path)
        else:
            folder = Path(seed_path)
            supported = sorted(
                list(folder.glob("*.nd2")) + list(folder.glob("*.ND2")) +
                list(folder.glob("*.lif")) + list(folder.glob("*.LIF")) +
                list(folder.glob("*.czi")) + list(folder.glob("*.CZI")) +
                list(folder.glob("*.tif")) + list(folder.glob("*.tiff")) +
                list(folder.glob("*.TIF")) + list(folder.glob("*.TIFF"))
            )
            if supported and hasattr(nd2_tab, "load_folder_info"):
                nd2_tab.load_folder_info(seed_path, supported)
        # Only run steps that were actually performed in the session that produced this log
        order = ("nd2", "segmentation", "analyze_masks", "tracking")
        steps = []
        for key in order:
            if key not in TAB_WIDGET_ATTR:
                continue
            tab_data = tabs_data.get(key, {})
            if not isinstance(tab_data, dict):
                tab_data = {}
            # Backward compatibility: if "performed" missing (old log), treat as True
            if not tab_data.get("performed", True):
                continue
            tab = getattr(self, TAB_WIDGET_ATTR.get(key), None)
            if tab is None or not hasattr(tab, "workflow_step_finished"):
                continue
            if key == "nd2":
                run_fn = getattr(tab, "convert_nd2", None)
            elif key == "segmentation":
                run_fn = getattr(tab, "process_file_or_folder", None)
            elif key == "analyze_masks":
                run_fn = getattr(tab, "analyze_masks", None)
            elif key == "tracking":
                run_fn = getattr(tab, "run_tracking", None)
            else:
                run_fn = None
            if callable(run_fn):
                steps.append((key, tab, run_fn))
        if not steps:
            QMessageBox.information(
                self,
                "Re-run analysis",
                "No steps were recorded as performed in this session log. Run at least one step (conversion, segment, analyze, or tracking) before saving a session to use Re-run.",
            )
            return
        self._workflow_replay_steps = steps
        self._workflow_replay_index = 0
        self._workflow_replay_connection = None
        self._workflow_replay_seed_path = seed_path

        def _conversion_output_path(seed: str) -> Path:
            p = Path(seed)
            base = p.parent if p.is_file() else p
            return base / "Tif_files"

        def _segment_output_path(segment_input_path: str) -> Path:
            # When segment input is Tif_files (or Tif_*), masks go to _segmentation_omni
            folder = Path(segment_input_path)
            if folder.name.startswith("Tif_"):
                return folder / "_segmentation_omni"
            return folder / "_segmentation_omni"

        def on_step_done(success: bool) -> None:
            idx = self._workflow_replay_index
            tab = self._workflow_replay_steps[idx][1]
            if self._workflow_replay_connection is not None:
                try:
                    tab.workflow_step_finished.disconnect(self._workflow_replay_connection)
                except Exception:
                    pass
                self._workflow_replay_connection = None
            if not success:
                QMessageBox.warning(
                    self, "Workflow stopped", "A step in the workflow failed. See the log for details."
                )
                return
            self._workflow_replay_index += 1
            if self._workflow_replay_index >= len(self._workflow_replay_steps):
                QMessageBox.information(self, "Re-run analysis", "Workflow completed successfully.")
                return
            # Chain: set next step's input path from previous step's output, then run
            seed = getattr(self, "_workflow_replay_seed_path", "")
            next_key, next_tab, next_run_fn = self._workflow_replay_steps[self._workflow_replay_index]
            if next_key == "segmentation":
                next_path = _conversion_output_path(seed)
                if next_path.exists():
                    next_tab.file_path.setText(str(next_path))
                else:
                    QMessageBox.warning(self, "Workflow stopped", f"Conversion output not found: {next_path}")
                    return
            elif next_key == "analyze_masks":
                seg_input = _conversion_output_path(seed)
                next_path = _segment_output_path(str(seg_input))
                if next_path.exists():
                    next_tab.file_path.setText(str(next_path))
                else:
                    QMessageBox.warning(self, "Workflow stopped", f"Segment output not found: {next_path}")
                    return
            elif next_key == "tracking":
                seg_input = _conversion_output_path(seed)
                next_path = _segment_output_path(str(seg_input))
                if next_path.exists():
                    next_tab.file_path.setText(str(next_path))
                else:
                    QMessageBox.warning(self, "Workflow stopped", f"Segment output not found: {next_path}")
                    return
            _run_next()

        def _run_next() -> None:
            key, tab, run_fn = self._workflow_replay_steps[self._workflow_replay_index]
            self._workflow_replay_connection = on_step_done
            tab.workflow_step_finished.connect(on_step_done)
            run_fn()

        # If first step is not conversion, set its path from seed (e.g. segment needs Tif_files from seed)
        first_key, first_tab, _ = self._workflow_replay_steps[0]
        if first_key == "segmentation":
            p = _conversion_output_path(seed_path)
            if p.exists():
                first_tab.file_path.setText(str(p))
            else:
                QMessageBox.warning(
                    self, "Workflow stopped",
                    f"Segment needs conversion output at {p}. Run conversion first or set path to an existing Tif_files folder.",
                )
                return
        elif first_key == "analyze_masks":
            seg_out = _segment_output_path(str(_conversion_output_path(seed_path)))
            if seg_out.exists():
                first_tab.file_path.setText(str(seg_out))
            else:
                QMessageBox.warning(self, "Workflow stopped", f"Analyze needs segment output at {seg_out}. Run conversion and segment first.")
                return
        elif first_key == "tracking":
            seg_out = _segment_output_path(str(_conversion_output_path(seed_path)))
            if seg_out.exists():
                first_tab.file_path.setText(str(seg_out))
            else:
                QMessageBox.warning(self, "Workflow stopped", f"Tracking needs segment output at {seg_out}. Run conversion and segment first.")
                return
        # first_key == "nd2" already has path set (seed_path)
        _run_next()

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
        if sys.platform == "darwin" and _running_from_app_bundle():
            pin_dock_action = QAction("Pin BactiMate to Dock (for Reopen)", self)
            pin_dock_action.triggered.connect(self._show_pin_to_dock_help)
            pin_dock_action.setToolTip("How to get Reopen from the Dock to work after quitting.")
            help_menu.addAction(pin_dock_action)
        check_env_action = QAction("Check environment", self)
        check_env_action.triggered.connect(self._check_environment)
        check_env_action.setToolTip("Verify required dependencies (numpy, PyQt6, tifffile, etc.) are installed.")
        help_menu.addAction(check_env_action)

    def _show_pin_to_dock_help(self) -> None:
        """Explain how to pin BactiMate so Reopen from the Dock works (macOS .app only)."""
        msg = (
            "To reopen BactiMate from the Dock after quitting:\n\n"
            "1. Remove any existing BactiMate icon from the Dock (drag it off).\n\n"
            "2. In Finder, go to where BactiMate.app is (e.g. Desktop or Applications).\n\n"
            "3. Drag BactiMate.app onto the Dock.\n\n"
            "4. Use that Dock icon to launch and to reopen after quit.\n\n"
            "Do not use \"Keep in Dock\" from the running app’s Dock menu — that pins the wrong target and Reopen will not work."
        )
        try:
            app_root = get_app_root()
            # App root is Contents/Resources; .app is Contents/../ so parent.parent
            app_bundle = app_root.parent.parent if "Contents" in str(app_root) else app_root.parent
            if app_bundle.suffix == ".app" and app_bundle.exists():
                open_folder = QMessageBox.question(
                    self,
                    "Pin BactiMate to Dock",
                    msg + "\n\nOpen the folder containing BactiMate.app in Finder?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if open_folder == QMessageBox.StandardButton.Yes:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(app_bundle.parent)))
            else:
                QMessageBox.information(self, "Pin BactiMate to Dock", msg)
        except Exception:
            QMessageBox.information(self, "Pin BactiMate to Dock", msg)

    def _show_about(self) -> None:
        try:
            from bactimate import __version__
            version = __version__
        except Exception:
            version = "1.5"
        QMessageBox.about(
            self,
            "About BactiMate",
            f"<b>BactiMate</b> – bacteria time-lapse batch processing and analysis.<br><br>Version {version}<br><br>"
            "Documentation: <b>Help → User manual</b>.",
        )

    def _check_environment(self) -> None:
        """Run environment check and show result in a dialog and in the log."""
        try:
            from bactimate.scripts.check_environment import check_environment
            report = check_environment()
            self.log_output(report + "\n\n")
            QMessageBox.information(self, "Check environment", report)
        except Exception as e:
            self.log_output(f"Check environment failed: {e}\n")
            QMessageBox.warning(self, "Check environment", f"Could not run check:\n{e}")

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
        # Condense old tqdm-style progress (bar + ###) to one short line: "28% (9/32) ETA 1:09"
        tqdm_match = re.match(
            r"^(\d+)%\s*\|[#\s]+\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]",
            s,
        )
        if tqdm_match:
            pct, current, total, rest = tqdm_match.groups()
            eta_match = re.search(r"<(\d+):(\d+)", rest)
            if eta_match:
                m, sec = eta_match.group(1), eta_match.group(2)
                return f"  {pct}% ({current}/{total})  ETA {int(m)}:{sec}"
            return f"  {pct}% ({current}/{total})"
        # Already simple format "91% (29/32)  ETA 00:11" – normalize for display
        simple_match = re.match(r"^(\d+)%\s*\(\s*(\d+)/(\d+)\s*\)\s*(?:ETA\s*(\d+):(\d+))?", s)
        if simple_match:
            pct, current, total = simple_match.group(1), simple_match.group(2), simple_match.group(3)
            eta_m, eta_s = simple_match.group(4), simple_match.group(5)
            if eta_m is not None and eta_s is not None:
                return f"  {pct}% ({current}/{total})  ETA {int(eta_m)}:{eta_s}"
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
        scrollbar = self.log_text.verticalScrollBar()
        # Only auto-scroll to bottom if user is already near the bottom (so scrolling up to read is not overridden)
        at_bottom = scrollbar.maximum() - scrollbar.value() <= 20
        self.log_text.append(text)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        message_lower = message.lower()
        # Don't show indeterminate bar when the message indicates run finished (would leave bar "moving" after done)
        if any(
            done in message_lower
            for done in [
                "merged ", "removed ", "analysis complete", "completed successfully",
                "task(s) succeeded", "cancelled", "combined summary saved",
            ]
        ):
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        elif any(
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
        try:
            import threading
            thread_name = getattr(threading.current_thread(), "name", "unknown")
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            block = "".join(lines)
            write_crash_log(
                f"\n--- Uncaught exception {datetime.now().isoformat()} ---\n"
                f"Thread: {thread_name}\n"
                f"(If this occurs when closing the Filter Effects Preview or during Analyze Masks, a background thread may still have been running.)\n"
                f"{block}"
            )
        except Exception:
            pass

    sys.excepthook = pyqt6_exception_handler

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("BactiMate")
        app.setApplicationDisplayName("BactiMate")
        if sys.platform == "darwin":
            try:
                import setproctitle
                setproctitle.setproctitle("BactiMate")
            except ImportError:
                pass
            _set_macos_process_name("BactiMate")
        icon_path = _get_app_icon_path()
        if icon_path is not None:
            app.setWindowIcon(QIcon(str(icon_path)))
            if sys.platform == "darwin":
                _set_macos_dock_icon(icon_path)
        app.setStyle("Fusion")
        app.setStyleSheet(get_modern_stylesheet())
        # Route Qt messages (e.g. QThread destroyed while running) to the same GUI log
        try:
            from PyQt6.QtCore import qInstallMessageHandler, QtMsgType
            def _qt_message_handler(msg_type, context, message):
                try:
                    level = "unknown"
                    if msg_type == QtMsgType.QtDebugMsg:
                        level = "QtDebug"
                    elif msg_type == QtMsgType.QtInfoMsg:
                        level = "QtInfo"
                    elif msg_type == QtMsgType.QtWarningMsg:
                        level = "QtWarning"
                    elif msg_type == QtMsgType.QtCriticalMsg:
                        level = "QtCritical"
                    elif msg_type == QtMsgType.QtFatalMsg:
                        level = "QtFatal"
                    loc = ""
                    if context is not None and getattr(context, "function", None):
                        loc = f" ({getattr(context, 'file', '')}:{getattr(context, 'line', '')} {context.function})"
                    write_gui_log_line(level, f"{message}{loc}")
                except Exception:
                    pass
            qInstallMessageHandler(_qt_message_handler)
        except Exception:
            pass
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
        # Bring to front (macOS often opens behind other windows when launched from Dock/Finder)
        if sys.platform == "darwin":
            try:
                from AppKit import NSApp  # type: ignore[import-untyped]
                NSApp.activateIgnoringOtherApps_(True)
            except Exception:
                pass
        window.raise_()
        window.activateWindow()
        log.info("BactiMate window shown.")
        sys.exit(app.exec())
    except Exception as e:
        log_exception(get_logger(), "Failed to start GUI")
        print(f"Error starting GUI: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
