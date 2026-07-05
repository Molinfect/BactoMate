"""
Main application window and entrypoint. Import tab classes from the main module to avoid circular imports.
"""
from __future__ import annotations
import base64
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, Tuple, Type
from bactomate.gui.log_config import get_logger, log_exception, write_crash_log, write_gui_log_line

def _startup_import_log(message: str) -> None:
    try:
        write_gui_log_line('main-window-import', message)
    except Exception:
        pass
_startup_import_log('before PyQt6 imports')
from PyQt6.QtCore import QByteArray, QEvent, Qt, QCoreApplication, QObject, QTimer, QUrl
from PyQt6.QtGui import QCursor, QFont, QIcon, QPixmap, QAction, QKeySequence, QDesktopServices, QGuiApplication
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from PyQt6.QtWidgets import QApplication, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMenu, QMenuBar, QProgressBar, QPushButton, QSizePolicy, QSplitter, QStackedWidget, QTextEdit, QVBoxLayout, QWidget
_startup_import_log('after PyQt6 imports')
_startup_import_log('before constants/preferences imports')
from bactomate.gui.constants import LOGS_DIR, MAX_SESSION_LOGS, SESSION_TAB_KEYS, TAB_WIDGET_ATTR
from bactomate.gui.session_schema import normalize_session_tabs
from bactomate.gui.preferences import get_default_browse_dir, get_last_dir, get_state, set_last_dir, set_state
_startup_import_log('after constants/preferences imports')

def _import_tab_shell_classes() -> tuple:
    """Lazy-import heavy tab classes to reduce startup import time."""
    from bactomate.gui.tabs.analyze_masks import AnalyzeTab, FilteringTab
    from bactomate.gui.tabs.file_conversion import ND2ConversionTab
    from bactomate.gui.tabs.omnipose_seg import OmniposeSegTab
    from bactomate.gui.tabs.post_processing import PostProcessingTab
    from bactomate.gui.tabs.segmentation import SegmentationTab
    from bactomate.gui.tabs.session_logs import SessionLogsPanel
    from bactomate.gui.tabs.tracking import TrackingTab
    from bactomate.gui.tabs.training import TrainingTab
    from bactomate.gui.tabs.unet_seg import UNetSegmentTab
    return (ND2ConversionTab, SegmentationTab, TrackingTab, PostProcessingTab, TrainingTab, AnalyzeTab, FilteringTab, OmniposeSegTab, UNetSegmentTab, SessionLogsPanel)

def _get_tab_classes() -> Tuple[Type[Any], ...]:
    """Return primary module tab classes (lazy import)."""
    (ND2ConversionTab, SegmentationTab, TrackingTab, PostProcessingTab, TrainingTab, *_rest) = _import_tab_shell_classes()
    return (ND2ConversionTab, SegmentationTab, TrackingTab, PostProcessingTab, TrainingTab)
_startup_import_log('before paths/styles imports')
from bactomate.gui.paths import get_app_root, get_omnipose_pretrained_cache_dir, get_omnipose_user_models_dir, get_unet_models_dir, _running_from_app_bundle
from bactomate.gui.styles import get_modern_stylesheet
_startup_import_log('after paths/styles imports')
_startup_import_log('before dashboard/project imports')
from bactomate.gui.dashboard import DashboardWidget
from bactomate.gui.module_shell import ModuleShellWidget
from bactomate.gui.project_model import MODULE_DASHBOARD, MODULE_FILE_CONVERSION, MODULE_POST_PROCESSING, MODULE_SEGMENTATION, MODULE_TRACKING, MODULE_TRAINING, default_project_dict, load_project, migrate_project, path_relative_to_workspace, resolve_maybe_relative, save_project, store_module_path
_startup_import_log('after dashboard/project imports')
from bactomate.gui.workspace_scanner import WorkspaceScanResult, list_convertible_microscopy_files, scan_workspace
from bactomate.gui.widgets import show_confirm_dialog, show_message_dialog
_startup_import_log('after workspace_scanner import')
_STACK_DASHBOARD = 0
_STACK_FILE_CONVERSION = 1
_STACK_SEGMENTATION = 2
_STACK_TRACKING = 3
_STACK_POST_PROCESSING = 4
_STACK_TRAINING = 5
_MODULE_ID_TO_STACK: dict[str, int] = {MODULE_DASHBOARD: _STACK_DASHBOARD, MODULE_FILE_CONVERSION: _STACK_FILE_CONVERSION, MODULE_SEGMENTATION: _STACK_SEGMENTATION, MODULE_TRACKING: _STACK_TRACKING, MODULE_POST_PROCESSING: _STACK_POST_PROCESSING, MODULE_TRAINING: _STACK_TRAINING}
_STACK_TO_MODULE_ID: dict[int, str] = {v: k for (k, v) in _MODULE_ID_TO_STACK.items()}
_PIPELINE_NAV_BAR: tuple[tuple[str, str], ...] = ((MODULE_FILE_CONVERSION, 'File conversion'), (MODULE_SEGMENTATION, 'Segmentation'), (MODULE_TRACKING, 'Tracking'), (MODULE_POST_PROCESSING, 'Post-processing'), (MODULE_TRAINING, 'Training'))
_STATUS_LOG_TEXT_STYLESHEET = 'QTextEdit { background-color: #1f1f1f; color: #dce6f2; border: 1px solid #2d3a47; border-radius: 6px; padding: 6px 8px;}QTextEdit:focus { border-color: #4a9eff; background-color: #252525; }'

def _get_app_icon_path() -> Path | None:
    """App icon from the bundle first, then the raw GUI logo as fallback."""
    root = get_app_root()
    for p in (root / 'BactoMate.icns', root / 'img' / 'logo2.png'):
        if p.exists():
            return p.resolve()
    return None

def _set_macos_dock_icon(icon_path: Path) -> None:
    """Set Dock icon on macOS (Qt setWindowIcon does not update Dock)."""
    try:
        from AppKit import NSApp, NSImage
        img = NSImage.alloc().initWithContentsOfFile_(str(icon_path.resolve()))
        if img is not None:
            NSApp.setApplicationIconImage_(img)
    except Exception:
        pass

def _set_macos_process_name(name: str) -> None:
    """Set process name on macOS so Dock/menu bar show 'BactoMate' not 'Python'."""
    if sys.platform != 'darwin':
        return
    try:
        from Foundation import NSProcessInfo
        NSProcessInfo.processInfo().setProcessName_(name)
    except Exception:
        pass

def _activate_macos_window(window: QMainWindow | None) -> None:
    """Bring the existing main window forward for Dock/Finder reopen requests."""
    if window is None:
        return
    try:
        window.show()
        window.raise_()
        window.activateWindow()
        if sys.platform == 'darwin':
            try:
                from AppKit import NSApp
                NSApp.activateIgnoringOtherApps_(True)
            except Exception:
                pass
    except Exception:
        pass

def _notify_existing_macos_instance(server_name: str) -> bool:
    """Return True when another BactoMate instance accepted an activation request."""
    if sys.platform != 'darwin':
        return False
    try:
        socket = QLocalSocket()
        socket.connectToServer(server_name)
        if not socket.waitForConnected(200):
            return False
        socket.write(b'activate\n')
        socket.flush()
        socket.waitForBytesWritten(200)
        socket.disconnectFromServer()
        return True
    except Exception:
        return False

def _start_macos_single_instance_server(server_name: str, window_getter) -> QLocalServer | None:
    """Listen for second launches and route them to the existing BactoMate window."""
    if sys.platform != 'darwin':
        return None
    try:
        server = QLocalServer()
        if not server.listen(server_name):
            QLocalServer.removeServer(server_name)
            if not server.listen(server_name):
                get_logger().warning('Could not start single-instance listener: %s', server.errorString())
                return None

        def _handle_connection() -> None:
            while server.hasPendingConnections():
                sock = server.nextPendingConnection()
                if sock is not None:
                    sock.disconnectFromServer()
            window = window_getter()
            if window is None:
                QTimer.singleShot(250, lambda : _activate_macos_window(window_getter()))
            else:
                _activate_macos_window(window)
        server.newConnection.connect(_handle_connection)
        return server
    except Exception:
        return None
_MAIN_WIN_MIN_W = 960
_MAIN_WIN_MIN_H = 640
_MAIN_WIN_PREFERRED_W = 1300
_MAIN_WIN_PREFERRED_H = 1050
_MAIN_WIN_SCREEN_MARGIN = 48
_MAIN_WIN_SIZE_RATIO = 0.92
_MAIN_WIN_LARGE_SCREEN_WIDTH_BONUS = 40

def _effective_preferred_caps(usable_w: int, usable_h: int, preferred_w: int, preferred_h: int) -> tuple[int, int]:
    """
    Raise the opening-size cap when the display has more room than a typical laptop.

    Base preferred (1300×1050) still applies on small screens; ultrawide / large monitors
    get a modestly larger default (e.g. 3400×1350 → up to ~1800×1150 instead of 1300×1050).
    """
    bonus_w = _MAIN_WIN_LARGE_SCREEN_WIDTH_BONUS if usable_w >= 1920 and usable_h >= 900 else 0
    if usable_w >= 3000 and usable_h >= 1200:
        return (min(1800 + bonus_w, usable_w), min(1150, usable_h))
    if usable_w >= 2400 and usable_h >= 1100:
        return (min(1650 + bonus_w, usable_w), min(1100, usable_h))
    if usable_w >= 1920 and usable_h >= 1000:
        return (min(1500 + bonus_w, usable_w), min(1080, usable_h))
    return (min(preferred_w + bonus_w, usable_w), min(preferred_h, usable_h))

def _initial_window_dimensions(min_w: int, min_h: int, preferred_w: int, preferred_h: int, screen_margin: int=48, size_ratio: float | None=None) -> tuple[int, int, int, int]:
    """
    Fit initial window size to the screen under the cursor (or primary screen).

    Returns (effective_min_w, effective_min_h, initial_w, initial_h).
    When size_ratio is provided, the initial size is that fraction of availableGeometry
    minus margin, capped at an effective preferred size (see _effective_preferred_caps).
    Effective minimums are capped by usable width/height so min size never exceeds the screen.
    """
    app = QApplication.instance()
    if app is None:
        return (min_w, min_h, preferred_w, preferred_h)
    screen = QGuiApplication.screenAt(QCursor.pos())
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    if screen is None:
        return (min_w, min_h, preferred_w, preferred_h)
    avail = screen.availableGeometry()
    uw = max(0, avail.width() - screen_margin)
    uh = max(0, avail.height() - screen_margin)
    eff_min_w = min(min_w, uw) if uw > 0 else min_w
    eff_min_h = min(min_h, uh) if uh > 0 else min_h
    (cap_w, cap_h) = _effective_preferred_caps(uw, uh, preferred_w, preferred_h)
    if size_ratio is not None:
        ratio = max(0.1, min(float(size_ratio), 1.0))
        init_w = min(cap_w, uw, max(eff_min_w, int(round(uw * ratio))))
        init_h = min(cap_h, uh, max(eff_min_h, int(round(uh * ratio))))
    else:
        init_w = min(cap_w, uw)
        init_h = min(cap_h, uh)
    init_w = max(eff_min_w, init_w)
    init_h = max(eff_min_h, init_h)
    init_w = min(init_w, uw) if uw > 0 else init_w
    init_h = min(init_h, uh) if uh > 0 else init_h
    return (eff_min_w, eff_min_h, init_w, init_h)

def _work_area_minimum_height(window_height: int) -> int:
    """Central stack min height: scales down on short windows (header + collapsed log reserved)."""
    return min(360, max(220, window_height - 260))

def _prune_old_session_logs(logs_dir: Path, keep_count: int) -> None:
    """Keep only the most recent keep_count session log files; delete older ones."""
    if keep_count <= 0 or not logs_dir.is_dir():
        return
    session_files = sorted(logs_dir.glob('*_session.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in session_files[keep_count:]:
        try:
            path.unlink()
        except OSError:
            log_exception(get_logger(), f'Failed to remove old session log {path.name}')

def _new_session_log_path(logs_dir: Path, started_at: datetime) -> Path:
    """Return a per-app-session recipe path without clobbering an existing session."""
    stem = started_at.strftime('%Y-%m-%d_%H-%M-%S')
    base = logs_dir / f'{stem}_session.json'
    if not base.exists():
        return base
    for idx in range(2, 1000):
        candidate = logs_dir / f'{stem}_{idx}_session.json'
        if not candidate.exists():
            return candidate
    return logs_dir / f"{stem}_{started_at.strftime('%f')}_session.json"

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self._session_log_started_at = datetime.now()
        self._session_log_path = _new_session_log_path(LOGS_DIR, self._session_log_started_at)
        self.shared_metadata = {'pixel_size': None, 'time_per_frame_min': None}
        self._log_panel_window_height_added = 0
        self._log_split_defer_count = 0
        self._workspace_root: Path | None = None
        self._project_data: dict[str, Any] = {}
        self._workspace_scan: WorkspaceScanResult | None = None
        self._module_shells: dict[str, ModuleShellWidget] = {}
        self.init_ui()
        self._session_autosave_timer = QTimer(self)
        self._session_autosave_timer.setInterval(30000)
        self._session_autosave_timer.timeout.connect(lambda : self._save_session_log(refresh_panel=False))
        self._session_autosave_timer.start()
        QTimer.singleShot(0, lambda : self._save_session_log(refresh_panel=True))
        from bactomate.gui.layout_helpers import patch_all_combo_popups
        QTimer.singleShot(0, lambda : patch_all_combo_popups(self))

    def init_ui(self) -> None:
        self.setWindowTitle('BactoMate')

        def _ensure_model_dirs() -> None:
            try:
                get_omnipose_user_models_dir()
                get_omnipose_pretrained_cache_dir()
                get_unet_models_dir()
            except Exception:
                pass
        QTimer.singleShot(0, _ensure_model_dirs)
        (eff_min_w, eff_min_h, init_w, init_h) = _initial_window_dimensions(_MAIN_WIN_MIN_W, _MAIN_WIN_MIN_H, _MAIN_WIN_PREFERRED_W, _MAIN_WIN_PREFERRED_H, screen_margin=_MAIN_WIN_SCREEN_MARGIN, size_ratio=_MAIN_WIN_SIZE_RATIO)
        self._window_init_w = init_w
        self._window_init_h = init_h
        self.setMinimumSize(eff_min_w, eff_min_h)
        self.resize(init_w, init_h)
        self._center_window()
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        title_bar = QWidget()
        title_bar.setProperty('class', 'title_bar')
        title_bar.setStyleSheet("QWidget[class='title_bar'] {background: transparent;padding: 8px;border-bottom: 1px solid #4a9eff;}")
        title_bar_layout = QHBoxLayout()
        title_bar_layout.setContentsMargins(12, 6, 12, 6)
        title_bar_layout.addStretch()
        logo_label = QLabel()
        logo_path = get_app_root() / 'img' / 'logo2.png'
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if pixmap.height() > 60:
                pixmap = pixmap.scaledToHeight(60, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setStyleSheet('background: transparent;')
        else:
            logo_label.setText('BactoMate')
            logo_label.setStyleSheet('font-size: 18pt; font-weight: bold; color: #ECF0F1; background: transparent;')
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        title_bar_layout.addWidget(logo_label)
        title_text = QLabel('BactoMate - an integrated platform for quantitative microscopy image analysis')
        title_text.setStyleSheet('QLabel {font-size: 20pt;font-weight: bold;color: #ECF0F1;background: transparent;padding-left: 12px;}')
        title_text.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        title_bar_layout.addWidget(title_text)
        title_bar_layout.addStretch()
        title_bar.setLayout(title_bar_layout)
        main_layout.addWidget(title_bar)
        (ND2ConversionTab, SegmentationTab, TrackingTab, PostProcessingTab, TrainingTab, AnalyzeTab, FilteringTab, _OmniposeSegTab, UNetSegmentTab, SessionLogsPanel) = _import_tab_shell_classes()
        _startup_import_log('after tabs import')
        self.nd2_tab = ND2ConversionTab(self)
        self.unet_segment_tab = UNetSegmentTab(self)
        self.filtering_tab = FilteringTab(self, embedded=True)
        self.analyze_tab = AnalyzeTab(self, embedded=True)
        self.segmentation_tab = SegmentationTab(self, unet_tab=self.unet_segment_tab, filtering_tab=self.filtering_tab, analyze_tab=self.analyze_tab)
        self.tracking_tab = TrackingTab(self)
        self.post_processing_tab = PostProcessingTab(self)
        self.training_tab = TrainingTab(self, omnipose_tab=self.segmentation_tab.omnipose_tab, unet_tab=self.unet_segment_tab)
        self.dashboard_widget = DashboardWidget(self, on_open_workspace=self._open_workspace_dialog, on_drop_workspace=self._open_workspace_from_drop, on_clear_workspace=self._clear_workspace, on_go_module=self._go_module_from_dashboard)
        self._shell_file = ModuleShellWidget(self, module_id=MODULE_FILE_CONVERSION, content=self.nd2_tab, on_back=self._show_dashboard, pipeline_navigation=_PIPELINE_NAV_BAR, on_open_module=self._go_module_from_dashboard)
        self._shell_segmentation = ModuleShellWidget(self, module_id=MODULE_SEGMENTATION, content=self.segmentation_tab, on_back=self._show_dashboard, pipeline_navigation=_PIPELINE_NAV_BAR, on_open_module=self._go_module_from_dashboard)
        self._shell_tracking = ModuleShellWidget(self, module_id=MODULE_TRACKING, content=self.tracking_tab, on_back=self._show_dashboard, pipeline_navigation=_PIPELINE_NAV_BAR, on_open_module=self._go_module_from_dashboard)
        self._shell_post = ModuleShellWidget(self, module_id=MODULE_POST_PROCESSING, content=self.post_processing_tab, on_back=self._show_dashboard, pipeline_navigation=_PIPELINE_NAV_BAR, on_open_module=self._go_module_from_dashboard)
        self._shell_training = ModuleShellWidget(self, module_id=MODULE_TRAINING, content=self.training_tab, on_back=self._show_dashboard, pipeline_navigation=_PIPELINE_NAV_BAR, on_open_module=self._go_module_from_dashboard, show_preview_picker=False, show_preview_panel=False)
        self._module_shells = {MODULE_FILE_CONVERSION: self._shell_file, MODULE_SEGMENTATION: self._shell_segmentation, MODULE_TRACKING: self._shell_tracking, MODULE_POST_PROCESSING: self._shell_post, MODULE_TRAINING: self._shell_training}
        self.nd2_tab.attach_module_preview_mount(self._shell_file.preview_surface().mount_widget())
        self.segmentation_tab.attach_module_preview_mount(self._shell_segmentation.preview_surface().mount_widget())
        self.tracking_tab.attach_module_preview_mount(self._shell_tracking.preview_surface().mount_widget())
        self.post_processing_tab.attach_module_preview_mount(self._shell_post.preview_surface().mount_widget())
        try:
            self.nd2_tab.workflow_step_finished.connect(self._on_file_conversion_finished)
        except Exception:
            pass
        self._central_stack = QStackedWidget()
        self._central_stack.addWidget(self.dashboard_widget)
        self._central_stack.addWidget(self._shell_file)
        self._central_stack.addWidget(self._shell_segmentation)
        self._central_stack.addWidget(self._shell_tracking)
        self._central_stack.addWidget(self._shell_post)
        self._central_stack.addWidget(self._shell_training)
        self._sync_pipeline_nav_highlights(MODULE_DASHBOARD)
        self._work_area_split = QSplitter(Qt.Orientation.Vertical)
        self._work_area_split.setObjectName('MainWorkAreaSplit')
        self._work_area_split.setChildrenCollapsible(False)
        self._central_stack.setMinimumHeight(_work_area_minimum_height(init_h))
        self._work_area_split.addWidget(self._central_stack)
        log_group = QGroupBox('Status && Session logs  ▲')
        log_layout = QVBoxLayout()
        self._log_content = QWidget()
        log_content_layout = QHBoxLayout(self._log_content)
        log_content_layout.setContentsMargins(0, 0, 0, 0)
        log_content_layout.setSpacing(14)
        left_log = QWidget()
        left_log_layout = QVBoxLayout(left_log)
        left_log_layout.setContentsMargins(0, 0, 0, 0)
        left_log_layout.setSpacing(6)
        status_title = QLabel('Status (all modules)')
        status_title.setStyleSheet('font-weight: bold; color: #aeb6c2;')
        status_title.setToolTip('Activity log from all modules: file conversion, segmentation, analyze masks, tracking, post-processing, etc.')
        left_log_layout.addWidget(status_title)
        self.status_message = QLabel('')
        self.status_message.setStyleSheet('color: #7F8C8D; font-size: 10pt;')
        self.status_message.setVisible(False)
        self.status_message.setWordWrap(True)
        left_log_layout.addWidget(self.status_message)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%p%')
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        left_log_layout.addWidget(self.progress_bar)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        monospace_font = QFont('Monaco', 11)
        if not monospace_font.exactMatch():
            monospace_font = QFont('Courier New', 11)
        self.log_text.setFont(monospace_font)
        self.log_text.setMinimumHeight(70)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log_text.setStyleSheet(_STATUS_LOG_TEXT_STYLESHEET)
        log_size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log_text.setSizePolicy(log_size_policy)
        left_log_layout.addWidget(self.log_text, 1)
        log_content_layout.addWidget(left_log, 1)
        self.logs_panel = SessionLogsPanel(self, on_rerun_session=self.open_rerun_session_dialog)
        log_content_layout.addWidget(self.logs_panel, 2)
        log_layout.addWidget(self._log_content)
        log_group.setLayout(log_layout)
        self._work_area_split.addWidget(log_group)
        self._work_area_split.setStretchFactor(0, 1)
        self._work_area_split.setStretchFactor(1, 0)
        main_layout.addWidget(self._work_area_split, 1)
        self._log_group = log_group
        _COLLAPSED_LINE_STYLE = 'QGroupBox { border: none; border-top: 1px solid #2d2d2d; border-radius: 0; margin-top: 0; padding-top: 6px; padding-bottom: 6px; padding-left: 0; background: #1a1a1a;} QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: 12px; padding: 8px 0 0 0; color: #4a9eff; background: transparent; }'

        def toggle_log_visibility():
            try:
                if not self._log_content or not log_group:
                    return
                visible = self._log_content.isVisible()
                if visible:
                    self._shrink_window_after_log_panel_collapse()
                    self._log_content.setVisible(False)
                    log_group.setTitle('Status && Session logs  ▲')
                    log_group.setFixedHeight(32)
                    log_group.setStyleSheet(_COLLAPSED_LINE_STYLE)
                    QTimer.singleShot(0, lambda : self._sync_log_panel_splitter_sizes(True))
                else:
                    self._log_content.setVisible(True)
                    log_group.setTitle('Status && Session logs  ▼')
                    log_group.setMinimumHeight(0)
                    log_group.setMaximumHeight(16777215)
                    log_group.setStyleSheet('')
                    QTimer.singleShot(0, self._prepare_expanded_log_panel_geometry)
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
        self._log_content.setVisible(False)
        log_group.setFixedHeight(32)
        log_group.setStyleSheet(_COLLAPSED_LINE_STYLE)
        main_layout.setStretchFactor(self._work_area_split, 1)
        margins = main_layout.contentsMargins()
        main_layout.setContentsMargins(margins.left(), margins.top(), margins.right(), 0)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self._add_menu_bar()
        self._restore_geometry_and_state()
        self.statusBar().hide()
        self._show_dashboard()
        self.log_output('Omnipose Batch Processing GUI started.\n')
        self.log_output('Open a workspace from the dashboard, pick a module, and configure parameters to begin.\n')
        self.log_output('Expand Status && Session logs at the bottom to watch progress and session snapshots.\n\n')
        QTimer.singleShot(0, lambda : self._sync_log_panel_splitter_sizes(True))
        QTimer.singleShot(120, lambda : self._sync_log_panel_splitter_sizes(True))

    def _sync_log_panel_splitter_sizes(self, collapsed: Optional[bool]=None) -> None:
        """Keep the bottom splitter pane from reserving space when logs are collapsed (avoids empty gap)."""
        split = getattr(self, '_work_area_split', None)
        if split is None:
            return
        log_c = getattr(self, '_log_content', None)
        if collapsed is None:
            collapsed = log_c is None or not log_c.isVisible()
        try:
            h = max(int(split.height()), 0)
            min_top = int(self._central_stack.minimumHeight())
            if h < 60:
                self._log_split_defer_count = self._log_split_defer_count + 1
                if self._log_split_defer_count <= 12:
                    QTimer.singleShot(50, lambda c=collapsed: self._sync_log_panel_splitter_sizes(c))
                return
            self._log_split_defer_count = 0
            if collapsed:
                bottom = 40
                top = max(min_top, h - bottom)
                split.setSizes([top, max(1, h - top)])
            else:
                reserved = min(360, max(160, h // 3))
                reserved = min(reserved, max(1, h - min_top))
                reserved = max(140, reserved)
                top = max(min_top, h - reserved)
                split.setSizes([top, max(1, h - top)])
        except Exception:
            pass

    def _prepare_expanded_log_panel_geometry(self) -> None:
        self._sync_log_panel_splitter_sizes(False)
        self._expand_window_for_log_panel()

    def _expand_window_for_log_panel(self) -> None:
        """After Status & Session logs is expanded, grow the main window so the work area is not squeezed."""
        if not getattr(self, '_log_content', None) or not self._log_content.isVisible():
            return
        lg = getattr(self, '_log_group', None)
        if lg is None:
            return
        self._log_panel_window_height_added = 0
        QApplication.processEvents()
        lg.updateGeometry()
        for _ in range(3):
            QApplication.processEvents()
        bar_h = 32
        content_h = max(lg.sizeHint().height(), lg.minimumSizeHint().height(), lg.height())
        delta = content_h - bar_h
        if delta < 2:
            return
        screen = self.screen()
        avail = screen.availableGeometry() if screen else None
        if avail is not None:
            room = avail.bottom() - self.frameGeometry().bottom() - 16
            delta = min(delta, max(0, room))
        if delta < 2:
            return
        self.resize(self.width(), self.height() + delta)
        self._log_panel_window_height_added = delta

    def _shrink_window_after_log_panel_collapse(self) -> None:
        """Restore main window height added when Status & Session logs is collapsed."""
        added = getattr(self, '_log_panel_window_height_added', 0) or 0
        self._log_panel_window_height_added = 0
        if added <= 0:
            return
        new_h = max(self.minimumHeight(), self.height() - added)
        self.resize(self.width(), new_h)

    def expand_status_log(self) -> None:
        """Expand the Status & Session logs panel if it is collapsed (e.g. when starting long-running tracking)."""
        if not getattr(self, '_log_content', None) or not getattr(self, '_log_group', None):
            return
        if self._log_content.isVisible():
            return
        self._log_content.setVisible(True)
        self._log_group.setTitle('Status && Session logs  ▼')
        self._log_group.setMinimumHeight(0)
        self._log_group.setMaximumHeight(16777215)
        self._log_group.setStyleSheet('')
        QTimer.singleShot(0, self._prepare_expanded_log_panel_geometry)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        try:
            if getattr(self, '_central_stack', None) is not None:
                self._central_stack.setMinimumHeight(_work_area_minimum_height(self.height()))
            log_c = getattr(self, '_log_content', None)
            if log_c is not None and (not log_c.isVisible()):
                self._sync_log_panel_splitter_sizes(True)
        except Exception:
            pass

    def closeEvent(self, event):

        def _wait_for_thread_worker(worker: Any, waiting_title: str, timeout_ms: int, warning_text: str) -> bool:
            """Wait for a running QThread-like worker and keep closeEvent logic compact."""
            if worker is None or not hasattr(worker, 'isRunning') or (not worker.isRunning()):
                return True
            old_title = self.windowTitle()
            self.setWindowTitle(waiting_title)
            QApplication.processEvents()
            ok = bool(worker.wait(timeout_ms))
            self.setWindowTitle(old_title)
            if not ok:
                show_message_dialog(self, 'Please wait', warning_text, kind='warning')
            return ok
        worker_wait_specs = [(getattr(self, 'tracking_tab', None), 'worker', 'Waiting for tracking to finish…', 300000, 'Tracking is still running. Wait for it to finish or cancel it, then close the application.'), (getattr(self, 'analyze_tab', None), 'worker', 'Waiting for Analyze Masks to finish…', 300000, 'Analyze Masks is still running. Wait for it to finish or cancel it, then close the application.'), (getattr(getattr(self, 'segmentation_tab', None), 'omnipose_tab', None), 'worker', 'Waiting for Omnipose processing to finish…', 300000, 'Omnipose processing is still running. Wait for it to finish or cancel it, then close the application.'), (getattr(self.segmentation_tab, '_cellpose_owner', None), 'worker', 'Waiting for Cellpose processing to finish…', 300000, 'Cellpose processing is still running. Wait for it to finish or cancel it, then close the application.'), (getattr(self, 'unet_segment_tab', None), 'worker', 'Waiting for U-Net processing to finish…', 300000, 'U-Net processing is still running. Wait for it to finish or cancel it, then close the application.'), (getattr(getattr(self, 'segmentation_tab', None), 'omnipose_tab', None), '_omnipose_train_worker', 'Waiting for Omnipose training to finish…', 300000, 'Omnipose training is still running. Wait for it to finish, then close the application.'), (getattr(self, 'nd2_tab', None), 'worker', 'Waiting for file conversion to finish…', 300000, 'File conversion is still running. Wait for it to finish or cancel it, then close the application.'), (getattr(self, 'post_processing_tab', None), 'worker', 'Waiting for post-processing to finish…', 300000, 'Post-processing is still running. Wait for it to finish, then close the application.'), (getattr(self, 'unet_segment_tab', None), '_unet_train_worker', 'Waiting for U-Net training to finish…', 300000, 'U-Net training is still running. Wait for it to finish, then close the application.')]
        for (tab, worker_attr, waiting_title, timeout_ms, warning_text) in worker_wait_specs:
            if tab is None:
                continue
            worker = getattr(tab, worker_attr, None)
            if not _wait_for_thread_worker(worker, waiting_title, timeout_ms, warning_text):
                event.ignore()
                return
        analyze_tab = getattr(self, 'analyze_tab', None)
        preview_dialog = getattr(analyze_tab, '_filter_preview_dialog', None) if analyze_tab else None
        if preview_dialog is not None:
            w = getattr(preview_dialog, '_preview_thread', None) or getattr(preview_dialog, '_preview_worker', None)
            if not _wait_for_thread_worker(w, 'Waiting for filter preview to finish…', 120000, 'Filter Effects Preview is still computing. Close the preview window first or wait for it to finish, then close the application.'):
                event.ignore()
                return
        self._save_geometry_and_state()
        self._save_session_log()
        super().closeEvent(event)

    def _restore_geometry_and_state(self) -> None:
        """Restore saved UI state (e.g. toolbars). Window size/position are not restored — each launch uses defaults + center."""
        state = get_state()
        if state and (not state.isEmpty()):
            self.restoreState(state)

    def _save_geometry_and_state(self) -> None:
        """Save UI state to QSettings. Window geometry is not persisted so each session starts at the default size."""
        set_state(self.saveState())

    def get_last_dir_for_tab(self, key: str) -> str:
        """Return last used directory for the given tab (e.g. preferences.LAST_DIR_ND2)."""
        return get_last_dir(key)

    def set_last_dir_for_tab(self, key: str, path: str) -> None:
        """Save last used directory for the given tab after a file/dir selection."""
        set_last_dir(key, path)

    def _save_session_log(self, *, refresh_panel: bool=True):
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            now = datetime.now()
            ts_display = now.strftime('%Y-%m-%d %H:%M:%S')
            started_display = self._session_log_started_at.strftime('%Y-%m-%d %H:%M:%S')
            path = Path(getattr(self, '_session_log_path', None) or _new_session_log_path(LOGS_DIR, now))
            self._session_log_path = path
            data = {'schema_version': 2, 'timestamp': ts_display, 'created_at': started_display, 'updated_at': ts_display, 'kind': 'analysis_recipe_session', 'tabs': self.gather_session_params()}
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            _prune_old_session_logs(LOGS_DIR, MAX_SESSION_LOGS)
            if refresh_panel and hasattr(self, 'logs_panel') and (self.logs_panel is not None):
                self.logs_panel._refresh_list()
        except Exception:
            log_exception(get_logger(), 'Failed to save session log')

    def gather_session_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for (key, _) in SESSION_TAB_KEYS:
            tab = getattr(self, TAB_WIDGET_ATTR[key])
            if hasattr(tab, 'get_session_params'):
                out[key] = tab.get_session_params()
        return out
    _SESSION_PATH_KEYS = frozenset({'file_path', 'single_fi_path'})

    def apply_session_params(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        data = normalize_session_tabs(data)
        for (key, _) in SESSION_TAB_KEYS:
            if key not in data:
                continue
            tab = getattr(self, TAB_WIDGET_ATTR[key])
            if hasattr(tab, 'set_session_params'):
                tab_data = dict(data[key])
                for path_key in self._SESSION_PATH_KEYS:
                    tab_data.pop(path_key, None)
                tab.set_session_params(tab_data)

    def open_rerun_session_dialog(self) -> None:
        """Choose a session JSON and run the full workflow on the current File conversion path (global action)."""
        (log_path, _) = QFileDialog.getOpenFileName(self, 'Choose session log', str(LOGS_DIR) if LOGS_DIR.exists() else get_default_browse_dir(), 'JSON (*.json);;All files (*)')
        if not log_path:
            return
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            show_message_dialog(self, 'Session log', f'Could not load session file:\n{e}', kind='warning')
            return
        if not callable(getattr(self, 'run_workflow_from_session', None)):
            show_message_dialog(self, 'Re-run analysis', 'Workflow replay is not available.', kind='warning')
            return
        self.run_workflow_from_session(data)

    def _path_is_within_workspace(self, path_str: str) -> bool:
        """True if ``path_str`` resolves to the workspace root or a path under it."""
        root = self._workspace_root
        if root is None or not (path_str or '').strip():
            return False
        try:
            p = Path(str(path_str).strip()).expanduser().resolve()
            r = root.resolve()
        except (OSError, ValueError):
            return False
        if p == r:
            return True
        try:
            p.relative_to(r)
            return True
        except ValueError:
            return False

    def _should_apply_workspace_hint_to_path_widget(self, widget: Any) -> bool:
        """Apply scan hint when the field is empty or points outside the current workspace."""
        if widget is None or not hasattr(widget, 'text'):
            return False
        text = str(widget.text() or '').strip()
        if not text:
            return True
        if self._workspace_root is None:
            return True
        return not self._path_is_within_workspace(text)

    def _apply_workspace_hints_for_module(self, module_id: str) -> None:
        """Apply workspace scan hints to module path fields when empty or stale (outside current workspace)."""
        scan = self._workspace_scan
        if scan is None:
            return
        hints = scan.hints
        if module_id == MODULE_FILE_CONVERSION:
            if hasattr(self.nd2_tab, 'file_path') and self._should_apply_workspace_hint_to_path_widget(self.nd2_tab.file_path):
                p = hints.get('file_conversion')
                if p and Path(p).exists():
                    self.nd2_tab.file_path.setText(p)
                    pp = Path(p)
                    if pp.is_file() and hasattr(self.nd2_tab, 'load_file_info'):
                        try:
                            self.nd2_tab.load_file_info(str(pp))
                        except Exception:
                            log_exception(get_logger(), 'Auto-load file conversion path')
                    elif pp.is_dir() and hasattr(self.nd2_tab, 'load_folder_info'):
                        supported = list_convertible_microscopy_files(pp, max_files=120)
                        if supported:
                            try:
                                self.nd2_tab.load_folder_info(str(pp), supported)
                            except Exception:
                                log_exception(get_logger(), 'Auto-load folder for file conversion')
            if hasattr(self.nd2_tab, 'refresh_conversion_candidate_list'):
                self.nd2_tab.refresh_conversion_candidate_list()
            QTimer.singleShot(150, lambda : getattr(self.nd2_tab, '_refresh_embedded_conversion_preview', lambda : None)())
        elif module_id == MODULE_SEGMENTATION:
            seg_path = getattr(self.segmentation_tab, 'file_path', None)
            if seg_path is not None and self._should_apply_workspace_hint_to_path_widget(seg_path):
                p = hints.get('segmentation')
                if p and Path(p).exists():
                    seg_path.setText(p)
            md = getattr(scan, 'mask_or_project_dir', None)
            if md is not None and md.exists():
                ms = str(md)
                if hasattr(self.filtering_tab, 'file_path') and self._should_apply_workspace_hint_to_path_widget(self.filtering_tab.file_path):
                    self.filtering_tab.file_path.setText(ms)
                if hasattr(self.analyze_tab, 'file_path') and self._should_apply_workspace_hint_to_path_widget(self.analyze_tab.file_path):
                    self.analyze_tab.file_path.setText(ms)
        elif module_id == MODULE_TRACKING:
            if hasattr(self.tracking_tab, 'file_path') and self._should_apply_workspace_hint_to_path_widget(self.tracking_tab.file_path):
                p = hints.get('tracking')
                if p and Path(p).exists():
                    self.tracking_tab.file_path.setText(p)
            if hasattr(self.tracking_tab, 'refresh_tracking_candidate_list'):
                self.tracking_tab.refresh_tracking_candidate_list()
        elif module_id == MODULE_POST_PROCESSING:
            pdir = hints.get('post_processing')
            if pdir and Path(pdir).exists():
                pp = Path(pdir)
                tifs = sorted(list(pp.glob('*.tif')) + list(pp.glob('*.tiff')))
                first = str(tifs[0]) if tifs else pdir
                if hasattr(self.post_processing_tab, 'montage_path') and self._should_apply_workspace_hint_to_path_widget(self.post_processing_tab.montage_path):
                    self.post_processing_tab.montage_path.setText(str(pp))
        elif module_id == MODULE_TRAINING:
            if hasattr(self.training_tab, 'train_dir_edit') and self._should_apply_workspace_hint_to_path_widget(self.training_tab.train_dir_edit):
                p = hints.get('training')
                if p:
                    self.training_tab.train_dir_edit.setText(p)

    def _sync_pipeline_nav_highlights(self, active_module_id: str) -> None:
        """Keep module shell top bars in sync: active module uses the same accent as the main header."""
        for shell in self._module_shells.values():
            shell.set_active_pipeline_module(active_module_id)

    def _get_current_module_id(self) -> str:
        idx = self._central_stack.currentIndex()
        return _STACK_TO_MODULE_ID.get(idx, MODULE_DASHBOARD)

    def _show_dashboard(self) -> None:
        self._central_stack.setCurrentIndex(_STACK_DASHBOARD)
        self._sync_pipeline_nav_highlights(MODULE_DASHBOARD)
        if isinstance(self._project_data, dict):
            self._project_data.setdefault('ui', {})['last_module'] = MODULE_DASHBOARD

    def _go_module_from_dashboard(self, module_id: str) -> None:
        idx = _MODULE_ID_TO_STACK.get(module_id, _STACK_DASHBOARD)
        if idx == _STACK_DASHBOARD:
            self._show_dashboard()
            return
        current_idx = self._central_stack.currentIndex()
        if current_idx != idx and current_idx != _STACK_DASHBOARD:
            self._dismiss_all_embedded_previews()
        self._central_stack.setCurrentIndex(idx)
        self._sync_pipeline_nav_highlights(module_id)
        if isinstance(self._project_data, dict):
            self._project_data.setdefault('ui', {})['last_module'] = module_id
        wr = str(self._workspace_root) if self._workspace_root else ''
        shell = self._module_shells.get(module_id)
        if shell is not None:
            shell.set_workspace_line(wr)
        self._apply_workspace_hints_for_module(module_id)

    def _update_all_shell_workspace_lines(self) -> None:
        wr = str(self._workspace_root) if self._workspace_root else ''
        for shell in self._module_shells.values():
            shell.set_workspace_line(wr)

    def _open_workspace_dialog(self) -> None:
        start = str(self._workspace_root) if self._workspace_root else get_default_browse_dir()
        picked = QFileDialog.getExistingDirectory(self, 'Open workspace folder', start)
        if not picked:
            return
        self._set_workspace_root(Path(picked))

    def _open_workspace_from_drop(self, dropped_path: str) -> None:
        """Open workspace from a dropped folder, or the parent folder of a dropped file."""
        try:
            p = Path(str(dropped_path)).expanduser()
            if not p.exists():
                return
            root = p if p.is_dir() else p.parent
            self._set_workspace_root(root)
        except Exception:
            log_exception(get_logger(), 'Open workspace from drag-and-drop failed')

    def _on_file_conversion_finished(self, ok: bool) -> None:
        """Use the generated Tif_files folder as the next Segmentation input."""
        if not ok:
            return
        try:
            out_dir = getattr(self.nd2_tab, '_last_conversion_output_dir', None)
            if out_dir is None:
                from bactomate.gui.widgets import tif_files_output_dir
                raw = (self.nd2_tab.file_path.text() or '').strip()
                if not raw:
                    return
                out_dir = tif_files_output_dir(raw)
            out_path = Path(out_dir).resolve()
            if not out_path.exists():
                return
            self.segmentation_tab._sync_segmentation_input_path(str(out_path))
            self.segmentation_tab.refresh_segmentation_candidate_list()
            self.log_output(f'Segmentation input set to converted TIFF folder: {out_path}\n')
        except Exception:
            log_exception(get_logger(), 'Set segmentation input from conversion output')

    def _clear_line_edit(self, widget: Any) -> None:
        if widget is None or not hasattr(widget, 'clear'):
            return
        try:
            widget.blockSignals(True)
            widget.clear()
        finally:
            widget.blockSignals(False)

    def _dismiss_all_embedded_previews(self) -> None:
        """Close previews embedded in module-shell right panes."""
        try:
            if hasattr(self.segmentation_tab, '_dismiss_embedded_previews'):
                self.segmentation_tab._dismiss_embedded_previews()
        except Exception:
            log_exception(get_logger(), 'Dismiss segmentation embedded previews')
        try:
            prev = getattr(self.filtering_tab, '_filter_preview_dialog', None)
            if prev is not None:
                prev.close()
                prev.deleteLater()
            self.filtering_tab._filter_preview_dialog = None
        except Exception:
            log_exception(get_logger(), 'Dismiss filtering embedded preview')
        try:
            emb = getattr(self.nd2_tab, '_embed_fc_preview_widget', None)
            if emb is not None:
                emb.close()
                emb.deleteLater()
            self.nd2_tab._embed_fc_preview_widget = None
        except Exception:
            log_exception(get_logger(), 'Dismiss file-conversion embedded preview')
        try:
            if hasattr(self.post_processing_tab, '_close_embedded_post_preview'):
                self.post_processing_tab._close_embedded_post_preview()
        except Exception:
            log_exception(get_logger(), 'Dismiss post-processing embedded preview')
        try:
            if hasattr(self.tracking_tab, '_dismiss_embedded_tracking_preview'):
                self.tracking_tab._dismiss_embedded_tracking_preview()
        except Exception:
            log_exception(get_logger(), 'Dismiss tracking embedded preview')
        for shell in self._module_shells.values():
            try:
                surf = shell.preview_surface()
                surf.set_preview_context('file_conversion')
                surf.clear_mount()
            except Exception:
                pass

    def _clear_all_module_paths(self) -> None:
        """Clear path fields in every module (does not delete files on disk)."""
        self._clear_line_edit(getattr(self.nd2_tab, 'file_path', None))
        if hasattr(self.nd2_tab, 'file_info'):
            self.nd2_tab.file_info = None
        for tab in getattr(self.segmentation_tab, '_segmentation_path_tabs', lambda: ())():
            if tab is not None:
                self._clear_line_edit(getattr(tab, 'file_path', None))
        self._clear_line_edit(getattr(self.filtering_tab, 'file_path', None))
        self._clear_line_edit(getattr(self.analyze_tab, 'file_path', None))
        self._clear_line_edit(getattr(self.tracking_tab, 'file_path', None))
        pp = self.post_processing_tab
        for attr in ('video_path', 'excel_path', 'montage_path'):
            self._clear_line_edit(getattr(pp, attr, None))
        self._clear_line_edit(getattr(self.training_tab, 'train_dir_edit', None))
        self.shared_metadata = {'pixel_size': None, 'time_per_frame_min': None}
        try:
            self.nd2_tab.refresh_conversion_candidate_list()
        except Exception:
            pass
        try:
            self.segmentation_tab.refresh_segmentation_candidate_list()
        except Exception:
            pass
        try:
            self.segmentation_tab.refresh_filtering_candidate_list()
        except Exception:
            pass

    def _clear_workspace(self) -> None:
        """Reset to a fresh session: no workspace, empty paths, closed embedded previews."""
        self._dismiss_all_embedded_previews()
        self._clear_all_module_paths()
        self._workspace_root = None
        self._project_data = {}
        self._workspace_scan = None
        self.dashboard_widget.set_workspace_display(None)
        self._update_all_shell_workspace_lines()
        try:
            self.set_status_message('')
        except Exception:
            pass
        try:
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
        except Exception:
            pass
        self._show_dashboard()
        self.log_output('Workspace cleared — all module paths and previews reset. Open a workspace or set paths manually to start again.\n')

    def _set_workspace_root(self, root: Path) -> None:
        self._workspace_root = root.resolve()
        loaded = load_project(self._workspace_root)
        if loaded is None:
            loaded = default_project_dict(display_name=self._workspace_root.name)
            save_project(self._workspace_root, loaded)
        self._project_data = loaded
        try:
            self._workspace_scan = scan_workspace(self._workspace_root)
            self.log_output(f"Workspace scan: segmentation hint={self._workspace_scan.hints.get('segmentation', '—')}, tracking hint={self._workspace_scan.hints.get('tracking', '—')}\n")
        except Exception:
            self._workspace_scan = None
            log_exception(get_logger(), 'Workspace folder scan failed')
        self._apply_project_paths_to_tabs()
        self._restore_module_splitters()
        self.dashboard_widget.set_workspace_display(self._workspace_root)
        self._update_all_shell_workspace_lines()
        ui = self._project_data.get('ui') or {}
        last = ui.get('last_module', MODULE_DASHBOARD)
        if last in _MODULE_ID_TO_STACK and last != MODULE_DASHBOARD:
            self._go_module_from_dashboard(str(last))
        else:
            self._show_dashboard()
        self.log_output(f'Workspace opened: {self._workspace_root}\n')
        QTimer.singleShot(0, lambda : getattr(self.nd2_tab, 'refresh_conversion_candidate_list', lambda : None)())

    def _apply_project_paths_to_tabs(self) -> None:
        if not self._workspace_root or not isinstance(self._project_data, dict):
            return
        self._project_data = migrate_project(self._project_data)
        paths = self._project_data.get('paths') or {}
        raw = paths.get('raw_input')
        if raw and hasattr(self.nd2_tab, 'file_path'):
            p = resolve_maybe_relative(self._workspace_root, str(raw))
            if p.exists():
                current = (self.nd2_tab.file_path.text() or '').strip()
                if not current or not self._path_is_within_workspace(current):
                    self.nd2_tab.file_path.setText(str(p))
                    if p.is_file() and hasattr(self.nd2_tab, 'load_file_info'):
                        self.nd2_tab.load_file_info(str(p))
                    elif p.is_dir() and hasattr(self.nd2_tab, 'load_folder_info'):
                        supported = list_convertible_microscopy_files(p, max_files=120)
                        if supported:
                            self.nd2_tab.load_folder_info(str(p), supported)
        seg_paths = paths.get('segmentation') or {}
        if isinstance(seg_paths, dict):
            backend = str(seg_paths.get('backend') or '').strip()
            if backend and hasattr(self.segmentation_tab, 'set_session_params'):
                try:
                    self.segmentation_tab.set_session_params({'segmentation_backend': backend})
                except Exception:
                    pass
            seg_input = str(seg_paths.get('input') or '').strip()
            if seg_input:
                p = resolve_maybe_relative(self._workspace_root, seg_input)
                if p.exists() and hasattr(self.segmentation_tab, '_sync_segmentation_input_path'):
                    self.segmentation_tab._sync_segmentation_input_path(str(p))
        for (section, tab, attr) in (('filtering', self.filtering_tab, 'file_path'), ('analyze', self.analyze_tab, 'file_path')):
            block = paths.get(section) or {}
            if isinstance(block, dict):
                stored = str(block.get('masks') or '').strip()
                if stored:
                    p = resolve_maybe_relative(self._workspace_root, stored)
                    if p.exists():
                        widget = getattr(tab, attr, None)
                        if widget is not None and hasattr(widget, 'setText'):
                            widget.setText(str(p))
        track_paths = paths.get('tracking') or {}
        if isinstance(track_paths, dict) and hasattr(self.tracking_tab, 'file_path'):
            for (field, attr) in (('masks', 'file_path'), ('source', 'single_fi_path')):
                stored = str(track_paths.get(field) or '').strip()
                if not stored:
                    continue
                p = resolve_maybe_relative(self._workspace_root, stored)
                if p.exists():
                    widget = getattr(self.tracking_tab, attr, None)
                    if widget is not None and hasattr(widget, 'setText'):
                        widget.setText(str(p))
        pp_paths = paths.get('post_processing') or {}
        if isinstance(pp_paths, dict):
            pp = self.post_processing_tab
            for (field, attr) in (('video', 'video_path'), ('montage', 'montage_path'), ('excel', 'excel_path')):
                stored = str(pp_paths.get(field) or '').strip()
                if stored:
                    p = resolve_maybe_relative(self._workspace_root, stored)
                    widget = getattr(pp, attr, None)
                    if widget is not None and hasattr(widget, 'setText'):
                        widget.setText(str(p) if p.exists() else stored)
        train_paths = paths.get('training') or {}
        if isinstance(train_paths, dict):
            stored = str(train_paths.get('dataset_dir') or '').strip()
            if stored and hasattr(self.training_tab, 'train_dir_edit'):
                p = resolve_maybe_relative(self._workspace_root, stored)
                if p.exists():
                    self.training_tab.train_dir_edit.setText(str(p))

    def _restore_module_splitters(self) -> None:
        if not isinstance(self._project_data, dict):
            return
        ui = self._project_data.get('ui') or {}
        states = ui.get('splitter_states') or {}
        if not isinstance(states, dict):
            return
        for (key, b64) in states.items():
            sh = self._module_shells.get(str(key))
            if sh is None or not isinstance(b64, str):
                continue
            try:
                ba = QByteArray.fromBase64(b64.encode('ascii'))
                sh.restore_splitter_state(ba)
            except Exception:
                pass

    def _save_project_workspace(self) -> None:
        if not self._workspace_root:
            show_message_dialog(self, 'Save project', 'Open a workspace folder first (Dashboard → Open workspace, or File → Open workspace).', kind='info')
            return
        if not isinstance(self._project_data, dict):
            self._project_data = default_project_dict(display_name=self._workspace_root.name)
        paths = self._project_data.setdefault('paths', default_project_dict()['paths'])
        ui = self._project_data.setdefault('ui', {})
        fp = (self.nd2_tab.file_path.text() or '').strip() if hasattr(self.nd2_tab, 'file_path') else ''
        if fp and self._workspace_root:
            try:
                paths['raw_input'] = path_relative_to_workspace(self._workspace_root, Path(fp))
            except Exception:
                paths['raw_input'] = fp
        if hasattr(self.segmentation_tab, '_active_backend_key'):
            seg_block = paths.setdefault('segmentation', {})
            if isinstance(seg_block, dict):
                seg_block['backend'] = self.segmentation_tab._active_backend_key()
                seg_input = (self.segmentation_tab.file_path.text() or '').strip()
                if seg_input and self._workspace_root:
                    store_module_path(paths, self._workspace_root, 'segmentation', 'input', seg_input)
        if self._workspace_root:
            store_module_path(paths, self._workspace_root, 'filtering', 'masks', (self.filtering_tab.file_path.text() or '').strip())
            store_module_path(paths, self._workspace_root, 'analyze', 'masks', (self.analyze_tab.file_path.text() or '').strip())
            store_module_path(paths, self._workspace_root, 'tracking', 'masks', (self.tracking_tab.file_path.text() or '').strip())
            store_module_path(paths, self._workspace_root, 'tracking', 'source', (getattr(self.tracking_tab, 'single_fi_path').text() or '').strip() if hasattr(self.tracking_tab, 'single_fi_path') else '')
            pp = self.post_processing_tab
            store_module_path(paths, self._workspace_root, 'post_processing', 'video', (pp.video_path.text() or '').strip())
            store_module_path(paths, self._workspace_root, 'post_processing', 'montage', (pp.montage_path.text() or '').strip())
            store_module_path(paths, self._workspace_root, 'post_processing', 'excel', (pp.excel_path.text() or '').strip())
            store_module_path(paths, self._workspace_root, 'training', 'dataset_dir', (self.training_tab.train_dir_edit.text() or '').strip())
        ui['last_module'] = self._get_current_module_id()
        states: dict[str, str] = {}
        for (key, shell) in self._module_shells.items():
            states[key] = base64.b64encode(shell.splitter_state()).decode('ascii')
        ui['splitter_states'] = states
        save_project(self._workspace_root, self._project_data)
        self.log_output(f"Project saved: {self._workspace_root / 'bactomate_project.json'}\n")

    def run_workflow_from_session(self, log_data: dict[str, Any]) -> None:
        """Apply session params from log (FI, z-stack, foci, model, filters, etc.); run full pipeline (conversion → segment → analyze → tracking) on the current TIFF Conversion path only. That path is the new file/folder; downstream steps use outputs from the previous step (Tif_files → _segmentation_<model> → analyze → tracking). So you can run the same analysis type from a session on a different file or folder."""
        tabs_data = log_data.get('tabs', log_data)
        if not isinstance(tabs_data, dict):
            show_message_dialog(self, 'Re-run analysis', 'Session log has no tab data.', kind='warning')
            return
        tabs_data = normalize_session_tabs(tabs_data)
        nd2_tab = getattr(self, TAB_WIDGET_ATTR.get('nd2'), None)
        if nd2_tab is None or not hasattr(nd2_tab, 'file_path'):
            show_message_dialog(self, 'Re-run analysis', 'File conversion module not available.', kind='warning')
            return
        seed_path = (nd2_tab.file_path.text() or '').strip()
        if not seed_path or not Path(seed_path).exists():
            show_message_dialog(self, 'Re-run analysis', 'Set the path to your new file or folder in File conversion. Re-run will apply the session parameters and run conversion → segment → analyze → tracking on that path (outputs go next to the input).', kind='info')
            return
        self.apply_session_params(tabs_data)
        seg_tab = getattr(self, 'segmentation_tab', None)
        unet_data = tabs_data.get('unet_segment', {})
        seg_data = tabs_data.get('segmentation', {})
        if seg_tab is not None and isinstance(unet_data, dict) and unet_data.get('performed') and isinstance(seg_data, dict) and (not seg_data.get('performed', False)) and (not seg_data.get('segmentation_backend')):
            try:
                seg_tab.set_session_params({'segmentation_backend': 'unet', **unet_data})
            except Exception:
                pass
        nd2_tab.file_path.setText(seed_path)
        if Path(seed_path).is_file():
            nd2_tab.load_file_info(seed_path)
        else:
            folder = Path(seed_path)
            supported = list_convertible_microscopy_files(folder, max_files=120)
            if supported and hasattr(nd2_tab, 'load_folder_info'):
                nd2_tab.load_folder_info(seed_path, supported)
        nd2_data = tabs_data.get('nd2', {})
        if isinstance(nd2_data, dict) and hasattr(nd2_tab, 'apply_channel_options_from_session'):
            nd2_tab.apply_channel_options_from_session(nd2_data)
        order = ('nd2', 'segmentation', 'analyze_masks', 'tracking')
        steps = []
        for key in order:
            if key not in TAB_WIDGET_ATTR:
                continue
            tab_data = tabs_data.get(key, {})
            if not isinstance(tab_data, dict):
                tab_data = {}
            performed = tab_data.get('performed', True)
            if key == 'segmentation' and (not performed):
                legacy_unet = tabs_data.get('unet_segment', {})
                if isinstance(legacy_unet, dict) and legacy_unet.get('performed'):
                    performed = True
            if not performed:
                continue
            tab = getattr(self, TAB_WIDGET_ATTR.get(key), None)
            if tab is None or not hasattr(tab, 'workflow_step_finished'):
                continue
            if key == 'nd2':
                run_fn = getattr(tab, 'convert_nd2', None)
            elif key == 'segmentation':
                run_fn = getattr(tab, 'process_file_or_folder', None)
            elif key == 'analyze_masks':
                run_fn = getattr(tab, 'analyze_masks', None)
            elif key == 'tracking':
                run_fn = getattr(tab, 'run_tracking', None)
            else:
                run_fn = None
            if callable(run_fn):
                steps.append((key, tab, run_fn))
        if not steps:
            show_message_dialog(self, 'Re-run analysis', 'No steps were recorded as performed in this session log. Run at least one step (conversion, segment, analyze, or tracking) before saving a session to use Re-run.', kind='info')
            return
        self._workflow_replay_steps = steps
        self._workflow_replay_index = 0
        self._workflow_replay_connection = None
        self._workflow_replay_seed_path = seed_path

        def _conversion_output_path(seed: str) -> Path:
            p = Path(seed)
            base = p.parent if p.is_file() else p
            return base / 'Tif_files'

        def _segment_output_path(segment_input_path: str) -> Path:
            seg_tab = getattr(self, 'segmentation_tab', None)
            if seg_tab is not None and hasattr(seg_tab, 'infer_segment_output_path'):
                return seg_tab.infer_segment_output_path(segment_input_path)
            folder = Path(segment_input_path)
            omnipose_tab = getattr(getattr(self, 'segmentation_tab', None), 'omnipose_tab', None)
            folder_name = omnipose_tab.get_segmentation_output_folder_name() if omnipose_tab and hasattr(omnipose_tab, 'get_segmentation_output_folder_name') else '_segmentation_omni'
            return folder / folder_name

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
                show_message_dialog(self, 'Workflow stopped', 'A step in the workflow failed. See the log for details.', kind='warning')
                return
            self._workflow_replay_index += 1
            if self._workflow_replay_index >= len(self._workflow_replay_steps):
                show_message_dialog(self, 'Re-run analysis', 'Workflow completed successfully.', kind='info')
                return
            seed = getattr(self, '_workflow_replay_seed_path', '')
            (next_key, next_tab, next_run_fn) = self._workflow_replay_steps[self._workflow_replay_index]
            if next_key == 'segmentation':
                next_path = _conversion_output_path(seed)
                if next_path.exists():
                    next_tab.file_path.setText(str(next_path))
                else:
                    show_message_dialog(self, 'Workflow stopped', f'Conversion output not found: {next_path}', kind='warning')
                    return
            elif next_key == 'analyze_masks':
                seg_input = _conversion_output_path(seed)
                next_path = _segment_output_path(str(seg_input))
                if next_path.exists():
                    next_tab.file_path.setText(str(next_path))
                else:
                    show_message_dialog(self, 'Workflow stopped', f'Segment output not found: {next_path}', kind='warning')
                    return
            elif next_key == 'tracking':
                seg_input = _conversion_output_path(seed)
                next_path = _segment_output_path(str(seg_input))
                if next_path.exists():
                    next_tab.file_path.setText(str(next_path))
                else:
                    show_message_dialog(self, 'Workflow stopped', f'Segment output not found: {next_path}', kind='warning')
                    return
            _run_next()

        def _run_next() -> None:
            (key, tab, run_fn) = self._workflow_replay_steps[self._workflow_replay_index]
            self._workflow_replay_connection = on_step_done
            tab.workflow_step_finished.connect(on_step_done)
            run_fn()
        (first_key, first_tab, _) = self._workflow_replay_steps[0]
        if first_key == 'segmentation':
            p = _conversion_output_path(seed_path)
            if p.exists():
                first_tab.file_path.setText(str(p))
            else:
                show_message_dialog(self, 'Workflow stopped', f'Segment needs conversion output at {p}. Run conversion first or set path to an existing Tif_files folder.', kind='warning')
                return
        elif first_key == 'analyze_masks':
            seg_out = _segment_output_path(str(_conversion_output_path(seed_path)))
            if seg_out.exists():
                first_tab.file_path.setText(str(seg_out))
            else:
                show_message_dialog(self, 'Workflow stopped', f'Analyze needs segment output at {seg_out}. Run conversion and segment first.', kind='warning')
                return
        elif first_key == 'tracking':
            seg_out = _segment_output_path(str(_conversion_output_path(seed_path)))
            if seg_out.exists():
                first_tab.file_path.setText(str(seg_out))
            else:
                show_message_dialog(self, 'Workflow stopped', f'Tracking needs segment output at {seg_out}. Run conversion and segment first.', kind='warning')
                return
        _run_next()

    def _add_menu_bar(self) -> None:
        """Minimal menu bar: app menu (About, Minimize, Quit), File, Help. On macOS the first menu is the app menu and shows the app name once."""
        menubar = self.menuBar()
        app_menu = menubar.addMenu('' if sys.platform == 'darwin' else 'BactoMate')
        about_action = QAction('About BactoMate', self)
        about_action.triggered.connect(self._show_about)
        app_menu.addAction(about_action)
        app_menu.addSeparator()
        minimize_action = QAction('Minimize', self)
        minimize_action.setShortcut(QKeySequence('Ctrl+M'))

        def minimize_active_or_self():
            active = QApplication.activeWindow()
            if active and active is not self and active.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint:
                active.showMinimized()
            else:
                self.showMinimized()
        minimize_action.triggered.connect(minimize_active_or_self)
        app_menu.addAction(minimize_action)
        quit_action = QAction('Quit BactoMate', self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        app_menu.addAction(quit_action)
        file_menu = menubar.addMenu('File')
        open_ws = QAction('Open workspace…', self)
        open_ws.triggered.connect(self._open_workspace_dialog)
        open_ws.setToolTip('Choose a folder; create or load bactomate_project.json')
        file_menu.addAction(open_ws)
        clear_ws = QAction('Clear workspace', self)
        clear_ws.triggered.connect(self._clear_workspace)
        clear_ws.setToolTip('Reset paths and previews without deleting files on disk (same as dashboard Clear workspace)')
        file_menu.addAction(clear_ws)
        save_proj = QAction('Save project', self)
        save_proj.triggered.connect(self._save_project_workspace)
        save_proj.setToolTip('Write bactomate_project.json in the current workspace folder')
        file_menu.addAction(save_proj)
        file_menu.addSeparator()
        file_quit = QAction('Quit', self)
        file_quit.setShortcut(QKeySequence.StandardKey.Quit)
        file_quit.triggered.connect(self.close)
        file_menu.addAction(file_quit)
        help_menu = menubar.addMenu('Help')
        user_manual_action = QAction('User manual', self)
        user_manual_action.triggered.connect(self._open_user_manual)
        help_menu.addAction(user_manual_action)
        if sys.platform == 'darwin' and _running_from_app_bundle():
            pin_dock_action = QAction('Pin BactoMate to Dock (for Reopen)', self)
            pin_dock_action.triggered.connect(self._show_pin_to_dock_help)
            pin_dock_action.setToolTip('How to get Reopen from the Dock to work after quitting.')
            help_menu.addAction(pin_dock_action)
        check_env_action = QAction('Check environment', self)
        check_env_action.triggered.connect(self._check_environment)
        check_env_action.setToolTip('Verify required dependencies (numpy, PyQt6, tifffile, etc.) are installed.')
        help_menu.addAction(check_env_action)

    def _show_pin_to_dock_help(self) -> None:
        """Explain how to pin BactoMate so Reopen from the Dock works (macOS .app only)."""
        msg = 'To reopen BactoMate from the Dock after quitting:\n\n1. Remove any existing BactoMate icon from the Dock (drag it off).\n\n2. In Finder, go to where BactoMate.app is (e.g. Desktop or Applications).\n\n3. Drag BactoMate.app onto the Dock.\n\n4. Use that Dock icon to launch and to reopen after quit.\n\nDo not use "Keep in Dock" from the running app’s Dock menu — that pins the wrong target and Reopen will not work.'
        try:
            app_root = get_app_root()
            app_bundle = app_root.parent.parent if 'Contents' in str(app_root) else app_root.parent
            if app_bundle.suffix == '.app' and app_bundle.exists():
                if show_confirm_dialog(self, title='Pin BactoMate to Dock', message=msg + '\n\nOpen the folder containing BactoMate.app in Finder?', confirm_label='Yes', cancel_label='No', default_confirm=True):
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(app_bundle.parent)))
            else:
                show_message_dialog(self, 'Pin BactoMate to Dock', msg, kind='info')
        except Exception:
            show_message_dialog(self, 'Pin BactoMate to Dock', msg, kind='info')

    def _show_about(self) -> None:
        try:
            from bactomate import __version__
            version = __version__
        except Exception:
            version = '2.8.3'
        show_message_dialog(self, 'About BactoMate', f'BactoMate – bacteria time-lapse batch processing and analysis.\n\nVersion {version}\n\nDocumentation: Help → User manual.', kind='info')

    def _check_environment(self) -> None:
        """Run environment check and show result in a dialog and in the log."""
        try:
            from bactomate.scripts.check_environment import check_environment
            report = check_environment()
            self.log_output(report + '\n\n')
            show_message_dialog(self, 'Check environment', report, kind='info')
        except Exception as e:
            self.log_output(f'Check environment failed: {e}\n')
            show_message_dialog(self, 'Check environment', f'Could not run check:\n{e}', kind='warning')

    def _open_user_manual(self) -> None:
        """Open the user manual (docs/BACTOMATE_USER_MANUAL.md) with the system default app."""
        app_root = get_app_root()
        manual_path = app_root / 'docs' / 'BACTOMATE_USER_MANUAL.md'
        if manual_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(manual_path)))
        else:
            docs_dir = app_root / 'docs'
            if docs_dir.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(docs_dir)))
            else:
                show_message_dialog(self, 'User manual', f'Manual not found at:\n{manual_path}\n\nOpen the docs/ folder from the app or installation.', kind='info')

    def _center_window(self):
        frame_geometry = self.frameGeometry()
        cursor_pos = QCursor.pos()
        screen = QApplication.screenAt(cursor_pos)
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen:
            center_point = screen.availableGeometry().center()
            frame_geometry.moveCenter(center_point)
            self.move(frame_geometry.topLeft())

    @staticmethod
    def _format_status_line(line: str) -> str:
        """Strip log prefixes (timestamp, [INFO]) and condense tqdm progress to one short line for the status window."""
        if not line or not line.strip():
            return line
        s = line.strip()
        s = re.sub('^\\d{4}-\\d{2}-\\d{2}\\s+\\d{2}:\\d{2}:\\d{2},\\d{3}\\s+\\[(?:INFO|DEBUG|WARNING|ERROR)\\]\\s*', '', s)
        s = re.sub('^\\[(?:INFO|DEBUG|WARNING|ERROR)\\]\\s*', '', s)
        s = re.sub('^[\\w.]+:(?:INFO|DEBUG|WARNING|ERROR):\\s*', '', s)
        s = s.strip()
        s = re.sub('^line\\s+\\d+\\s*[\\t ]*', '', s)
        s = s.strip()
        tqdm_match = re.match('^(\\d+)%\\s*\\|[#\\s]+\\|\\s*(\\d+)/(\\d+)\\s*\\[([^\\]]*)\\]', s)
        if tqdm_match:
            (pct, current, total, rest) = tqdm_match.groups()
            eta_match = re.search('<(\\d+):(\\d+)', rest)
            if eta_match:
                (m, sec) = (eta_match.group(1), eta_match.group(2))
                return f'  {pct}% ({current}/{total})  ETA {int(m)}:{sec}'
            return f'  {pct}% ({current}/{total})'
        simple_match = re.match('^(\\d+)%\\s*\\(\\s*(\\d+)/(\\d+)\\s*\\)\\s*(?:ETA\\s*(\\d+):(\\d+))?', s)
        if simple_match:
            (pct, current, total) = (simple_match.group(1), simple_match.group(2), simple_match.group(3))
            (eta_m, eta_s) = (simple_match.group(4), simple_match.group(5))
            if eta_m is not None and eta_s is not None:
                return f'  {pct}% ({current}/{total})  ETA {int(eta_m)}:{eta_s}'
            return f'  {pct}% ({current}/{total})'
        return s if s else ''

    def set_status_message(self, text: str) -> None:
        """Set the one-line status message above the log (e.g. 'Omnipose: file 3 of 11 — image.tif'). Empty string hides it."""
        if not hasattr(self, 'status_message'):
            return
        self.status_message.setText((text or '').strip())
        self.status_message.setVisible(bool((text or '').strip()))

    def log_output(self, message: str) -> None:

        def _persist_lines_to_gui_log(raw_message: str) -> None:
            try:
                for line in raw_message.splitlines():
                    if line.strip():
                        write_gui_log_line('GUI', line)
            except Exception:
                pass
        _persist_lines_to_gui_log(message)
        lines = message.split('\n')
        formatted = [self._format_status_line(line) for line in lines]
        formatted = [f for f in formatted if f]
        text = '\n'.join(formatted)
        if not text:
            return
        if not text.endswith('\n') and message.endswith('\n'):
            text += '\n'
        scrollbar = self.log_text.verticalScrollBar()
        at_bottom = scrollbar.maximum() - scrollbar.value() <= 20
        self.log_text.append(text)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        message_lower = message.lower()
        if any((done in message_lower for done in ['merged ', 'removed ', 'analysis complete', 'completed successfully', 'task(s) succeeded', 'cancelled', 'combined summary saved'])):
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.set_status_message('')
        elif any((keyword in message_lower for keyword in ['applying drift correction', 'applying bleach correction', 'processing frame', 'segmenting', 'analyzing masks', 'reading frames', 'extracting frames', 'processing file', 'starting training', 'epoch ', 'u-net', 'running u-net', 'starting tracking', 'tracking completed', 'starting nd2 conversion', 'converted:', 'starting mask analysis', 'starting batch', 'create growth curve'])):
            if not self.progress_bar.isVisible():
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)

    def update_all_tabs_with_metadata(self):
        pixel_size = self.shared_metadata.get('pixel_size')
        time_per_frame_min = self.shared_metadata.get('time_per_frame_min')
        if pixel_size is None and time_per_frame_min is None:
            return
        if pixel_size is not None:
            omnipose_tab = getattr(getattr(self, 'segmentation_tab', None), 'omnipose_tab', None)
            if omnipose_tab is not None and hasattr(omnipose_tab, 'pixel_size_um'):
                omnipose_tab.pixel_size_um.setValue(pixel_size)
                self.log_output(f'✓ Updated pixel size in Omnipose Segment tab: {pixel_size:.4f} µm\n')
            for (tab_name, tab) in (('Analyze', getattr(self, 'analyze_tab', None)), ('Filtering', getattr(self, 'filtering_tab', None))):
                if tab is None or not hasattr(tab, 'pixel_size'):
                    continue
                if hasattr(tab, '_apply_pixel_size_from_calibration'):
                    tab._apply_pixel_size_from_calibration(float(pixel_size), indicator='(from metadata)')
                else:
                    tab.pixel_size.setValue(pixel_size)
                    if hasattr(tab, 'pixel_size_metadata_loaded'):
                        tab.pixel_size_metadata_loaded = True
                self.log_output(f'✓ Updated pixel size in {tab_name} tab: {pixel_size:.4f} µm\n')
            if hasattr(self, 'tracking_tab') and hasattr(self.tracking_tab, 'filter_pixel_size'):
                self.tracking_tab.filter_pixel_size.setValue(pixel_size)
                self.log_output(f'✓ Updated pixel size in Tracking tab: {pixel_size:.4f} µm\n')
        if time_per_frame_min is not None and time_per_frame_min > 0:
            if hasattr(self, 'tracking_tab') and hasattr(self.tracking_tab, 'filter_time_per_frame'):
                self.tracking_tab.filter_time_per_frame.setValue(time_per_frame_min)
                if hasattr(self.tracking_tab, 'filter_time_per_frame_unit'):
                    self.tracking_tab.filter_time_per_frame_unit.setCurrentText('min')
                self.log_output(f'✓ Updated time per frame in Tracking tab: {time_per_frame_min:.4f} min\n')

def main() -> None:
    log = get_logger()
    t0 = perf_counter()
    t_last = t0

    def _mark(label: str) -> None:
        nonlocal t_last
        try:
            now = perf_counter()
            dt_total_ms = (now - t0) * 1000.0
            dt_step_ms = (now - t_last) * 1000.0
            t_last = now
            log.info('[startup] %s step_ms=%.1f total_ms=%.1f', label, dt_step_ms, dt_total_ms)
        except Exception:
            pass
    log.info('Creating application and main window...')
    _mark('main_window_main_enter')
    if sys.platform == 'darwin':
        plugins_dir = os.environ.get('QT_PLUGIN_PATH')
        if plugins_dir:
            abs_plugins_dir = os.path.abspath(plugins_dir)
            if abs_plugins_dir.endswith('platforms'):
                abs_plugins_dir = os.path.dirname(abs_plugins_dir)
            os.environ['QT_PLUGIN_PATH'] = abs_plugins_dir
            QCoreApplication.setLibraryPaths([abs_plugins_dir])
    original_excepthook = sys.excepthook

    def pyqt6_exception_handler(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        try:
            import threading
            thread_name = getattr(threading.current_thread(), 'name', 'unknown')
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            block = ''.join(lines)
            write_crash_log(f'\n--- Uncaught exception {datetime.now().isoformat()} ---\nThread: {thread_name}\n(If this occurs when closing the Filter Effects Preview or during Analyze Masks, a background thread may still have been running.)\n{block}')
        except Exception:
            pass
    sys.excepthook = pyqt6_exception_handler
    try:
        _mark('before_QApplication')
        app = QApplication(sys.argv)
        _mark('after_QApplication')
        app.setApplicationName('BactoMate')
        app.setApplicationDisplayName('BactoMate')
        single_instance_server = None
        window_ref: list[QMainWindow | None] = [None]
        if sys.platform == 'darwin':
            server_name = 'de.hu-berlin.bactomate.single-instance'
            if _notify_existing_macos_instance(server_name):
                log.info('Existing BactoMate instance activated; exiting duplicate launch.')
                sys.exit(0)
            single_instance_server = _start_macos_single_instance_server(server_name, lambda : window_ref[0])
            if single_instance_server is not None:
                app._bactomate_single_instance_server = single_instance_server
        if sys.platform == 'darwin':
            try:
                import setproctitle
                setproctitle.setproctitle('BactoMate')
            except ImportError:
                pass
            _set_macos_process_name('BactoMate')
        icon_path = _get_app_icon_path()
        if icon_path is not None:
            app.setWindowIcon(QIcon(str(icon_path)))
            if sys.platform == 'darwin':
                _set_macos_dock_icon(icon_path)
        app.setStyle('Fusion')
        _mark('before_stylesheet')
        app.setStyleSheet(get_modern_stylesheet())
        _mark('after_stylesheet')
        try:
            from PyQt6.QtCore import qInstallMessageHandler, QtMsgType

            def _qt_message_handler(msg_type, context, message):
                try:
                    level = 'unknown'
                    if msg_type == QtMsgType.QtDebugMsg:
                        level = 'QtDebug'
                    elif msg_type == QtMsgType.QtInfoMsg:
                        level = 'QtInfo'
                    elif msg_type == QtMsgType.QtWarningMsg:
                        level = 'QtWarning'
                    elif msg_type == QtMsgType.QtCriticalMsg:
                        level = 'QtCritical'
                    elif msg_type == QtMsgType.QtFatalMsg:
                        level = 'QtFatal'
                    loc = ''
                    if context is not None and getattr(context, 'function', None):
                        loc = f" ({getattr(context, 'file', '')}:{getattr(context, 'line', '')} {context.function})"
                    write_gui_log_line(level, f'{message}{loc}')
                except Exception:
                    pass
            qInstallMessageHandler(_qt_message_handler)
        except Exception:
            pass
        if sys.platform == 'win32':
            try:
                from bactomate.gui.windows_patches import apply_after_ui
                apply_after_ui(app)
            except Exception:
                pass
        _mark('before_MainWindow')
        window = MainWindow()
        window_ref[0] = window
        _mark('after_MainWindow')
        if icon_path is not None:
            window.setWindowIcon(QIcon(str(icon_path)))
        _mark('before_window_show')
        window.show()
        _mark('after_window_show')
        if sys.platform == 'darwin':
            _activate_macos_window(window)
        else:
            window.raise_()
            window.activateWindow()
        log.info('BactoMate window shown.')
        sys.exit(app.exec())
    except Exception as e:
        log_exception(get_logger(), 'Failed to start GUI')
        print(f'Error starting GUI: {e}', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
