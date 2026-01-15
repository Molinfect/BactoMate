#!/usr/bin/env python3
"""
Omnipose Batch Processing GUI

A user-friendly graphical interface for batch processing TIFF files with Omnipose
and analyzing cell areas with morphological filtering.

Usage:
    python omnipose_batch_gui.py
"""

import sys
import os
from pathlib import Path
import traceback
import warnings

# Suppress deprecation warnings from omnipose/pkg_resources
# This is harmless - omnipose uses pkg_resources which is deprecated in newer setuptools
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')

# CRITICAL: Patch pbr.version.VersionInfo early to prevent versioning errors in frozen builds
if getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__'):
    try:
        import pbr.version
        # Check if already patched (by runtime hook) - if calling it works, it's patched
        try:
            test_result = pbr.version.VersionInfo('test')
            if not hasattr(test_result, 'release_string'):
                raise AttributeError("Not properly patched")
        except (Exception, SystemExit):
            # Not patched yet or patching failed, patch it now
            class FakeVersionInfo:
                def __init__(self, project_name):
                    self.project_name = project_name
                    default_versions = {'mgen': '1.2.1'}
                    self.version = default_versions.get(project_name, "0.0.0")
                def release_string(self): return self.version
                def version_string(self): return self.version
                def semantic_version(self):
                    class FakeSemVer:
                        def __init__(self, v): 
                            self.version_string = v
                            parts = v.split('.')
                            self.major = int(parts[0]) if len(parts) > 0 else 0
                            self.minor = int(parts[1]) if len(parts) > 1 else 0
                            self.patch = int(parts[2]) if len(parts) > 2 else 0
                        def release_string(self): return self.version_string
                        def version_tuple(self): return (self.major, self.minor, self.patch)
                    return FakeSemVer(self.version)
            pbr.version.VersionInfo = lambda name: FakeVersionInfo(name)
    except (ImportError, Exception):
        # pbr not available or patching failed, continue anyway
        pass

# CRITICAL: Patch omnipose.core.get_masks to handle bd=None in frozen builds
# This must be done early, before any omnipose imports, to catch all code paths
def _patch_omnipose_get_masks():
        """Patch omnipose.core.get_masks to handle None boundary field"""
        try:
            import omnipose.core as oc
            import numpy as np
            
            # Only patch if not already patched
            if hasattr(oc, 'get_masks') and not hasattr(oc.get_masks, '_bactimate_patched'):
                _orig_get_masks = oc.get_masks
                
                def get_masks_safe(*args, **kwargs):
                    """
                    Patched get_masks that handles bd=None in frozen builds.
                    The boundary field (bd) can be None, causing "None > -1" errors.
                    This patch provides a fallback array.
                    """
                    # bd is typically the second positional argument (index 1)
                    if len(args) >= 2:
                        bd = args[1]
                        if bd is None:
                            # Need to create a fallback array with the right shape
                            # Try to get shape from dist (3rd arg), mask (4th arg), or p (1st arg)
                            shape = None
                            
                            # Try dist field (dt_pad) - 3rd argument
                            if len(args) >= 3 and args[2] is not None and hasattr(args[2], 'shape'):
                                shape = args[2].shape
                            # Try mask field (iscell_pad) - 4th argument  
                            elif len(args) >= 4 and args[3] is not None and hasattr(args[3], 'shape'):
                                shape = args[3].shape
                            # Try p field - 1st argument
                            elif len(args) >= 1 and args[0] is not None and hasattr(args[0], 'shape'):
                                shape = args[0].shape
                            
                            # ALWAYS create a fallback if bd is None, even if shape detection fails
                            if shape is not None:
                                # Create fallback array with values < -1 so (bd > -1) is False everywhere
                                # This effectively disables boundary-based suppression
                                bd = np.full(shape, -2, dtype=np.float32)
                            else:
                                # Last resort: try to infer shape from any available array
                                # This should rarely happen, but better than crashing
                                for arg in args:
                                    if arg is not None and hasattr(arg, 'shape') and len(arg.shape) >= 2:
                                        shape = arg.shape
                                        bd = np.full(shape, -2, dtype=np.float32)
                                        break
                                
                                # If still no shape, create a minimal fallback (shouldn't happen in practice)
                                if bd is None:
                                    # This is a last resort - create a small array
                                    # The original function will likely fail anyway, but at least we tried
                                    bd = np.array([[-2.0]], dtype=np.float32)
                            
                            # Replace bd in args - CRITICAL: must do this even if shape detection failed
                            args = list(args)
                            args[1] = bd
                            args = tuple(args)
                    
                    # Final safety check: ensure bd is never None before calling original function
                    if len(args) >= 2 and args[1] is None:
                        # This should never happen, but if it does, create a minimal fallback
                        args = list(args)
                        args[1] = np.array([[-2.0]], dtype=np.float32)
                        args = tuple(args)
                    
                    return _orig_get_masks(*args, **kwargs)
                
                # Mark as patched to avoid recursion
                get_masks_safe._bactimate_patched = True
                get_masks_safe.__wrapped__ = _orig_get_masks
                oc.get_masks = get_masks_safe
        except (ImportError, AttributeError, Exception):
            # omnipose not available yet or patching failed - will retry later
            pass

# Try to patch immediately if in frozen build (Nuitka)
# Nuitka uses __compiled__ or sys.frozen
is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
if is_frozen:
    _patch_omnipose_get_masks()
    
    # Also register a hook to patch when omnipose is imported
    import importlib.util
    _original_import = __import__
    
    def _patched_import(name, *args, **kwargs):
        module = _original_import(name, *args, **kwargs)
        if name == 'omnipose.core' or (hasattr(module, '__name__') and module.__name__ == 'omnipose.core'):
            _patch_omnipose_get_masks()
        return module
    
    # Note: We can't easily monkeypatch __import__ without breaking things
    # Instead, we'll patch it when omnipose is actually used (in load_model)

# CRITICAL: Set Qt paths BEFORE any Qt imports (PyQt6 or matplotlib Qt backend)
# This prevents conflicts with system Qt (Homebrew/conda) installations
# Also handles Nuitka frozen executables

# Check if running as compiled executable (Nuitka)
if getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__'):
    # Running as compiled executable (Nuitka)
    # Nuitka: use directory containing the executable
    base_path = os.path.dirname(sys.executable)
else:
    # Running as script
    base_path = os.path.dirname(os.path.abspath(__file__))

# Initialize variables
abs_plugin_path = None
abs_lib_dir = None

if sys.platform in ['darwin', 'win32']:
    # For compiled bundles (Nuitka), plugins are in the bundle
    if getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__'):
        # In compiled bundle, plugins are at PyQt6/Qt6/plugins relative to base_path
        plugin_candidate = os.path.join(base_path, 'PyQt6', 'Qt6', 'plugins')
        if os.path.exists(plugin_candidate):
            abs_plugin_path = os.path.abspath(plugin_candidate)
            if sys.platform == 'darwin':
                lib_candidate = os.path.join(base_path, 'PyQt6', 'Qt6', 'lib')
                abs_lib_dir = os.path.abspath(lib_candidate) if os.path.exists(lib_candidate) else None
    else:
        # Running as script - use original logic
        script_dir = base_path
        
        # Detect Python version and venv directory dynamically
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        
        # Try to find PyQt6 in the current Python environment
        # First, try to find it relative to the script (for venv scenarios)
        venv_candidates = [
            os.path.join(script_dir, 'venv_py310'),  # Python 3.10+ venv
            os.path.join(script_dir, 'venv'),  # Original Python 3.9 venv
        ]
        
        # Also check if we're in a venv (sys.prefix will point to venv)
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            # We're in a virtual environment
            venv_base = sys.prefix
            venv_candidates.insert(0, venv_base)
        
        # Search for PyQt6 plugins in candidate locations
        for venv_dir in venv_candidates:
            if not os.path.exists(venv_dir):
                continue
            
            # Try the detected Python version first, then try common versions
            for py_ver in [python_version, 'python3.11', 'python3.10', 'python3.9']:
                candidate = os.path.join(venv_dir, 'lib', py_ver, 'site-packages', 'PyQt6', 'Qt6', 'plugins')
                if os.path.exists(candidate):
                    abs_plugin_path = os.path.abspath(candidate)
                    qt_lib_candidate = os.path.join(venv_dir, 'lib', py_ver, 'site-packages', 'PyQt6', 'Qt6', 'lib')
                    if os.path.exists(qt_lib_candidate):
                        abs_lib_dir = os.path.abspath(qt_lib_candidate)
                    break
            
            if abs_plugin_path:
                break
        
        # If not found in venv candidates, try to find PyQt6 in site-packages
        if not abs_plugin_path:
            try:
                import PyQt5
                pyqt5_path = os.path.dirname(PyQt5.__file__)
                candidate = os.path.join(pyqt5_path, 'Qt5', 'plugins')
                if os.path.exists(candidate):
                    abs_plugin_path = os.path.abspath(candidate)
                    qt_lib_candidate = os.path.join(pyqt5_path, 'Qt5', 'lib')
                    if os.path.exists(qt_lib_candidate):
                        abs_lib_dir = os.path.abspath(qt_lib_candidate)
            except ImportError:
                pass

# Set Qt paths if found (for both macOS and Windows)
if abs_plugin_path:
    if sys.platform == 'darwin':
        # Clear any conflicting Qt paths from system installations
        conflicting_paths = ['/opt/homebrew', '/usr/local', '/anaconda', '/miniconda']
        
        # Clean QT_PLUGIN_PATH - prioritize our venv path
        if 'QT_PLUGIN_PATH' in os.environ:
            current_val = os.environ['QT_PLUGIN_PATH']
            # If it contains system Qt paths, replace with our venv path
            if any(conflict in current_val for conflict in conflicting_paths):
                os.environ['QT_PLUGIN_PATH'] = abs_plugin_path
            elif abs_plugin_path not in current_val:
                # Prepend our path to ensure it's checked first
                os.environ['QT_PLUGIN_PATH'] = abs_plugin_path + ':' + current_val
        else:
            os.environ['QT_PLUGIN_PATH'] = abs_plugin_path
        
        # Set DYLD_FRAMEWORK_PATH - prioritize our venv Qt
        if abs_lib_dir and os.path.exists(abs_lib_dir):
            current_dyld = os.environ.get('DYLD_FRAMEWORK_PATH', '')
            # Remove any system Qt paths
            if current_dyld:
                paths = current_dyld.split(':')
                filtered_paths = [p for p in paths if not any(conflict in p for conflict in conflicting_paths)]
                if filtered_paths:
                    os.environ['DYLD_FRAMEWORK_PATH'] = abs_lib_dir + ':' + ':'.join(filtered_paths)
                else:
                    os.environ['DYLD_FRAMEWORK_PATH'] = abs_lib_dir
            else:
                os.environ['DYLD_FRAMEWORK_PATH'] = abs_lib_dir
        
        # Clear QT_QPA_PLATFORM_PLUGIN_PATH if it points to system Qt
        if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
            qpa_val = os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
            if any(conflict in qpa_val for conflict in conflicting_paths):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = abs_plugin_path
    elif sys.platform == 'win32':
        # Windows: Set QT_PLUGIN_PATH
        os.environ['QT_PLUGIN_PATH'] = abs_plugin_path

# Note: Models are cached in ~/.cellpose/models by default
# The cache persists between sessions - models are only downloaded once
# If you want to use a custom models directory, set CELLPOSE_LOCAL_MODELS_PATH environment variable

# Import PyQt5 - paths should already be set above
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
        QGroupBox, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QTextEdit,
        QProgressBar, QProgressDialog, QScrollArea, QDialog, QDialogButtonBox, QRadioButton, QListWidget, QSizePolicy
    )
    
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer, QRect, QObject, QCoreApplication, QSize
    from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QColor, QPen, QKeySequence, QShortcut, QCursor
except ImportError as e:
    print(f"ERROR: Failed to import PyQt6: {e}", file=sys.stderr)
    print("This usually means PyQt6 is not installed or there's a Qt plugin issue.", file=sys.stderr)
    print("Try running: ./repair_pyqt6.sh", file=sys.stderr)
    sys.exit(1)

# Import our existing scripts - we'll call them via subprocess to handle argparse properly
import subprocess
import threading
import numpy as np


class ClickableTooltip(QLabel):
    """A tooltip-like widget that appears on click and disappears on click elsewhere"""
    _current_tooltip = None
    _event_filter = None
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(
            "QLabel {"
            "background-color: #2b2b2b;"
            "color: #ffffff;"
            "padding: 8px;"
            "border: 1px solid #555555;"
            "border-radius: 4px;"
            "font-size: 10pt;"
            "}"
        )
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.hide()
    
    @classmethod
    def show_tooltip(cls, text, icon_widget, parent_widget=None):
        """Show a tooltip next to the icon widget"""
        # If same tooltip is already showing, hide it (toggle behavior)
        if cls._current_tooltip and cls._current_tooltip.isVisible():
            cls.hide_current()
            return
        
        # Hide any existing tooltip
        if cls._current_tooltip:
            cls.hide_current()
        
        # Create and show new tooltip
        tooltip = ClickableTooltip(text, parent_widget)
        tooltip.setText(text)
        
        # Calculate position (to the right of the icon)
        icon_global_pos = icon_widget.mapToGlobal(QPoint(0, 0))
        tooltip_pos = QPoint(icon_global_pos.x() + icon_widget.width() + 5, icon_global_pos.y())
        
        # Set maximum width for word wrapping
        tooltip.setMaximumWidth(400)
        tooltip.adjustSize()
        
        # Ensure tooltip stays on screen (PyQt6: use QScreen instead of QDesktopWidget)
        screen = QApplication.primaryScreen()
        if screen:
            screen_rect = screen.geometry()
            if tooltip_pos.x() + tooltip.width() > screen_rect.width():
                tooltip_pos.setX(screen_rect.width() - tooltip.width() - 10)
            if tooltip_pos.y() + tooltip.height() > screen_rect.height():
                tooltip_pos.setY(icon_global_pos.y() - tooltip.height())
        
        tooltip.move(tooltip_pos)
        tooltip.show()
        cls._current_tooltip = tooltip
        
        # Install event filter on the application to hide on click elsewhere
        if cls._event_filter:
            QApplication.instance().removeEventFilter(cls._event_filter)
        cls._event_filter = TooltipClickFilter(tooltip)
        QApplication.instance().installEventFilter(cls._event_filter)
    
    @classmethod
    def hide_current(cls):
        """Hide the currently shown tooltip"""
        if cls._current_tooltip:
            cls._current_tooltip.hide()
            cls._current_tooltip = None
        if cls._event_filter:
            QApplication.instance().removeEventFilter(cls._event_filter)
            cls._event_filter = None


class TooltipClickFilter(QObject):
    """Event filter to hide tooltip when clicking elsewhere"""
    def __init__(self, tooltip, parent=None):
        super().__init__(parent)
        self.tooltip = tooltip
    
    def eventFilter(self, obj, event):
        if event.type() == event.MouseButtonPress:
            # Get widget under mouse
            widget_under_mouse = QApplication.widgetAt(event.globalPos())
            
            # Hide if clicking outside the tooltip
            if self.tooltip and self.tooltip.isVisible():
                # Check if click is on the tooltip or its children
                if widget_under_mouse != self.tooltip:
                    # Also check if it's a child of the tooltip
                    parent = widget_under_mouse
                    is_tooltip_child = False
                    while parent:
                        if parent == self.tooltip:
                            is_tooltip_child = True
                            break
                        parent = parent.parent()
                    
                    if not is_tooltip_child:
                        ClickableTooltip.hide_current()
        return False


def create_info_icon(info_text, parent_widget=None):
    """Create a small (?) info icon label that shows tooltip-style info on click"""
    info_label = QLabel("(?)")
    info_label.setStyleSheet(
        "QLabel {"
        "color: #7F8C8D;"
        "font-size: 10pt;"
        "font-weight: normal;"
        "text-decoration: none;"
        "}"
        "QLabel:hover {"
        "color: #3498DB;"
        "text-decoration: underline;"
        "}"
    )
    info_label.setCursor(Qt.PointingHandCursor)
    
    def show_info(event):
        # Toggle tooltip - if already showing, hide it; otherwise show it
        if ClickableTooltip._current_tooltip and ClickableTooltip._current_tooltip.isVisible():
            ClickableTooltip.hide_current()
        else:
            ClickableTooltip.show_tooltip(info_text, info_label, parent_widget)
        event.accept()
    
    info_label.mousePressEvent = show_info
    return info_label


class WorkerThread(QThread):
    """Thread for running long-running tasks without freezing the GUI"""
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(str)  # status message
    
    def __init__(self, script_path, cmd_args, parent=None):
        super().__init__(parent)
        self.script_path = script_path
        self.cmd_args = cmd_args
    
    def run(self):
        try:
            # Build command
            cmd = [sys.executable, self.script_path] + self.cmd_args
            
            # Run process and capture output with unbuffered output
            # Use unbuffered mode to get real-time output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,  # Unbuffered
                universal_newlines=True,
                env=env
            )
            
            output_lines = []
            # Read line by line in real-time
            while True:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                
                line = line.rstrip()
                if line:
                    output_lines.append(line)
                    self.progress.emit(line)
            
            process.wait()
            output = '\n'.join(output_lines)
            
            if process.returncode == 0:
                self.finished.emit(True, output)
            else:
                self.finished.emit(False, output)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class ND2ConversionTab(QWidget):
    """Tab for converting microscopy files (ND2, LIF, CZI) to TIFF with metadata extraction"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.file_info = None  # Store file info
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        
        # File selection
        file_group = QGroupBox("Microscopy File Selection")
        file_layout = QVBoxLayout()
        
        file_select_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a microscopy file (ND2, LIF, CZI) or folder...")
        self.file_path.setMinimumWidth(500)  # Increased for better visibility
        # Set larger font for better readability
        file_font = QFont()
        file_font.setPointSize(11)
        self.file_path.setFont(file_font)
        # Fix size to minimum - prevent expansion
        size_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.file_path.setSizePolicy(size_policy)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)  # Fixed width for buttons
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)  # Fixed width for buttons
        folder_btn.clicked.connect(self.select_folder)
        file_select_layout.addWidget(self.file_path, 0)  # No stretch - keep at minimum size
        file_select_layout.addWidget(file_btn, 0)  # No stretch for buttons
        file_select_layout.addWidget(folder_btn, 0)  # No stretch for buttons
        file_layout.addLayout(file_select_layout)
        
        file_group.setLayout(file_layout)
        scroll_layout.addWidget(file_group)
        
        # Conversion parameters
        params_group = QGroupBox("Conversion Parameters")
        params_layout = QVBoxLayout()
        
        # Frame selection
        frame_layout = QHBoxLayout()
        frame_label = QLabel("Number of Frames:")
        frame_info_icon = create_info_icon(
            "Number of time frames to extract from the microscopy file.\n\n"
            "Default: 100 frames\n"
            "Source: Hardcoded default value\n\n"
            "The value will automatically update to the maximum number of frames when a file is loaded.\n\n"
            "Note: Extracting more frames takes longer and uses more disk space.",
            parent_widget=self
        )
        frame_layout.addWidget(frame_label)
        frame_layout.addWidget(frame_info_icon)
        self.max_frames = QSpinBox()
        self.max_frames.setRange(1, 10000)
        self.max_frames.setValue(100)
        frame_layout.addWidget(self.max_frames)
        frame_layout.addStretch()
        params_layout.addLayout(frame_layout)
        
        # Channel options
        channel_layout = QHBoxLayout()
        channel_label = QLabel("Channel Handling:")
        channel_info_icon = create_info_icon(
            "Save each channel as a separate TIFF file.\n\n"
            "Default: Enabled (checked)\n"
            "Source: Recommended approach for multi-channel data\n\n"
            "When enabled: Each channel is saved as its own TIFF file with T timepoints.\n"
            "When disabled: All channels saved in one OME-TIFF file (T×C pages).\n\n"
            "Note: Standard TIFF cannot store multiple grayscale channels per page,\n"
            "so separate files is the most compatible approach.",
            parent_widget=self
        )
        channel_layout.addWidget(channel_label)
        channel_layout.addWidget(channel_info_icon)
        self.separate_channels = QCheckBox("Save channels separately (recommended)")
        self.separate_channels.setChecked(True)
        self.separate_channels.setToolTip(
            "Save each channel as a separate TIFF file.\n\n"
            "Default: Enabled (checked)\n"
            "Source: Recommended approach for multi-channel data\n\n"
            "When enabled: Each channel is saved as its own TIFF file with T timepoints.\n"
            "When disabled: All channels saved in one OME-TIFF file (T×C pages).\n\n"
            "Note: Standard TIFF cannot store multiple grayscale channels per page,\n"
            "so separate files is the most compatible approach."
        )
        channel_layout.addWidget(self.separate_channels)
        channel_layout.addStretch()
        params_layout.addLayout(channel_layout)
        
        self.channel_select = QComboBox()
        self.channel_select.addItem("All channels")
        self.channel_select.setToolTip(
            "Select which channel(s) to extract from the microscopy file.\n\n"
            "Default: 'All channels'\n"
            "Source: User selection\n\n"
            "Options:\n"
            "- 'All channels': Extract all channels (saved separately if 'Save channels separately' is checked)\n"
            "- 'Channel N: [name]': Extract only the specified channel\n\n"
            "Note: Only enabled when the file has multiple channels."
        )
        self.channel_select.setEnabled(False)
        params_layout.addWidget(self.channel_select)
        
        # Corrections
        corrections_group = QGroupBox("Image Corrections")
        corrections_layout = QVBoxLayout()
        
        # Drift correction with info icon
        drift_layout = QHBoxLayout()
        self.drift_correct = QCheckBox("Apply drift correction (X,Y registration)")
        self.drift_correct.setChecked(False)
        drift_info_icon = create_info_icon(
            "Apply X,Y drift correction to compensate for stage movement during time-lapse imaging.\n\n"
            "Default: Disabled (unchecked)\n"
            "Source: Optional correction step\n\n"
            "Method: Phase cross-correlation (frame-by-frame alignment)\n"
            "Algorithm: Each frame is aligned to the PREVIOUS frame (incremental correction).\n"
            "This approach is more robust to large cumulative drifts because:\n"
            "- Smaller shifts between consecutive frames improve alignment accuracy\n"
            "- Phase cross-correlation works best with small shifts\n"
            "- Naturally handles gradual drift over time\n\n"
            "Use when: You notice the sample drifting over time.\n"
            "Note: Increases processing time but improves temporal analysis accuracy.\n"
            "The first frame remains unchanged; subsequent frames are incrementally aligned.",
            parent_widget=self
        )
        drift_layout.addWidget(self.drift_correct)
        drift_layout.addWidget(drift_info_icon)
        drift_layout.addStretch()
        corrections_layout.addLayout(drift_layout)
        
        # Photobleaching correction with info icon
        bleach_layout = QHBoxLayout()
        self.bleach_correct = QCheckBox("Apply photobleaching correction")
        self.bleach_correct.setChecked(False)
        bleach_info_icon = create_info_icon(
            "Apply photobleaching correction to compensate for fluorescence signal decay over time.\n\n"
            "Default: Disabled (unchecked)\n"
            "Source: Optional correction step\n\n"
            "Method: Exponential decay correction (fits I(t) = I₀ × exp(-t/τ))\n"
            "Normalizes intensity to the first frame.\n\n"
            "Use when: Working with fluorescence channels that show signal decay.\n"
            "Note: Only affects fluorescence channels, not phase contrast.",
            parent_widget=self
        )
        bleach_layout.addWidget(self.bleach_correct)
        bleach_layout.addWidget(bleach_info_icon)
        bleach_layout.addStretch()
        corrections_layout.addLayout(bleach_layout)
        
        corrections_group.setLayout(corrections_layout)
        params_layout.addWidget(corrections_group)
        
        params_group.setLayout(params_layout)
        scroll_layout.addWidget(params_group)
        
        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QVBoxLayout()
        
        output_select_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Auto: creates 'Tif_1-N' folder in input directory")
        self.output_dir.setMinimumWidth(500)  # Increased for better visibility
        # Set larger font for better readability
        output_font = QFont()
        output_font.setPointSize(11)
        self.output_dir.setFont(output_font)
        # Fix size to minimum - prevent expansion
        size_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.output_dir.setSizePolicy(size_policy)
        output_btn = QPushButton("Browse...")
        output_btn.setFixedWidth(120)  # Fixed width for buttons
        output_btn.clicked.connect(self.select_output_dir)
        output_select_layout.addWidget(self.output_dir, 0)  # No stretch - keep at minimum size
        output_select_layout.addWidget(output_btn, 0)  # No stretch for buttons
        output_layout.addLayout(output_select_layout)
        
        output_group.setLayout(output_layout)
        scroll_layout.addWidget(output_group)
        
        # Metadata info
        metadata_group = QGroupBox("Extracted Metadata")
        metadata_layout = QVBoxLayout()
        
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMinimumHeight(120)  # Increased for better visibility
        self.metadata_text.setMaximumHeight(200)  # Prevent excessive expansion
        self.metadata_text.setPlaceholderText("Metadata will be displayed here after loading microscopy file...")
        # Set larger, more readable font
        metadata_font = QFont()
        metadata_font.setPointSize(13)  # Increased font size for better readability
        self.metadata_text.setFont(metadata_font)
        # Prevent vertical expansion - only allow horizontal
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.metadata_text.setSizePolicy(size_policy)
        metadata_layout.addWidget(self.metadata_text)
        
        metadata_group.setLayout(metadata_layout)
        scroll_layout.addWidget(metadata_group)
        
        # Preview button (moved here to match Single File tab layout)
        self.preview_btn = QPushButton("Preview File")
        self.preview_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #5D6D7E;"
            "color: white;"
            "font-weight: bold;"
            "padding: 8px;"
            "border: none;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "background-color: #34495E;"
            "}"
        )
        self.preview_btn.clicked.connect(self.preview_file)
        self.preview_btn.setEnabled(False)
        scroll_layout.addWidget(self.preview_btn)
        
        # Convert button
        self.convert_btn = QPushButton("Convert to TIFF")
        self.convert_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #5D6D7E;"
            "color: white;"
            "font-weight: bold;"
            "padding: 10px;"
            "border: none;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "background-color: #34495E;"
            "}"
            "QPushButton:disabled {"
            "background-color: #7F8C8D;"
            "color: #BDC3C7;"
            "}"
        )
        self.convert_btn.clicked.connect(self.convert_nd2)
        self.convert_btn.setEnabled(False)
        scroll_layout.addWidget(self.convert_btn)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Microscopy File", "", 
            "Microscopy Files (*.nd2 *.ND2 *.lif *.LIF *.czi *.CZI);;"
            "ND2 Files (*.nd2 *.ND2);;"
            "LIF Files (*.lif *.LIF);;"
            "CZI Files (*.czi *.CZI);;"
            "All Files (*)"
        )
        if file_path:
            self.file_path.setText(file_path)
            self.load_file_info(file_path)
    
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with Microscopy Files")
        if folder_path:
            self.file_path.setText(folder_path)
            # Load info from all supported files in the folder
            folder = Path(folder_path)
            supported_files = sorted(
                list(folder.glob("*.nd2")) + list(folder.glob("*.ND2")) +
                list(folder.glob("*.lif")) + list(folder.glob("*.LIF")) +
                list(folder.glob("*.czi")) + list(folder.glob("*.CZI"))
            )
            if supported_files:
                self.load_folder_info(folder_path, supported_files)
            else:
                self.metadata_text.setText("No supported microscopy files (ND2, LIF, CZI) found in selected folder")
                self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 12pt;")
    
    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir.setText(dir_path)
    
    def load_folder_info(self, folder_path, supported_files):
        """Load and display information for all files in a folder"""
        folder = Path(folder_path)
        all_files_info = []
        errors = []
        
        self.parent_window.log_output(f"Scanning folder: {folder_path}\n")
        self.parent_window.log_output(f"Found {len(supported_files)} file(s) to analyze\n")
        
        # Load info from each file
        for file_path in supported_files:
            try:
                file_ext = file_path.suffix.lower()
                file_info = None
                
                if file_ext == '.nd2':
                    file_info = self._get_nd2_info(str(file_path))
                elif file_ext == '.lif':
                    file_info = self._get_lif_info(str(file_path))
                elif file_ext == '.czi':
                    file_info = self._get_czi_info(str(file_path))
                
                if file_info:
                    all_files_info.append((file_path.name, file_info))
                    self.parent_window.log_output(f"✓ {file_path.name}: {file_info['num_time']} frames, {file_info['num_channels']} channels\n")
                else:
                    errors.append(f"{file_path.name}: Could not read file info")
                    
            except Exception as e:
                errors.append(f"{file_path.name}: {str(e)}")
                self.parent_window.log_output(f"✗ {file_path.name}: Error - {str(e)}\n")
        
        # Display summary of all files
        if all_files_info:
            self._display_folder_summary(all_files_info, errors)
            # Store info from first file for conversion parameters (use first as default)
            first_file_info = all_files_info[0][1]
            self.file_info = {
                'path': folder_path,
                'file_type': first_file_info['file_type'],
                'num_time': first_file_info['num_time'],
                'num_channels': first_file_info['num_channels'],
                'num_z': first_file_info['num_z'],
                'channel_names': first_file_info['channel_names'],
                'pixel_size': first_file_info['pixel_size']
            }
            # Update UI with first file's parameters (user can adjust)
            self.max_frames.setMaximum(first_file_info['num_time'])
            self.max_frames.setValue(first_file_info['num_time'])  # Automatically set to maximum
            self.channel_select.clear()
            self.channel_select.addItem("All channels")
            for i, name in enumerate(first_file_info['channel_names']):
                self.channel_select.addItem(f"Channel {i+1}: {name}")
            self.channel_select.setEnabled(first_file_info['num_channels'] > 1)
            self.preview_btn.setEnabled(True)
            self.convert_btn.setEnabled(True)
            
            # Use pixel size from first file if available
            if first_file_info['pixel_size']:
                self.parent_window.shared_metadata['pixel_size'] = first_file_info['pixel_size']
                self.parent_window.update_all_tabs_with_metadata()
        else:
            self.metadata_text.setText("No files could be read from the selected folder.\n" + 
                                     "\n".join(errors) if errors else "")
            self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ff6b6b; font-size: 12pt;")
    
    def _display_folder_summary(self, all_files_info, errors):
        """Display a summary of all files in the folder"""
        metadata_content = f"Folder contains {len(all_files_info)} file(s):\n"
        metadata_content += "=" * 60 + "\n\n"
        
        for i, (filename, info) in enumerate(all_files_info, 1):
            metadata_content += f"File {i}: {filename}\n"
            metadata_content += f"  Type: {info['file_type'].upper()}\n"
            metadata_content += f"  Dimensions: T={info['num_time']} frames, C={info['num_channels']} channels, Z={info['num_z']} slices\n"
            if info['num_channels'] > 1:
                metadata_content += f"  Channels: {', '.join(info['channel_names'])}\n"
            if info['pixel_size']:
                metadata_content += f"  Pixel Size: {info['pixel_size']:.4f} µm\n"
            else:
                metadata_content += f"  Pixel Size: Not found in metadata\n"
            
            # Show differences if files have different properties
            if i > 1:
                first_info = all_files_info[0][1]
                differences = []
                if info['num_time'] != first_info['num_time']:
                    differences.append(f"frames ({info['num_time']} vs {first_info['num_time']})")
                if info['num_channels'] != first_info['num_channels']:
                    differences.append(f"channels ({info['num_channels']} vs {first_info['num_channels']})")
                if info['pixel_size'] and first_info['pixel_size']:
                    if abs(info['pixel_size'] - first_info['pixel_size']) > 0.0001:
                        differences.append(f"pixel size ({info['pixel_size']:.4f} vs {first_info['pixel_size']:.4f} µm)")
                if differences:
                    metadata_content += f"  ⚠ Different from first file: {', '.join(differences)}\n"
            
            metadata_content += "\n"
        
        if errors:
            metadata_content += f"\nErrors ({len(errors)} file(s)):\n"
            for error in errors:
                metadata_content += f"  ✗ {error}\n"
        
        self.metadata_text.setText(metadata_content)
        self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 10pt;")
    
    def _get_nd2_info(self, file_path):
        """Extract ND2 file info without updating UI (for folder scanning)"""
        try:
            import nd2
            with nd2.ND2File(file_path) as nd2_reader:
                sizes = nd2_reader.sizes
                try:
                    shape = nd2_reader.shape
                    num_time = shape[0] if len(shape) > 0 else sizes.get('T', 1)
                except Exception:
                    num_time = sizes.get('T', 1)
                
                num_channels = sizes.get('C', 1)
                num_z = sizes.get('Z', 1)
                
                channel_names = []
                try:
                    if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                        meta = nd2_reader.metadata
                        if hasattr(meta, 'channels') and meta.channels:
                            channel_names = [ch.channel.name if hasattr(ch, 'channel') and hasattr(ch.channel, 'name') 
                                           else f'Channel_{i+1}' 
                                           for i, ch in enumerate(meta.channels)]
                except Exception:
                    pass
                
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                pixel_size = None
                try:
                    if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                        meta = nd2_reader.metadata
                        if hasattr(meta, 'channels') and meta.channels:
                            first_channel = meta.channels[0]
                            if hasattr(first_channel, 'volume') and first_channel.volume:
                                vol = first_channel.volume
                                if hasattr(vol, 'axesCalibration') and vol.axesCalibration:
                                    if len(vol.axesCalibration) >= 2:
                                        pixel_size = vol.axesCalibration[0]
                except Exception:
                    pass
                
                return {
                    'file_type': 'nd2',
                    'num_time': num_time,
                    'num_channels': num_channels,
                    'num_z': num_z,
                    'channel_names': channel_names,
                    'pixel_size': pixel_size
                }
        except Exception:
            return None
    
    def _get_lif_info(self, file_path):
        """Extract LIF file info without updating UI (for folder scanning)"""
        try:
            from aicsimageio import AICSImage
            with AICSImage(file_path) as lif_reader:
                dims = lif_reader.dims
                shape = lif_reader.shape
                num_time = shape[dims.index('T')] if 'T' in dims else 1
                num_channels = shape[dims.index('C')] if 'C' in dims else 1
                num_z = shape[dims.index('Z')] if 'Z' in dims else 1
                
                channel_names = []
                try:
                    if hasattr(lif_reader, 'channel_names'):
                        channel_names = lif_reader.channel_names
                except:
                    pass
                
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                pixel_size = None
                try:
                    if hasattr(lif_reader, 'physical_pixel_sizes'):
                        if lif_reader.physical_pixel_sizes and len(lif_reader.physical_pixel_sizes) >= 3:
                            pixel_size = float(lif_reader.physical_pixel_sizes[2])
                except:
                    pass
                
                return {
                    'file_type': 'lif',
                    'num_time': num_time,
                    'num_channels': num_channels,
                    'num_z': num_z,
                    'channel_names': channel_names,
                    'pixel_size': pixel_size
                }
        except Exception:
            return None
    
    def _get_czi_info(self, file_path):
        """Extract CZI file info without updating UI (for folder scanning)"""
        try:
            from aicsimageio import AICSImage
            with AICSImage(file_path) as czi_reader:
                dims = czi_reader.dims
                shape = czi_reader.shape
                num_time = shape[dims.index('T')] if 'T' in dims else 1
                num_channels = shape[dims.index('C')] if 'C' in dims else 1
                num_z = shape[dims.index('Z')] if 'Z' in dims else 1
                
                channel_names = []
                try:
                    if hasattr(czi_reader, 'channel_names'):
                        channel_names = czi_reader.channel_names
                except:
                    pass
                
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                pixel_size = None
                try:
                    if hasattr(czi_reader, 'physical_pixel_sizes'):
                        if czi_reader.physical_pixel_sizes and len(czi_reader.physical_pixel_sizes) >= 3:
                            pixel_size = float(czi_reader.physical_pixel_sizes[2])
                except:
                    pass
                
                return {
                    'file_type': 'czi',
                    'num_time': num_time,
                    'num_channels': num_channels,
                    'num_z': num_z,
                    'channel_names': channel_names,
                    'pixel_size': pixel_size
                }
        except Exception:
            return None
    
    def load_file_info(self, file_path):
        """Load and display microscopy file information (ND2, LIF, or CZI)"""
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        
        try:
            if file_ext == '.nd2':
                self._load_nd2_info(file_path)
            elif file_ext == '.lif':
                self._load_lif_info(file_path)
            elif file_ext == '.czi':
                self._load_czi_info(file_path)
            else:
                QMessageBox.warning(self, "Unsupported Format", 
                                  f"File format '{file_ext}' is not supported.\n"
                                  "Supported formats: ND2, LIF, CZI")
                return
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", 
                               f"Could not load file information:\n{str(e)}\n\n{traceback.format_exc()}")
            self.metadata_text.setText(f"Error loading file: {str(e)}")
            self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 10pt;")
    
    def _load_nd2_info(self, file_path):
        """Load and display ND2 file information"""
        try:
            # Lazy import to avoid loading if not needed
            import nd2
            
            self.parent_window.log_output(f"Loading ND2 file info: {file_path}\n")
            
            with nd2.ND2File(file_path) as nd2_reader:
                sizes = nd2_reader.sizes
                # Verify actual frame count by checking shape
                # This ensures we get the correct count (sometimes sizes.get('T') can be off by one)
                try:
                    shape = nd2_reader.shape
                    # First dimension is typically T (time)
                    if len(shape) > 0:
                        actual_t = shape[0]
                        num_time = actual_t
                    else:
                        num_time = sizes.get('T', 1)
                except Exception:
                    # Fallback to reported value
                    num_time = sizes.get('T', 1)
                
                num_channels = sizes.get('C', 1)
                num_z = sizes.get('Z', 1)
                
                # Get channel names
                channel_names = []
                try:
                    if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                        meta = nd2_reader.metadata
                        # Metadata is an object, not a dict - access channels attribute
                        if hasattr(meta, 'channels') and meta.channels:
                            channel_names = [ch.channel.name if hasattr(ch, 'channel') and hasattr(ch.channel, 'name') 
                                           else f'Channel_{i+1}' 
                                           for i, ch in enumerate(meta.channels)]
                except Exception as e:
                    # Silently fail - will use default names
                    pass
                
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                # Extract pixel size from metadata
                pixel_size = None
                metadata_str = "Metadata:\n"
                
                try:
                    if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                        meta = nd2_reader.metadata
                        
                        # Pixel size is in volume.axesCalibration (first element is X, second is Y)
                        # Both should be the same for square pixels
                        if hasattr(meta, 'channels') and meta.channels:
                            # Get pixel size from first channel's volume
                            first_channel = meta.channels[0]
                            if hasattr(first_channel, 'volume') and first_channel.volume:
                                vol = first_channel.volume
                                if hasattr(vol, 'axesCalibration') and vol.axesCalibration:
                                    # axesCalibration is a list: [X, Y, Z] pixel sizes
                                    if len(vol.axesCalibration) >= 2:
                                        # Use X pixel size (should be same as Y for square pixels)
                                        pixel_size = vol.axesCalibration[0]
                        
                        # Format metadata for display
                        if pixel_size:
                            metadata_str += f"  Pixel Size: {pixel_size:.4f} µm\n"
                        else:
                            metadata_str += "  Pixel Size: Not found in metadata\n"
                        
                        # Add other metadata if available
                        if hasattr(meta, 'contents') and meta.contents:
                            if hasattr(meta.contents, 'frameCount'):
                                metadata_str += f"  Frame Count: {meta.contents.frameCount}\n"
                            if hasattr(meta.contents, 'channelCount'):
                                metadata_str += f"  Channel Count: {meta.contents.channelCount}\n"
                        
                except Exception as e:
                    metadata_str += f"  Error reading metadata: {str(e)}\n"
                
                # Use common UI update function
                self._update_file_info_ui(file_path, 'nd2', num_time, num_channels, num_z, 
                                        channel_names, pixel_size, metadata_str)
                
        except ImportError:
            QMessageBox.warning(self, "ND2 Library Missing", 
                              "The 'nd2' library is not installed.\n\n"
                              "Install it with: pip install nd2")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading ND2", f"Could not load ND2 file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _load_lif_info(self, file_path):
        """Load and display LIF file information"""
        try:
            # Try using aicsimageio for LIF files
            try:
                from aicsimageio import AICSImage
                self.parent_window.log_output(f"Loading LIF file info: {file_path}\n")
                
                with AICSImage(file_path) as lif_reader:
                    dims = lif_reader.dims
                    shape = lif_reader.shape
                    
                    # AICSImage uses dimension order like 'STCZYX' or similar
                    # Find time, channel, and z dimensions
                    num_time = shape[dims.index('T')] if 'T' in dims else 1
                    num_channels = shape[dims.index('C')] if 'C' in dims else 1
                    num_z = shape[dims.index('Z')] if 'Z' in dims else 1
                    
                    # Get channel names if available
                    channel_names = []
                    try:
                        if hasattr(lif_reader, 'channel_names'):
                            channel_names = lif_reader.channel_names
                    except:
                        pass
                    
                    if not channel_names or len(channel_names) != num_channels:
                        channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                    
                    # Extract pixel size
                    pixel_size = None
                    try:
                        if hasattr(lif_reader, 'physical_pixel_sizes'):
                            # physical_pixel_sizes is typically (Z, Y, X)
                            if lif_reader.physical_pixel_sizes and len(lif_reader.physical_pixel_sizes) >= 3:
                                pixel_size = float(lif_reader.physical_pixel_sizes[2])  # X dimension
                    except:
                        pass
                    
                    self._update_file_info_ui(file_path, 'lif', num_time, num_channels, num_z, 
                                            channel_names, pixel_size)
                    
            except ImportError:
                # Fallback to readlif if aicsimageio not available
                try:
                    from readlif.reader import LifFile
                    self.parent_window.log_output(f"Loading LIF file info: {file_path}\n")
                    
                    lif_file = LifFile(file_path)
                    # Get first image series
                    if len(lif_file.get_image_count()) > 0:
                        image = lif_file.get_image(0)
                        num_time = image.get_dims()['t'] if 't' in image.get_dims() else 1
                        num_channels = image.get_dims()['c'] if 'c' in image.get_dims() else 1
                        num_z = image.get_dims()['z'] if 'z' in image.get_dims() else 1
                        
                        channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                        pixel_size = None  # readlif doesn't easily provide pixel size
                        
                        self._update_file_info_ui(file_path, 'lif', num_time, num_channels, num_z,
                                                channel_names, pixel_size)
                    else:
                        raise ValueError("No image series found in LIF file")
                except ImportError:
                    QMessageBox.warning(self, "LIF Library Missing",
                                      "The 'aicsimageio' or 'readlif' library is not installed.\n\n"
                                      "Install with: pip install aicsimageio readlif")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading LIF", f"Could not load LIF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _load_czi_info(self, file_path):
        """Load and display CZI file information"""
        try:
            # Try using aicsimageio for CZI files
            try:
                from aicsimageio import AICSImage
                self.parent_window.log_output(f"Loading CZI file info: {file_path}\n")
                
                with AICSImage(file_path) as czi_reader:
                    dims = czi_reader.dims
                    shape = czi_reader.shape
                    
                    num_time = shape[dims.index('T')] if 'T' in dims else 1
                    num_channels = shape[dims.index('C')] if 'C' in dims else 1
                    num_z = shape[dims.index('Z')] if 'Z' in dims else 1
                    
                    # Get channel names if available
                    channel_names = []
                    try:
                        if hasattr(czi_reader, 'channel_names'):
                            channel_names = czi_reader.channel_names
                    except:
                        pass
                    
                    if not channel_names or len(channel_names) != num_channels:
                        channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                    
                    # Extract pixel size
                    pixel_size = None
                    try:
                        if hasattr(czi_reader, 'physical_pixel_sizes'):
                            if czi_reader.physical_pixel_sizes and len(czi_reader.physical_pixel_sizes) >= 3:
                                pixel_size = float(czi_reader.physical_pixel_sizes[2])  # X dimension
                    except:
                        pass
                    
                    self._update_file_info_ui(file_path, 'czi', num_time, num_channels, num_z,
                                            channel_names, pixel_size)
                    
            except ImportError:
                # Fallback to czifile if aicsimageio not available
                try:
                    import czifile
                    self.parent_window.log_output(f"Loading CZI file info: {file_path}\n")
                    
                    with czifile.CziFile(file_path) as czi_file:
                        # Get metadata
                        metadata = czi_file.metadata()
                        # Parse dimensions from metadata
                        # This is simplified - czifile metadata parsing can be complex
                        num_time = 1
                        num_channels = 1
                        num_z = 1
                        channel_names = ['Channel_1']
                        pixel_size = None
                        
                        # Try to extract from shape
                        if hasattr(czi_file, 'shape'):
                            shape = czi_file.shape
                            # CZI shape is typically (T, C, Z, Y, X) or similar
                            if len(shape) >= 3:
                                num_time = shape[0] if len(shape) > 0 else 1
                                num_channels = shape[1] if len(shape) > 1 else 1
                                num_z = shape[2] if len(shape) > 2 else 1
                        
                        channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                        
                        self._update_file_info_ui(file_path, 'czi', num_time, num_channels, num_z,
                                                channel_names, pixel_size)
                except ImportError:
                    QMessageBox.warning(self, "CZI Library Missing",
                                      "The 'aicsimageio' or 'czifile' library is not installed.\n\n"
                                      "Install with: pip install aicsimageio czifile")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading CZI", f"Could not load CZI file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _update_file_info_ui(self, file_path, file_type, num_time, num_channels, num_z, 
                            channel_names, pixel_size, metadata_str=None):
        """Update UI with file information (common for all file types)"""
        # Store info
        self.file_info = {
            'path': file_path,
            'file_type': file_type,
            'num_time': num_time,
            'num_channels': num_channels,
            'num_z': num_z,
            'channel_names': channel_names,
            'pixel_size': pixel_size
        }
        
        # Update frame selector - automatically set to max frames
        self.max_frames.setMaximum(num_time)
        self.max_frames.setValue(num_time)  # Automatically set to maximum
        
        # Update channel selector
        self.channel_select.clear()
        self.channel_select.addItem("All channels")
        for i, name in enumerate(channel_names):
            self.channel_select.addItem(f"Channel {i+1}: {name}")
        self.channel_select.setEnabled(num_channels > 1)
        
        # Update metadata display (now includes file name and dimensions)
        metadata_content = f"File: {Path(file_path).name} ({file_type.upper()})\n"
        metadata_content += f"Dimensions: T={num_time} frames, C={num_channels} channels, Z={num_z} slices\n"
        if num_channels > 1:
            metadata_content += f"Channels: {', '.join(channel_names)}\n"
        metadata_content += "\n"
        
        if metadata_str is None:
            metadata_content += f"Metadata:\n  File Type: {file_type.upper()}\n"
            if pixel_size:
                metadata_content += f"  Pixel Size: {pixel_size:.4f} µm\n"
            else:
                metadata_content += "  Pixel Size: Not found in metadata\n"
        else:
            metadata_content += metadata_str
        
        self.metadata_text.setText(metadata_content)
        self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 10pt;")
        
        if pixel_size:
            # Update shared metadata and propagate to all tabs
            self.parent_window.shared_metadata['pixel_size'] = pixel_size
            self.parent_window.update_all_tabs_with_metadata()
        else:
            # Clear shared metadata
            self.parent_window.shared_metadata['pixel_size'] = None
        
        self.preview_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)
        
        self.parent_window.log_output(f"✓ Loaded: {num_time} frames, {num_channels} channels\n")
        if pixel_size:
            self.parent_window.log_output(f"✓ Pixel size extracted: {pixel_size:.4f} µm (updated in all tabs)\n")
    
    def preview_file(self):
        """Preview microscopy file in a dialog with frame navigation"""
        if not self.file_path.text() or not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "No File", "Please select a valid microscopy file first.")
            return
        
        if not self.file_info:
            QMessageBox.warning(self, "File Not Loaded", "Please load file information first.")
            return
        
        file_type = self.file_info.get('file_type', 'nd2')
        
        if file_type == 'nd2':
            self._preview_nd2()
        elif file_type == 'lif':
            self._preview_lif()
        elif file_type == 'czi':
            self._preview_czi()
        else:
            QMessageBox.warning(self, "Unsupported Format", f"Preview not yet implemented for {file_type} files.")
    
    def _preview_nd2(self):
        """Preview ND2 file in a dialog with frame navigation"""
        
        try:
            import nd2
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            file_path = self.file_path.text()
            
            # Create preview dialog (non-modal so user can access main window)
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("File Preview")
            preview_dialog.setMinimumSize(1000, 800)
            preview_dialog.setModal(False)  # Make non-modal so main window is accessible
            
            layout = QVBoxLayout()
            
            # Load ND2 file info first
            with nd2.ND2File(file_path) as nd2_reader:
                sizes = nd2_reader.sizes
                num_time = sizes.get('T', 1)
                num_channels = sizes.get('C', 1)
                num_z = sizes.get('Z', 1)
                
                # Load all frames (for navigation)
                all_frames = nd2_reader.asarray()
            
            # Matplotlib figure with navigation toolbar
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, preview_dialog)
            # Style toolbar buttons to be white/more visible
            toolbar.setStyleSheet(
                "QToolButton {"
                "color: white;"
                "background-color: #5D6D7E;"
                "border: 1px solid #34495E;"
                "border-radius: 3px;"
                "padding: 4px;"
                "}"
                "QToolButton:hover {"
                "background-color: #34495E;"
                "}"
            )
            
            layout.addWidget(toolbar)
            
            # Frame navigation (moved to top, below toolbar, to match segmentation preview)
            nav_layout = QHBoxLayout()
            prev_btn = QPushButton("Previous")
            nav_layout.addWidget(prev_btn)
            
            nav_layout.addWidget(QLabel("Frame:"))
            frame_spin = QSpinBox()
            frame_spin.setRange(1, num_time)
            frame_spin.setValue(1)
            nav_layout.addWidget(frame_spin)
            nav_layout.addWidget(QLabel(f"of {num_time}"))
            
            next_btn = QPushButton("Next")
            nav_layout.addWidget(next_btn)
            
            nav_layout.addStretch()
            
            # Frame info label
            frame_info = QLabel(f"Timepoint 1 of {num_time} | Channel: 1 | Z-slice: 1")
            frame_info.setStyleSheet("font-weight: bold; color: #2196F3;")
            nav_layout.addWidget(frame_info)
            
            layout.addLayout(nav_layout)
            
            # LUT controls (min/max sliders)
            lut_group = QGroupBox("LUT (Lookup Table) Controls")
            lut_layout = QHBoxLayout()
            
            lut_layout.addWidget(QLabel("Min:"))
            lut_min_slider = QDoubleSpinBox()
            lut_min_slider.setRange(0, 65535)
            lut_min_slider.setDecimals(1)
            lut_min_slider.setSingleStep(10)
            lut_min_slider.setToolTip("Minimum intensity value for display")
            lut_layout.addWidget(lut_min_slider)
            
            lut_layout.addWidget(QLabel("Max:"))
            lut_max_slider = QDoubleSpinBox()
            lut_max_slider.setRange(0, 65535)
            lut_max_slider.setDecimals(1)
            lut_max_slider.setSingleStep(10)
            lut_max_slider.setToolTip("Maximum intensity value for display")
            lut_layout.addWidget(lut_max_slider)
            
            auto_lut_btn = QPushButton("Auto LUT")
            auto_lut_btn.setToolTip("Auto-adjust min/max intensity for better contrast (1st-99th percentile)")
            lut_layout.addWidget(auto_lut_btn)
            
            lut_layout.addStretch()
            lut_group.setLayout(lut_layout)
            layout.addWidget(lut_group)
            
            layout.addWidget(canvas)
            
            # Store current frame data and LUT settings
            current_frame_data = {'frame': None, 'vmin': None, 'vmax': None, 'frame_min': None, 'frame_max': None}
            use_auto_lut = True  # Track if we should use auto LUT or manual values
            is_updating_lut = False  # Flag to prevent recursive updates during LUT adjustment
            im_handle = None  # Store image handle for LUT updates
            
            # Function to update displayed frame
            def update_frame(frame_idx, vmin=None, vmax=None, update_sliders=True, force_auto=False):
                """Update the displayed frame (frame_idx is 0-indexed internally)"""
                # Get frame at index (0-indexed)
                if len(all_frames.shape) >= 3:
                    frame = all_frames[frame_idx]
                else:
                    frame = all_frames
                
                # Handle different shapes
                if len(frame.shape) == 3:  # CYX or ZYX
                    if num_channels > 1:
                        frame = frame[0]  # Take first channel
                    else:
                        frame = frame[0]  # Take first Z-slice
                elif len(frame.shape) == 2:  # YX
                    pass
                elif len(frame.shape) == 4:  # TCYX
                    frame = frame[0]  # Take first channel
                
                # Store frame data and actual min/max
                current_frame_data['frame'] = frame
                frame_min = float(frame.min())
                frame_max = float(frame.max())
                current_frame_data['frame_min'] = frame_min
                current_frame_data['frame_max'] = frame_max
                
                # Determine LUT values
                nonlocal use_auto_lut
                if force_auto:
                    # Force auto LUT (when Auto LUT button is clicked)
                    vmin = np.percentile(frame, 1)  # Use 1st percentile for min
                    vmax = np.percentile(frame, 99)  # Use 99th percentile for max
                    use_auto_lut = True
                elif vmin is not None and vmax is not None:
                    # Explicit values provided (manual adjustment from sliders)
                    use_auto_lut = False
                elif vmin is None or vmax is None:
                    # No values provided - check if we should use auto or manual
                    if use_auto_lut:
                        # Auto mode - calculate from frame
                        vmin = np.percentile(frame, 1)
                        vmax = np.percentile(frame, 99)
                    else:
                        # Manual mode - use current slider values
                        vmin = lut_min_slider.value()
                        vmax = lut_max_slider.value()
                
                current_frame_data['vmin'] = vmin
                current_frame_data['vmax'] = vmax
                
                # Update sliders (block signals to prevent recursive calls)
                if update_sliders:
                    nonlocal is_updating_lut
                    is_updating_lut = True
                    try:
                        lut_min_slider.blockSignals(True)
                        lut_max_slider.blockSignals(True)
                        # Store current values before changing range (to preserve manual adjustments)
                        current_min_val = lut_min_slider.value()
                        current_max_val = lut_max_slider.value()
                        
                        # Only update range if it actually changed (to avoid unnecessary resets)
                        if lut_min_slider.minimum() != frame_min or lut_min_slider.maximum() != frame_max:
                            lut_min_slider.setRange(frame_min, frame_max)
                            lut_max_slider.setRange(frame_min, frame_max)
                        
                        # Set values - prioritize provided vmin/vmax, then preserve current if in manual mode
                        if vmin is not None and vmax is not None:
                            # Use the provided values (either from manual adjustment or auto calculation)
                            lut_min_slider.setValue(vmin)
                            lut_max_slider.setValue(vmax)
                        else:
                            # No explicit values - preserve current manual values if they're still in range
                            # This happens when navigating frames in manual mode
                            if frame_min <= current_min_val <= frame_max:
                                lut_min_slider.setValue(current_min_val)
                            else:
                                # Clamp to range if out of bounds
                                lut_min_slider.setValue(max(frame_min, min(frame_max, current_min_val)))
                            if frame_min <= current_max_val <= frame_max:
                                lut_max_slider.setValue(current_max_val)
                            else:
                                lut_max_slider.setValue(max(frame_min, min(frame_max, current_max_val)))
                        lut_min_slider.blockSignals(False)
                        lut_max_slider.blockSignals(False)
                    finally:
                        is_updating_lut = False
                
                # Clear and redraw
                fig.clear()
                ax = fig.add_subplot(111)
                nonlocal im_handle
                im_handle = ax.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title(f"Frame {frame_idx + 1} of {num_time}: {Path(file_path).name}")
                ax.axis('off')
                canvas.draw()
                
                # Update frame info (1-indexed for display)
                frame_info.setText(f"Timepoint {frame_idx + 1} of {num_time} | Channel: 1 | Z-slice: 1 | LUT: [{vmin:.1f}, {vmax:.1f}]")
            
            # Manual LUT adjustment from sliders
            def update_lut_from_sliders():
                """Update display when sliders are moved"""
                # Prevent recursive calls
                nonlocal use_auto_lut, is_updating_lut
                if is_updating_lut:
                    return
                
                is_updating_lut = True
                try:
                    # Explicitly set manual mode when user adjusts sliders
                    use_auto_lut = False
                    
                    # Block signals to prevent recursive updates
                    lut_min_slider.blockSignals(True)
                    lut_max_slider.blockSignals(True)
                    
                    vmin = lut_min_slider.value()
                    vmax = lut_max_slider.value()
                    
                    # Ensure min < max for valid display
                    if vmin >= vmax:
                        vmax = vmin + 1.0
                        lut_max_slider.setValue(vmax)
                    
                    # Update the image display directly using matplotlib's set_clim
                    # This is the proper way to update LUT without redrawing the entire figure
                    if current_frame_data['frame'] is not None and im_handle is not None:
                        # Update the color limits on the existing image
                        im_handle.set_clim(vmin=vmin, vmax=vmax)
                        # Force immediate redraw (draw_idle might not work reliably)
                        canvas.draw()
                        
                        # Update frame info
                        frame_info.setText(f"Timepoint {preview_dialog.current_frame + 1} of {num_time} | Channel: 1 | Z-slice: 1 | LUT: [{vmin:.1f}, {vmax:.1f}]")
                        
                        # Store the current LUT values so they persist
                        current_frame_data['vmin'] = vmin
                        current_frame_data['vmax'] = vmax
                    else:
                        # Fallback: full update if image handle doesn't exist
                        update_frame(preview_dialog.current_frame, vmin=vmin, vmax=vmax, update_sliders=False)
                    
                    # Unblock signals
                    lut_min_slider.blockSignals(False)
                    lut_max_slider.blockSignals(False)
                finally:
                    is_updating_lut = False
            
            # Auto LUT function
            def auto_lut():
                """Auto-adjust LUT based on current frame"""
                if current_frame_data['frame'] is not None:
                    update_frame(preview_dialog.current_frame, force_auto=True)
            
            # Debounce timer for valueChanged signals (to avoid too many updates while dragging)
            lut_update_timer = QTimer()
            lut_update_timer.setSingleShot(True)
            lut_update_timer.timeout.connect(update_lut_from_sliders)
            
            def on_lut_value_changed():
                """Debounced handler for valueChanged signal"""
                # Only trigger if not currently updating (to avoid conflicts)
                if not is_updating_lut:
                    lut_update_timer.start(50)  # 50ms debounce
            
            # Connect LUT sliders
            # Use debounced valueChanged for smooth real-time updates when using arrows
            lut_min_slider.valueChanged.connect(on_lut_value_changed)
            lut_max_slider.valueChanged.connect(on_lut_value_changed)
            # Use editingFinished for immediate update when user finishes typing (Enter key)
            # Make sure it doesn't get reset by blocking signals during update
            def on_lut_editing_finished():
                """Handle when user finishes editing (Enter key or focus loss)"""
                if not is_updating_lut:
                    update_lut_from_sliders()
            
            lut_min_slider.editingFinished.connect(on_lut_editing_finished)
            lut_max_slider.editingFinished.connect(on_lut_editing_finished)
            auto_lut_btn.clicked.connect(auto_lut)
            
            # Store current frame
            preview_dialog.current_frame = 0
            
            # Connect navigation (matching Preview Segmentation style)
            def go_to_frame(idx):
                if 0 <= idx < num_time:
                    preview_dialog.current_frame = idx
                    # Block signals to prevent recursive calls
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(idx + 1)
                    frame_spin.blockSignals(False)
                    try:
                        # Always preserve current slider values when navigating (don't force auto)
                        # User can click Auto LUT button if they want auto for new frame
                        # Only use manual values if we're in manual mode, otherwise use auto
                        if not use_auto_lut:
                            # Manual mode: preserve current slider values
                            vmin = lut_min_slider.value()
                            vmax = lut_max_slider.value()
                            update_frame(idx, vmin=vmin, vmax=vmax, update_sliders=False)
                        else:
                            # Auto mode: let update_frame calculate auto LUT
                            update_frame(idx, vmin=None, vmax=None, update_sliders=True)
                    except Exception as e:
                        QMessageBox.warning(preview_dialog, "Preview Error", 
                                          f"Could not display frame {idx + 1}:\n{str(e)}")
            
            prev_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame - 1))
            next_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame + 1))
            
            # Use editingFinished for when user finishes editing (presses Enter or leaves field)
            # and a debounced valueChanged for when using arrow keys
            def on_frame_changed():
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_time:
                    go_to_frame(frame_num)
            
            frame_spin.editingFinished.connect(on_frame_changed)
            # Debounce valueChanged to avoid triggering while typing
            frame_spin._change_timer = QTimer()
            frame_spin._change_timer.setSingleShot(True)
            frame_spin._change_timer.timeout.connect(on_frame_changed)
            frame_spin.valueChanged.connect(lambda v: frame_spin._change_timer.start(300))
            
            # Add Cmd+W (or Ctrl+W) shortcut to close dialog
            close_shortcut = QShortcut(QKeySequence("Ctrl+W"), preview_dialog)
            close_shortcut.activated.connect(preview_dialog.close)
            # Also handle Cmd+W on macOS (Qt uses Ctrl modifier for Cmd on macOS)
            if sys.platform == 'darwin':
                close_shortcut_mac = QShortcut(QKeySequence("Meta+W"), preview_dialog)
                close_shortcut_mac.activated.connect(preview_dialog.close)
            
            # Display first frame (with auto LUT applied by default)
            update_frame(0)
            
            preview_dialog.setLayout(layout)
            preview_dialog.show()  # Use show() instead of exec_() for non-modal dialog
            
        except ImportError as e:
            error_msg = str(e)
            if 'torch.testing' in error_msg:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  "Preview requires torch.testing module.\n\n"
                                  "This is a known issue with frozen builds.\n"
                                  "Try running from source: python omnipose_batch_gui.py")
            else:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  f"Preview requires matplotlib:\n\npip install matplotlib")
        except Exception as e:
            error_msg = str(e)
            if 'RpcBackendOptions' in error_msg or 'generic_type' in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "PyTorch initialization error.\n\n"
                                  "This is a known issue with frozen builds and PyTorch.\n"
                                  "Try restarting the application or running from source.")
            else:
                QMessageBox.critical(self, "Preview Error", f"Could not preview file:\n{error_msg}")
            import traceback
            traceback.print_exc()
    
    def _preview_lif(self):
        """Preview LIF file (stub - to be implemented)"""
        QMessageBox.information(self, "LIF Preview", 
                              "LIF file preview is not yet fully implemented.\n"
                              "Please use ND2 or CZI files for preview.")
    
    def _preview_czi(self):
        """Preview CZI file (stub - to be implemented)"""
        QMessageBox.information(self, "CZI Preview", 
                              "CZI file preview is not yet fully implemented.\n"
                              "Please use ND2 files for preview.")
    
    def convert_nd2(self):
        """Convert microscopy file(s) to TIFF"""
        if not self.file_path.text():
            QMessageBox.warning(self, "No File Selected", "Please select a microscopy file or folder first.")
            return
        
        if not self.file_info:
            QMessageBox.warning(self, "File Not Loaded", "Please load file information first.")
            return
        
        file_type = self.file_info.get('file_type', 'nd2')
        
        if file_type == 'nd2':
            self._convert_nd2()
        elif file_type == 'lif':
            QMessageBox.information(self, "LIF Conversion", 
                                  "LIF file conversion is not yet fully implemented.\n"
                                  "Please use ND2 or CZI files, or convert LIF files manually.")
        elif file_type == 'czi':
            QMessageBox.information(self, "CZI Conversion", 
                                  "CZI file conversion is not yet fully implemented.\n"
                                  "Please use ND2 files, or convert CZI files manually.")
        else:
            QMessageBox.warning(self, "Unsupported Format", f"Conversion not yet implemented for {file_type} files.")
    
    def _convert_nd2(self):
        """Convert file(s) to TIFF"""
        
        # Check if convert_nd2_to_tif.py exists
        script_dir = Path(__file__).parent
        convert_script = script_dir / "convert_nd2_to_tif.py"
        
        # Check if script exists in parent directory (Dru_analysis)
        if not convert_script.exists():
            # Try parent directory
            parent_dir = script_dir.parent
            convert_script = parent_dir / "Dru_analysis" / "convert_nd2_to_tif.py"
        
        if not convert_script.exists():
            QMessageBox.warning(self, "Script Not Found", 
                              f"convert_nd2_to_tif.py not found.\n\n"
                              f"Expected at: {script_dir / 'convert_nd2_to_tif.py'}\n"
                              f"Or: {parent_dir / 'Dru_analysis' / 'convert_nd2_to_tif.py'}")
            return
        
        # Build command arguments
        cmd_args = [
            self.file_path.text(),
            '--frames', str(self.max_frames.value()),
        ]
        
        if self.output_dir.text():
            cmd_args.extend(['--output', self.output_dir.text()])
        
        if self.separate_channels.isChecked():
            cmd_args.append('--separate-channels')
        else:
            cmd_args.append('--combine-channels')
        
        # Channel selection
        selected_channel = self.channel_select.currentIndex() - 1  # -1 means all
        if selected_channel >= 0:
            cmd_args.extend(['--channel', str(selected_channel)])
        
        if self.drift_correct.isChecked():
            cmd_args.append('--drift-correct')
        
        if self.bleach_correct.isChecked():
            cmd_args.append('--bleach-correct')
        
        # Pass pixel size if extracted
        if self.file_info and self.file_info.get('pixel_size'):
            cmd_args.extend(['--pixel-size', str(self.file_info['pixel_size'])])
        
        # Run in thread
        self.convert_btn.setEnabled(False)
        self.convert_btn.setText("Converting...")
        self.parent_window.log_output(f"Starting ND2 conversion: {self.file_path.text()}\n")
        
        # Show progress bar with proper range
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        
        # Track progress state
        self.conversion_progress = {
            'total_files': 0,
            'current_file': 0,  # 0-indexed: 0 = first file, 1 = second file, etc.
            'file_progress': 0  # 0-100 for current file
        }
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(str(convert_script), cmd_args, parent=self)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self._update_conversion_progress)
        self.worker.start()
    
    def _update_conversion_progress(self, message):
        """Parse conversion output and update progress bar"""
        # Log the message
        self.parent_window.log_output(message + "\n")
        
        # Parse progress from message
        msg_lower = message.lower()
        import re
        
        # Detect total number of files - try multiple patterns
        if "found" in msg_lower:
            # Try different patterns: "Found 3 ND2 file(s) to process" or "Found 3 file(s)"
            match = re.search(r'found (\d+)', msg_lower, re.IGNORECASE)
            if match:
                self.conversion_progress['total_files'] = int(match.group(1))
                # Reset progress when we detect files
                self.conversion_progress['current_file'] = 0
                self.conversion_progress['file_progress'] = 0
        
        # Detect file processing start
        if "processing:" in msg_lower or ("processing" in msg_lower and ".nd2" in msg_lower):
            # New file starting - increment current_file index
            # This happens when we see "Processing: filename.nd2"
            if self.conversion_progress['file_progress'] >= 90:  # Previous file was nearly done
                # Move to next file
                if self.conversion_progress['total_files'] > 0:
                    self.conversion_progress['current_file'] = min(
                        self.conversion_progress['current_file'] + 1,
                        self.conversion_progress['total_files'] - 1
                    )
            self.conversion_progress['file_progress'] = 5  # Start new file at 5%
        
        # Estimate progress based on stage (use if statements to update progressively)
        # Each stage sets a minimum progress, so progress can only increase
        if "reading frames" in msg_lower or "reading" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 20)
        if "extracting frames" in msg_lower or "extracting" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 25)
        if "dimensions:" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 15)
        if "applying drift correction" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 40)
        if "drift correction complete" in msg_lower or "✓ drift" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 45)
        if "applying bleach correction" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 50)
        if "bleach correction complete" in msg_lower or "✓ bleach" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 55)
        
        # Parse channel saving progress (e.g., "Saving channel 1/3" or "channel 1/3:")
        if "saving" in msg_lower:
            channel_match = re.search(r'channel (\d+)/(\d+)', msg_lower)
            if channel_match:
                current_ch = int(channel_match.group(1))
                total_ch = int(channel_match.group(2))
                # Saving is 60-95% of file progress
                save_progress = 60 + int(35 * (current_ch / total_ch))
                self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], save_progress)
            else:
                # General saving stage
                if self.conversion_progress['file_progress'] < 60:
                    self.conversion_progress['file_progress'] = 60
        
        # Detect completion
        if "complete" in msg_lower or "✓" in message or "successfully" in msg_lower:
            self.conversion_progress['file_progress'] = 100
        
        # Fallback: if we see any meaningful output, increment progress slightly
        # This ensures the bar moves even if pattern matching fails
        if len(message.strip()) > 5:
            # Increment by 1% for each meaningful message (capped at 95% to leave room for completion)
            if self.conversion_progress['file_progress'] < 95:
                self.conversion_progress['file_progress'] = min(
                    95, 
                    self.conversion_progress['file_progress'] + 1
                )
        
        # Calculate overall progress
        if self.conversion_progress['total_files'] > 0:
            # Progress = (completed files + current file progress) / total files
            # current_file is 0-indexed, so file 0 = first file
            completed_files = self.conversion_progress['current_file']  # Files already completed (0 = working on first)
            current_file_progress = self.conversion_progress['file_progress'] / 100.0
            
            # Overall = (files_done + current_file_progress) / total_files
            # If we're on file 0 at 50%, that's 0.5/3 = 16.7% overall (for 3 files)
            overall_progress = int(((completed_files + current_file_progress) / self.conversion_progress['total_files']) * 100)
        else:
            # Fallback: if we don't know total files, use file_progress directly
            # This ensures progress shows even if detection fails
            overall_progress = self.conversion_progress['file_progress']
        
        # Clamp to 0-100
        overall_progress = max(0, min(100, overall_progress))
        
        # Update progress bar (ensure it's visible and has correct range)
        if not self.parent_window.progress_bar.isVisible():
            self.parent_window.progress_bar.setVisible(True)
        if self.parent_window.progress_bar.maximum() != 100:
            self.parent_window.progress_bar.setRange(0, 100)
        
        # Update the value - always update even if 0 to ensure it's set
        self.parent_window.progress_bar.setValue(overall_progress)
        
        # Force GUI update to ensure the bar redraws
        QApplication.processEvents()
    
    def on_finished(self, success, message):
        self.convert_btn.setEnabled(True)
        self.convert_btn.setText("Convert to TIFF")
        self.parent_window.log_output(message)
        # Hide progress bar
        self.parent_window.progress_bar.setVisible(False)
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        if success:
            self.parent_window.log_output("✓ ND2 conversion completed successfully!\n")
        else:
            QMessageBox.critical(self, "Error", f"Conversion failed:\n{message}")


class ProcessingTab(QWidget):
    """Unified tab for processing single files or folders"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # File/Folder selection
        file_group = QGroupBox("Input File/Folder")
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a TIFF file or folder...")
        self.file_path.setMinimumWidth(500)  # Increased for better visibility
        # Set larger font for better readability
        file_font = QFont()
        file_font.setPointSize(11)
        self.file_path.setFont(file_font)
        # Fix size to minimum - prevent expansion
        size_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.file_path.setSizePolicy(size_policy)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)  # Fixed width for buttons
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)  # Fixed width for buttons
        folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.file_path, 0)  # No stretch - keep at minimum size
        file_layout.addWidget(file_btn, 0)  # No stretch for buttons
        file_layout.addWidget(folder_btn, 0)  # No stretch for buttons
        file_group.setLayout(file_layout)
        scroll_layout.addWidget(file_group)
        
        # Segmentation parameters (same for both modes)
        seg_group = QGroupBox("Segmentation Parameters")
        seg_layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_info_icon = create_info_icon(
            "Omnipose model type for cell segmentation.\n\n"
            "Default: 'bact_phase_omni'\n"
            "Source: Optimized for bacterial phase contrast images\n\n"
            "Available models:\n"
            "- bact_phase_omni: Bacterial phase contrast (default, best for bacteria)\n"
            "- cyto2_omni: General cytoplasm segmentation\n"
            "- cyto_omni: Original cytoplasm model\n"
            "- bact_omni: General bacterial cells\n"
            "- bact_fluor_omni: Bacterial fluorescence\n"
            "- worm_omni: C. elegans and similar\n"
            "- nuclei: Nucleus segmentation only\n"
            "- cyto2/cyto: Cellpose models (no Omnipose)\n\n"
            "Tip: Choose based on your cell type and imaging modality."
        , parent_widget=self)
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_info_icon)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'bact_phase_omni', 'cyto2_omni', 'cyto_omni', 'bact_omni',
            'bact_fluor_omni', 'worm_omni', 'nuclei', 'cyto2', 'cyto'
        ])
        self.model_combo.setCurrentText('bact_phase_omni')
        self.model_combo.setToolTip(
            "Omnipose model type for cell segmentation.\n\n"
            "Default: 'bact_phase_omni'\n"
            "Source: Optimized for bacterial phase contrast images\n\n"
            "Available models:\n"
            "- bact_phase_omni: Bacterial phase contrast (default, best for bacteria)\n"
            "- cyto2_omni: General cytoplasm segmentation\n"
            "- cyto_omni: Original cytoplasm model\n"
            "- bact_omni: General bacterial cells\n"
            "- bact_fluor_omni: Bacterial fluorescence\n"
            "- worm_omni: C. elegans and similar\n"
            "- nuclei: Nucleus segmentation only\n"
            "- cyto2/cyto: Cellpose models (no Omnipose)\n\n"
            "Tip: Choose based on your cell type and imaging modality."
        )
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        seg_layout.addLayout(model_layout)
        
        # Diameter
        diam_layout = QHBoxLayout()
        diam_label = QLabel("Diameter (0=auto):")
        diam_info_icon = create_info_icon(
            "Expected cell diameter in pixels.\n\n"
            "Default: 0 (auto-detect)\n"
            "Source: Omnipose default (auto-detection)\n\n"
            "Set to 0 for automatic detection (recommended for first try).\n"
            "Specify a value (e.g., 20-50 pixels for bacteria) for more consistent results.\n\n"
            "Typical values:\n"
            "- Bacteria: 20-50 pixels\n"
            "- Mammalian cells: 30-100 pixels\n"
            "- Nuclei: 15-30 pixels\n\n"
            "Tip: Auto-detection works well, but specifying diameter often gives better results."
        , parent_widget=self)
        diam_layout.addWidget(diam_label)
        diam_layout.addWidget(diam_info_icon)
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0, 1000)
        self.diameter.setValue(0)
        self.diameter.setDecimals(1)
        self.diameter.setToolTip(
            "Expected cell diameter in pixels.\n\n"
            "Default: 0 (auto-detect)\n"
            "Source: Omnipose default (auto-detection)\n\n"
            "Set to 0 for automatic detection (recommended for first try).\n"
            "Note: Auto-detection requires a pre-trained size model. If you see a warning\n"
            "'no pretrained size model specified', set a specific diameter value instead.\n\n"
            "Specify a value (e.g., 20-50 pixels for bacteria) for more consistent results.\n\n"
            "Typical values:\n"
            "- Bacteria: 20-50 pixels\n"
            "- Mammalian cells: 30-100 pixels\n"
            "- Nuclei: 15-30 pixels\n\n"
            "Tip: If auto-detection fails, try setting diameter to 30 pixels as a starting point."
        )
        diam_layout.addWidget(self.diameter)
        diam_layout.addStretch()
        seg_layout.addLayout(diam_layout)
        
        # Flow threshold
        flow_layout = QHBoxLayout()
        flow_label = QLabel("Flow Threshold:")
        flow_info_icon = create_info_icon(
            "Flow error threshold for cell detection.\n\n"
            "Default: 0.4\n"
            "Source: Omnipose default value\n\n"
            "Controls sensitivity of cell detection:\n"
            "- Lower values (0.2-0.3): More sensitive, detects more cells (may include false positives)\n"
            "- Higher values (0.5-0.7): More strict, fewer cells (may miss some cells)\n\n"
            "Range: 0.0 to 1.0\n"
            "Tip: If missing cells, try lowering to 0.3. If too many false positives, raise to 0.5."
        , parent_widget=self)
        flow_layout.addWidget(flow_label)
        flow_layout.addWidget(flow_info_icon)
        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setRange(0.0, 1.0)
        self.flow_threshold.setValue(0.4)
        self.flow_threshold.setSingleStep(0.1)
        self.flow_threshold.setDecimals(2)
        self.flow_threshold.setToolTip(
            "Flow error threshold for cell detection.\n\n"
            "Default: 0.4\n"
            "Source: Omnipose default value\n\n"
            "Controls sensitivity of cell detection:\n"
            "- Lower values (0.2-0.3): More sensitive, detects more cells (may include false positives)\n"
            "- Higher values (0.5-0.7): More strict, fewer cells (may miss some cells)\n\n"
            "Range: 0.0 to 1.0\n"
            "Tip: If missing cells, try lowering to 0.3. If too many false positives, raise to 0.5."
        )
        flow_layout.addWidget(self.flow_threshold)
        flow_layout.addStretch()
        seg_layout.addLayout(flow_layout)
        
        # Options
        gpu_layout = QHBoxLayout()
        self.use_gpu = QCheckBox("Use GPU (if available)")
        self.use_gpu.setChecked(True)
        gpu_info_icon = create_info_icon(
            "Use GPU acceleration for faster processing.\n\n"
            "Default: Enabled (checked)\n"
            "Source: Recommended for performance\n\n"
            "When enabled: Uses GPU if available (10-100x faster than CPU).\n"
            "When disabled: Forces CPU processing (slower but works without GPU).\n\n"
            "Note: Automatically falls back to CPU if no GPU is available.\n"
            "Tip: Disable if you encounter out-of-memory errors.",
            parent_widget=self
        )
        gpu_layout.addWidget(self.use_gpu)
        gpu_layout.addWidget(gpu_info_icon)
        gpu_layout.addStretch()
        seg_layout.addLayout(gpu_layout)
        
        cluster_layout = QHBoxLayout()
        self.cluster = QCheckBox("Use DBSCAN clustering")
        self.cluster.setChecked(False)
        cluster_info_icon = create_info_icon(
            "Use DBSCAN clustering to help segment thin/elongated cells.\n\n"
            "Default: Disabled (unchecked)\n"
            "Source: Optional Omnipose feature\n\n"
            "When enabled: Applies DBSCAN clustering to improve segmentation of:\n"
            "- Thin, elongated cells (e.g., bacteria, worms)\n"
            "- Cells with complex shapes\n"
            "- Overlapping cells\n\n"
            "Use when: Standard segmentation struggles with elongated or thin structures.\n"
            "Note: Slightly increases processing time.",
            parent_widget=self
        )
        cluster_layout.addWidget(self.cluster)
        cluster_layout.addWidget(cluster_info_icon)
        cluster_layout.addStretch()
        seg_layout.addLayout(cluster_layout)
        
        # Reset to defaults button for segmentation parameters
        reset_seg_btn = QPushButton("Reset to Default")
        reset_seg_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #7D8C9E;"
            "color: white;"
            "font-weight: bold;"
            "padding: 6px;"
            "border: none;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "background-color: #6C7B8C;"
            "}"
        )
        reset_seg_btn.setToolTip("Reset all segmentation parameters to their default values")
        reset_seg_btn.clicked.connect(self.reset_segmentation_defaults)
        seg_layout.addWidget(reset_seg_btn)
        
        seg_group.setLayout(seg_layout)
        scroll_layout.addWidget(seg_group)
        
        # Preview button
        preview_btn = QPushButton("Preview Segmentation")
        preview_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #5D6D7E;"
            "color: white;"
            "font-weight: bold;"
            "padding: 8px;"
            "border: none;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "background-color: #34495E;"
            "}"
        )
        preview_btn.clicked.connect(self.preview_segmentation)
        scroll_layout.addWidget(preview_btn)
        
        # Process button
        self.process_btn = QPushButton("Process File or Folder")
        self.process_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #5D6D7E;"
            "color: white;"
            "font-weight: bold;"
            "padding: 10px;"
            "border: none;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "background-color: #34495E;"
            "}"
            "QPushButton:disabled {"
            "background-color: #7F8C8D;"
            "color: #BDC3C7;"
            "}"
        )
        self.process_btn.clicked.connect(self.process_file_or_folder)
        scroll_layout.addWidget(self.process_btn)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def select_file(self):
        """Open file dialog to select a TIFF file"""
        try:
            # Get the current file path if one exists, to use as starting directory
            start_dir = ""
            if self.file_path.text() and Path(self.file_path.text()).exists():
                start_dir = str(Path(self.file_path.text()).parent)
            elif self.file_path.text():
                # If path exists but file doesn't, use parent directory
                start_dir = str(Path(self.file_path.text()).parent)
            else:
                # Default to home directory
                start_dir = str(Path.home())
            
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select TIFF File", 
                start_dir,
                "TIFF Files (*.tif *.tiff);;All Files (*)"
            )
            if file_path:
                self.file_path.setText(file_path)
                if self.parent_window and hasattr(self.parent_window, 'log_output'):
                    self.parent_window.log_output(f"Selected file: {file_path}\n")
        except Exception as e:
            QMessageBox.critical(
                self, 
                "File Selection Error", 
                f"Could not open file dialog:\n{str(e)}\n\n{traceback.format_exc()}"
            )
            if self.parent_window and hasattr(self.parent_window, 'log_output'):
                self.parent_window.log_output(f"Error selecting file: {str(e)}\n{traceback.format_exc()}\n")
    
    def select_folder(self):
        """Open folder dialog to select a folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.file_path.setText(folder_path)
    
    def _is_folder_mode(self):
        """Check if current path is a folder"""
        if not self.file_path.text():
            return False
        path = Path(self.file_path.text())
        return path.exists() and path.is_dir()
    
    def reset_segmentation_defaults(self):
        """Reset all segmentation parameters to their default values"""
        self.model_combo.setCurrentText('bact_phase_omni')
        self.diameter.setValue(0)
        self.flow_threshold.setValue(0.4)
        self.use_gpu.setChecked(True)
        self.cluster.setChecked(False)
    
    def preview_segmentation(self):
        """Preview segmentation on a single frame or file from folder"""
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a TIFF file or folder first.")
            return
        
        if not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return
        
        is_folder = self._is_folder_mode()
        if is_folder:
            # Folder mode: show file selection dialog first
            folder = Path(self.file_path.text())
            # Look for TIFF files (not mask files - this is for processing, not analyzing)
            # Use set to avoid duplicates from multiple glob patterns
            tiff_files_set = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                            set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out mask files (we want original files for segmentation preview)
            tiff_files = sorted([f for f in tiff_files_set if '_masks' not in f.name.lower()], 
                               key=lambda x: x.name.lower())
            
            if not tiff_files:
                QMessageBox.warning(self, "No TIFF Files Found", 
                                  "No TIFF files found in the selected folder.\n\n"
                                  "Please select a folder containing TIFF files to process.")
                return
            
            # Let user choose which file to preview - use custom dialog
            file_dialog = QDialog(self)
            file_dialog.setWindowTitle("Select File for Preview")
            file_dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout()
            
            # Label
            label = QLabel("Select a file to preview segmentation:")
            label.setStyleSheet("color: #ffffff; font-size: 11pt; padding: 10px;")
            layout.addWidget(label)
            
            # List widget with file names
            file_list = QListWidget()
            file_list.setStyleSheet(
                "QListWidget {"
                "background-color: #2b2b2b;"
                "color: #ffffff;"
                "font-size: 11pt;"
                "border: 1px solid #555555;"
                "border-radius: 4px;"
                "padding: 5px;"
                "}"
                "QListWidget::item {"
                "padding: 8px;"
                "border-bottom: 1px solid #3a3a3a;"
                "}"
                "QListWidget::item:selected {"
                "background-color: #5D6D7E;"
                "color: #ffffff;"
                "}"
                "QListWidget::item:hover {"
                "background-color: #3a3a3a;"
                "}"
            )
            
            # Add file names to list
            for f in tiff_files:
                file_list.addItem(f.name)
            
            # Select first item by default
            if file_list.count() > 0:
                file_list.setCurrentRow(0)
            
            # Enable double-click to accept
            def on_item_double_clicked(item):
                file_dialog.accept()
            
            file_list.itemDoubleClicked.connect(on_item_double_clicked)
            
            layout.addWidget(file_list)
            
            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(file_dialog.accept)
            button_box.rejected.connect(file_dialog.reject)
            button_box.setStyleSheet(
                "QPushButton {"
                "background-color: #5D6D7E;"
                "color: white;"
                "font-weight: bold;"
                "padding: 8px 20px;"
                "border: none;"
                "border-radius: 4px;"
                "min-width: 80px;"
                "}"
                "QPushButton:hover {"
                "background-color: #34495E;"
                "}"
            )
            layout.addWidget(button_box)
            
            file_dialog.setLayout(layout)
            
            # Show dialog
            if file_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            
            # Get selected file
            selected_item = file_list.currentItem()
            if not selected_item:
                return
            
            selected_name = selected_item.text()
            selected_file = None
            for f in tiff_files:
                if f.name == selected_name:
                    selected_file = f
                    break
            
            if not selected_file:
                return
            
            # Use selected file for preview
            file_path = str(selected_file)
        else:
            # Single file mode: use the selected file directly
            file_path = self.file_path.text()
        
        # Show progress dialog while loading
        loading_progress = QProgressDialog("Loading file and initializing preview...", None, 0, 0, self)
        loading_progress.setWindowTitle("Loading")
        loading_progress.setWindowModality(Qt.WindowModal)
        loading_progress.setAutoClose(False)
        loading_progress.setAutoReset(False)
        loading_progress.setCancelButton(None)
        loading_progress.show()
        QApplication.processEvents()
        
        # Load TIFF file
        try:
            loading_progress.setLabelText("Loading TIFF file...")
            QApplication.processEvents()
            
            import tifffile
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            
            # CRITICAL: Patch omnipose.core.OMNI_MODELS before cellpose_omni imports it
            # This fixes "cannot import name 'OMNI_MODELS' from 'omnipose.core'" error in frozen builds
            # Also ensure get_masks patch is applied
            try:
                import omnipose.core
                # Re-apply patch in case it wasn't applied earlier (omnipose imported late)
                is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                if is_frozen:
                    _patch_omnipose_get_masks()
                if not hasattr(omnipose.core, 'OMNI_MODELS'):
                    # Construct OMNI_MODELS from available model constants
                    OMNI_MODELS = []
                    if hasattr(omnipose.core, 'C1_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C1_MODELS)
                    if hasattr(omnipose.core, 'C2_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C2_MODELS)
                    if hasattr(omnipose.core, 'C1_BD_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C1_BD_MODELS)
                    if hasattr(omnipose.core, 'C2_BD_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C2_BD_MODELS)
                    # Remove duplicates
                    seen = set()
                    OMNI_MODELS = [x for x in OMNI_MODELS if not (x in seen or seen.add(x))]
                    # Default fallback if still empty
                    if not OMNI_MODELS:
                        OMNI_MODELS = ['cyto_omni', 'cyto2_omni', 'bact_omni', 'bact_phase_omni',
                                      'bact_fluor_omni', 'worm_omni', 'nuclei_omni']
                    omnipose.core.OMNI_MODELS = OMNI_MODELS
            except Exception:
                # If patching fails, continue anyway - cellpose_omni has a fallback
                pass
            
            from cellpose_omni import models
            from cellpose_omni.io import imread
            import numpy as np
            
            images = tifffile.imread(file_path)
            
            loading_progress.setLabelText("Processing image data...")
            QApplication.processEvents()
            
            # Convert to float32 for MPS compatibility (Apple Silicon GPU doesn't support float64)
            if images.dtype != np.float32:
                images = images.astype(np.float32)
            
            # Handle different input formats
            if images.ndim == 2:
                images = images[np.newaxis, ...]
            elif images.ndim == 3:
                if images.shape[-1] < 10:  # Likely channels last
                    images = images.transpose(2, 0, 1)
                    images = images[np.newaxis, ...]
            elif images.ndim == 4:
                if images.shape[-1] < 10:  # Channels last
                    images = images.transpose(0, 3, 1, 2)
                elif images.shape[0] < 10:  # Channels first
                    images = images.transpose(1, 0, 2, 3)
            
            # Check if we have any frames
            if images.size == 0 or images.shape[0] == 0:
                QMessageBox.critical(preview_dialog, "Empty File", 
                                   "The TIFF file appears to be empty or has no frames.")
                return
            
            num_frames = images.shape[0]
            
            # Create preview dialog (non-modal so user can access main window)
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("Segmentation Preview")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)  # Make non-modal so main window is accessible
            
            layout = QVBoxLayout()
            
            # Matplotlib figure
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # Add navigation toolbar for zoom, pan, etc.
            toolbar = NavigationToolbar(canvas, preview_dialog)
            # Style toolbar buttons to be white/more visible
            toolbar.setStyleSheet(
                "QToolButton {"
                "color: white;"
                "background-color: #5D6D7E;"
                "border: 1px solid #34495E;"
                "border-radius: 3px;"
                "padding: 4px;"
                "}"
                "QToolButton:hover {"
                "background-color: #34495E;"
                "}"
            )
            layout.addWidget(toolbar)
            
            # Store data for navigation
            preview_dialog.images = images
            preview_dialog.current_frame = 0
            preview_dialog.masks = None
            preview_dialog.model = None
            
            # Status label for model loading messages (defined early so load_model can use it)
            status_label = QLabel("Ready")
            status_label.setStyleSheet("color: #7F8C8D; font-size: 9pt; font-style: italic;")
            
            def find_model_file(model_type, model_index=0, use_torch=True):
                """
                Find model file in the default cache location.
                Models are cached in ~/.cellpose/models and persist between sessions.
                Returns the path to the model file if found, None otherwise.
                """
                from pathlib import Path
                from cellpose_omni.models import MODEL_DIR
                
                # Construct model filename
                torch_str = 'torch' if use_torch else ''
                basename = f'{model_type}{torch_str}_{model_index}'
                
                # Check default cache location
                cache_model_file = Path(MODEL_DIR) / basename
                
                if cache_model_file.exists():
                    return str(cache_model_file)
                
                # Not found
                return None
            
            def load_model():
                """Load Omnipose model (only reloads if model type or GPU setting changed)"""
                # Get current parameters
                params = get_current_parameters()
                current_model_type = params['model_type']
                current_use_gpu = params['use_gpu']
                
                # Check if model needs to be reloaded (only when model type or GPU changes)
                needs_reload = (preview_dialog.model is None or 
                               (hasattr(preview_dialog, '_last_model_type') and preview_dialog._last_model_type != current_model_type) or
                               (hasattr(preview_dialog, '_last_use_gpu') and preview_dialog._last_use_gpu != current_use_gpu))
                
                if needs_reload:
                    # Clear cache when model changes (old results are invalid)
                    preview_dialog.segmentation_cache.clear()
                
                if needs_reload:
                    try:
                        from cellpose_omni.models import model_path, MODEL_DIR
                        import os
                        from urllib.error import HTTPError
                        from pathlib import Path
                        
                        # Verify MODEL_DIR is set correctly (should be local models directory if env var was set)
                        # This is just for debugging - MODEL_DIR is set when module is imported
                        expected_model_dir = Path(__file__).parent / 'models'
                        if expected_model_dir.exists() and str(MODEL_DIR) != str(expected_model_dir):
                            # Environment variable might not have been set before import
                            # This shouldn't happen, but log it if it does
                            pass
                        
                        # Check for model in local directory first, then cache
                        model_file = find_model_file(current_model_type, 0, True)
                        
                        if model_file:
                            # Model found in cache
                            status_label.setText("Loading model from cache...")
                            QApplication.processEvents()
                        else:
                            # Model not found - need to download
                            status_label.setText("Downloading model (this may take a moment)...")
                            QApplication.processEvents()
                            # Only call model_path if we need to download (it will check and download if needed)
                            try:
                                model_file = model_path(current_model_type, 0, True)
                            except HTTPError as http_err:
                                # Server error during download - check local directory one more time
                                model_file = find_model_file(current_model_type, 0, True)
                                if model_file:
                                    status_label.setText("Download failed, using local/cached model...")
                                    QApplication.processEvents()
                                else:
                                    # Re-raise to be handled by outer exception handler
                                    raise
                        
                        preview_dialog.model = models.Cellpose(
                            gpu=current_use_gpu, 
                            model_type=current_model_type, 
                            omni=True
                        )
                        
                        # CRITICAL: Patch omnipose.core.get_masks to handle bd=None
                        # In frozen builds, the boundary field (bd) can be None if model files
                        # aren't found, causing "None > -1" errors. This patch provides a fallback.
                        try:
                            import omnipose.core as oc
                            import sys
                            import numpy as np
                            
                            # First try the global patch function
                            is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                            if is_frozen:
                                _patch_omnipose_get_masks()
                            
                            # Also apply local patch if global one didn't work
                            if is_frozen and not hasattr(oc.get_masks, '_bactimate_patched'):
                                _orig_get_masks = oc.get_masks
                                
                                def get_masks_safe(*args, **kwargs):
                                    """
                                    Patched get_masks that handles bd=None in frozen builds.
                                    The boundary field (bd) can be None, causing "None > -1" errors.
                                    This patch provides a fallback array.
                                    """
                                    # bd is typically the second positional argument (index 1)
                                    if len(args) >= 2:
                                        bd = args[1]
                                        if bd is None:
                                            # Need to create a fallback array with the right shape
                                            # Try to get shape from dist (3rd arg), mask (4th arg), or p (1st arg)
                                            shape = None
                                            
                                            # Try dist field (dt_pad) - 3rd argument
                                            if len(args) >= 3 and args[2] is not None and hasattr(args[2], 'shape'):
                                                shape = args[2].shape
                                            # Try mask field (iscell_pad) - 4th argument  
                                            elif len(args) >= 4 and args[3] is not None and hasattr(args[3], 'shape'):
                                                shape = args[3].shape
                                            # Try p field - 1st argument
                                            elif len(args) >= 1 and args[0] is not None and hasattr(args[0], 'shape'):
                                                shape = args[0].shape
                                            
                                            # ALWAYS create a fallback if bd is None, even if shape detection fails
                                            if shape is not None:
                                                # Create fallback array with values < -1 so (bd > -1) is False everywhere
                                                # This effectively disables boundary-based suppression
                                                bd = np.full(shape, -2, dtype=np.float32)
                                            else:
                                                # Last resort: try to infer shape from any available array
                                                # This should rarely happen, but better than crashing
                                                for arg in args:
                                                    if arg is not None and hasattr(arg, 'shape') and len(arg.shape) >= 2:
                                                        shape = arg.shape
                                                        bd = np.full(shape, -2, dtype=np.float32)
                                                        break
                                                
                                                # If still no shape, create a minimal fallback (shouldn't happen in practice)
                                                if bd is None:
                                                    # This is a last resort - create a small array
                                                    # The original function will likely fail anyway, but at least we tried
                                                    bd = np.array([[-2.0]], dtype=np.float32)
                                            
                                            # Replace bd in args - CRITICAL: must do this even if shape detection failed
                                            args = list(args)
                                            args[1] = bd
                                            args = tuple(args)
                                    
                                    # Final safety check: ensure bd is never None before calling original function
                                    if len(args) >= 2 and args[1] is None:
                                        # This should never happen, but if it does, create a minimal fallback
                                        args = list(args)
                                        args[1] = np.array([[-2.0]], dtype=np.float32)
                                        args = tuple(args)
                                    
                                    return _orig_get_masks(*args, **kwargs)
                                
                                # Mark as patched to avoid recursion
                                get_masks_safe._bactimate_patched = True
                                oc.get_masks = get_masks_safe
                        except Exception:
                            # If patching fails, continue anyway
                            pass
                        
                        preview_dialog._last_model_type = current_model_type
                        preview_dialog._last_use_gpu = current_use_gpu
                        status_label.setText("Model loaded")
                        QApplication.processEvents()
                    except RuntimeError as e:
                        error_msg = str(e)
                        # Check if it's a model architecture mismatch
                        if "size mismatch" in error_msg or "state_dict" in error_msg or "checkpoint" in error_msg.lower() or "Unexpected key" in error_msg:
                            # Check if there's a corrupted or wrong model file (e.g., cp3 symlinked as omni)
                            from cellpose_omni.models import model_path, MODEL_DIR
                            import os
                            from pathlib import Path
                            expected_basename = f'{current_model_type}torch_0'
                            model_file_path = Path(MODEL_DIR) / expected_basename
                            
                            # Check if model file is a symlink pointing to wrong model (e.g., cp3 instead of omni)
                            if model_file_path.exists() and model_file_path.is_symlink():
                                try:
                                    target = model_file_path.readlink()
                                    # If symlink points to cp3 model but we need omni, remove it
                                    if '_cp3' in str(target) and '_omni' in current_model_type:
                                        model_file_path.unlink()
                                        status_label.setText("Removed incorrect model symlink...")
                                        QApplication.processEvents()
                                except:
                                    pass
                            
                            # Try to fix by clearing model cache and retrying
                            try:
                                model_file = model_path(current_model_type, 0, True)
                                if os.path.exists(model_file):
                                    status_label.setText("Fixing model architecture mismatch...")
                                    QApplication.processEvents()
                                    # Backup the corrupted model
                                    backup_file = model_file + ".corrupted"
                                    if os.path.exists(backup_file):
                                        os.remove(backup_file)
                                    os.rename(model_file, backup_file)
                                    # Retry loading model (will download if needed)
                                    status_label.setText("Re-loading model...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_type = current_model_type
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Model reloaded successfully!")
                                    QApplication.processEvents()
                                else:
                                    raise RuntimeError("Model file not found after clearing cache")
                            except Exception as retry_error:
                                error_details = f"Original error: {error_msg}\nRetry error: {retry_error}"
                                # Provide helpful error message
                                help_text = (
                                    f"The model checkpoint has a different architecture than expected.\n\n"
                                    f"{error_details}\n\n"
                                )
                                
                                # Check if user has cp3 models that might be causing confusion
                                script_dir = Path(__file__).parent
                                local_models_dir = script_dir / 'models'
                                has_cp3 = (local_models_dir / 'bact_phase_cp3').exists() if local_models_dir.exists() else False
                                
                                if has_cp3 and '_omni' in current_model_type:
                                    help_text += (
                                        f"⚠️ IMPORTANT: You have 'bact_phase_cp3' in your models directory, "
                                        f"but this is a Cellpose model, NOT an Omnipose model.\n"
                                        f"They have different architectures and cannot be used interchangeably.\n\n"
                                    )
                                
                                help_text += (
                                    f"Please try:\n"
                                    f"1. Wait for the model server to come back online (currently HTTP 500 error)\n"
                                    f"2. Try a different model type (e.g., 'cyto2_omni' if available)\n"
                                    f"3. If you have cp3 models, use them with Cellpose model types, not Omnipose\n"
                                    f"4. Delete corrupted models: rm -rf {MODEL_DIR}/*.corrupted\n"
                                    f"5. Restart the application"
                                )
                                
                                QMessageBox.critical(
                                    preview_dialog, 
                                    "Model Architecture Mismatch",
                                    help_text
                                )
                                status_label.setText("Model load failed")
                                return False
                        else:
                            QMessageBox.critical(preview_dialog, "Model Error", 
                                               f"Could not load model:\n{error_msg}")
                            status_label.setText("Model load failed")
                        return False
                    except HTTPError as http_err:
                        # Handle HTTP errors (network issues when downloading models)
                        error_msg = str(http_err)
                        # Check for model in local directory or cache without triggering another download
                        model_file = find_model_file(current_model_type, 0, True)
                        cached_model_exists = (model_file is not None)
                        
                        if cached_model_exists:
                                # Try to use cached model even if download failed
                                try:
                                    status_label.setText("Trying cached model...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_type = current_model_type
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Using cached model")
                                    QApplication.processEvents()
                                    return True  # Success with cached model
                                except Exception as cache_error:
                                    # Cached model also failed
                                    QMessageBox.warning(
                                        preview_dialog, 
                                        "Network Error & Cache Issue",
                                        f"Could not download model (server error) and cached model also failed:\n\n"
                                        f"Download error: {error_msg}\n"
                                        f"Cache error: {cache_error}\n\n"
                                        f"Please try:\n"
                                        f"1. Wait a few minutes and try again (server may be temporarily down)\n"
                                        f"2. Try a different model type\n"
                                        f"3. Check your internet connection\n"
                                        f"4. If persistent, clear cache: rm -rf ~/.cellpose/models/"
                                    )
                                    status_label.setText("Network & cache error")
                                    return False
                        else:
                            # No cached model available
                            QMessageBox.warning(
                                preview_dialog, 
                                "Network Error - Model Server Unavailable",
                                f"Could not download model due to network error:\n\n{error_msg}\n\n"
                                f"The model server appears to be temporarily unavailable (HTTP 500).\n\n"
                                f"Please try:\n"
                                f"1. Wait 5-10 minutes and try again (server may be temporarily down)\n"
                                f"2. Check your internet connection\n"
                                f"3. Try a different model type (may already be cached)\n"
                                f"4. Check the Cellpose website status: https://www.cellpose.org\n\n"
                                f"Note: Once downloaded, models are cached locally and won't need to be re-downloaded."
                            )
                            status_label.setText("Server unavailable - try again later")
                            return False
                    except Exception as e:
                        error_msg = str(e)
                        # Handle other HTTP/network errors that might be wrapped
                        if "HTTP" in error_msg or "500" in error_msg or "connection" in error_msg.lower() or "network" in error_msg.lower() or "urllib" in error_msg.lower() or "download" in error_msg.lower():
                            # Check for model in local directory or cache without triggering another download
                            model_file = find_model_file(current_model_type, 0, True)
                            cached_model_exists = (model_file is not None)
                            
                            if cached_model_exists:
                                # Try to use cached model even if download failed
                                try:
                                    status_label.setText("Trying cached model...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_type = current_model_type
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Using cached model")
                                    QApplication.processEvents()
                                    return True  # Success with cached model
                                except Exception as cache_error:
                                    # Cached model also failed
                                    QMessageBox.warning(
                                        preview_dialog, 
                                        "Network Error & Cache Issue",
                                        f"Could not download model (server error) and cached model also failed:\n\n"
                                        f"Download error: {error_msg}\n"
                                        f"Cache error: {cache_error}\n\n"
                                        f"Please try:\n"
                                        f"1. Wait a few minutes and try again (server may be temporarily down)\n"
                                        f"2. Try a different model type\n"
                                        f"3. Check your internet connection\n"
                                        f"4. If persistent, clear cache: rm -rf ~/.cellpose/models/"
                                    )
                                    status_label.setText("Network & cache error")
                                    return False
                            else:
                                # No cached model available
                                QMessageBox.warning(
                                    preview_dialog, 
                                    "Network Error - Model Server Unavailable",
                                    f"Could not download model due to network error:\n\n{error_msg}\n\n"
                                    f"The model server appears to be temporarily unavailable (HTTP 500).\n\n"
                                    f"Please try:\n"
                                    f"1. Wait 5-10 minutes and try again (server may be temporarily down)\n"
                                    f"2. Check your internet connection\n"
                                    f"3. Try a different model type (may already be cached)\n"
                                    f"4. Check the Cellpose website status: https://www.cellpose.org\n\n"
                                    f"Note: Once downloaded, models are cached locally and won't need to be re-downloaded."
                                )
                                status_label.setText("Server unavailable - try again later")
                                return False
                        else:
                            QMessageBox.critical(preview_dialog, "Model Error", 
                                               f"Could not load model:\n{error_msg}")
                            status_label.setText("Model load failed")
                        return False
                return True
            
            def get_current_parameters():
                """Get current parameters from main window (works because dialog is non-modal)"""
                # CRITICAL: In frozen builds, diameter=0 (auto-detect) causes None comparison errors
                # in omnipose/core.py. Use a default diameter instead of 0.
                import sys
                is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                
                diameter_value = self.diameter.value()
                model_type = self.model_combo.currentText()
                
                # If diameter is 0 and we're in a frozen build, use a model-specific default
                if diameter_value == 0 and is_frozen:
                    # Default diameters based on model type (common values for bacteria)
                    default_diameters = {
                        'bact_phase_omni': 30.0,
                        'bact_omni': 30.0,
                        'bact_fluor_omni': 30.0,
                        'cyto2_omni': 30.0,
                        'cyto_omni': 30.0,
                        'worm_omni': 30.0,
                        'nuclei_omni': 20.0,
                    }
                    diameter_value = default_diameters.get(model_type, 30.0)
                
                return {
                    'diameter': diameter_value if diameter_value > 0 else 0,  # Use 0 instead of None
                    'flow_threshold': self.flow_threshold.value(),
                    'cluster': self.cluster.isChecked(),
                    'model_type': model_type,
                    'use_gpu': self.use_gpu.isChecked(),
                }
            
            # Cache for segmentation results: key = (frame_idx, diameter, flow_threshold, cluster, model_type, use_gpu)
            preview_dialog.segmentation_cache = {}
            
            def get_cache_key(frame_idx, params):
                """Generate cache key from frame index and parameters"""
                # Ensure diameter is always a number (0 if None) for cache key consistency
                diameter = params.get('diameter', 0)
                if diameter is None or not isinstance(diameter, (int, float)):
                    diameter = 0.0
                diameter = float(diameter)
                
                flow_threshold = params.get('flow_threshold', 0.4)
                if flow_threshold is None or not isinstance(flow_threshold, (int, float)):
                    flow_threshold = 0.4
                flow_threshold = float(flow_threshold)
                
                return (frame_idx, diameter, flow_threshold, 
                       bool(params.get('cluster', False)), 
                       str(params.get('model_type', 'bact_phase_omni')), 
                       bool(params.get('use_gpu', True)))
            
            # Segmentation worker thread class
            class SegmentationWorker(QThread):
                finished = pyqtSignal(object, object)  # mask, error
                
                def __init__(self, model, frame, diameter, flow_threshold, cluster):
                    super().__init__()
                    self.model = model
                    self.frame = frame
                    self.diameter = diameter
                    self.flow_threshold = flow_threshold
                    self.cluster = cluster
                
                def run(self):
                    try:
                        # CRITICAL: Ensure all parameters are valid numbers, not None
                        # This prevents comparison errors in model.eval() which expects numbers
                        diameter = self.diameter
                        if diameter is None or not isinstance(diameter, (int, float)):
                            diameter = 0
                        diameter = float(diameter)
                        
                        flow_threshold = self.flow_threshold
                        if flow_threshold is None or not isinstance(flow_threshold, (int, float)):
                            flow_threshold = 0.4
                        flow_threshold = float(flow_threshold)
                        
                        mask, flows, styles, diams = self.model.eval(
                            self.frame,
                            channels=[0, 0],
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            mask_threshold=0.0,  # Explicitly set mask_threshold
                            cellprob_threshold=0.0,
                            do_3D=False,
                            omni=True,
                            cluster=self.cluster,
                            net_avg=False,  # Faster, omni models don't support net_avg anyway
                            resample=True,
                            verbose=False
                        )
                        self.finished.emit(mask, None)
                    except Exception as e:
                        self.finished.emit(None, str(e))
            
            def run_segmentation(frame_idx, use_cache=True):
                """Run segmentation on a specific frame with caching and background threading"""
                # Get current parameters
                params = get_current_parameters()
                
                # CRITICAL: Clean parameters IMMEDIATELY - ensure no None values anywhere
                # This must happen before cache key generation or any other operations
                if params.get('diameter') is None or not isinstance(params.get('diameter'), (int, float)):
                    params['diameter'] = 0.0
                else:
                    params['diameter'] = float(params['diameter'])
                
                if params.get('flow_threshold') is None or not isinstance(params.get('flow_threshold'), (int, float)):
                    params['flow_threshold'] = 0.4
                else:
                    params['flow_threshold'] = float(params['flow_threshold'])
                
                # Now generate cache key with clean parameters
                cache_key = get_cache_key(frame_idx, params)
                
                # Check cache first (safely - handle any old cache entries with None values)
                if use_cache:
                    try:
                        if cache_key in preview_dialog.segmentation_cache:
                            return preview_dialog.segmentation_cache[cache_key]
                    except (TypeError, ValueError) as e:
                        # Cache key comparison failed (likely old entries with None values)
                        # Clean cache and continue
                        preview_dialog.segmentation_cache.clear()
                
                # Load model if needed (only when model type or GPU setting changes)
                if not load_model():
                    return None
                
                frame = images[frame_idx]
                # Handle multi-channel: use first channel if needed
                if frame.ndim == 3:
                    frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                
                # Ensure frame is float32 for MPS compatibility (Apple Silicon GPU doesn't support float64)
                if frame.dtype != np.float32:
                    frame = frame.astype(np.float32)
                
                # Extract parameters (already validated and converted above)
                diameter = params['diameter']
                flow_threshold = params['flow_threshold']
                cluster = params['cluster']
                
                try:
                    # Final validation before calling model.eval()
                    # Double-check all parameters are valid numbers (defensive programming)
                    assert isinstance(diameter, (int, float)) and diameter is not None, f"Invalid diameter: {diameter}"
                    assert isinstance(flow_threshold, (int, float)) and flow_threshold is not None, f"Invalid flow_threshold: {flow_threshold}"
                    
                    # Run segmentation (synchronously for now, but we can make it async later)
                    # Pass all parameters explicitly to match working batch_process_tiff_sequence.py
                    # Use a wrapper that catches the None comparison error and retries with workaround
                    import sys
                    is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                    
                    if is_frozen:
                        # In frozen builds, wrap the call to handle the omnipose bug
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                mask, flows, styles, diams = preview_dialog.model.eval(
                                    frame,
                                    channels=[0, 0],
                                    diameter=diameter if diameter > 0 else 30.0,  # Never use 0 in frozen builds
                                    flow_threshold=flow_threshold,
                                    mask_threshold=0.0,
                                    cellprob_threshold=0.0,
                                    do_3D=False,
                                    omni=True,
                                    cluster=cluster,
                                    net_avg=False,
                                    resample=True,
                                    verbose=False
                                )
                                break  # Success, exit retry loop
                            except TypeError as e:
                                if "'>' not supported between instances of 'NoneType'" in str(e) or "'<' not supported between instances of 'NoneType'" in str(e):
                                    if attempt < max_retries - 1:
                                        # Try with a different diameter
                                        diameter = 30.0 + (attempt * 5.0)  # Try 30, 35, 40
                                        continue
                                    else:
                                        # Last attempt failed, raise the error
                                        raise
                                else:
                                    # Different TypeError, raise immediately
                                    raise
                    else:
                        # Not frozen build, use normal call
                        mask, flows, styles, diams = preview_dialog.model.eval(
                            frame,
                            channels=[0, 0],
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            mask_threshold=0.0,
                            cellprob_threshold=0.0,
                            do_3D=False,
                            omni=True,
                            cluster=cluster,
                            net_avg=False,
                            resample=True,
                            verbose=False
                        )
                    # Cache the result
                    preview_dialog.segmentation_cache[cache_key] = mask
                    return mask
                except RecursionError as e:
                    # Handle recursion errors (e.g., from broken runtime hooks)
                    import traceback
                    error_details = f"Recursion error detected:\n{str(e)}\n\n"
                    error_details += f"This may be caused by a bug in the frozen build.\n"
                    error_details += f"Try rebuilding the application.\n\n"
                    error_details += f"Parameters:\n  diameter: {diameter} (type: {type(diameter)})\n  flow_threshold: {flow_threshold} (type: {type(flow_threshold)})\n\n"
                    error_details += f"Traceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error - Recursion", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
                except TypeError as e:
                    # Handle the specific "'>' not supported between instances of 'NoneType' and 'int'" error
                    # This is a known bug in omnipose when running in frozen builds
                    if "'>' not supported between instances of 'NoneType'" in str(e) or "'<' not supported between instances of 'NoneType'" in str(e):
                        import sys
                        is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                        
                        # This error occurs in frozen builds (Nuitka) due to a bug in omnipose/core.py
                        # Try different diameter values as workaround
                        retry_diameters = [30.0, 25.0, 35.0, 20.0, 40.0] if is_frozen else []
                        
                        if retry_diameters:
                            for retry_diameter in retry_diameters:
                                if retry_diameter == diameter:
                                    continue  # Skip if we already tried this value
                                
                                try:
                                    QMessageBox.warning(preview_dialog, "Omnipose Bug Workaround", 
                                                      f"Encountered a known bug in omnipose (frozen build compatibility issue).\n"
                                                      f"Retrying with diameter={retry_diameter} pixels...")
                                    mask, flows, styles, diams = preview_dialog.model.eval(
                                        frame,
                                        channels=[0, 0],
                                        diameter=retry_diameter,
                                        flow_threshold=flow_threshold,
                                        mask_threshold=0.0,
                                        cellprob_threshold=0.0,
                                        do_3D=False,
                                        omni=True,
                                        cluster=cluster,
                                        net_avg=False,
                                        resample=True,
                                        verbose=False
                                    )
                                    # Cache the result with the new diameter
                                    cache_key_retry = get_cache_key(frame_idx, {'diameter': retry_diameter, 'flow_threshold': flow_threshold, 
                                                                              'cluster': cluster, 'model_type': params.get('model_type'), 
                                                                              'use_gpu': params.get('use_gpu')})
                                    preview_dialog.segmentation_cache[cache_key_retry] = mask
                                    return mask
                                except (TypeError, Exception) as e2:
                                    # If this diameter also fails, try next one
                                    if retry_diameter == retry_diameters[-1]:
                                        # Last attempt failed, show error
                                        import traceback
                                        error_details = f"Known omnipose bug in frozen builds.\n\n"
                                        error_details += f"Original error: {str(e)}\n\n"
                                        error_details += f"Tried diameters: {[diameter] + retry_diameters}\n"
                                        error_details += f"All attempts failed. Last error: {str(e2)}\n\n"
                                        error_details += f"Workaround: Please set a specific diameter value (e.g., 30 pixels) instead of 0 (auto-detect).\n\n"
                                        error_details += f"Full traceback:\n{traceback.format_exc()}"
                                        QMessageBox.critical(preview_dialog, "Segmentation Error - Omnipose Bug", 
                                                           f"Could not segment frame:\n{error_details}")
                                        return None
                                    continue
                        else:
                            # Not a frozen build, or no retry diameters
                            import traceback
                            error_details = f"TypeError in omnipose: {str(e)}\n\n"
                            error_details += f"This may be a bug in omnipose. Please try setting a specific diameter value.\n\n"
                            error_details += f"Traceback:\n{traceback.format_exc()}"
                            QMessageBox.critical(preview_dialog, "Segmentation Error", 
                                               f"Could not segment frame:\n{error_details}")
                            return None
                    else:
                        # Different TypeError, re-raise
                        raise
                except AssertionError as e:
                    # Parameter validation error
                    import traceback
                    error_details = f"Parameter validation error:\n{str(e)}\n\nParameters:\n  diameter: {diameter} (type: {type(diameter)})\n  flow_threshold: {flow_threshold} (type: {type(flow_threshold)})\n\nTraceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
                except Exception as e:
                    # Other errors - include full traceback for debugging
                    import traceback
                    error_details = f"{str(e)}\n\nParameters:\n  diameter: {diameter} (type: {type(diameter)})\n  flow_threshold: {flow_threshold} (type: {type(flow_threshold)})\n\nTraceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
            
            def update_display(frame_idx, show_progress=True, force_recalc=False):
                """Update the display with segmentation overlay"""
                # Check cache first
                params = get_current_parameters()
                cache_key = get_cache_key(frame_idx, params)
                has_cache = cache_key in preview_dialog.segmentation_cache and not force_recalc
                
                # Show progress dialog if we need to calculate
                progress_dialog = None
                if show_progress and not has_cache:
                    progress_dialog = QProgressDialog("Segmenting frame...", None, 0, 0, preview_dialog)
                    progress_dialog.setWindowTitle("Processing")
                    progress_dialog.setWindowModality(Qt.WindowModal)
                    progress_dialog.setAutoClose(False)
                    progress_dialog.setAutoReset(False)
                    progress_dialog.setCancelButton(None)  # Don't allow cancellation
                    progress_dialog.show()
                    QApplication.processEvents()
                
                # Disable buttons during processing
                if progress_dialog:
                    prev_btn.setEnabled(False)
                    next_btn.setEnabled(False)
                    resegment_btn.setEnabled(False)
                    clear_cache_btn.setEnabled(False)
                    frame_spin.setEnabled(False)
                
                try:
                    frame = images[frame_idx]
                    # Handle multi-channel: use first channel if needed
                    if frame.ndim == 3:
                        display_frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                    else:
                        display_frame = frame
                    
                    # Update progress message
                    if progress_dialog:
                        progress_dialog.setLabelText("Running segmentation...")
                        QApplication.processEvents()
                    
                    # Run segmentation (with caching)
                    mask = run_segmentation(frame_idx, use_cache=not force_recalc)
                    if mask is None:
                        if progress_dialog:
                            progress_dialog.close()
                        status_label.setText("Segmentation failed")
                        status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
                        return
                    
                    preview_dialog.masks = mask
                    
                    # Update progress message
                    if progress_dialog:
                        progress_dialog.setLabelText("Rendering display...")
                        QApplication.processEvents()
                    
                    # Clear and redraw
                    ax.clear()
                    
                    # Normalize image for display
                    img_norm = (display_frame - display_frame.min()) / (display_frame.max() - display_frame.min() + 1e-10)
                    ax.imshow(img_norm, cmap='gray', alpha=0.7)
                    
                    # Overlay mask
                    if mask.max() > 0:
                        # Create colored overlay
                        from matplotlib.colors import ListedColormap
                        mask_overlay = np.ma.masked_where(mask == 0, mask)
                        ax.imshow(mask_overlay, cmap='gist_ncar', alpha=0.5, interpolation='nearest')
                    
                    file_name = Path(file_path).name
                    num_cells = len(np.unique(mask)) - 1
                    ax.set_title(f"Frame {frame_idx + 1} of {num_frames} | Cells detected: {num_cells} | {file_name}")
                    ax.axis('off')
                    canvas.draw()
                    
                    # Close progress dialog
                    if progress_dialog:
                        progress_dialog.close()
                    
                    # Re-enable buttons
                    prev_btn.setEnabled(True)
                    next_btn.setEnabled(True)
                    resegment_btn.setEnabled(True)
                    clear_cache_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    
                    # Update status
                    cache_size = len(preview_dialog.segmentation_cache)
                    if has_cache:
                        status_label.setText(f"Ready (from cache)")
                    else:
                        status_label.setText(f"Ready (cached: {cache_size})")
                    status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
                except Exception as e:
                    if progress_dialog:
                        progress_dialog.close()
                    # Re-enable buttons on error
                    prev_btn.setEnabled(True)
                    next_btn.setEnabled(True)
                    resegment_btn.setEnabled(True)
                    clear_cache_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    status_label.setText(f"Error: {str(e)}")
                    status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
                    QMessageBox.warning(preview_dialog, "Preview Error", f"Could not update display:\n{str(e)}")
            
            # Frame navigation
            nav_layout = QHBoxLayout()
            prev_btn = QPushButton("Previous")
            prev_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame - 1))
            nav_layout.addWidget(prev_btn)
            
            frame_spin = QSpinBox()
            frame_spin.setRange(1, num_frames)
            frame_spin.setValue(1)
            nav_layout.addWidget(QLabel("Frame:"))
            nav_layout.addWidget(frame_spin)
            nav_layout.addWidget(QLabel(f"of {num_frames}"))
            
            next_btn = QPushButton("Next")
            next_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame + 1))
            nav_layout.addWidget(next_btn)
            
            # Add status label to navigation layout
            nav_layout.addStretch()
            nav_layout.addWidget(status_label)
            
            def go_to_frame(idx, force_recalc=False):
                if 0 <= idx < num_frames:
                    preview_dialog.current_frame = idx
                    # Block signals to prevent recursive calls
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(idx + 1)
                    frame_spin.blockSignals(False)
                    try:
                        # Check if we have cached result for this frame with current parameters
                        params = get_current_parameters()
                        cache_key = get_cache_key(idx, params)
                        has_cache = cache_key in preview_dialog.segmentation_cache
                        
                        # Only show progress if we need to recalculate
                        update_display(idx, show_progress=(not has_cache or force_recalc))
                    except Exception as e:
                        QMessageBox.warning(preview_dialog, "Preview Error", 
                                          f"Could not display frame {idx + 1}:\n{str(e)}")
            
            # Use editingFinished for when user finishes editing (presses Enter or leaves field)
            # and a debounced valueChanged for when using arrow keys
            def on_frame_changed():
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_frames:
                    go_to_frame(frame_num)
            
            frame_spin.editingFinished.connect(on_frame_changed)
            # Debounce valueChanged to avoid triggering while typing
            frame_spin._change_timer = QTimer()
            frame_spin._change_timer.setSingleShot(True)
            frame_spin._change_timer.timeout.connect(on_frame_changed)
            frame_spin.valueChanged.connect(lambda v: frame_spin._change_timer.start(300))
            
            nav_layout.addStretch()
            
            # Re-segment button (to update with current parameters from main window)
            resegment_btn = QPushButton("Re-segment")
            resegment_btn.setToolTip(
                "Re-run segmentation on the current frame using the current parameters from the main window.\n"
                "You can change parameters in the main window (it stays accessible) and click this button to see the effect.\n"
                "Results are cached - changing parameters and clicking again will use cached results if parameters match."
            )
            def on_resegment():
                # Clear cache for current frame to force recalculation
                params = get_current_parameters()
                cache_key = get_cache_key(preview_dialog.current_frame, params)
                if cache_key in preview_dialog.segmentation_cache:
                    del preview_dialog.segmentation_cache[cache_key]
                # Force recalculation with progress
                update_display(preview_dialog.current_frame, show_progress=True, force_recalc=True)
            resegment_btn.clicked.connect(on_resegment)
            nav_layout.addWidget(resegment_btn)
            
            # Clear cache button
            clear_cache_btn = QPushButton("Clear Cache")
            clear_cache_btn.setToolTip(
                "Clear all cached segmentation results.\n"
                "Use this if you want to force recalculation of all frames."
            )
            def on_clear_cache():
                preview_dialog.segmentation_cache.clear()
                status_label.setText("Cache cleared")
                status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
                QTimer.singleShot(2000, lambda: status_label.setText("Ready") or status_label.setStyleSheet("color: #00FF00; font-weight: bold;"))
            clear_cache_btn.clicked.connect(on_clear_cache)
            nav_layout.addWidget(clear_cache_btn)
            
            layout.addLayout(nav_layout)
            layout.addWidget(canvas)
            
            # Close loading progress dialog now that preview dialog is ready
            loading_progress.close()
            
            # Display first frame (will use cache if available)
            update_display(0, show_progress=True)
            
            # Show cache info in status
            def update_cache_status():
                cache_size = len(preview_dialog.segmentation_cache)
                if cache_size > 0:
                    status_label.setText(f"Ready (cached: {cache_size})")
                else:
                    status_label.setText("Ready")
            
            # Update cache status after display
            QTimer.singleShot(100, update_cache_status)
            
            # Add Cmd+W (or Ctrl+W) shortcut to close dialog
            close_shortcut = QShortcut(QKeySequence("Ctrl+W"), preview_dialog)
            close_shortcut.activated.connect(preview_dialog.close)
            # Also handle Cmd+W on macOS (Qt uses Ctrl modifier for Cmd on macOS)
            if sys.platform == 'darwin':
                close_shortcut_mac = QShortcut(QKeySequence("Meta+W"), preview_dialog)
                close_shortcut_mac.activated.connect(preview_dialog.close)
            
            # Close button
            button_box = QDialogButtonBox(QDialogButtonBox.Close)
            button_box.rejected.connect(preview_dialog.close)
            layout.addWidget(button_box)
            
            preview_dialog.setLayout(layout)
            preview_dialog.show()  # Use show() instead of exec_() to make it non-modal
            
        except ImportError as e:
            loading_progress.close()
            error_msg = str(e)
            # Handle specific import errors
            if 'torch.testing' in error_msg:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  "Preview requires torch.testing module.\n\n"
                                  "This is a known issue with frozen builds.\n"
                                  "The preview feature may not work in the packaged application.\n\n"
                                  "Try running from source: python omnipose_batch_gui.py")
            else:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  f"Preview requires additional libraries:\n\n{error_msg}\n\n"
                                  f"Install with: pip install matplotlib cellpose-omni")
        except Exception as e:
            loading_progress.close()
            error_msg = str(e)
            # Handle PyTorch RpcBackendOptions duplicate initialization error
            if 'RpcBackendOptions' in error_msg or 'generic_type' in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "PyTorch initialization error detected.\n\n"
                                  "This is a known issue with frozen builds and PyTorch.\n"
                                  "The preview feature may not work in the packaged application.\n\n"
                                  "Try:\n"
                                  "1. Restart the application\n"
                                  "2. Run from source: python omnipose_batch_gui.py\n"
                                  "3. Use the processing features instead of preview")
            # Handle pbr versioning error (mgen package version detection issue)
            elif 'Versioning for this project requires' in error_msg or 'pbr.version.VersionInfo' in error_msg or 'mgen' in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "Package versioning error detected.\n\n"
                                  "This is a known issue with frozen builds and the mgen package.\n"
                                  "The preview feature may not work in the packaged application.\n\n"
                                  "Try:\n"
                                  "1. Restart the application\n"
                                  "2. Run from source: python omnipose_batch_gui.py\n"
                                  "3. Use the processing features instead of preview\n\n"
                                  "This error is harmless and doesn't affect actual processing.")
            # Handle TorchScript source access error
            elif 'TorchScript requires source access' in error_msg or "Can't get source for" in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "TorchScript compilation error detected.\n\n"
                                  "This is a known issue with frozen builds and PyTorch's TorchScript.\n"
                                  "TorchScript requires access to original source files which aren't\n"
                                  "available in packaged applications.\n\n"
                                  "Try:\n"
                                  "1. Restart the application (the runtime hook should fix this)\n"
                                  "2. Run from source: python omnipose_batch_gui.py\n"
                                  "3. Use the processing features instead of preview\n\n"
                                  "Note: This has been fixed in the latest build. Please rebuild the app.")
            else:
                QMessageBox.critical(self, "Preview Error", f"Could not preview segmentation:\n{error_msg}")
            import traceback
            traceback.print_exc()
    
    def process_file_or_folder(self):
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a TIFF file or folder first.")
            return
        
        if not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return
        
        script_dir = Path(__file__).parent
        is_folder = self._is_folder_mode()
        
        if not is_folder:
            # Single file mode
            script_path = str(script_dir / "batch_process_tiff_sequence.py")
            cmd_args = [
                self.file_path.text(),
                '--model', self.model_combo.currentText(),
                '--diameter', str(self.diameter.value()),
                '--flow-threshold', str(self.flow_threshold.value()),
            ]
            
            if not self.use_gpu.isChecked():
                cmd_args.append('--no-gpu')
            if self.cluster.isChecked():
                cmd_args.append('--cluster')
            
            self.process_btn.setEnabled(False)
            self.process_btn.setText("Processing...")
            self.parent_window.log_output(f"Starting single file processing: {self.file_path.text()}\n")
        else:
            # Folder mode - only segmentation, no analysis
            script_path = str(script_dir / "batch_process_folder.py")
            cmd_args = [
                self.file_path.text(),
                '--model', self.model_combo.currentText(),
                '--diameter', str(self.diameter.value()),
                '--flow-threshold', str(self.flow_threshold.value()),
            ]
            
            if not self.use_gpu.isChecked():
                cmd_args.append('--no-gpu')
            if self.cluster.isChecked():
                cmd_args.append('--cluster')
            
            self.process_btn.setEnabled(False)
            self.process_btn.setText("Processing...")
            self.parent_window.log_output(f"Starting batch folder processing (segmentation only): {self.file_path.text()}\n")
        
        # Show progress bar
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self.parent_window.log_output)
        self.worker.start()
    
    def on_finished(self, success, message):
        self.process_btn.setEnabled(True)
        is_folder = self._is_folder_mode()
        if is_folder:
            self.process_btn.setText("Process All Files in Folder")
        else:
            self.process_btn.setText("Process File")
        self.parent_window.log_output(message)
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()  # Wait for thread to finish
            self.worker = None
        # Hide progress bar
        self.parent_window.progress_bar.setVisible(False)
        if success:
            if is_folder:
                self.parent_window.log_output("✓ Folder processed successfully!\n")
            else:
                self.parent_window.log_output("✓ File processed successfully!\n")
        else:
            QMessageBox.critical(self, "Error", f"Processing failed:\n{message}")


class AnalyzeMasksTab(QWidget):
    """Tab for analyzing existing masks"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # File/Folder selection
        file_group = QGroupBox("Mask File/Folder")
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a mask TIFF file or folder...")
        self.file_path.setMinimumWidth(500)  # Increased for better visibility
        # Set larger font for better readability
        file_font = QFont()
        file_font.setPointSize(11)
        self.file_path.setFont(file_font)
        # Fix size to minimum - prevent expansion
        size_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.file_path.setSizePolicy(size_policy)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)  # Fixed width for buttons
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)  # Fixed width for buttons
        folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.file_path, 0)  # No stretch - keep at minimum size
        file_layout.addWidget(file_btn, 0)  # No stretch for buttons
        file_layout.addWidget(folder_btn, 0)  # No stretch for buttons
        file_group.setLayout(file_layout)
        scroll_layout.addWidget(file_group)
        
        # Analysis parameters
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QVBoxLayout()
        
        pixel_layout = QHBoxLayout()
        pixel_label = QLabel("Pixel Size (µm):")
        pixel_info_icon = create_info_icon(
            "Pixel size in micrometers (µm) for area calculations.\n\n"
            "Default: 0.108 µm (or from ND2 metadata if available)\n"
            "Source: Hardcoded default, or extracted from ND2 file metadata\n\n"
            "This value is used to convert pixel areas to µm².\n"
            "If you loaded an ND2 file with pixel size metadata, this will be auto-filled.\n\n"
            "Common values:\n"
            "- 100x objective: ~0.108 µm/pixel\n"
            "- 60x objective: ~0.18 µm/pixel\n"
            "- 40x objective: ~0.27 µm/pixel\n\n"
            "Tip: Check your microscope settings or ND2 metadata for the correct value."
        , parent_widget=self)
        pixel_layout.addWidget(pixel_label)
        pixel_layout.addWidget(pixel_info_icon)
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.001, 10.0)
        # Initialize with shared metadata if available, otherwise default to 0.108
        pixel_size_from_meta = self.parent_window.shared_metadata.get('pixel_size')
        default_pixel_size = pixel_size_from_meta if pixel_size_from_meta is not None else 0.108
        self.pixel_size.setValue(default_pixel_size)
        self.pixel_size.setDecimals(3)
        self.pixel_size.setSingleStep(0.001)
        pixel_layout.addWidget(self.pixel_size)
        # Add indicator label for auto-filled values
        self.pixel_size_indicator = QLabel("")
        self.pixel_size_indicator.setStyleSheet("color: #4CAF50; font-size: 9pt;")
        pixel_layout.addWidget(self.pixel_size_indicator)
        pixel_layout.addStretch()
        analysis_layout.addLayout(pixel_layout)
        
        # Morphological filters
        morph_group = QGroupBox("Morphological Filters (Rod-Shaped Cells)")
        morph_layout = QVBoxLayout()
        
        aspect_min_layout = QHBoxLayout()
        aspect_min_label = QLabel("Min Aspect Ratio:")
        aspect_min_info_icon = create_info_icon(
            "Minimum aspect ratio (length/width) for rod-shaped cell filtering.\n\n"
            "Default: 1.5\n"
            "Source: Hardcoded default for rod-shaped cell detection\n\n"
            "Aspect ratio = major_axis_length / minor_axis_length\n"
            "Filters out circular or nearly circular objects.\n\n"
            "Typical values:\n"
            "- Rod-shaped bacteria: 2.0-5.0\n"
            "- Very elongated cells: 3.0-10.0\n"
            "- Less strict: 1.5-2.0\n\n"
            "Range: 1.0 to 20.0\n"
            "Tip: Rod-shaped cells typically have aspect ratio > 2.0"
        , parent_widget=self)
        aspect_min_layout.addWidget(aspect_min_label)
        aspect_min_layout.addWidget(aspect_min_info_icon)
        self.min_aspect_ratio = QDoubleSpinBox()
        self.min_aspect_ratio.setRange(1.0, 20.0)
        self.min_aspect_ratio.setValue(1.5)
        self.min_aspect_ratio.setDecimals(1)
        self.min_aspect_ratio.setSingleStep(0.1)
        aspect_min_layout.addWidget(self.min_aspect_ratio)
        aspect_min_layout.addStretch()
        morph_layout.addLayout(aspect_min_layout)
        
        aspect_max_layout = QHBoxLayout()
        aspect_max_label = QLabel("Max Aspect Ratio:")
        aspect_max_info_icon = create_info_icon(
            "Maximum aspect ratio (length/width) for rod-shaped cell filtering.\n\n"
            "Default: 10.0\n"
            "Source: Hardcoded default to filter out extremely elongated artifacts\n\n"
            "Filters out objects that are too elongated (likely artifacts or debris chains).\n\n"
            "Range: 1.0 to 50.0\n"
            "Tip: Most rod-shaped cells have aspect ratio < 10.0. Very high values may indicate artifacts."
        , parent_widget=self)
        aspect_max_layout.addWidget(aspect_max_label)
        aspect_max_layout.addWidget(aspect_max_info_icon)
        self.max_aspect_ratio = QDoubleSpinBox()
        self.max_aspect_ratio.setRange(1.0, 50.0)
        self.max_aspect_ratio.setValue(10.0)
        self.max_aspect_ratio.setDecimals(1)
        self.max_aspect_ratio.setSingleStep(0.1)
        aspect_max_layout.addWidget(self.max_aspect_ratio)
        aspect_max_layout.addStretch()
        morph_layout.addLayout(aspect_max_layout)
        
        ecc_layout = QHBoxLayout()
        ecc_label = QLabel("Min Eccentricity:")
        ecc_info_icon = create_info_icon(
            "Minimum eccentricity for rod-shaped cell filtering.\n\n"
            "Default: 0.5\n"
            "Source: Hardcoded default for rod-shaped cell detection\n\n"
            "Eccentricity measures how elongated an object is:\n"
            "- 0.0 = perfect circle\n"
            "- 1.0 = line (infinitely elongated)\n\n"
            "Filters out circular or nearly circular objects.\n\n"
            "Range: 0.0 to 1.0\n"
            "Typical values: 0.5-0.9 for rod-shaped cells\n"
            "Tip: Higher values (0.7-0.9) are more strict, filtering out rounder objects."
        , parent_widget=self)
        ecc_layout.addWidget(ecc_label)
        ecc_layout.addWidget(ecc_info_icon)
        self.min_eccentricity = QDoubleSpinBox()
        self.min_eccentricity.setRange(0.0, 1.0)
        self.min_eccentricity.setValue(0.5)
        self.min_eccentricity.setDecimals(2)
        self.min_eccentricity.setSingleStep(0.05)
        ecc_layout.addWidget(self.min_eccentricity)
        ecc_layout.addStretch()
        morph_layout.addLayout(ecc_layout)
        
        sol_layout = QHBoxLayout()
        sol_label = QLabel("Min Solidity:")
        sol_info_icon = create_info_icon(
            "Minimum solidity for rod-shaped cell filtering.\n\n"
            "Default: 0.7\n"
            "Source: Hardcoded default for filtering irregular shapes\n\n"
            "Solidity = area / convex_hull_area\n"
            "Measures how 'solid' or 'filled' an object is:\n"
            "- 1.0 = perfectly solid (no indentations)\n"
            "- Lower values = more irregular, indented shapes\n\n"
            "Filters out irregular shapes, debris, and poorly segmented objects.\n\n"
            "Range: 0.0 to 1.0\n"
            "Tip: Higher values (0.8-0.9) filter out more irregular shapes and debris."
        , parent_widget=self)
        sol_layout.addWidget(sol_label)
        sol_layout.addWidget(sol_info_icon)
        self.min_solidity = QDoubleSpinBox()
        self.min_solidity.setRange(0.0, 1.0)
        self.min_solidity.setValue(0.7)
        self.min_solidity.setDecimals(2)
        self.min_solidity.setSingleStep(0.05)
        sol_layout.addWidget(self.min_solidity)
        sol_layout.addStretch()
        morph_layout.addLayout(sol_layout)
        
        morph_group.setLayout(morph_layout)
        analysis_layout.addWidget(morph_group)
        
        # Size filters
        size_group = QGroupBox("Size Filters")
        size_layout = QVBoxLayout()
        
        max_area_layout = QHBoxLayout()
        max_area_label = QLabel("Max Area (µm², 0=no limit):")
        max_area_info_icon = create_info_icon(
            "Maximum cell area in square micrometers (µm²).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Filters out cells larger than this value (likely mis-segmented debris or clumps).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for bacteria: 10-50 µm²\n"
            "Range: 0 to 10000 µm²\n\n"
            "Tip: Use to filter out oversized objects that are likely artifacts."
        , parent_widget=self)
        max_area_layout.addWidget(max_area_label)
        max_area_layout.addWidget(max_area_info_icon)
        self.max_area = QDoubleSpinBox()
        self.max_area.setRange(0, 10000)
        self.max_area.setValue(0)
        self.max_area.setDecimals(1)
        self.max_area.setSingleStep(1.0)
        max_area_layout.addWidget(self.max_area)
        max_area_layout.addStretch()
        size_layout.addLayout(max_area_layout)
        
        max_width_layout = QHBoxLayout()
        max_width_label = QLabel("Max Width (µm, 0=no limit):")
        max_width_info_icon = create_info_icon(
            "Maximum cell width in micrometers (µm).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Width is measured as the minor axis (perpendicular to length).\n"
            "Filters out cells wider than this value (likely mis-segmented debris or clumps).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for rod-shaped bacteria: 0.5-2.0 µm\n"
            "Range: 0 to 100 µm\n\n"
            "Tip: Use to filter out objects that are too wide to be single cells."
        , parent_widget=self)
        max_width_layout.addWidget(max_width_label)
        max_width_layout.addWidget(max_width_info_icon)
        self.max_width = QDoubleSpinBox()
        self.max_width.setRange(0, 100)
        self.max_width.setValue(0)
        self.max_width.setDecimals(1)
        self.max_width.setSingleStep(0.1)
        max_width_layout.addWidget(self.max_width)
        max_width_layout.addStretch()
        size_layout.addLayout(max_width_layout)
        
        size_group.setLayout(size_layout)
        analysis_layout.addWidget(size_group)
        
        # Options
        detailed_layout = QHBoxLayout()
        self.detailed = QCheckBox("Include detailed per-cell data")
        self.detailed.setChecked(False)
        detailed_info_icon = create_info_icon(
            "Include detailed per-cell measurements in the Excel output.\n\n"
            "Default: Disabled (unchecked)\n"
            "Source: Optional output mode\n\n"
            "When enabled: Excel file includes a separate sheet with measurements for each cell:\n"
            "- Cell ID, area, width, length, aspect ratio, eccentricity, solidity, etc.\n\n"
            "When disabled: Only summary data (total area per frame) is included.\n\n"
            "Use when: You need individual cell measurements for further analysis.\n"
            "Note: Creates larger Excel files but provides more detailed information.",
            parent_widget=self
        )
        detailed_layout.addWidget(self.detailed)
        detailed_layout.addWidget(detailed_info_icon)
        detailed_layout.addStretch()
        analysis_layout.addLayout(detailed_layout)
        
        vis_layout = QHBoxLayout()
        self.create_vis = QCheckBox("Create filter visualization")
        self.create_vis.setChecked(True)
        vis_info_icon = create_info_icon(
            "Create visualization TIFF showing which cells were accepted/rejected.\n\n"
            "Default: Enabled (checked)\n"
            "Source: Recommended for quality control\n\n"
            "When enabled: Creates a TIFF file where:\n"
            "- Green = accepted cells (passed all filters)\n"
            "- Red = rejected cells (failed one or more filters)\n\n"
            "Saved as: filename_filter_visualization.tif\n\n"
            "Use to: Verify that filters are working correctly and adjust parameters if needed.\n"
            "Note: Increases processing time but very useful for validation.",
            parent_widget=self
        )
        vis_layout.addWidget(self.create_vis)
        vis_layout.addWidget(vis_info_icon)
        vis_layout.addStretch()
        analysis_layout.addLayout(vis_layout)
        
        # Reset to defaults button for analysis parameters
        reset_analysis_btn = QPushButton("Reset to Default")
        reset_analysis_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #7D8C9E;"
            "color: white;"
            "font-weight: bold;"
            "padding: 6px;"
            "border: none;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "background-color: #6C7B8C;"
            "}"
        )
        reset_analysis_btn.setToolTip("Reset all analysis parameters to their default values")
        reset_analysis_btn.clicked.connect(self.reset_analysis_defaults)
        analysis_layout.addWidget(reset_analysis_btn)
        
        analysis_group.setLayout(analysis_layout)
        scroll_layout.addWidget(analysis_group)
        
        # Analyze button
        # Preview button
        preview_btn = QPushButton("Preview Filter Effects")
        preview_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #5D6D7E;"
            "color: white;"
            "font-weight: bold;"
            "padding: 8px;"
            "border: none;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "background-color: #5D6D7E;"
            "}"
        )
        preview_btn.setToolTip("Preview how current filter parameters would affect the selected mask file")
        preview_btn.clicked.connect(self.preview_filter_effects)
        scroll_layout.addWidget(preview_btn)
        
        self.analyze_btn = QPushButton("Analyze Masks")
        self.analyze_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #5D6D7E;"
            "color: white;"
            "font-weight: bold;"
            "padding: 10px;"
            "border: none;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "background-color: #34495E;"
            "}"
            "QPushButton:disabled {"
            "background-color: #7F8C8D;"
            "color: #BDC3C7;"
            "}"
        )
        self.analyze_btn.clicked.connect(self.analyze_masks)
        scroll_layout.addWidget(self.analyze_btn)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def select_file(self):
        """Open file dialog to select a mask TIFF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask TIFF File", "", "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.file_path.setText(file_path)
    
    def select_folder(self):
        """Open folder dialog to select a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Mask Files", ""
        )
        if folder_path:
            self.file_path.setText(folder_path)
    
    def _is_folder_mode(self):
        """Check if current path is a folder"""
        if not self.file_path.text():
            return False
        path = Path(self.file_path.text())
        return path.exists() and path.is_dir()
    
    def reset_analysis_defaults(self):
        """Reset all analysis parameters to their default values"""
        # Pixel size: use shared metadata if available, otherwise 0.108
        pixel_size_from_meta = self.parent_window.shared_metadata.get('pixel_size')
        default_pixel_size = pixel_size_from_meta if pixel_size_from_meta is not None else 0.108
        self.pixel_size.setValue(default_pixel_size)
        self.pixel_size_indicator.setText("")  # Clear indicator
        
        # Morphological filters
        self.min_aspect_ratio.setValue(1.5)
        self.max_aspect_ratio.setValue(10.0)
        self.min_eccentricity.setValue(0.5)
        self.min_solidity.setValue(0.7)
        
        # Size filters
        self.max_area.setValue(0)
        self.max_width.setValue(0)
        
        # Options
        self.detailed.setChecked(False)
        self.create_vis.setChecked(True)
    
    def preview_filter_effects(self):
        """Preview how current filter parameters would affect mask segmentation"""
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a mask file or folder first.")
            return
        
        if not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return
        
        is_folder = self._is_folder_mode()
        if is_folder:
            # Folder mode: show file selection dialog first
            folder = Path(self.file_path.text())
            # Look for mask files - use set to avoid duplicates
            mask_files_set = set()
            
            # First, collect all files with _masks in the name (preferred)
            mask_files_set.update(folder.glob("*_masks.tif"))
            mask_files_set.update(folder.glob("*_masks.tiff"))
            mask_files_set.update(folder.glob("*_masks.TIF"))
            mask_files_set.update(folder.glob("*_masks.TIFF"))
            
            # Then, collect other .tif files that don't have _masks in the name
            all_tif_files = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                           set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out files that already have _masks in name and filter visualization files
            for f in all_tif_files:
                if "_masks" not in f.name.lower() and "_filter_visualization" not in f.name.lower():
                    mask_files_set.add(f)
            
            # Convert to sorted list
            mask_files = sorted(mask_files_set, key=lambda x: x.name.lower())
            
            if not mask_files:
                QMessageBox.warning(self, "No Mask Files Found", 
                                  "No mask files found in the selected folder.\n\n"
                                  "Please select a folder containing mask TIFF files.")
                return
            
            # Let user choose which file to preview - use custom dialog
            file_dialog = QDialog(self)
            file_dialog.setWindowTitle("Select Mask File for Preview")
            file_dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout()
            
            # Label
            label = QLabel("Select a mask file to preview filter effects:")
            label.setStyleSheet("color: #ffffff; font-size: 11pt; padding: 10px;")
            layout.addWidget(label)
            
            # List widget with file names
            file_list = QListWidget()
            file_list.setStyleSheet(
                "QListWidget {"
                "background-color: #2b2b2b;"
                "color: #ffffff;"
                "font-size: 11pt;"
                "border: 1px solid #555555;"
                "border-radius: 4px;"
                "padding: 5px;"
                "}"
                "QListWidget::item {"
                "padding: 8px;"
                "border-bottom: 1px solid #3a3a3a;"
                "}"
                "QListWidget::item:selected {"
                "background-color: #5D6D7E;"
                "color: #ffffff;"
                "}"
                "QListWidget::item:hover {"
                "background-color: #3a3a3a;"
                "}"
            )
            
            # Add file names to list
            for f in mask_files:
                file_list.addItem(f.name)
            
            # Select first item by default
            if file_list.count() > 0:
                file_list.setCurrentRow(0)
            
            # Enable double-click to accept
            def on_item_double_clicked(item):
                file_dialog.accept()
            
            file_list.itemDoubleClicked.connect(on_item_double_clicked)
            
            layout.addWidget(file_list)
            
            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(file_dialog.accept)
            button_box.rejected.connect(file_dialog.reject)
            button_box.setStyleSheet(
                "QPushButton {"
                "background-color: #5D6D7E;"
                "color: white;"
                "font-weight: bold;"
                "padding: 8px 20px;"
                "border: none;"
                "border-radius: 4px;"
                "min-width: 80px;"
                "}"
                "QPushButton:hover {"
                "background-color: #34495E;"
                "}"
            )
            layout.addWidget(button_box)
            
            file_dialog.setLayout(layout)
            
            # Show dialog
            if file_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            
            # Get selected file
            selected_item = file_list.currentItem()
            if not selected_item:
                return
            
            selected_name = selected_item.text()
            selected_file = None
            for f in mask_files:
                if f.name == selected_name:
                    selected_file = f
                    break
            
            if not selected_file:
                return
            
            # Show preview dialog for selected file
            self._show_filter_preview(str(selected_file))
        else:
            # Single file mode: show preview directly
            self._show_filter_preview(self.file_path.text())
    
    def _show_filter_preview(self, mask_path):
        """Show preview dialog with filter effects"""
        # Show loading progress
        loading_progress = QProgressDialog("Loading mask file and initializing preview...", None, 0, 0, self)
        loading_progress.setWindowTitle("Loading")
        loading_progress.setWindowModality(Qt.WindowModal)
        loading_progress.setAutoClose(False)
        loading_progress.setAutoReset(False)
        loading_progress.setCancelButton(None)
        loading_progress.show()
        QApplication.processEvents()
        
        try:
            loading_progress.setLabelText("Loading mask file...")
            QApplication.processEvents()
            
            import tifffile
            from skimage.measure import regionprops
            from skimage.morphology import remove_small_objects
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure
            
            # Load masks
            masks = tifffile.imread(mask_path)
            
            loading_progress.setLabelText("Processing mask data...")
            QApplication.processEvents()
            if masks.ndim == 2:
                masks = masks[np.newaxis, :, :]
            num_frames = masks.shape[0]
            
            # Create preview dialog (non-modal so user can access main window)
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("Filter Effects Preview")
            preview_dialog.setMinimumSize(1000, 800)
            preview_dialog.setModal(False)  # Make non-modal so main window is accessible
            
            layout = QVBoxLayout()
            
            # Matplotlib figure
            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, preview_dialog)
            # Style toolbar buttons to be white/more visible
            toolbar.setStyleSheet(
                "QToolButton {"
                "color: white;"
                "background-color: #5D6D7E;"
                "border: 1px solid #34495E;"
                "border-radius: 3px;"
                "padding: 4px;"
                "}"
                "QToolButton:hover {"
                "background-color: #34495E;"
                "}"
            )
            
            layout.addWidget(toolbar)
            
            # Frame navigation (moved to top, below toolbar, to match segmentation preview)
            nav_layout = QHBoxLayout()
            prev_btn = QPushButton("Previous")
            nav_layout.addWidget(prev_btn)
            
            nav_layout.addWidget(QLabel("Frame:"))
            frame_spin = QSpinBox()
            frame_spin.setRange(1, num_frames)
            frame_spin.setValue(1)
            nav_layout.addWidget(frame_spin)
            nav_layout.addWidget(QLabel(f"of {num_frames}"))
            
            next_btn = QPushButton("Next")
            nav_layout.addWidget(next_btn)
            
            nav_layout.addStretch()
            
            layout.addLayout(nav_layout)
            layout.addWidget(canvas)
            
            # Info label will be added to nav_layout after reapply button
            info_label = QLabel("")
            info_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            
            # Import filter functions from analyze_cell_areas
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from analyze_cell_areas import calculate_cell_properties, filter_rod_shaped_cells, filter_by_size
            
            # Store current frame
            preview_dialog.current_frame = 0
            
            # Info label (will be added to nav_layout after reapply button)
            info_label = QLabel("")
            info_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            
            def get_current_parameters():
                """Get current filter parameters from main window (works because dialog is non-modal)"""
                return {
                    'pixel_size': self.pixel_size.value(),
                    'min_aspect_ratio': self.min_aspect_ratio.value(),
                    'max_aspect_ratio': self.max_aspect_ratio.value(),
                    'min_eccentricity': self.min_eccentricity.value(),
                    'min_solidity': self.min_solidity.value(),
                    'max_area': self.max_area.value() if self.max_area.value() > 0 else None,
                    'max_width': self.max_width.value() if self.max_width.value() > 0 else None,
                }
            
            # Function to update preview
            def update_preview(frame_idx, show_progress=True):
                if frame_idx < 0 or frame_idx >= num_frames:
                    return
                
                # Show progress dialog
                progress_dialog = None
                if show_progress:
                    progress_dialog = QProgressDialog("Applying filters...", None, 0, 0, preview_dialog)
                    progress_dialog.setWindowTitle("Processing")
                    progress_dialog.setWindowModality(Qt.WindowModal)
                    progress_dialog.setAutoClose(False)
                    progress_dialog.setAutoReset(False)
                    progress_dialog.setCancelButton(None)
                    progress_dialog.show()
                    QApplication.processEvents()
                
                # Disable buttons during processing
                if progress_dialog:
                    prev_btn.setEnabled(False)
                    next_btn.setEnabled(False)
                    reapply_btn.setEnabled(False)
                    frame_spin.setEnabled(False)
                
                try:
                    preview_dialog.current_frame = frame_idx
                    mask = masks[frame_idx]
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Cleaning mask...")
                        QApplication.processEvents()
                    
                    # Remove small objects
                    mask_cleaned = remove_small_objects(mask, min_size=10)
                    
                    # Get current parameters from main window
                    params = get_current_parameters()
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Calculating cell properties...")
                        QApplication.processEvents()
                    
                    # Calculate properties
                    properties = calculate_cell_properties(mask_cleaned, pixel_size=params['pixel_size'])
                    all_labels = {prop['label'] for prop in properties}
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Applying morphological filters...")
                        QApplication.processEvents()
                    
                    # Apply filters with current parameters
                    rod_shaped = filter_rod_shaped_cells(
                        properties,
                        min_aspect_ratio=params['min_aspect_ratio'],
                        max_aspect_ratio=params['max_aspect_ratio'],
                        min_eccentricity=params['min_eccentricity'],
                        min_solidity=params['min_solidity']
                    )
                    rod_shaped_labels = {cell['label'] for cell in rod_shaped}
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Applying size filters...")
                        QApplication.processEvents()
                    
                    filtered = filter_by_size(
                        rod_shaped,
                        max_area_um2=params['max_area'],
                        max_width_um=params['max_width']
                    )
                    accepted_labels = {cell['label'] for cell in filtered}
                    rejected_labels = all_labels - accepted_labels
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Creating visualization...")
                        QApplication.processEvents()
                    
                    # Create visualization
                    height, width = mask.shape
                    vis = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Green for accepted
                    for label in accepted_labels:
                        vis[mask == label, 1] = 255
                    
                    # Red for rejected
                    for label in rejected_labels:
                        vis[mask == label, 0] = 255
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Rendering display...")
                        QApplication.processEvents()
                    
                    # Update plot
                    fig.clear()
                    ax = fig.add_subplot(111)
                    ax.imshow(vis)
                    ax.set_title(f"Frame {frame_idx + 1}/{num_frames}: Green=Accepted ({len(accepted_labels)}), Red=Rejected ({len(rejected_labels)})")
                    ax.axis('off')
                    canvas.draw()
                    
                    info_label.setText(f"Accepted: {len(accepted_labels)} | Rejected: {len(rejected_labels)} | Total: {len(all_labels)}")
                    
                    # Close progress dialog
                    if progress_dialog:
                        progress_dialog.close()
                    
                    # Re-enable buttons
                    if progress_dialog:
                        prev_btn.setEnabled(True)
                        next_btn.setEnabled(True)
                        reapply_btn.setEnabled(True)
                        frame_spin.setEnabled(True)
                except Exception as e:
                    if progress_dialog:
                        progress_dialog.close()
                    # Re-enable buttons on error
                    prev_btn.setEnabled(True)
                    next_btn.setEnabled(True)
                    reapply_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    QMessageBox.warning(preview_dialog, "Preview Error", f"Could not update preview:\n{str(e)}")
            
            def go_to_frame(idx):
                if 0 <= idx < num_frames:
                    preview_dialog.current_frame = idx
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(idx + 1)
                    frame_spin.blockSignals(False)
                    try:
                        # Only show progress if it's a new frame (not cached)
                        update_preview(idx, show_progress=False)  # Filter preview is usually fast
                    except Exception as e:
                        QMessageBox.warning(preview_dialog, "Preview Error", 
                                          f"Could not display frame {idx + 1}:\n{str(e)}")
            
            prev_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame - 1))
            next_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame + 1))
            
            # Use editingFinished for when user finishes editing (presses Enter or leaves field)
            # and a debounced valueChanged for when using arrow keys
            def on_frame_changed():
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_frames:
                    go_to_frame(frame_num)
            
            frame_spin.editingFinished.connect(on_frame_changed)
            # Debounce valueChanged to avoid triggering while typing
            frame_spin._change_timer = QTimer()
            frame_spin._change_timer.setSingleShot(True)
            frame_spin._change_timer.timeout.connect(on_frame_changed)
            frame_spin.valueChanged.connect(lambda v: frame_spin._change_timer.start(300))
            
            nav_layout.addStretch()
            
            # Re-apply filters button (moved to match segmentation preview layout)
            reapply_btn = QPushButton("Re-apply Filters with Current Parameters")
            reapply_btn.setToolTip(
                "Re-apply filters on the current frame using the current parameters from the main window.\n"
                "You can change filter parameters in the main window (it stays accessible) and click this button to see the effect."
            )
            reapply_btn.setStyleSheet(
                "QPushButton {"
                "background-color: #5D6D7E;"
                "color: white;"
                "font-weight: bold;"
                "padding: 6px;"
                "border: none;"
                "border-radius: 4px;"
                "}"
                "QPushButton:hover {"
                "background-color: #34495E;"
                "}"
            )
            def on_reapply():
                update_preview(preview_dialog.current_frame, show_progress=True)
            reapply_btn.clicked.connect(on_reapply)
            nav_layout.addWidget(reapply_btn)
            
            # Info label (moved after reapply button to match segmentation preview)
            nav_layout.addWidget(info_label)
            
            # Close loading progress dialog now that preview dialog is ready
            loading_progress.close()
            
            # Initial preview with progress
            update_preview(0, show_progress=True)
            
            # Add Cmd+W (or Ctrl+W) shortcut to close dialog
            close_shortcut = QShortcut(QKeySequence("Ctrl+W"), preview_dialog)
            close_shortcut.activated.connect(preview_dialog.close)
            # Also handle Cmd+W on macOS (Qt uses Ctrl modifier for Cmd on macOS)
            if sys.platform == 'darwin':
                close_shortcut_mac = QShortcut(QKeySequence("Meta+W"), preview_dialog)
                close_shortcut_mac.activated.connect(preview_dialog.close)
            
            # Close button
            button_box = QDialogButtonBox(QDialogButtonBox.Close)
            button_box.rejected.connect(preview_dialog.close)
            layout.addWidget(button_box)
            
            preview_dialog.setLayout(layout)
            preview_dialog.show()  # Use show() instead of exec_() to make it non-modal
            
        except Exception as e:
            loading_progress.close()
            QMessageBox.critical(self, "Preview Error", f"Could not create preview:\n{str(e)}\n\n{traceback.format_exc()}")
    
    def analyze_masks(self):
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a mask file or folder first.")
            return
        
        if not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return
        
        script_dir = Path(__file__).parent
        script_path = str(script_dir / "analyze_cell_areas.py")
        
        is_folder = self._is_folder_mode()
        if is_folder:
            # Folder mode: process all mask files in the folder
            folder = Path(self.file_path.text())
            # Look for mask files - use set to avoid duplicates
            mask_files_set = set()
            
            # First, collect all files with _masks in the name (preferred)
            mask_files_set.update(folder.glob("*_masks.tif"))
            mask_files_set.update(folder.glob("*_masks.tiff"))
            mask_files_set.update(folder.glob("*_masks.TIF"))
            mask_files_set.update(folder.glob("*_masks.TIFF"))
            
            # Then, collect other .tif files that don't have _masks in the name
            all_tif_files = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                           set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out files that already have _masks in name and filter visualization files
            for f in all_tif_files:
                if "_masks" not in f.name.lower() and "_filter_visualization" not in f.name.lower():
                    mask_files_set.add(f)
            
            # Convert to sorted list
            mask_files = sorted(mask_files_set, key=lambda x: x.name.lower())
            
            if not mask_files:
                QMessageBox.warning(self, "No Mask Files Found", 
                                  "No mask files found in the selected folder.\n\n"
                                  "Please select a folder containing mask TIFF files.")
                return
            
            # Process each file sequentially
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText(f"Analyzing... (0/{len(mask_files)})")
            self.parent_window.log_output(f"Starting mask analysis for {len(mask_files)} file(s) in folder: {folder}\n")
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, len(mask_files))
            self.parent_window.progress_bar.setValue(0)
            
            # Store files and current index
            self.folder_files = mask_files
            self.current_file_index = 0
            self.folder_results = []
            
            # Process first file
            self._process_next_folder_file(script_path)
        else:
            # Single file mode
            # Build command arguments
            cmd_args = [
                self.file_path.text(),
                '--pixel-size', str(self.pixel_size.value()),
                '--min-aspect-ratio', str(self.min_aspect_ratio.value()),
                '--max-aspect-ratio', str(self.max_aspect_ratio.value()),
                '--min-eccentricity', str(self.min_eccentricity.value()),
                '--min-solidity', str(self.min_solidity.value()),
            ]
            
            if self.max_area.value() > 0:
                cmd_args.extend(['--max-area', str(self.max_area.value())])
            if self.max_width.value() > 0:
                cmd_args.extend(['--max-width', str(self.max_width.value())])
            if self.detailed.isChecked():
                cmd_args.append('--detailed')
            if self.create_vis.isChecked():
                cmd_args.append('--visualization')
            
            # Run in thread
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText("Analyzing...")
            self.parent_window.log_output(f"Starting mask analysis: {self.file_path.text()}\n")
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Store reference to prevent garbage collection
            self.worker = WorkerThread(script_path, cmd_args, parent=self)
            self.worker.finished.connect(self.on_finished)
            self.worker.progress.connect(self.parent_window.log_output)
            self.worker.start()
    
    def _process_next_folder_file(self, script_path):
        """Process next file in folder mode"""
        if self.current_file_index >= len(self.folder_files):
            # All files processed
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("Analyze Masks")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output(f"\n✓ Analysis complete for all {len(self.folder_files)} file(s)!\n")
            QMessageBox.information(self, "Analysis Complete", 
                                  f"Successfully analyzed {len(self.folder_files)} mask file(s).")
            return
        
        mask_file = self.folder_files[self.current_file_index]
        self.parent_window.log_output(f"\nProcessing file {self.current_file_index + 1}/{len(self.folder_files)}: {mask_file.name}\n")
        
        # Build command arguments
        cmd_args = [
            str(mask_file),
            '--pixel-size', str(self.pixel_size.value()),
            '--min-aspect-ratio', str(self.min_aspect_ratio.value()),
            '--max-aspect-ratio', str(self.max_aspect_ratio.value()),
            '--min-eccentricity', str(self.min_eccentricity.value()),
            '--min-solidity', str(self.min_solidity.value()),
        ]
        
        if self.max_area.value() > 0:
            cmd_args.extend(['--max-area', str(self.max_area.value())])
        if self.max_width.value() > 0:
            cmd_args.extend(['--max-width', str(self.max_width.value())])
        if self.detailed.isChecked():
            cmd_args.append('--detailed')
        if self.create_vis.isChecked():
            cmd_args.append('--visualization')
        
        # Update progress
        self.parent_window.progress_bar.setValue(self.current_file_index)
        self.analyze_btn.setText(f"Analyzing... ({self.current_file_index + 1}/{len(self.folder_files)})")
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(lambda success, msg: self._on_folder_file_finished(success, msg, script_path))
        self.worker.progress.connect(self.parent_window.log_output)
        self.worker.start()
    
    def _on_folder_file_finished(self, success, message, script_path):
        """Handle completion of a single file in folder mode"""
        self.parent_window.log_output(message)
        if success:
            self.folder_results.append(True)
            self.parent_window.log_output(f"✓ File {self.current_file_index + 1} completed successfully\n")
        else:
            self.folder_results.append(False)
            self.parent_window.log_output(f"✗ File {self.current_file_index + 1} failed: {message}\n")
        
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        
        # Process next file
        self.current_file_index += 1
        self._process_next_folder_file(script_path)
    
    def on_finished(self, success, message):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze Masks")
        self.parent_window.log_output(message)
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()  # Wait for thread to finish
            self.worker = None
        # Hide progress bar
        self.parent_window.progress_bar.setVisible(False)
        if success:
            self.parent_window.log_output("✓ Analysis completed successfully!\n")
        else:
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{message}")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        # Shared metadata storage (pixel size, etc.) that can be accessed by all tabs
        self.shared_metadata = {
            'pixel_size': None,  # Will be updated when ND2 file is loaded
        }
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("BactiMate")  # Short title for window bar only
        # Set minimum size for better usability on larger screens
        self.setMinimumSize(1000, 700)
        # Set default size (will be resizable)
        self.resize(1200, 900)
        # Center window on screen
        self._center_window()
        
        # Central widget with tabs
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Custom title bar with logo and text
        title_bar = QWidget()
        title_bar.setStyleSheet(
            "QWidget {"
            "background-color: #34495E;"
            "padding: 10px;"
            "border-bottom: 2px solid #2C3E50;"
            "}"
        )
        title_bar_layout = QHBoxLayout()
        title_bar_layout.setContentsMargins(15, 10, 15, 10)
        
        # Add stretch to center content
        title_bar_layout.addStretch()
        
        # Logo
        logo_label = QLabel()
        logo_path = Path(__file__).parent / "logo2.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            # Scale logo to fit title bar (max height 50px, maintain aspect ratio)
            if pixmap.height() > 100:
                pixmap = pixmap.scaledToHeight(100, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("BactiMate")
            logo_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: #ECF0F1;")
        logo_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        title_bar_layout.addWidget(logo_label)
        
        # Title text
        
        title_text = QLabel("BactiMate - your mate for bacteria time-lapse data")
        title_text.setStyleSheet(
            "QLabel {"
            "font-size: 22pt;"
            "font-weight: bold;"
            "color: #ECF0F1;"
            "padding-left: 15px;"
            "}"
        )
        title_text.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        title_bar_layout.addWidget(title_text)
        
        # Add stretch to center content
        title_bar_layout.addStretch()
        
        title_bar.setLayout(title_bar_layout)
        main_layout.addWidget(title_bar)
        
        # Create tabs
        tabs = QTabWidget()
        self.nd2_tab = ND2ConversionTab(self)
        self.processing_tab = ProcessingTab(self)
        self.analyze_tab = AnalyzeMasksTab(self)
        
        tabs.addTab(self.nd2_tab, "TIFF Conversion")
        tabs.addTab(self.processing_tab, "Segmentation")
        tabs.addTab(self.analyze_tab, "Analyze Masks")
        
        main_layout.addWidget(tabs)
        
        # Log output area - smaller portion (10% of window)
        log_group = QGroupBox("Status & Log")
        log_layout = QVBoxLayout()
        
        # Progress bar (hidden by default, shown during long operations)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")  # Show percentage
        self.progress_bar.setTextVisible(True)  # Show text
        self.progress_bar.setVisible(False)
        log_layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # Use a monospace font that exists on macOS
        monospace_font = QFont("Monaco", 11)  # Monaco is standard on macOS, increased size
        if not monospace_font.exactMatch():
            # Fallback to Courier New or system monospace
            monospace_font = QFont("Courier New", 11)
        self.log_text.setFont(monospace_font)
        self.log_text.setMinimumHeight(100)  # Increased for better visibility
        self.log_text.setMaximumHeight(150)  # Prevent excessive expansion
        # Prevent vertical expansion
        log_size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.log_text.setSizePolicy(log_size_policy)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Set stretch factors: tabs expand to fill space, log stays at minimum size
        main_layout.setStretchFactor(tabs, 1)  # Tabs expand to fill all available space
        main_layout.setStretchFactor(log_group, 0)  # Log stays at minimum size
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        self.log_output("Omnipose Batch Processing GUI started.\n")
        self.log_output("Select a tab and configure parameters to begin.\n\n")
    
    def _center_window(self):
        """Center the window on the screen (PyQt6: use QScreen instead of QDesktopWidget)"""
        frame_geometry = self.frameGeometry()
        # Get screen at cursor position, or primary screen as fallback
        cursor_pos = QCursor.pos()
        screen = QApplication.screenAt(cursor_pos)
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen:
            center_point = screen.geometry().center()
            frame_geometry.moveCenter(center_point)
            self.move(frame_geometry.topLeft())
    
    def log_output(self, message):
        """Append message to log output and update progress bar if needed"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Show progress bar for long operations
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in [
            'applying drift correction', 'applying bleach correction',
            'processing frame', 'segmenting', 'analyzing masks',
            'reading frames', 'extracting frames', 'processing file'
        ]):
            if not self.progress_bar.isVisible():
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate
    
    def update_all_tabs_with_metadata(self):
        """Update all tabs with shared metadata (e.g., pixel size from ND2)"""
        pixel_size = self.shared_metadata.get('pixel_size')
        
        if pixel_size is not None:
            # Update Processing tab
            if hasattr(self, 'processing_tab') and hasattr(self.processing_tab, 'pixel_size'):
                self.processing_tab.pixel_size.setValue(pixel_size)
                # Show indicator that value was auto-filled
                if hasattr(self.processing_tab, 'pixel_size_indicator'):
                    self.processing_tab.pixel_size_indicator.setText("(from ND2 metadata)")
                self.log_output(f"✓ Updated pixel size in Processing tab: {pixel_size:.4f} µm\n")
            
            # Update Analyze Masks tab
            if hasattr(self, 'analyze_tab') and hasattr(self.analyze_tab, 'pixel_size'):
                self.analyze_tab.pixel_size.setValue(pixel_size)
                # Show indicator that value was auto-filled
                if hasattr(self.analyze_tab, 'pixel_size_indicator'):
                    self.analyze_tab.pixel_size_indicator.setText("(from ND2 metadata)")
                self.log_output(f"✓ Updated pixel size in Analyze Masks tab: {pixel_size:.4f} µm\n")


def main():
    # Qt plugin paths should already be set by launch_gui.sh or module-level code
    # Set library paths programmatically as well (Qt reads this)
    if sys.platform == 'darwin':
        plugins_dir = os.environ.get('QT_PLUGIN_PATH')
        if plugins_dir:
            abs_plugins_dir = os.path.abspath(plugins_dir)
            # Ensure it's the plugins directory, not platforms subdirectory
            if abs_plugins_dir.endswith('platforms'):
                abs_plugins_dir = os.path.dirname(abs_plugins_dir)
            os.environ['QT_PLUGIN_PATH'] = abs_plugins_dir
            QCoreApplication.setLibraryPaths([abs_plugins_dir])
    
    try:
        app = QApplication(sys.argv)
        
        app.setStyle('Fusion')  # Modern look
        
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting GUI: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
