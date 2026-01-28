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
import json
from pathlib import Path
from datetime import datetime
import traceback
import warnings

# Session log storage: Logs folder next to the app
LOGS_DIR = Path(__file__).resolve().parent / "Logs"

# Suppress deprecation warnings from omnipose/pkg_resources
# This is harmless - omnipose uses pkg_resources which is deprecated in newer setuptools
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')
# Suppress FutureWarning from omnipose about deprecated area_threshold parameter
# This is from omnipose's internal code using deprecated scikit-image API
warnings.filterwarnings('ignore', category=FutureWarning, message='.*area_threshold.*')

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

# Try to patch immediately if in frozen/build environment
# Check for frozen executables or compiled builds
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
# Also handles frozen executables and app bundles

# Check if running as frozen/compiled executable or app bundle
if getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__'):
    # Running as frozen executable or app bundle
    # Use directory containing the executable
    base_path = os.path.dirname(sys.executable)
else:
    # Running as script
    base_path = os.path.dirname(os.path.abspath(__file__))

# Initialize variables
abs_plugin_path = None
abs_lib_dir = None

if sys.platform in ['darwin', 'win32']:
    # For frozen executables or app bundles, plugins are in the bundle
    if getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__'):
        # In app bundle or frozen executable, plugins are at PyQt6/Qt6/plugins relative to base_path
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
            os.path.join(script_dir, 'venv'),  # Python 3.11 venv
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
            for py_ver in [python_version, 'python3.11', 'python3.10']:
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
        QProgressBar, QProgressDialog, QScrollArea, QDialog, QDialogButtonBox, QRadioButton, QListWidget, QListWidgetItem, QSizePolicy
    )
    
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer, QRect, QObject, QCoreApplication, QSize, QEvent, QUrl
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
import tifffile

# Compatibility wrapper for NumPy 2.0
def safe_imread(file_path):
    """
    Safely read TIFF file with NumPy 2.0 compatibility.
    Handles the newbyteorder() issue with tifffile.
    """
    try:
        # Try standard reading first
        return tifffile.imread(file_path)
    except (AttributeError, TypeError) as e:
        if 'newbyteorder' in str(e):
            # NumPy 2.0 compatibility issue - use PIL as fallback
            try:
                from PIL import Image
                # Use PIL to read TIFF (handles NumPy 2.0 fine)
                img = Image.open(file_path)
                frames = []
                try:
                    while True:
                        frames.append(np.array(img))
                        img.seek(img.tell() + 1)
                except EOFError:
                    pass
                
                if len(frames) == 1:
                    return frames[0]
                elif len(frames) > 1:
                    return np.stack(frames, axis=0)
                else:
                    # Single frame case
                    img.seek(0)
                    return np.array(img)
            except ImportError:
                # PIL not available - raise original error with helpful message
                raise RuntimeError(
                    f"NumPy 2.0 compatibility issue reading TIFF file. "
                    f"Please install Pillow: pip install Pillow"
                ) from e
            except Exception as pil_error:
                # If PIL also fails, raise original error
                raise e from pil_error
        else:
            raise


class ClickableTooltip(QLabel):
    """A tooltip-like widget that appears on click and disappears on click elsewhere"""
    _current_tooltip = None
    _event_filter = None
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(
            "QLabel {"
            "background-color: #2d2d2d;"
            "color: #ffffff;"
            "padding: 10px;"
            "border: 1px solid #4a9eff;"
            "border-radius: 8px;"
            "font-size: 11pt;"
            "}"
        )
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
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
        
        # Calculate position (to the right of the icon, closer to input boxes)
        icon_global_pos = icon_widget.mapToGlobal(QPoint(0, 0))
        # Position tooltip further to the right (closer to input boxes) with larger offset
        tooltip_pos = QPoint(icon_global_pos.x() + icon_widget.width() + 20, icon_global_pos.y())
        
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
        if event.type() == QEvent.Type.MouseButtonPress:
            # Get widget under mouse
            # PyQt6: globalPos() -> globalPosition().toPoint()
            global_pos = event.globalPosition().toPoint()
            widget_under_mouse = QApplication.widgetAt(global_pos)
            
            # Hide if clicking outside the tooltip
            if self.tooltip and self.tooltip.isVisible():
                # Check if click is on the tooltip itself
                tooltip_rect = QRect(
                    self.tooltip.mapToGlobal(QPoint(0, 0)),
                    self.tooltip.size()
                )
                click_in_tooltip = tooltip_rect.contains(global_pos)
                
                if not click_in_tooltip:
                    # Check if click is on any (?) icon (to allow toggling)
                    click_on_icon = False
                    for widget in QApplication.allWidgets():
                        if isinstance(widget, QLabel) and widget.text() == "(?)":
                            icon_rect = QRect(
                                widget.mapToGlobal(QPoint(0, 0)),
                                widget.size()
                            )
                            if icon_rect.contains(global_pos):
                                click_on_icon = True
                                break
                    
                    if not click_on_icon:
                        # Click is outside tooltip and not on an icon - hide tooltip
                        ClickableTooltip.hide_current()
        return False


def get_modern_stylesheet():
    """
    Returns a comprehensive modern stylesheet for BactiMate with:
    - Modern color palette with gradients
    - Card-based design with shadows
    - Pill-shaped buttons with smooth transitions
    - Modern tabs with rounded corners
    - Enhanced input fields
    - Modern progress bars
    """
    return """
    /* ===== GLOBAL STYLES ===== */
    QMainWindow {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    QWidget {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: "Helvetica Neue", Helvetica, Arial;
        font-size: 11pt;
    }
    
    /* ===== MODERN TABS ===== */
    QTabWidget::pane {
        border: 1px solid #2d2d2d;
        border-radius: 6px;
        background-color: #1f1f1f;
        top: -1px;
    }
    
    QTabBar::tab {
        background-color: #2d2d2d;
        color: #b0b0b0;
        border: none;
        padding: 8px 16px;
        margin-right: 3px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 500;
        font-size: 11pt;
        min-width: 80px;
    }
    
    QTabBar::tab:hover {
        background-color: #3a3a3a;
        color: #ffffff;
    }
    
    QTabBar::tab:selected {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a9eff, stop:1 #357abd);
        color: #ffffff;
        font-weight: 600;
    }
    
    QTabBar::tab:!selected {
        margin-top: 2px;
    }
    
    /* ===== MODERN GROUP BOXES (CARDS) ===== */
    QGroupBox {
        border: 1px solid #2d2d2d;
        border-radius: 8px;
        margin-top: 10px;
        padding-top: 14px;
        background-color: #1a1a1a;
        font-weight: 600;
        font-size: 11pt;
        color: #e0e0e0;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 2px 8px;
        background-color: #1a1a1a;
        color: #4a9eff;
        border-radius: 4px;
    }
    
    /* ===== MODERN BUTTONS ===== */
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #5a9eff, stop:1 #4a7fdd);
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 6px 16px;
        font-weight: 600;
        font-size: 11pt;
        min-height: 28px;
    }
    
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #6aaeff, stop:1 #5a8fed);
    }
    
    QPushButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a8eef, stop:1 #3a6fcd);
    }
    
    QPushButton:disabled {
        background-color: #3a3a3a;
        color: #666666;
    }
    
    /* Secondary button style */
    QPushButton[class="secondary"] {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a4a4a, stop:1 #3a3a3a);
    }
    
    QPushButton[class="secondary"]:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #5a5a5a, stop:1 #4a4a4a);
    }
    
    /* ===== MODERN INPUT FIELDS ===== */
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
        background-color: #1f1f1f;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        padding: 6px 10px;
        color: #ffffff;
        font-size: 11pt;
        selection-background-color: #4a9eff;
    }
    
    QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #4a9eff;
        background-color: #252525;
    }
    
    QLineEdit:hover, QTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
        border: 1px solid #4a4a4a;
    }
    
    /* ===== MODERN COMBO BOXES ===== */
    QComboBox {
        background-color: #1f1f1f;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        padding: 6px 10px;
        color: #ffffff;
        font-size: 11pt;
        min-height: 28px;
    }
    
    QComboBox:hover {
        border: 1px solid #4a4a4a;
    }
    
    QComboBox:focus {
        border: 1px solid #4a9eff;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 24px;
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
        background-color: #3a3a3a;
    }
    
    QComboBox::drop-down:hover {
        background-color: #4a4a4a;
    }
    
    QComboBox QAbstractItemView {
        background-color: #1f1f1f;
        border: 1px solid #4a9eff;
        border-radius: 6px;
        selection-background-color: #4a9eff;
        selection-color: #ffffff;
        padding: 2px;
    }
    
    /* ===== MODERN CHECKBOXES ===== */
    QCheckBox {
        color: #e0e0e0;
        font-size: 11pt;
        spacing: 6px;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #4a4a4a;
        border-radius: 4px;
        background-color: #2d2d2d;
    }
    
    QCheckBox::indicator:hover {
        border: 1px solid #4a9eff;
    }
    
    QCheckBox::indicator:checked {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a9eff, stop:1 #357abd);
        border: 1px solid #4a9eff;
    }
    
    
    /* ===== MODERN RADIO BUTTONS ===== */
    QRadioButton {
        color: #e0e0e0;
        font-size: 11pt;
        spacing: 6px;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #4a4a4a;
        border-radius: 9px;
        background-color: #2d2d2d;
    }
    
    QRadioButton::indicator:hover {
        border: 1px solid #4a9eff;
    }
    
    QRadioButton::indicator:checked {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a9eff, stop:1 #357abd);
        border: 1px solid #4a9eff;
    }
    
    /* ===== MODERN PROGRESS BAR ===== */
    QProgressBar {
        border: 1px solid #2d2d2d;
        border-radius: 6px;
        text-align: center;
        color: #ffffff;
        font-weight: 600;
        font-size: 10pt;
        background-color: #1f1f1f;
        height: 24px;
    }
    
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #4a9eff, stop:0.5 #5aaeff, stop:1 #4a9eff);
        border-radius: 5px;
    }
    
    /* ===== MODERN SCROLLBAR ===== */
    QScrollBar:vertical {
        border: none;
        background-color: #1f1f1f;
        width: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #4a4a4a;
        border-radius: 6px;
        min-height: 30px;
        margin: 2px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #5a5a5a;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QScrollBar:horizontal {
        border: none;
        background-color: #1f1f1f;
        height: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #4a4a4a;
        border-radius: 6px;
        min-width: 30px;
        margin: 2px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #5a5a5a;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    
    /* ===== MODERN LIST WIDGET ===== */
    QListWidget {
        background-color: #1f1f1f;
        border: 1px solid #2d2d2d;
        border-radius: 6px;
        padding: 3px;
        color: #ffffff;
    }
    
    QListWidget::item {
        padding: 6px;
        border-radius: 4px;
        margin: 1px;
    }
    
    QListWidget::item:hover {
        background-color: #2d2d2d;
    }
    
    QListWidget::item:selected {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a9eff, stop:1 #357abd);
        color: #ffffff;
    }
    
    /* ===== MODERN DIALOG BUTTONS ===== */
    QDialogButtonBox QPushButton {
        min-width: 100px;
    }
    
    /* ===== MODERN TOOLTIP ===== */
    QToolTip {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #4a9eff;
        border-radius: 6px;
        padding: 8px;
        font-size: 11pt;
    }
    
    /* ===== MODERN LABELS ===== */
    QLabel {
        color: #e0e0e0;
    }
    
    /* ===== MODERN TITLE BAR ===== */
    QWidget[class="title_bar"] {
        background: transparent;
        border-bottom: 1px solid #4a9eff;
    }
    """
    
def create_info_icon(info_text, parent_widget=None):
    """Create a small (?) info icon label that shows tooltip-style info on click"""
    info_label = QLabel("(?)")
    info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    info_label.setStyleSheet(
        "QLabel {"
        "color: #7F8C8D;"
        "font-size: 8pt;"
        "font-weight: 600;"
        "text-decoration: none;"
        "background-color: #2d2d2d;"
        "border: 1px solid #4a4a4a;"
        "border-radius: 6px;"
        "padding: 0px 3px;"
        "min-width: 14px;"
        "min-height: 14px;"
        "max-width: 14px;"
        "max-height: 14px;"
        "}"
        "QLabel:hover {"
        "color: #4a9eff;"
        "background-color: #3a3a3a;"
        "border: 1px solid #4a9eff;"
        "}"
    )
    info_label.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def show_info(event):
        # Toggle tooltip - if already showing, hide it; otherwise show it
        if ClickableTooltip._current_tooltip and ClickableTooltip._current_tooltip.isVisible():
            ClickableTooltip.hide_current()
        else:
            ClickableTooltip.show_tooltip(info_text, info_label, parent_widget)
        event.accept()
    
    info_label.mousePressEvent = show_info
    return info_label


def show_file_selection_dialog(parent, files, title="Select Files", message="Select files to process (all selected by default):"):
    """
    Show a dialog with checkboxes for file selection.
    
    Parameters:
    -----------
    parent : QWidget
        Parent widget
    files : list of Path
        List of file paths
    title : str
        Dialog title
    message : str
        Dialog message
        
    Returns:
    --------
    selected_files : list of Path
        List of selected file paths, or None if cancelled
    """
    if not files:
        return None
    
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumSize(600, 500)
    
    layout = QVBoxLayout()
    
    # Message label
    label = QLabel(message)
    label.setStyleSheet("color: #ffffff; font-size: 11pt; padding: 10px;")
    layout.addWidget(label)
    
    # Select All / Deselect All buttons - modern styling applied via global stylesheet
    button_layout = QHBoxLayout()
    select_all_btn = QPushButton("Select All")
    deselect_all_btn = QPushButton("Deselect All")
    button_layout.addWidget(select_all_btn)
    button_layout.addWidget(deselect_all_btn)
    button_layout.addStretch()
    layout.addLayout(button_layout)
    
    # List widget with checkboxes - modern styling applied via global stylesheet
    file_list = QListWidget()
    
    # Add files with checkboxes (all checked by default)
    file_items = []
    for f in files:
        item = QListWidgetItem(f.name)
        item.setCheckState(Qt.CheckState.Checked)  # All checked by default
        item.setData(Qt.ItemDataRole.UserRole, f)  # Store full path
        file_list.addItem(item)
        file_items.append(item)
    
    # Connect select all / deselect all buttons
    def select_all():
        for item in file_items:
            item.setCheckState(Qt.CheckState.Checked)
    
    def deselect_all():
        for item in file_items:
            item.setCheckState(Qt.CheckState.Unchecked)
    
    select_all_btn.clicked.connect(select_all)
    deselect_all_btn.clicked.connect(deselect_all)
    
    layout.addWidget(file_list)
    
    # Buttons - modern styling applied via global stylesheet
    button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)
    
    dialog.setLayout(layout)
    
    # Show dialog
    if dialog.exec() == QDialog.DialogCode.Accepted:
        # Return selected files
        selected_files = []
        for i in range(file_list.count()):
            item = file_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_files.append(item.data(Qt.ItemDataRole.UserRole))
        return selected_files if selected_files else None  # Return None if nothing selected
    else:
        return None


class WorkerThread(QThread):
    """Thread for running long-running tasks without freezing the GUI"""
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(str)  # status message
    
    def __init__(self, script_path, cmd_args, parent=None):
        super().__init__(parent)
        self.script_path = script_path
        self.cmd_args = cmd_args
    
    @staticmethod
    def strip_ansi_codes(text):
        """Remove ANSI escape codes from text (e.g., [A for cursor movement)"""
        import re
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    @staticmethod
    def is_tqdm_progress_line(text):
        """Check if a line is a tqdm progress bar (e.g., "12%|#2        | 4/32")"""
        import re
        # Pattern matches tqdm progress bars: "XX%|##...| X/Y [time<time, time/it]"
        tqdm_pattern = re.compile(r'^\s*\d+%\s*\|[#\s]+\|\s*\d+/\d+\s*\[.*\]\s*$')
        return bool(tqdm_pattern.match(text.strip()))
    
    def run(self):
        try:
            # Build command
            cmd = [sys.executable, self.script_path] + self.cmd_args
            
            # Run process and capture output with unbuffered output
            # Use unbuffered mode to get real-time output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            # Disable tqdm progress bars when output is redirected (cleaner logs)
            env['TQDM_DISABLE'] = '1'
            
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
                    # Strip ANSI escape codes (like [A for cursor movement)
                    cleaned_line = self.strip_ansi_codes(line)
                    # Skip tqdm progress bar lines (they're too noisy in logs)
                    if self.is_tqdm_progress_line(cleaned_line):
                        continue  # Skip tqdm progress bars
                    # Skip empty lines after cleaning
                    if cleaned_line.strip():
                        output_lines.append(cleaned_line)
                        self.progress.emit(cleaned_line)
            
            process.wait()
            output = '\n'.join(output_lines)
            
            if process.returncode == 0:
                self.finished.emit(True, output)
            else:
                self.finished.emit(False, output)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class DragDropGroupBox(QGroupBox):
    """QGroupBox with drag-and-drop support for files and folders"""
    
    def __init__(self, title, parent=None, on_drop_callback=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)
        self.on_drop_callback = on_drop_callback
        # Visual feedback when dragging over
        self._drag_over = False
    
    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drag_over = True
            # Optional: change appearance to show drag is active
            self.setStyleSheet("QGroupBox { border: 2px dashed #2196F3; }")
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event"""
        self._drag_over = False
        self.setStyleSheet("")  # Reset to default style
    
    def dropEvent(self, event):
        """Handle drop event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.on_drop_callback:
                dropped_path = urls[0].toLocalFile()
                if dropped_path:
                    self.on_drop_callback(dropped_path)
            event.acceptProposedAction()
            self._drag_over = False
            self.setStyleSheet("")  # Reset to default style
        else:
            event.ignore()


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
        
        
        # File selection with drag-and-drop support
        file_group = DragDropGroupBox(
            "Microscopy File Selection",
            parent=self,
            on_drop_callback=self.handle_dropped_path
        )
        file_layout = QVBoxLayout()
        
        file_select_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a microscopy file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(500)  # Increased for better visibility
        # Set larger font for better readability
        file_font = QFont()
        file_font.setPointSize(12)
        self.file_path.setFont(file_font)
        # Fix size to minimum - prevent expansion
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
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
        channel_layout.addWidget(channel_label)
        self.separate_channels = QCheckBox("Save channels separately (recommended)")
        self.separate_channels.setChecked(True)
        channel_layout.addWidget(self.separate_channels)
        
        # Max intensity projection for Z-stack
        self.max_z_projection = QCheckBox("Max intensity projection for multiple Z stack")
        self.max_z_projection.setChecked(False)
        channel_layout.addWidget(self.max_z_projection)
        
        # Add info icon for Z-projection (click to show info, not hover)
        z_projection_info_icon = create_info_icon(
            "Apply maximum intensity projection across Z-slices when Z > 1.\n\n"
            "Default: Disabled (unchecked)\n"
            "Source: User selection\n\n"
            "When enabled: For each pixel, takes the maximum intensity value across all Z-slices.\n"
            "This creates a single 2D image from a 3D Z-stack.\n\n"
            "When disabled: Uses only the first Z-slice (Z=1).\n\n"
            "Use when: You have Z-stack data and want to combine all slices into one image.",
            parent_widget=self
        )
        channel_layout.addWidget(z_projection_info_icon)
        channel_layout.addStretch()
        params_layout.addLayout(channel_layout)
        
        self.channel_select = QComboBox()
        self.channel_select.addItem("All channels")
        self.channel_select.setEnabled(False)
        params_layout.addWidget(self.channel_select)
        
        # Corrections (one row to save vertical space)
        corrections_group = QGroupBox("Image Corrections")
        corrections_layout = QHBoxLayout()
        
        # Drift correction with info icon
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
        corrections_layout.addWidget(self.drift_correct)
        corrections_layout.addWidget(drift_info_icon)
        
        # Photobleaching correction with info icon
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
        corrections_layout.addWidget(self.bleach_correct)
        corrections_layout.addWidget(bleach_info_icon)
        corrections_layout.addStretch()
        
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
        output_font.setPointSize(12)
        self.output_dir.setFont(output_font)
        # Fix size to minimum - prevent expansion
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
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
        metadata_font.setPointSize(12)  # Increased font size for better readability
        self.metadata_text.setFont(metadata_font)
        # Prevent vertical expansion - only allow horizontal
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.metadata_text.setSizePolicy(size_policy)
        metadata_layout.addWidget(self.metadata_text)
        
        metadata_group.setLayout(metadata_layout)
        scroll_layout.addWidget(metadata_group)
        
        # Preview button (moved here to match Single File tab layout)
        self.preview_btn = QPushButton("Preview File")
        # Modern styling applied via global stylesheet
        self.preview_btn.clicked.connect(self.preview_file)
        self.preview_btn.setEnabled(False)
        scroll_layout.addWidget(self.preview_btn)
        
        # Convert button
        self.convert_btn = QPushButton("Convert to TIFF")
        # Modern styling applied via global stylesheet
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
            "Microscopy Files (*.nd2 *.ND2 *.lif *.LIF *.czi *.CZI *.tif *.tiff *.TIF *.TIFF);;"
            "ND2 Files (*.nd2 *.ND2);;"
            "LIF Files (*.lif *.LIF);;"
            "CZI Files (*.czi *.CZI);;"
            "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;"
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
                list(folder.glob("*.czi")) + list(folder.glob("*.CZI")) +
                list(folder.glob("*.tif")) + list(folder.glob("*.tiff")) +
                list(folder.glob("*.TIF")) + list(folder.glob("*.TIFF"))
            )
            if supported_files:
                self.load_folder_info(folder_path, supported_files)
            else:
                self.metadata_text.setText("No supported microscopy files (ND2, LIF, CZI, TIFF) found in selected folder")
                self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 12pt;")
    
    def handle_dropped_path(self, dropped_path):
        """Handle dropped file or folder path"""
        path_obj = Path(dropped_path)
        # Update file path
        self.file_path.setText(dropped_path)
        
        # Check if it's a file or folder and load accordingly
        if path_obj.is_file():
            # Single file
            self.load_file_info(dropped_path)
        elif path_obj.is_dir():
            # Folder - load folder info
            folder = path_obj
            supported_files = sorted(
                list(folder.glob("*.nd2")) + list(folder.glob("*.ND2")) +
                list(folder.glob("*.lif")) + list(folder.glob("*.LIF")) +
                list(folder.glob("*.czi")) + list(folder.glob("*.CZI")) +
                list(folder.glob("*.tif")) + list(folder.glob("*.tiff")) +
                list(folder.glob("*.TIF")) + list(folder.glob("*.TIFF"))
            )
            if supported_files:
                self.load_folder_info(str(folder), supported_files)
            else:
                self.metadata_text.setText("No supported microscopy files (ND2, LIF, CZI, TIFF) found in dropped folder")
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
                elif file_ext in ['.tif', '.tiff']:
                    file_info = self._get_tiff_info(str(file_path))
                
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
            import numpy as np
            from aicsimageio import AICSImage
            # AICSImage doesn't support context manager protocol, use direct instantiation
            lif_reader = AICSImage(file_path)
            try:
                # Get dimensions - handle both old and new aicsimageio API
                dims = lif_reader.dims
                
                # Try to get dimension order as string
                if hasattr(dims, 'order'):
                    dims_str = dims.order
                elif isinstance(dims, str):
                    dims_str = dims
                elif isinstance(dims, (list, tuple)):
                    dims_str = ''.join(str(d) for d in dims)
                else:
                    dims_str = str(dims)
                
                # Get shape - try different methods depending on aicsimageio version
                try:
                    # Try accessing as dictionary-like (newer versions)
                    if hasattr(lif_reader, 'get_xarray_data'):
                        xdata = lif_reader.get_xarray_data()
                        num_time = int(xdata.sizes.get('T', 1))
                        num_channels = int(xdata.sizes.get('C', 1))
                        num_z = int(xdata.sizes.get('Z', 1))
                    else:
                        # Fallback to shape attribute
                        shape = lif_reader.shape
                        # Convert shape to tuple/list if needed
                        if not isinstance(shape, (tuple, list)):
                            shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                        
                        # Access dimensions by index in dims string
                        num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                        num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                        num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                except (KeyError, IndexError, AttributeError, TypeError) as e:
                    # If dimension access fails, try to get from shape directly
                    shape = lif_reader.shape
                    if isinstance(shape, dict):
                        # Shape is a dictionary - access by dimension name
                        num_time = shape.get('T', 1) if isinstance(shape.get('T'), (int, np.integer)) else 1
                        num_channels = shape.get('C', 1) if isinstance(shape.get('C'), (int, np.integer)) else 1
                        num_z = shape.get('Z', 1) if isinstance(shape.get('Z'), (int, np.integer)) else 1
                    else:
                        # Fallback: assume standard order and try indexing
                        shape_list = list(shape) if hasattr(shape, '__iter__') else [shape]
                        num_time = shape_list[dims_str.index('T')] if 'T' in dims_str and len(shape_list) > dims_str.index('T') else 1
                        num_channels = shape_list[dims_str.index('C')] if 'C' in dims_str and len(shape_list) > dims_str.index('C') else 1
                        num_z = shape_list[dims_str.index('Z')] if 'Z' in dims_str and len(shape_list) > dims_str.index('Z') else 1
                
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
            finally:
                # Clean up if needed (AICSImage may have resources to close)
                if hasattr(lif_reader, 'close'):
                    try:
                        lif_reader.close()
                    except:
                        pass
        except Exception:
            return None
    
    def _get_tiff_info(self, file_path):
        """Extract TIFF file info without updating UI (for folder scanning)"""
        try:
            import tifffile
            import numpy as np
            
            # Read TIFF file to get dimensions
            with tifffile.TiffFile(file_path) as tif:
                # Get shape from first page
                first_page = tif.pages[0]
                shape = first_page.shape
                
                # Try to get metadata
                metadata = {}
                if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                    metadata = tif.imagej_metadata
                elif hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                    # OME metadata parsing would be more complex
                    pass
                
                # Determine dimensions from shape
                # Common formats: (T, Y, X), (T, C, Y, X), (C, Y, X), (Y, X), (T, Z, C, Y, X)
                ndim = len(shape)
                
                if ndim == 2:
                    # Single 2D image
                    num_time = 1
                    num_channels = 1
                    num_z = 1
                elif ndim == 3:
                    # Could be (T, Y, X) or (C, Y, X) or (Z, Y, X)
                    # Check if first dimension is small (likely channels or z-slices)
                    if shape[0] < 10:
                        num_time = 1
                        num_channels = shape[0]
                        num_z = 1
                    else:
                        # Likely (T, Y, X)
                        num_time = shape[0]
                        num_channels = 1
                        num_z = 1
                elif ndim == 4:
                    # Could be (T, C, Y, X) or (T, Z, Y, X) or (Z, C, Y, X)
                    if shape[1] < 10:
                        # Likely (T, C, Y, X)
                        num_time = shape[0]
                        num_channels = shape[1]
                        num_z = 1
                    elif shape[0] < 10:
                        # Likely (Z, C, Y, X)
                        num_time = 1
                        num_channels = shape[1]
                        num_z = shape[0]
                    else:
                        # Assume (T, Z, Y, X)
                        num_time = shape[0]
                        num_channels = 1
                        num_z = shape[1]
                else:
                    # 5D or more - assume (T, Z, C, Y, X)
                    num_time = shape[0] if len(shape) > 0 else 1
                    num_z = shape[1] if len(shape) > 1 else 1
                    num_channels = shape[2] if len(shape) > 2 else 1
                
                # Check if multi-page TIFF (time series)
                num_pages = len(tif.pages)
                if num_pages > 1 and ndim == 2:
                    # Multi-page 2D images - treat as time series
                    num_time = num_pages
                    num_channels = 1
                    num_z = 1
                
                # Get pixel size from metadata if available
                pixel_size = None
                if 'pixelSize' in metadata:
                    pixel_size = float(metadata['pixelSize'])
                elif 'unit' in metadata and 'pixelSize' in metadata:
                    pixel_size = float(metadata['pixelSize'])
                
                channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                return {
                    'file_type': 'tiff',
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
            import numpy as np
            from aicsimageio import AICSImage
            # AICSImage doesn't support context manager protocol, use direct instantiation
            czi_reader = AICSImage(file_path)
            try:
                # Get dimensions - handle both old and new aicsimageio API
                dims = czi_reader.dims
                
                # Try to get dimension order as string
                if hasattr(dims, 'order'):
                    dims_str = dims.order
                elif isinstance(dims, str):
                    dims_str = dims
                elif isinstance(dims, (list, tuple)):
                    dims_str = ''.join(str(d) for d in dims)
                else:
                    dims_str = str(dims)
                
                # Get shape - try different methods depending on aicsimageio version
                try:
                    # Try accessing as dictionary-like (newer versions)
                    if hasattr(czi_reader, 'get_xarray_data'):
                        xdata = czi_reader.get_xarray_data()
                        num_time = int(xdata.sizes.get('T', 1))
                        num_channels = int(xdata.sizes.get('C', 1))
                        num_z = int(xdata.sizes.get('Z', 1))
                    else:
                        # Fallback to shape attribute
                        shape = czi_reader.shape
                        # Convert shape to tuple/list if needed
                        if not isinstance(shape, (tuple, list)):
                            shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                        
                        # Access dimensions by index in dims string
                        num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                        num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                        num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                except (KeyError, IndexError, AttributeError, TypeError) as e:
                    # If dimension access fails, try to get from shape directly
                    shape = czi_reader.shape
                    if isinstance(shape, dict):
                        # Shape is a dictionary - access by dimension name
                        num_time = shape.get('T', 1) if isinstance(shape.get('T'), (int, np.integer)) else 1
                        num_channels = shape.get('C', 1) if isinstance(shape.get('C'), (int, np.integer)) else 1
                        num_z = shape.get('Z', 1) if isinstance(shape.get('Z'), (int, np.integer)) else 1
                    else:
                        # Fallback: assume standard order and try indexing
                        shape_list = list(shape) if hasattr(shape, '__iter__') else [shape]
                        num_time = shape_list[dims_str.index('T')] if 'T' in dims_str and len(shape_list) > dims_str.index('T') else 1
                        num_channels = shape_list[dims_str.index('C')] if 'C' in dims_str and len(shape_list) > dims_str.index('C') else 1
                        num_z = shape_list[dims_str.index('Z')] if 'Z' in dims_str and len(shape_list) > dims_str.index('Z') else 1
                
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
            finally:
                # Clean up if needed (AICSImage may have resources to close)
                if hasattr(czi_reader, 'close'):
                    try:
                        czi_reader.close()
                    except:
                        pass
        except Exception:
            return None
    
    def load_file_info(self, file_path):
        """Load and display microscopy file information (ND2, LIF, CZI, or TIFF)"""
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        
        try:
            if file_ext == '.nd2':
                self._load_nd2_info(file_path)
            elif file_ext == '.lif':
                self._load_lif_info(file_path)
            elif file_ext == '.czi':
                self._load_czi_info(file_path)
            elif file_ext in ['.tif', '.tiff']:
                self._load_tiff_info(file_path)
            else:
                QMessageBox.warning(self, "Unsupported Format", 
                                  f"File format '{file_ext}' is not supported.\n"
                                  "Supported formats: ND2, LIF, CZI, TIFF")
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
                # Use sizes dictionary directly - it properly maps dimension names to counts
                # Don't assume dimension order from shape[0] as ND2 files can have different orders
                num_time = sizes.get('T', 1)
                num_channels = sizes.get('C', 1)
                num_z = sizes.get('Z', 1)
                
                # Verify by checking actual data if needed (but trust sizes first)
                # Sometimes ND2 files report dimensions incorrectly, so double-check with shape
                try:
                    shape = nd2_reader.shape
                    # Get dimension order from ND2 reader
                    if hasattr(nd2_reader, 'sizes') and hasattr(nd2_reader.sizes, 'keys'):
                        # Check if shape matches reported sizes
                        # If shape[0] matches channels instead of time, sizes might be wrong
                        # But we'll trust sizes.get() as it's the proper way
                        pass
                except Exception:
                    pass
                
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
                import numpy as np
                from aicsimageio import AICSImage
                self.parent_window.log_output(f"Loading LIF file info: {file_path}\n")
                
                # AICSImage doesn't support context manager protocol, use direct instantiation
                lif_reader = AICSImage(file_path)
                try:
                    # Get dimensions - handle both old and new aicsimageio API
                    dims = lif_reader.dims
                    
                    # Try to get dimension order as string
                    if hasattr(dims, 'order'):
                        dims_str = dims.order
                    elif isinstance(dims, str):
                        dims_str = dims
                    elif isinstance(dims, (list, tuple)):
                        dims_str = ''.join(str(d) for d in dims)
                    else:
                        dims_str = str(dims)
                    
                    # Get shape - try different methods depending on aicsimageio version
                    try:
                        # Try accessing as dictionary-like (newer versions)
                        if hasattr(lif_reader, 'get_xarray_data'):
                            xdata = lif_reader.get_xarray_data()
                            num_time = int(xdata.sizes.get('T', 1))
                            num_channels = int(xdata.sizes.get('C', 1))
                            num_z = int(xdata.sizes.get('Z', 1))
                        else:
                            # Fallback to shape attribute
                            shape = lif_reader.shape
                            # Convert shape to tuple/list if needed
                            if not isinstance(shape, (tuple, list)):
                                shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                            
                            # Access dimensions by index in dims string
                            num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                            num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                            num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                    except (KeyError, IndexError, AttributeError, TypeError) as e:
                        # If dimension access fails, try to get from shape directly
                        shape = lif_reader.shape
                        if isinstance(shape, dict):
                            # Shape is a dictionary - access by dimension name
                            num_time = shape.get('T', 1) if isinstance(shape.get('T'), (int, np.integer)) else 1
                            num_channels = shape.get('C', 1) if isinstance(shape.get('C'), (int, np.integer)) else 1
                            num_z = shape.get('Z', 1) if isinstance(shape.get('Z'), (int, np.integer)) else 1
                        else:
                            # Fallback: assume standard order and try indexing
                            shape_list = list(shape) if hasattr(shape, '__iter__') else [shape]
                            num_time = shape_list[dims_str.index('T')] if 'T' in dims_str and len(shape_list) > dims_str.index('T') else 1
                            num_channels = shape_list[dims_str.index('C')] if 'C' in dims_str and len(shape_list) > dims_str.index('C') else 1
                            num_z = shape_list[dims_str.index('Z')] if 'Z' in dims_str and len(shape_list) > dims_str.index('Z') else 1
                    
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
                finally:
                    # Clean up if needed (AICSImage may have resources to close)
                    if hasattr(lif_reader, 'close'):
                        try:
                            lif_reader.close()
                        except:
                            pass
                    
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
                import numpy as np
                from aicsimageio import AICSImage
                self.parent_window.log_output(f"Loading CZI file info: {file_path}\n")
                
                # AICSImage doesn't support context manager protocol, use direct instantiation
                czi_reader = AICSImage(file_path)
                try:
                    # Get dimensions - handle both old and new aicsimageio API
                    dims = czi_reader.dims
                    
                    # Try to get dimension order as string
                    if hasattr(dims, 'order'):
                        dims_str = dims.order
                    elif isinstance(dims, str):
                        dims_str = dims
                    elif isinstance(dims, (list, tuple)):
                        dims_str = ''.join(str(d) for d in dims)
                    else:
                        dims_str = str(dims)
                    
                    # Get shape - try different methods depending on aicsimageio version
                    try:
                        # Try accessing as dictionary-like (newer versions)
                        if hasattr(czi_reader, 'get_xarray_data'):
                            xdata = czi_reader.get_xarray_data()
                            num_time = int(xdata.sizes.get('T', 1))
                            num_channels = int(xdata.sizes.get('C', 1))
                            num_z = int(xdata.sizes.get('Z', 1))
                        else:
                            # Fallback to shape attribute
                            shape = czi_reader.shape
                            # Convert shape to tuple/list if needed
                            if not isinstance(shape, (tuple, list)):
                                shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                            
                            # Access dimensions by index in dims string
                            num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                            num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                            num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                    except (KeyError, IndexError, AttributeError, TypeError) as e:
                        # If dimension access fails, try to get from shape directly
                        shape = czi_reader.shape
                        if isinstance(shape, dict):
                            # Shape is a dictionary - access by dimension name
                            num_time = shape.get('T', 1) if isinstance(shape.get('T'), (int, np.integer)) else 1
                            num_channels = shape.get('C', 1) if isinstance(shape.get('C'), (int, np.integer)) else 1
                            num_z = shape.get('Z', 1) if isinstance(shape.get('Z'), (int, np.integer)) else 1
                        else:
                            # Fallback: assume standard order and try indexing
                            shape_list = list(shape) if hasattr(shape, '__iter__') else [shape]
                            num_time = shape_list[dims_str.index('T')] if 'T' in dims_str and len(shape_list) > dims_str.index('T') else 1
                            num_channels = shape_list[dims_str.index('C')] if 'C' in dims_str and len(shape_list) > dims_str.index('C') else 1
                            num_z = shape_list[dims_str.index('Z')] if 'Z' in dims_str and len(shape_list) > dims_str.index('Z') else 1
                    
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
                finally:
                    # Clean up if needed (AICSImage may have resources to close)
                    if hasattr(czi_reader, 'close'):
                        try:
                            czi_reader.close()
                        except:
                            pass
                    
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
    
    def _load_tiff_info(self, file_path):
        """Load and display TIFF file information"""
        try:
            import tifffile
            import numpy as np
            self.parent_window.log_output(f"Loading TIFF file info: {file_path}\n")
            
            # Read TIFF file to get dimensions
            with tifffile.TiffFile(file_path) as tif:
                # Get shape from first page
                first_page = tif.pages[0]
                shape = first_page.shape
                
                # Try to get metadata
                metadata = {}
                if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                    metadata = tif.imagej_metadata
                
                # Determine dimensions from shape
                ndim = len(shape)
                num_pages = len(tif.pages)
                
                if ndim == 2:
                    # Single 2D image or multi-page
                    if num_pages > 1:
                        num_time = num_pages
                        num_channels = 1
                        num_z = 1
                    else:
                        num_time = 1
                        num_channels = 1
                        num_z = 1
                elif ndim == 3:
                    # Could be (T, Y, X) or (C, Y, X) or (Z, Y, X)
                    if shape[0] < 10:
                        num_time = 1
                        num_channels = shape[0]
                        num_z = 1
                    else:
                        num_time = shape[0]
                        num_channels = 1
                        num_z = 1
                elif ndim == 4:
                    # Could be (T, C, Y, X) or (T, Z, Y, X) or (Z, C, Y, X)
                    if shape[1] < 10:
                        num_time = shape[0]
                        num_channels = shape[1]
                        num_z = 1
                    elif shape[0] < 10:
                        num_time = 1
                        num_channels = shape[1]
                        num_z = shape[0]
                    else:
                        num_time = shape[0]
                        num_channels = 1
                        num_z = shape[1]
                else:
                    # 5D or more - assume (T, Z, C, Y, X)
                    num_time = shape[0] if len(shape) > 0 else 1
                    num_z = shape[1] if len(shape) > 1 else 1
                    num_channels = shape[2] if len(shape) > 2 else 1
                
                # Get pixel size from metadata if available
                pixel_size = None
                if 'pixelSize' in metadata:
                    pixel_size = float(metadata['pixelSize'])
                
                channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                self._update_file_info_ui(file_path, 'tiff', num_time, num_channels, num_z,
                                        channel_names, pixel_size)
                
        except Exception as e:
            QMessageBox.critical(self, "Error Loading TIFF", f"Could not load TIFF file:\n{str(e)}")
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
        
        file_path_obj = Path(self.file_path.text())
        
        # Check if it's a folder - if so, show file selection dialog
        if file_path_obj.is_dir():
            folder = file_path_obj
            # Find all supported files in the folder
            supported_files = sorted(
                list(folder.glob("*.nd2")) + list(folder.glob("*.ND2")) +
                list(folder.glob("*.lif")) + list(folder.glob("*.LIF")) +
                list(folder.glob("*.czi")) + list(folder.glob("*.CZI")) +
                list(folder.glob("*.tif")) + list(folder.glob("*.tiff")) +
                list(folder.glob("*.TIF")) + list(folder.glob("*.TIFF"))
            )
            
            if not supported_files:
                QMessageBox.warning(self, "No Files Found", 
                                  "No supported microscopy files (ND2, LIF, CZI, TIFF) found in the selected folder.")
                return
            
            # Show file selection dialog
            file_dialog = QDialog(self)
            file_dialog.setWindowTitle("Select File for Preview")
            file_dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout()
            
            # Label
            label = QLabel("Select a file to preview:")
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
            for f in supported_files:
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
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setStyleSheet(
                "QPushButton {"
                "background-color: #5D6D7E;"
                "color: white;"
                "font-weight: bold;"
                "padding: 6px 12px;"
                "border: none;"
                "border-radius: 4px;"
                "}"
                "QPushButton:hover {"
                "background-color: #34495E;"
                "}"
            )
            cancel_btn.clicked.connect(file_dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            ok_btn = QPushButton("Preview")
            ok_btn.setStyleSheet(
                "QPushButton {"
                "background-color: #2196F3;"
                "color: white;"
                "font-weight: bold;"
                "padding: 6px 12px;"
                "border: none;"
                "border-radius: 4px;"
                "}"
                "QPushButton:hover {"
                "background-color: #1976D2;"
                "}"
            )
            ok_btn.clicked.connect(file_dialog.accept)
            button_layout.addWidget(ok_btn)
            
            layout.addLayout(button_layout)
            file_dialog.setLayout(layout)
            
            # Show dialog and get selected file
            if file_dialog.exec() == QDialog.DialogCode.Accepted:
                current_item = file_list.currentItem()
                if current_item:
                    selected_file_name = current_item.text()
                    selected_file = folder / selected_file_name
                    
                    # Temporarily update file_path and load info for selected file
                    original_path = self.file_path.text()
                    self.file_path.setText(str(selected_file))
                    self.load_file_info(str(selected_file))
                    
                    # Now preview the selected file
                    if self.file_info:
                        file_type = self.file_info.get('file_type', 'nd2')
                        
                        if file_type == 'nd2':
                            self._preview_nd2()
                        elif file_type == 'lif':
                            self._preview_lif()
                        elif file_type == 'czi':
                            self._preview_czi()
                        elif file_type == 'tiff':
                            self._preview_tiff()
                        else:
                            QMessageBox.warning(self, "Unsupported Format", f"Preview not yet implemented for {file_type} files.")
                    
                    # Restore original path (optional - user might want to keep the selected file)
                    # self.file_path.setText(original_path)
            return
        
        # Single file mode - check if file info is loaded
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
        elif file_type == 'tiff':
            self._preview_tiff()
        else:
            QMessageBox.warning(self, "Unsupported Format", f"Preview not yet implemented for {file_type} files.")
    
    def _create_unified_preview_ui(self, preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path):
        """Create unified preview UI with improved histogram layout (used by all file types)"""
        import numpy as np
        import sys
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure
        from PyQt6.QtCore import QTimer
        from PyQt6.QtGui import QKeySequence, QShortcut
        
        # Matplotlib figure with navigation toolbar
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
        
        # Channel selection (if multiple channels)
        if num_channels > 1:
            nav_layout.addWidget(QLabel("Channel:"))
            channel_spin = QSpinBox()
            channel_spin.setRange(1, num_channels)
            channel_spin.setValue(1)
            channel_spin.setToolTip(f"Select channel to preview (1-{num_channels})")
            nav_layout.addWidget(channel_spin)
        else:
            channel_spin = None
        
        nav_layout.addStretch()
        
        # Frame info label
        frame_info = QLabel(f"Timepoint 1 of {num_time} | Channel: 1 | Z-slice: 1")
        frame_info.setStyleSheet("font-weight: bold; color: #2196F3;")
        nav_layout.addWidget(frame_info)
        
        layout.addLayout(nav_layout)
        
        # LUT controls with histogram - IMPROVED LAYOUT
        lut_group = QGroupBox("LUT (Lookup Table) Controls")
        lut_layout = QVBoxLayout()
        lut_layout.setSpacing(8)  # Reduce spacing between elements
        
        # Histogram plot - optimized size and appearance
        hist_fig = Figure(figsize=(10, 3.0))  # Increased height for better visibility
        hist_canvas = FigureCanvas(hist_fig)
        hist_canvas.setMinimumHeight(180)  # Increased to prevent cutoff
        hist_canvas.setMaximumHeight(180)  # Increased to prevent cutoff
        hist_ax = hist_fig.add_subplot(111)
        hist_fig.patch.set_facecolor('#2b2b2b')  # Dark background
        hist_ax.set_facecolor('#2b2b2b')
        hist_ax.tick_params(colors='white', labelsize=9)  # Slightly reduced for cleaner look
        hist_ax.spines['bottom'].set_color('white')
        hist_ax.spines['top'].set_color('white')
        hist_ax.spines['left'].set_color('white')
        hist_ax.spines['right'].set_color('white')
        hist_ax.xaxis.label.set_color('white')
        hist_ax.yaxis.label.set_color('white')
        # Set subplot parameters to ensure nothing gets cut off
        hist_fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.15)
        hist_canvas.setStyleSheet("background-color: #2b2b2b;")
        lut_layout.addWidget(hist_canvas)
        
        # Min/Max value displays and manual input - more compact layout
        value_layout = QHBoxLayout()
        value_layout.setSpacing(10)  # Reduce spacing
        value_layout.addWidget(QLabel("Min:"))
        lut_min_slider = QDoubleSpinBox()
        lut_min_slider.setRange(0, 65535)
        lut_min_slider.setDecimals(1)
        lut_min_slider.setSingleStep(10)
        lut_min_slider.setToolTip("Minimum intensity value for display (or adjust via histogram)")
        lut_min_slider.setMaximumWidth(120)  # Limit width to reduce white space
        value_layout.addWidget(lut_min_slider)
        
        value_layout.addWidget(QLabel("Max:"))
        lut_max_slider = QDoubleSpinBox()
        lut_max_slider.setRange(0, 65535)
        lut_max_slider.setDecimals(1)
        lut_max_slider.setSingleStep(10)
        lut_max_slider.setToolTip("Maximum intensity value for display (or adjust via histogram)")
        lut_max_slider.setMaximumWidth(120)  # Limit width to reduce white space
        value_layout.addWidget(lut_max_slider)
        
        value_layout.addStretch()
        lut_layout.addLayout(value_layout)
        
        lut_group.setLayout(lut_layout)
        lut_group.setMaximumHeight(280)  # Increased to match new histogram height
        layout.addWidget(lut_group)
        
        layout.addWidget(canvas)
        
        # Store current frame data and LUT settings
        current_frame_data = {'frame': None, 'vmin': None, 'vmax': None, 'frame_min': None, 'frame_max': None, 'histogram': None}
        is_updating_lut = False  # Flag to prevent recursive updates during LUT adjustment
        im_handle = None  # Store image handle for LUT updates
        hist_vmin_line = None  # Vertical line for min value in histogram
        hist_vmax_line = None  # Vertical line for max value in histogram
        hist_plot = None  # Histogram bar plot
        dragging_line = None  # Track which line is being dragged
        
        # Initialize LUT values (will be set from first frame)
        current_vmin = None
        current_vmax = None
        
        def update_histogram(frame, vmin, vmax):
            """Update histogram plot with min/max lines"""
            nonlocal hist_vmin_line, hist_vmax_line, hist_plot
            
            # Clear axes instead of trying to remove lines
            hist_ax.clear()
            
            # Calculate histogram
            flat_frame = frame.flatten()
            flat_frame = flat_frame[flat_frame > 0]  # Exclude zeros for better visualization
            if len(flat_frame) == 0:
                flat_frame = frame.flatten()
            
            frame_min = float(frame.min())
            frame_max = float(frame.max())
            
            # Create histogram
            bins = min(256, int(frame_max - frame_min + 1))
            hist, bin_edges = np.histogram(flat_frame, bins=bins, range=(frame_min, frame_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot histogram (log scale for better visualization)
            hist_plot = hist_ax.bar(bin_centers, np.log1p(hist), width=bin_edges[1] - bin_edges[0], 
                                   color='#4A90E2', alpha=0.8, edgecolor='none')
            
            # Draw min/max lines (cleaner appearance)
            hist_vmin_line = hist_ax.axvline(vmin, color='#FF6B6B', linewidth=2, linestyle='--', label=f'Min: {vmin:.1f}')
            hist_vmax_line = hist_ax.axvline(vmax, color='#4ECDC4', linewidth=2, linestyle='--', label=f'Max: {vmax:.1f}')
            
            # Subtle fill region between min and max (lighter, less intrusive)
            hist_ax.axvspan(vmin, vmax, alpha=0.15, color='yellow')
            
            hist_ax.set_xlabel('Intensity', color='white', fontsize=10)
            hist_ax.set_ylabel('Log(Count+1)', color='white', fontsize=10)
            hist_ax.set_title('Click/drag lines to adjust min/max', color='white', fontsize=9, pad=10)
            hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
            hist_ax.grid(True, alpha=0.25, color='white', linestyle='-', linewidth=0.5)
            
            # Ensure proper margins to prevent cutoff
            hist_ax.margins(x=0.02, y=0.08)
            
            # Reapply tight layout after drawing to prevent cutoff
            hist_fig.tight_layout(pad=2.0)
            
            hist_canvas.draw()
            current_frame_data['histogram'] = (hist, bin_edges, frame_min, frame_max)
        
        def on_histogram_press(event):
            """Handle mouse press on histogram to start dragging"""
            if event.inaxes != hist_ax:
                return
            
            x = event.xdata
            if x is None:
                return
            
            nonlocal dragging_line, current_vmin, current_vmax
            
            # Determine which line is closest (within 5% of range for easier grabbing)
            if current_vmin is not None and current_vmax is not None:
                frame_min = current_frame_data.get('frame_min', 0)
                frame_max = current_frame_data.get('frame_max', 65535)
                range_size = frame_max - frame_min
                if range_size > 0:
                    threshold = max(range_size * 0.05, (frame_max - frame_min) * 0.01)  # 5% of range, or 1% minimum
                else:
                    threshold = 100  # Default threshold
                
                dist_to_min = abs(x - current_vmin)
                dist_to_max = abs(x - current_vmax)
                
                # Check if clicking near a line
                if dist_to_min < threshold and dist_to_min < dist_to_max:
                    dragging_line = 'min'
                elif dist_to_max < threshold:
                    dragging_line = 'max'
                else:
                    # If not near a line, create a new line at click position
                    # Determine which side of the midpoint to adjust
                    midpoint = (current_vmin + current_vmax) / 2
                    if x < midpoint:
                        dragging_line = 'min'
                    else:
                        dragging_line = 'max'
        
        def on_histogram_motion(event):
            """Handle mouse motion to drag lines"""
            if event.inaxes != hist_ax:
                return
            
            x = event.xdata
            if x is None or dragging_line is None:
                return
            
            nonlocal current_vmin, current_vmax, is_updating_lut
            if is_updating_lut:
                return
            
            is_updating_lut = True
            try:
                frame_min = current_frame_data.get('frame_min', 0)
                frame_max = current_frame_data.get('frame_max', 65535)
                
                if dragging_line == 'min':
                    new_min = max(frame_min, min(x, (current_vmax or frame_max) - 1))
                    current_vmin = new_min
                    # Block signals to prevent recursive updates
                    lut_min_slider.blockSignals(True)
                    lut_min_slider.setValue(new_min)
                    lut_min_slider.blockSignals(False)
                elif dragging_line == 'max':
                    new_max = min(frame_max, max(x, (current_vmin or frame_min) + 1))
                    current_vmax = new_max
                    # Block signals to prevent recursive updates
                    lut_max_slider.blockSignals(True)
                    lut_max_slider.setValue(new_max)
                    lut_max_slider.blockSignals(False)
                
                # Update display directly without calling update_lut_from_sliders (to avoid recursion)
                vmin = current_vmin
                vmax = current_vmax
                
                # Update image display
                if current_frame_data['frame'] is not None and im_handle is not None:
                    im_handle.set_clim(vmin=vmin, vmax=vmax)
                    canvas.draw()
                    
                    # Update frame info
                    current_channel = (channel_spin.value() if channel_spin is not None else 1)
                    frame_info.setText(f"Timepoint {preview_dialog.current_frame + 1} of {num_time} | Channel: {current_channel} | Z-slice: 1 | LUT: [{vmin:.1f}, {vmax:.1f}]")
                    
                    # Update histogram lines
                    if hist_vmin_line is not None:
                        hist_vmin_line.set_xdata([vmin, vmin])
                    if hist_vmax_line is not None:
                        hist_vmax_line.set_xdata([vmax, vmax])
                    # Update fill region
                    for patch in list(hist_ax.patches):
                        if hasattr(patch, 'get_alpha') and patch.get_alpha() == 0.2:
                            patch.remove()
                    hist_ax.axvspan(vmin, vmax, alpha=0.2, color='yellow')
                    # Update legend
                    if hist_vmin_line is not None and hist_vmax_line is not None:
                        hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
                    hist_canvas.draw()
                    
                    # Store values
                    current_frame_data['vmin'] = vmin
                    current_frame_data['vmax'] = vmax
            finally:
                is_updating_lut = False
        
        def on_histogram_release(event):
            """Handle mouse release to stop dragging"""
            nonlocal dragging_line
            dragging_line = None
        
        # Connect histogram interaction handlers
        hist_canvas.mpl_connect('button_press_event', on_histogram_press)
        hist_canvas.mpl_connect('motion_notify_event', on_histogram_motion)
        hist_canvas.mpl_connect('button_release_event', on_histogram_release)
        
        # Function to update displayed frame
        def update_frame(frame_idx, channel_idx=0, vmin=None, vmax=None, update_sliders=True):
            """Update the displayed frame (frame_idx is 0-indexed internally, channel_idx is 0-indexed)"""
            # Get current channel selection if available
            if channel_spin is not None:
                channel_idx = channel_spin.value() - 1  # Convert to 0-indexed
            else:
                channel_idx = 0  # Default to first channel
            
            # Handle different all_frames shapes directly (more robust)
            # ND2 files can have various shapes: (T, C, Z, Y, X), (T, C, Y, X), (T, Y, X), etc.
            if len(all_frames.shape) == 5:  # TCZYX
                # Extract frame, channel, and first Z-slice
                if channel_idx < all_frames.shape[1]:
                    frame = all_frames[frame_idx, channel_idx, 0, :, :]  # T=frame_idx, C=channel_idx, Z=0
                else:
                    frame = all_frames[frame_idx, 0, 0, :, :]  # Fallback
            elif len(all_frames.shape) == 4:
                # Could be TCYX or TZYX
                if num_channels > 1:
                    # TCYX: extract frame and channel
                    if channel_idx < all_frames.shape[1]:
                        frame = all_frames[frame_idx, channel_idx, :, :]  # T=frame_idx, C=channel_idx
                    else:
                        frame = all_frames[frame_idx, 0, :, :]  # Fallback
                else:
                    # TZYX: extract frame and first Z-slice
                    frame = all_frames[frame_idx, 0, :, :] if num_z > 1 else all_frames[frame_idx, :, :]
            elif len(all_frames.shape) == 3:
                # Could be TYX (single channel, single Z) or CYX (single frame, multi-channel)
                if all_frames.shape[0] == num_time:
                    # TYX: single channel, single Z
                    frame = all_frames[frame_idx, :, :]
                elif num_channels > 1:
                    # CYX: single frame, multi-channel - extract channel
                    if channel_idx < all_frames.shape[0]:
                        frame = all_frames[channel_idx, :, :]
                    else:
                        frame = all_frames[0, :, :]
                else:
                    # Fallback: assume TYX
                    frame = all_frames[frame_idx, :, :]
            elif len(all_frames.shape) == 2:
                # YX: single frame, single channel
                frame = all_frames
            else:
                # Fallback: try to extract frame
                frame = all_frames[frame_idx] if frame_idx < len(all_frames) else all_frames[0]
                # If still multi-dimensional, try to extract channel
                if len(frame.shape) > 2 and num_channels > 1:
                    if channel_idx < frame.shape[0]:
                        frame = frame[channel_idx]
                    else:
                        frame = frame[0]
            
            # Store frame data and actual min/max
            current_frame_data['frame'] = frame
            frame_min = float(frame.min())
            frame_max = float(frame.max())
            current_frame_data['frame_min'] = frame_min
            current_frame_data['frame_max'] = frame_max
            
            # Determine LUT values - preserve current values or use defaults
            nonlocal current_vmin, current_vmax
            if vmin is not None and vmax is not None:
                # Explicit values provided
                current_vmin = vmin
                current_vmax = vmax
            elif current_vmin is None or current_vmax is None:
                # First time - use percentile-based defaults
                current_vmin = float(np.percentile(frame, 1))
                current_vmax = float(np.percentile(frame, 99))
            # Otherwise, keep current_vmin and current_vmax (preserve user settings)
            
            vmin = current_vmin
            vmax = current_vmax
            
            current_frame_data['vmin'] = vmin
            current_frame_data['vmax'] = vmax
            
            # Update sliders (block signals to prevent recursive calls)
            if update_sliders:
                nonlocal is_updating_lut
                is_updating_lut = True
                try:
                    lut_min_slider.blockSignals(True)
                    lut_max_slider.blockSignals(True)
                    
                    # Update range based on current frame (but don't change values!)
                    # Only expand range if needed, never shrink it
                    current_min_range = lut_min_slider.minimum()
                    current_max_range = lut_min_slider.maximum()
                    new_min_range = min(current_min_range, frame_min) if current_min_range > 0 else frame_min
                    new_max_range = max(current_max_range, frame_max)
                    
                    lut_min_slider.setRange(new_min_range, new_max_range)
                    lut_max_slider.setRange(new_min_range, new_max_range)
                    
                    # DON'T change the values - preserve what user set!
                    # Only clamp if absolutely necessary (out of new range)
                    current_min_val = lut_min_slider.value()
                    current_max_val = lut_max_slider.value()
                    
                    if vmin is not None and vmax is not None:
                        # Use provided values
                        lut_min_slider.setValue(max(new_min_range, min(new_max_range, vmin)))
                        lut_max_slider.setValue(max(new_min_range, min(new_max_range, vmax)))
                    else:
                        # Preserve current values if they're still valid
                        if new_min_range <= current_min_val <= new_max_range:
                            # Keep current value
                            pass
                        else:
                            # Clamp to new range
                            lut_min_slider.setValue(max(new_min_range, min(new_max_range, current_min_val)))
                        
                        if new_min_range <= current_max_val <= new_max_range:
                            # Keep current value
                            pass
                        else:
                            # Clamp to new range
                            lut_max_slider.setValue(max(new_min_range, min(new_max_range, current_max_val)))
                    
                    lut_min_slider.blockSignals(False)
                    lut_max_slider.blockSignals(False)
                finally:
                    is_updating_lut = False
            
            # Update histogram
            update_histogram(frame, vmin, vmax)
            
            # Clear and redraw
            fig.clear()
            ax = fig.add_subplot(111)
            nonlocal im_handle
            im_handle = ax.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"Frame {frame_idx + 1} of {num_time}: {Path(file_path).name}")
            ax.axis('off')
            canvas.draw()
            
            # Update frame info (1-indexed for display)
            current_channel = (channel_spin.value() if channel_spin is not None else 1)
            frame_info.setText(f"Timepoint {frame_idx + 1} of {num_time} | Channel: {current_channel} | Z-slice: 1 | LUT: [{vmin:.1f}, {vmax:.1f}]")
        
        # Manual LUT adjustment from sliders
        def update_lut_from_sliders():
            """Update display when sliders are moved"""
            # Prevent recursive calls
            nonlocal is_updating_lut, current_vmin, current_vmax
            if is_updating_lut:
                return
            
            is_updating_lut = True
            try:
                vmin = lut_min_slider.value()
                vmax = lut_max_slider.value()
                
                # Ensure min < max for valid display
                if vmin >= vmax:
                    vmax = vmin + 1.0
                    lut_max_slider.blockSignals(True)
                    lut_max_slider.setValue(vmax)
                    lut_max_slider.blockSignals(False)
                
                # Update stored values
                current_vmin = vmin
                current_vmax = vmax
                
                # Update the image display directly using matplotlib's set_clim
                if current_frame_data['frame'] is not None and im_handle is not None:
                    # Update the color limits on the existing image
                    im_handle.set_clim(vmin=vmin, vmax=vmax)
                    # Force immediate redraw
                    canvas.draw()
                    
                    # Update frame info
                    current_channel = (channel_spin.value() if channel_spin is not None else 1)
                    frame_info.setText(f"Timepoint {preview_dialog.current_frame + 1} of {num_time} | Channel: {current_channel} | Z-slice: 1 | LUT: [{vmin:.1f}, {vmax:.1f}]")
                    
                    # Store the current LUT values so they persist
                    current_frame_data['vmin'] = vmin
                    current_frame_data['vmax'] = vmax
                    
                    # Update histogram lines (only update lines, don't redraw entire histogram)
                    if hist_vmin_line is not None and hist_vmax_line is not None:
                        # Update line positions
                        hist_vmin_line.set_xdata([vmin, vmin])
                        hist_vmax_line.set_xdata([vmax, vmax])
                        # Update the fill region - clear old and add new
                        for patch in list(hist_ax.patches):
                            if hasattr(patch, 'get_alpha') and patch.get_alpha() == 0.2:
                                patch.remove()
                        hist_ax.axvspan(vmin, vmax, alpha=0.2, color='yellow')
                        # Update legend
                        hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
                        hist_canvas.draw()
                    else:
                        # If lines don't exist yet, update full histogram
                        if current_frame_data['frame'] is not None:
                            update_histogram(current_frame_data['frame'], vmin, vmax)
                else:
                    # Fallback: full update if image handle doesn't exist
                    update_frame(preview_dialog.current_frame, vmin=vmin, vmax=vmax, update_sliders=False)
            finally:
                is_updating_lut = False
        
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
                    # Always preserve current slider values when navigating
                    # User's manual adjustments persist across frames
                    # Get current values from sliders (user's settings)
                    vmin = lut_min_slider.value()
                    vmax = lut_max_slider.value()
                    # Update frame with user's values - don't recalculate!
                    update_frame(idx, vmin=vmin, vmax=vmax, update_sliders=False)
                except Exception as e:
                    QMessageBox.warning(preview_dialog, "Preview Error", 
                                      f"Could not display frame {idx + 1}:\n{str(e)}")
        
        prev_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame - 1))
        next_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame + 1))
        
        # Connect channel selection (if available)
        if channel_spin is not None:
            def on_channel_changed():
                """Update frame when channel changes - auto-adjust LUT to new channel"""
                if preview_dialog.current_frame is not None:
                    # Reset LUT values so they recalculate for the new channel
                    nonlocal current_vmin, current_vmax
                    current_vmin = None
                    current_vmax = None
                    # This will auto-adjust min/max based on the new channel's intensity range
                    update_frame(preview_dialog.current_frame, vmin=None, vmax=None, update_sliders=True)
            channel_spin.valueChanged.connect(on_channel_changed)
        
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
        # Initialize current_frame to ensure it's set before channel changes
        preview_dialog.current_frame = 0
        update_frame(0)
        
        preview_dialog.setLayout(layout)
        preview_dialog.show()  # Use show() instead of exec_() for non-modal dialog
    
    def _preview_nd2(self):
        """Preview ND2 file in a dialog with frame navigation"""
        
        try:
            import nd2
            
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
            
            # Use unified preview UI
            self._create_unified_preview_ui(preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path)
            
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
        """Preview LIF file in a dialog with frame navigation"""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            from aicsimageio import AICSImage
            
            file_path = self.file_path.text()
            
            # Create preview dialog
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("File Preview")
            preview_dialog.setMinimumSize(1000, 800)
            preview_dialog.setModal(False)
            
            layout = QVBoxLayout()
            
            # Load LIF file
            lif_reader = AICSImage(file_path)
            try:
                # Get dimensions
                dims = lif_reader.dims
                if hasattr(dims, 'order'):
                    dims_str = dims.order
                elif isinstance(dims, str):
                    dims_str = dims
                else:
                    dims_str = str(dims)
                
                # Get data
                if hasattr(lif_reader, 'get_xarray_data'):
                    xdata = lif_reader.get_xarray_data()
                    all_frames = np.array(xdata)
                    num_time = int(xdata.sizes.get('T', 1))
                    num_channels = int(xdata.sizes.get('C', 1))
                    num_z = int(xdata.sizes.get('Z', 1))
                else:
                    all_frames = np.array(lif_reader.data)
                    shape = all_frames.shape
                    num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                    num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                    num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                
                # Reshape to (T, Y, X) for preview (take first channel and z-slice)
                if len(all_frames.shape) > 3:
                    # Take first channel and first z-slice
                    if 'C' in dims_str and 'Z' in dims_str:
                        c_idx = dims_str.index('C')
                        z_idx = dims_str.index('Z')
                        t_idx = dims_str.index('T') if 'T' in dims_str else 0
                        # Create indexing tuple
                        indices = [slice(None)] * len(all_frames.shape)
                        indices[c_idx] = 0  # First channel
                        indices[z_idx] = 0  # First z-slice
                        all_frames = all_frames[tuple(indices)]
                    elif 'C' in dims_str:
                        c_idx = dims_str.index('C')
                        indices = [slice(None)] * len(all_frames.shape)
                        indices[c_idx] = 0
                        all_frames = all_frames[tuple(indices)]
                    elif 'Z' in dims_str:
                        z_idx = dims_str.index('Z')
                        indices = [slice(None)] * len(all_frames.shape)
                        indices[z_idx] = 0
                        all_frames = all_frames[tuple(indices)]
                
                # Ensure we have (T, Y, X) shape
                if len(all_frames.shape) == 2:
                    all_frames = all_frames[np.newaxis, ...]
                elif len(all_frames.shape) == 3 and all_frames.shape[0] > 1000:
                    # Likely (Y, X, T) - transpose
                    all_frames = all_frames.transpose(2, 0, 1)
                
                # Use unified preview UI with full LUT controls
                self._create_unified_preview_ui(preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path)
                
            finally:
                if hasattr(lif_reader, 'close'):
                    try:
                        lif_reader.close()
                    except:
                        pass
                        
        except ImportError:
            QMessageBox.warning(self, "Library Missing", 
                              "The 'aicsimageio' library is required for LIF preview.\n\n"
                              "Install with: pip install aicsimageio")
        except Exception as e:
            QMessageBox.critical(self, "Error Previewing LIF", f"Could not preview LIF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _preview_czi(self):
        """Preview CZI file (stub - to be implemented)"""
        QMessageBox.information(self, "CZI Preview", 
                              "CZI file preview is not yet fully implemented.\n"
                              "Please use ND2, LIF, or TIFF files for preview.")
    
    def _preview_tiff(self):
        """Preview TIFF file in a dialog with frame navigation"""
        try:
            import tifffile
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            file_path = self.file_path.text()
            
            # Create preview dialog
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("File Preview")
            preview_dialog.setMinimumSize(1000, 800)
            preview_dialog.setModal(False)
            
            layout = QVBoxLayout()
            
            # Load TIFF file
            all_frames = safe_imread(file_path)
            
            # Determine dimensions
            ndim = len(all_frames.shape)
            num_pages = 1
            try:
                with tifffile.TiffFile(file_path) as tif:
                    num_pages = len(tif.pages)
            except:
                pass
            
            if ndim == 2:
                # Single 2D image or multi-page
                if num_pages > 1:
                    # Multi-page TIFF - load all pages
                    all_frames = []
                    with tifffile.TiffFile(file_path) as tif:
                        for page in tif.pages:
                            all_frames.append(page.asarray())
                    all_frames = np.array(all_frames)
                    num_time = len(all_frames)
                    num_channels = 1
                    num_z = 1
                else:
                    all_frames = all_frames[np.newaxis, ...]
                    num_time = 1
                    num_channels = 1
                    num_z = 1
            elif ndim == 3:
                # Could be (T, Y, X) or (C, Y, X)
                if all_frames.shape[0] < 10:
                    # Likely channels
                    num_time = 1
                    num_channels = all_frames.shape[0]
                    num_z = 1
                    # Take first channel for preview
                    all_frames = all_frames[0:1, ...]  # Keep time dimension
                else:
                    # Likely time series
                    num_time = all_frames.shape[0]
                    num_channels = 1
                    num_z = 1
            elif ndim == 4:
                # Could be (T, C, Y, X) or (T, Z, Y, X)
                if all_frames.shape[1] < 10:
                    # Likely (T, C, Y, X)
                    num_time = all_frames.shape[0]
                    num_channels = all_frames.shape[1]
                    num_z = 1
                    # Take first channel for preview
                    all_frames = all_frames[:, 0, ...]
                else:
                    # Likely (T, Z, Y, X)
                    num_time = all_frames.shape[0]
                    num_channels = 1
                    num_z = all_frames.shape[1]
                    # Take first z-slice for preview
                    all_frames = all_frames[:, 0, ...]
            else:
                # 5D or more - take first channel and z-slice
                num_time = all_frames.shape[0]
                num_channels = all_frames.shape[2] if len(all_frames.shape) > 2 else 1
                num_z = all_frames.shape[1] if len(all_frames.shape) > 1 else 1
                all_frames = all_frames[:, 0, 0, ...]  # Take first z and channel
            
            # Ensure we have (T, Y, X) shape
            if len(all_frames.shape) == 2:
                all_frames = all_frames[np.newaxis, ...]
            
            # Use unified preview UI with full LUT controls
            self._create_unified_preview_ui(preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path)
            
        except Exception as e:
            QMessageBox.critical(self, "Error Previewing TIFF", f"Could not preview TIFF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
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
            self._convert_lif()
        elif file_type == 'czi':
            self._convert_czi()
        elif file_type == 'tiff':
            QMessageBox.information(self, "TIFF File", 
                                  "The selected file is already a TIFF file.\n"
                                  "No conversion needed.")
        else:
            QMessageBox.warning(self, "Unsupported Format", f"Conversion not yet implemented for {file_type} files.")
    
    def _convert_nd2(self):
        """Convert file(s) to TIFF"""
        
        # Check for Z-stack and prompt user if max projection is not enabled
        if self.file_info and self.file_info.get('num_z', 1) > 1:
            if not self.max_z_projection.isChecked():
                reply = QMessageBox.information(
                    self, 
                    "Z-Stack Detected",
                    f"This file has Z={self.file_info['num_z']} slices.\n\n"
                    "By default, only the first Z-slice (Z=1) will be used.\n\n"
                    "To combine all Z-slices into one image, please check:\n"
                    "\"Max intensity projection for multiple Z stack\"\n\n"
                    "Would you like to enable max Z-projection now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.max_z_projection.setChecked(True)
        
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
        
        # Max Z projection
        if self.max_z_projection.isChecked():
            cmd_args.append('--max-z-projection')
        
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
    
    def _convert_lif(self):
        """Convert LIF file(s) to TIFF"""
        try:
            import numpy as np
            import tifffile
            from aicsimageio import AICSImage
            
            file_path = Path(self.file_path.text())
            
            # Check if it's a folder or single file
            if file_path.is_dir():
                # Folder mode: find all LIF files
                lif_files = sorted(
                    list(file_path.glob("*.lif")) + list(file_path.glob("*.LIF"))
                )
                
                if not lif_files:
                    QMessageBox.warning(self, "No LIF Files", 
                                      "No LIF files found in the selected folder.")
                    return
                
                total_files = len(lif_files)
                self.parent_window.log_output(f"Found {total_files} LIF file(s) to convert\n")
            else:
                # Single file mode
                lif_files = [file_path]
                total_files = 1
            
            # Determine output directory
            if self.output_dir.text():
                output_base = Path(self.output_dir.text())
            else:
                output_base = file_path.parent if file_path.is_file() else file_path
            
            # Get conversion parameters
            max_frames = self.max_frames.value()
            separate_channels = self.separate_channels.isChecked()
            selected_channel = self.channel_select.currentIndex() - 1  # -1 means all
            
            # Show progress
            self.convert_btn.setEnabled(False)
            self.convert_btn.setText("Converting...")
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(0)
            
            # Process each file
            for file_idx, lif_file in enumerate(lif_files):
                try:
                    self.parent_window.log_output(f"Processing: {lif_file.name}\n")
                    progress = int((file_idx / total_files) * 100)
                    self.parent_window.progress_bar.setValue(progress)
                    QApplication.processEvents()
                    
                    # Load LIF file
                    lif_reader = AICSImage(str(lif_file))
                    try:
                        # Get dimensions
                        dims = lif_reader.dims
                        if hasattr(dims, 'order'):
                            dims_str = dims.order
                        elif isinstance(dims, str):
                            dims_str = dims
                        else:
                            dims_str = str(dims)
                        
                        # Get data
                        if hasattr(lif_reader, 'get_xarray_data'):
                            xdata = lif_reader.get_xarray_data()
                            all_frames = np.array(xdata)
                            num_time = int(xdata.sizes.get('T', 1))
                            num_channels = int(xdata.sizes.get('C', 1))
                            num_z = int(xdata.sizes.get('Z', 1))
                        else:
                            all_frames = np.array(lif_reader.data)
                            shape = all_frames.shape
                            num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                            num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                            num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                        
                        # Limit frames
                        num_time = min(num_time, max_frames)
                        if 'T' in dims_str:
                            t_idx = dims_str.index('T')
                            indices = [slice(None)] * len(all_frames.shape)
                            indices[t_idx] = slice(0, num_time)
                            all_frames = all_frames[tuple(indices)]
                        
                        # Handle channel selection - validate bounds first
                        if selected_channel >= 0 and num_channels > 1:
                            if 'C' in dims_str:
                                c_idx = dims_str.index('C')
                                # Validate bounds before accessing
                                if c_idx < len(all_frames.shape) and selected_channel < all_frames.shape[c_idx]:
                                    indices = [slice(None)] * len(all_frames.shape)
                                    indices[c_idx] = selected_channel
                                    all_frames = all_frames[tuple(indices)]
                                    num_channels = 1
                                else:
                                    self.parent_window.log_output(f"  Warning: Selected channel {selected_channel + 1} is out of bounds (shape: {all_frames.shape}, channel dim: {c_idx}). Using all channels.\n")
                        
                        # Take first z-slice if multiple z-slices
                        if num_z > 1 and 'Z' in dims_str:
                            z_idx = dims_str.index('Z')
                            if z_idx < len(all_frames.shape):
                                indices = [slice(None)] * len(all_frames.shape)
                                indices[z_idx] = 0
                                all_frames = all_frames[tuple(indices)]
                                num_z = 1
                        
                        # Reshape to (T, C, Y, X) or (T, Y, X)
                        # CRITICAL: Don't use all_frames[0] as it might remove time dimension!
                        # Find time dimension and avoid removing it
                        if len(all_frames.shape) > 4:
                            # Find time dimension index
                            time_dim_idx = None
                            if 'T' in dims_str:
                                time_dim_idx = dims_str.index('T')
                            
                            # Remove dimensions that are not time, starting from the end
                            while len(all_frames.shape) > 4:
                                # Find a dimension to remove (not time dimension)
                                dim_to_remove = None
                                for i in range(len(all_frames.shape) - 1, -1, -1):  # Start from end
                                    if time_dim_idx is None or i != time_dim_idx:
                                        if all_frames.shape[i] == 1:
                                            dim_to_remove = i
                                            break
                                
                                if dim_to_remove is not None:
                                    # Remove singleton dimension
                                    all_frames = np.squeeze(all_frames, axis=dim_to_remove)
                                    # Update dims_str if needed
                                    if dim_to_remove < len(dims_str):
                                        dims_str = dims_str[:dim_to_remove] + dims_str[dim_to_remove+1:]
                                        if time_dim_idx is not None and dim_to_remove < time_dim_idx:
                                            time_dim_idx -= 1
                                else:
                                    # No singleton dimensions found, take first of a non-time dimension
                                    for i in range(len(all_frames.shape) - 1, -1, -1):
                                        if time_dim_idx is None or i != time_dim_idx:
                                            # Create indices to take first element of this dimension
                                            indices = [slice(None)] * len(all_frames.shape)
                                            indices[i] = 0
                                            all_frames = all_frames[tuple(indices)]
                                            # Update dims_str
                                            if i < len(dims_str):
                                                dims_str = dims_str[:i] + dims_str[i+1:]
                                                if time_dim_idx is not None and i < time_dim_idx:
                                                    time_dim_idx -= 1
                                            break
                        
                        # Create output filename
                        output_file = output_base / f"{lif_file.stem}.tif"
                        
                        # Save as TIFF - use more robust dimension handling
                        # Find channel dimension by looking at actual shape after all slicing
                        current_shape = all_frames.shape
                        channel_dim = None
                        
                        # Try to find channel dimension - look for dimension matching num_channels
                        # But only if we still have multiple channels
                        if num_channels > 1:
                            for i, size in enumerate(current_shape):
                                if size == num_channels:
                                    channel_dim = i
                                    break
                        
                        
                        # Find time dimension to preserve it
                        time_dim = None
                        if 'T' in dims_str:
                            t_idx = dims_str.index('T')
                            # After slicing, find which dimension corresponds to time
                            # Look for dimension with size matching num_time
                            for i, size in enumerate(current_shape):
                                if size == num_time and num_time > 1:
                                    time_dim = i
                                    break
                            # If time dimension not found by size, try original position
                            if time_dim is None and t_idx < len(current_shape):
                                time_dim = t_idx
                        
                        if separate_channels and num_channels > 1 and channel_dim is not None:
                            # Save each channel separately
                            max_channels = min(num_channels, current_shape[channel_dim])
                            for ch in range(max_channels):
                                # Create indexing that selects one channel
                                indices = [slice(None)] * len(current_shape)
                                indices[channel_dim] = ch
                                channel_data = all_frames[tuple(indices)]
                                
                                # Remove singleton dimensions but preserve time dimension
                                # Only squeeze dimensions that are size 1 and not the time dimension
                                axes_to_squeeze = []
                                for i, size in enumerate(channel_data.shape):
                                    if size == 1 and (time_dim is None or i != time_dim):
                                        axes_to_squeeze.append(i)
                                if axes_to_squeeze:
                                    channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                                
                                # Ensure we have at least 2D (add time dimension if needed)
                                if len(channel_data.shape) == 2:
                                    channel_data = channel_data[np.newaxis, ...]
                                
                                ch_output = output_base / f"{lif_file.stem}_ch{ch+1}.tif"
                                tifffile.imwrite(str(ch_output), channel_data, photometric='minisblack')
                                self.parent_window.log_output(f"  Saved channel {ch+1}: {ch_output.name} ({channel_data.shape})\n")
                        else:
                            # Combine channels or single channel
                            if channel_dim is not None and num_channels > 1 and current_shape[channel_dim] > 1:
                                # Use max projection across channels
                                combined = np.max(all_frames, axis=channel_dim)
                            else:
                                # Single channel or no channel dimension
                                combined = all_frames
                            
                            # Remove singleton dimensions but preserve time dimension
                            # Only squeeze dimensions that are size 1 and not the time dimension
                            axes_to_squeeze = []
                            for i, size in enumerate(combined.shape):
                                if size == 1 and (time_dim is None or i != time_dim):
                                    axes_to_squeeze.append(i)
                            if axes_to_squeeze:
                                combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                            
                            # Ensure we have at least 2D (add time dimension if needed)
                            if len(combined.shape) == 2:
                                combined = combined[np.newaxis, ...]
                            
                            # Log shape for debugging
                            self.parent_window.log_output(f"  Saving with shape: {combined.shape}\n")
                            
                            tifffile.imwrite(str(output_file), combined, photometric='minisblack')
                            self.parent_window.log_output(f"  Saved: {output_file.name}\n")
                        
                        self.parent_window.log_output(f"✓ Converted: {lif_file.name}\n")
                        
                    finally:
                        if hasattr(lif_reader, 'close'):
                            try:
                                lif_reader.close()
                            except:
                                pass
                
                except Exception as e:
                    self.parent_window.log_output(f"✗ Error converting {lif_file.name}: {str(e)}\n")
                    QMessageBox.warning(self, "Conversion Error", 
                                      f"Error converting {lif_file.name}:\n{str(e)}")
            
            # Complete
            self.parent_window.progress_bar.setValue(100)
            QMessageBox.information(self, "Conversion Complete", 
                                  f"Successfully converted {total_files} LIF file(s) to TIFF.")
            
        except ImportError:
            QMessageBox.warning(self, "Library Missing", 
                              "LIF conversion requires 'aicsimageio' and 'tifffile' libraries.\n\n"
                              "Install with: pip install aicsimageio tifffile")
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"LIF conversion failed:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText("Convert to TIFF")
            self.parent_window.progress_bar.setVisible(False)
    
    def _convert_czi(self):
        """Convert CZI file(s) to TIFF"""
        try:
            import numpy as np
            import tifffile
            from aicsimageio import AICSImage
            
            file_path = Path(self.file_path.text())
            
            # Check if it's a folder or single file
            if file_path.is_dir():
                # Folder mode: find all CZI files
                czi_files = sorted(
                    list(file_path.glob("*.czi")) + list(file_path.glob("*.CZI"))
                )
                
                if not czi_files:
                    QMessageBox.warning(self, "No CZI Files", 
                                      "No CZI files found in the selected folder.")
                    return
                
                total_files = len(czi_files)
                self.parent_window.log_output(f"Found {total_files} CZI file(s) to convert\n")
            else:
                # Single file mode
                czi_files = [file_path]
                total_files = 1
            
            # Determine output directory
            if self.output_dir.text():
                output_base = Path(self.output_dir.text())
            else:
                output_base = file_path.parent if file_path.is_file() else file_path
            
            # Get conversion parameters
            max_frames = self.max_frames.value()
            separate_channels = self.separate_channels.isChecked()
            selected_channel = self.channel_select.currentIndex() - 1  # -1 means all
            
            # Show progress
            self.convert_btn.setEnabled(False)
            self.convert_btn.setText("Converting...")
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(0)
            
            # Process each file
            for file_idx, czi_file in enumerate(czi_files):
                try:
                    self.parent_window.log_output(f"Processing: {czi_file.name}\n")
                    progress = int((file_idx / total_files) * 100)
                    self.parent_window.progress_bar.setValue(progress)
                    QApplication.processEvents()
                    
                    # Load CZI file
                    czi_reader = AICSImage(str(czi_file))
                    try:
                        # Get dimensions
                        dims = czi_reader.dims
                        if hasattr(dims, 'order'):
                            dims_str = dims.order
                        elif isinstance(dims, str):
                            dims_str = dims
                        else:
                            dims_str = str(dims)
                        
                        # Get data
                        if hasattr(czi_reader, 'get_xarray_data'):
                            xdata = czi_reader.get_xarray_data()
                            all_frames = np.array(xdata)
                            num_time = int(xdata.sizes.get('T', 1))
                            num_channels = int(xdata.sizes.get('C', 1))
                            num_z = int(xdata.sizes.get('Z', 1))
                        else:
                            all_frames = np.array(czi_reader.data)
                            shape = all_frames.shape
                            num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                            num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                            num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                        
                        # Limit frames
                        num_time = min(num_time, max_frames)
                        if 'T' in dims_str:
                            t_idx = dims_str.index('T')
                            indices = [slice(None)] * len(all_frames.shape)
                            indices[t_idx] = slice(0, num_time)
                            all_frames = all_frames[tuple(indices)]
                        
                        # Handle channel selection - validate bounds first
                        if selected_channel >= 0 and num_channels > 1:
                            if 'C' in dims_str:
                                c_idx = dims_str.index('C')
                                # Validate bounds before accessing
                                if c_idx < len(all_frames.shape) and selected_channel < all_frames.shape[c_idx]:
                                    indices = [slice(None)] * len(all_frames.shape)
                                    indices[c_idx] = selected_channel
                                    all_frames = all_frames[tuple(indices)]
                                    num_channels = 1
                                else:
                                    self.parent_window.log_output(f"  Warning: Selected channel {selected_channel + 1} is out of bounds (shape: {all_frames.shape}, channel dim: {c_idx}). Using all channels.\n")
                        
                        # Take first z-slice if multiple z-slices
                        if num_z > 1 and 'Z' in dims_str:
                            z_idx = dims_str.index('Z')
                            if z_idx < len(all_frames.shape):
                                indices = [slice(None)] * len(all_frames.shape)
                                indices[z_idx] = 0
                                all_frames = all_frames[tuple(indices)]
                                num_z = 1
                        
                        # Reshape to (T, C, Y, X) or (T, Y, X)
                        # Don't use all_frames[0] as it might remove time dimension!
                        # Instead, find and remove only non-time singleton dimensions
                        if len(all_frames.shape) > 4:
                            # Find time dimension first
                            time_dim_idx = None
                            if 'T' in dims_str:
                                time_dim_idx = dims_str.index('T')
                            
                            # Remove dimensions that are not time, starting from the end
                            while len(all_frames.shape) > 4:
                                # Find a dimension to remove (not time dimension)
                                dim_to_remove = None
                                for i in range(len(all_frames.shape) - 1, -1, -1):  # Start from end
                                    if time_dim_idx is None or i != time_dim_idx:
                                        if all_frames.shape[i] == 1:
                                            dim_to_remove = i
                                            break
                                
                                if dim_to_remove is not None:
                                    # Remove singleton dimension
                                    all_frames = np.squeeze(all_frames, axis=dim_to_remove)
                                    # Update dims_str if needed
                                    if dim_to_remove < len(dims_str):
                                        dims_str = dims_str[:dim_to_remove] + dims_str[dim_to_remove+1:]
                                        if time_dim_idx is not None and dim_to_remove < time_dim_idx:
                                            time_dim_idx -= 1
                                else:
                                    # No singleton dimensions found, take first of a non-time dimension
                                    for i in range(len(all_frames.shape) - 1, -1, -1):
                                        if time_dim_idx is None or i != time_dim_idx:
                                            # Create indices to take first element of this dimension
                                            indices = [slice(None)] * len(all_frames.shape)
                                            indices[i] = 0
                                            all_frames = all_frames[tuple(indices)]
                                            # Update dims_str
                                            if i < len(dims_str):
                                                dims_str = dims_str[:i] + dims_str[i+1:]
                                                if time_dim_idx is not None and i < time_dim_idx:
                                                    time_dim_idx -= 1
                                            break
                        
                        # Create output filename
                        output_file = output_base / f"{czi_file.stem}.tif"
                        
                        # Save as TIFF - use more robust dimension handling
                        current_shape = all_frames.shape
                        channel_dim = None
                        
                        # Try to find channel dimension - look for dimension matching num_channels
                        if num_channels > 1:
                            for i, size in enumerate(current_shape):
                                if size == num_channels:
                                    channel_dim = i
                                    break
                        
                        # Find time dimension to preserve it
                        time_dim = None
                        if 'T' in dims_str:
                            t_idx = dims_str.index('T')
                            # After slicing, find which dimension corresponds to time
                            # Look for dimension with size matching num_time
                            for i, size in enumerate(current_shape):
                                if size == num_time and num_time > 1:
                                    time_dim = i
                                    break
                            # If time dimension not found by size, try original position
                            if time_dim is None and t_idx < len(current_shape):
                                time_dim = t_idx
                        
                        if separate_channels and num_channels > 1 and channel_dim is not None:
                            # Save each channel separately
                            max_channels = min(num_channels, current_shape[channel_dim])
                            for ch in range(max_channels):
                                # Create indexing that selects one channel
                                indices = [slice(None)] * len(current_shape)
                                indices[channel_dim] = ch
                                channel_data = all_frames[tuple(indices)]
                                
                                # Remove singleton dimensions but preserve time dimension
                                # Only squeeze dimensions that are size 1 and not the time dimension
                                axes_to_squeeze = []
                                for i, size in enumerate(channel_data.shape):
                                    if size == 1 and (time_dim is None or i != time_dim):
                                        axes_to_squeeze.append(i)
                                if axes_to_squeeze:
                                    channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                                
                                # Ensure we have at least 2D (add time dimension if needed)
                                if len(channel_data.shape) == 2:
                                    channel_data = channel_data[np.newaxis, ...]
                                
                                ch_output = output_base / f"{czi_file.stem}_ch{ch+1}.tif"
                                tifffile.imwrite(str(ch_output), channel_data, photometric='minisblack')
                                self.parent_window.log_output(f"  Saved channel {ch+1}: {ch_output.name} ({channel_data.shape})\n")
                        else:
                            # Combine channels or single channel
                            if channel_dim is not None and num_channels > 1 and current_shape[channel_dim] > 1:
                                # Use max projection across channels
                                combined = np.max(all_frames, axis=channel_dim)
                            else:
                                # Single channel or no channel dimension
                                combined = all_frames
                            
                            # Remove singleton dimensions but preserve time dimension
                            # Only squeeze dimensions that are size 1 and not the time dimension
                            axes_to_squeeze = []
                            for i, size in enumerate(combined.shape):
                                if size == 1 and (time_dim is None or i != time_dim):
                                    axes_to_squeeze.append(i)
                            if axes_to_squeeze:
                                combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                            
                            # Ensure we have at least 2D (add time dimension if needed)
                            if len(combined.shape) == 2:
                                combined = combined[np.newaxis, ...]
                            
                            # Log shape for debugging
                            self.parent_window.log_output(f"  Saving with shape: {combined.shape}\n")
                            
                            tifffile.imwrite(str(output_file), combined, photometric='minisblack')
                            self.parent_window.log_output(f"  Saved: {output_file.name}\n")
                        
                        self.parent_window.log_output(f"✓ Converted: {czi_file.name}\n")
                        
                    finally:
                        if hasattr(czi_reader, 'close'):
                            try:
                                czi_reader.close()
                            except:
                                pass
                
                except Exception as e:
                    self.parent_window.log_output(f"✗ Error converting {czi_file.name}: {str(e)}\n")
                    QMessageBox.warning(self, "Conversion Error", 
                                      f"Error converting {czi_file.name}:\n{str(e)}")
            
            # Complete
            self.parent_window.progress_bar.setValue(100)
            QMessageBox.information(self, "Conversion Complete", 
                                  f"Successfully converted {total_files} CZI file(s) to TIFF.")
            
        except ImportError:
            QMessageBox.warning(self, "Library Missing", 
                              "CZI conversion requires 'aicsimageio' and 'tifffile' libraries.\n\n"
                              "Install with: pip install aicsimageio tifffile")
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"CZI conversion failed:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText("Convert to TIFF")
            self.parent_window.progress_bar.setVisible(False)
    
    def get_session_params(self):
        """Return current parameter values for session log (parameters only, no paths)."""
        return {
            "max_frames": self.max_frames.value(),
            "separate_channels": self.separate_channels.isChecked(),
            "max_z_projection": self.max_z_projection.isChecked(),
            "drift_correct": self.drift_correct.isChecked(),
            "bleach_correct": self.bleach_correct.isChecked(),
        }
    
    def set_session_params(self, data):
        """Apply saved parameter values (parameters only)."""
        if not isinstance(data, dict):
            return
        if "max_frames" in data and hasattr(self, "max_frames"):
            try:
                self.max_frames.setValue(int(data["max_frames"]))
            except (TypeError, ValueError):
                pass
        for k, w in [
            ("separate_channels", self.separate_channels), ("max_z_projection", self.max_z_projection),
            ("drift_correct", self.drift_correct), ("bleach_correct", self.bleach_correct),
        ]:
            if k in data and hasattr(w, "setChecked"):
                w.setChecked(bool(data[k]))


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
        
        # File/Folder selection with drag-and-drop support
        file_group = DragDropGroupBox(
            "Input File/Folder",
            parent=self,
            on_drop_callback=self.handle_dropped_path_segmentation
        )
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a TIFF file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(500)  # Increased for better visibility
        # Set larger font for better readability
        file_font = QFont()
        file_font.setPointSize(12)
        self.file_path.setFont(file_font)
        # Fix size to minimum - prevent expansion
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
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
        
        # Segmentation parameters (same for both modes) - side-by-side layout
        seg_group = QGroupBox("Segmentation Parameters")
        seg_container_layout = QHBoxLayout()
        
        # Left column: Basic parameters
        basic_params_group = QGroupBox("Basic Parameters")
        basic_params_layout = QVBoxLayout()
        seg_layout = basic_params_layout  # Use this for basic params
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_info_icon = create_info_icon(
            "Omnipose model type for cell segmentation.\n\n"
            "Default: 'bact_phase_omni'\n"
            "Source: Official Omnipose models (https://omnipose.readthedocs.io/models.html)\n\n"
            "Available Omnipose models:\n"
            "- bact_phase_omni: Bacterial phase contrast (recommended for bacteria)\n"
            "- bact_fluor_omni: Bacterial fluorescence\n"
            "- cyto2_omni: General cytoplasm segmentation\n"
            "- worm_omni: C. elegans and similar\n"
            "- plant_omni: Plant cells (3D)\n"
            "- bact_phase_affinity: Bacterial phase contrast (affinity-based)\n\n"
            "Cellpose models (no Omnipose):\n"
            "- cyto2: General cytoplasm (Cellpose)\n"
            "- cyto: Original cytoplasm (Cellpose)\n"
            "- nuclei: Nucleus segmentation (Cellpose)\n\n"
            "Tip: Choose based on your cell type and imaging modality."
        , parent_widget=self)
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_info_icon)
        # Model status indicator (on the left, before combo box)
        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet("color: #7F8C8D; font-size: 9pt; font-style: italic;")
        self.model_status_label.setFixedWidth(150)  # Fixed width for status text
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        model_layout.addWidget(self.model_status_label)
        model_layout.addStretch()  # Add stretch before combo box to align to right
        self.model_combo = QComboBox()
        # Validate and add only available models
        available_models = self._get_available_models()
        self.model_combo.addItems(available_models)
        if available_models:
            # Set default to bact_phase_omni if available, otherwise first available
            if 'bact_phase_omni' in available_models:
                self.model_combo.setCurrentText('bact_phase_omni')
            else:
                self.model_combo.setCurrentText(available_models[0])
        model_layout.addWidget(self.model_combo)
        
        # Check model availability when changed
        def check_model_availability():
            """Check if selected model is available (cached or downloadable)"""
            from pathlib import Path
            try:
                from cellpose_omni.models import MODEL_DIR
                model_type = self.model_combo.currentText()
                
                # Check if model exists in cache
                torch_str = 'torch'
                basename = f'{model_type}{torch_str}_0'
                cache_model_file = Path(MODEL_DIR) / basename
                
                if cache_model_file.exists():
                    self.model_status_label.setText("✓ Cached")
                    self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 9pt;")
                else:
                    self.model_status_label.setText("Will download on first use")
                    self.model_status_label.setStyleSheet("color: #FF9800; font-size: 9pt;")
            except Exception:
                # If check fails, just don't show status
                self.model_status_label.setText("")
        
        self.model_combo.currentTextChanged.connect(check_model_availability)
        # Check initial model
        check_model_availability()
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
        diam_layout.addStretch()  # Add stretch before spin box to align to right
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0, 1000)
        self.diameter.setValue(0)
        self.diameter.setDecimals(1)
        self.diameter.setFixedWidth(100)  # Fixed width for uniform appearance
        # Tooltip removed - info available via (?) icon only
        diam_layout.addWidget(self.diameter)
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
        flow_layout.addStretch()  # Add stretch before spin box to align to right
        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setRange(0.0, 1.0)
        self.flow_threshold.setValue(0.4)
        self.flow_threshold.setSingleStep(0.1)
        self.flow_threshold.setDecimals(2)
        self.flow_threshold.setFixedWidth(100)  # Fixed width for uniform appearance
        # Tooltip removed - info available via (?) icon only
        flow_layout.addWidget(self.flow_threshold)
        seg_layout.addLayout(flow_layout)
        
        basic_params_group.setLayout(basic_params_layout)
        basic_params_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        seg_container_layout.addWidget(basic_params_group)
        
        # Right column: Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # GPU option
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
        options_layout.addLayout(gpu_layout)
        
        # DBSCAN clustering option
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
        options_layout.addLayout(cluster_layout)
        
        # Overlay visualization option
        overlay_layout = QHBoxLayout()
        self.create_overlay = QCheckBox("Create overlay visualization")
        self.create_overlay.setChecked(True)
        overlay_info_icon = create_info_icon(
            "Create overlay visualization showing masks overlaid on original images.\n\n"
            "Default: Enabled (checked)\n"
            "Source: Optional output file\n\n"
            "When enabled: Creates a colored overlay TIFF file showing segmentation masks overlaid on the original images.\n"
            "When disabled: Skips overlay creation (faster processing, saves disk space).\n\n"
            "Saved as: filename_overlay.tif\n\n"
            "Tip: Disable to speed up processing when you only need masks.",
            parent_widget=self
        )
        overlay_layout.addWidget(self.create_overlay)
        overlay_layout.addWidget(overlay_info_icon)
        overlay_layout.addStretch()
        options_layout.addLayout(overlay_layout)
        
        options_group.setLayout(options_layout)
        options_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        seg_container_layout.addWidget(options_group)
        
        # Connect file path changes to update overlay checkbox state
        self.file_path.textChanged.connect(self._update_overlay_checkbox_state)
        # Initialize state
        self._update_overlay_checkbox_state()
        
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
        seg_container_layout.addWidget(reset_seg_btn)
        
        seg_group.setLayout(seg_container_layout)
        scroll_layout.addWidget(seg_group)
        
        # Preview button - modern styling applied via global stylesheet
        self.preview_btn = QPushButton("Preview Segmentation")
        self.preview_btn.setEnabled(False)  # Disabled until file/folder is selected
        # Wrap preview_segmentation to catch all exceptions and prevent PyQt6 abort()
        def safe_preview_segmentation():
            try:
                self.preview_segmentation()
            except BaseException as e:
                # PyQt6 calls abort() on unhandled exceptions in slots
                # This wrapper catches ALL exceptions to prevent crashes
                import traceback
                error_msg = f"Error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                try:
                    QMessageBox.critical(self, "Preview Error", error_msg)
                except:
                    # If showing dialog fails, at least print to console
                    print(f"CRITICAL ERROR: {error_msg}", file=sys.stderr)
                # Don't re-raise - PyQt6 will abort if we do
        self.preview_btn.clicked.connect(safe_preview_segmentation)
        scroll_layout.addWidget(self.preview_btn)
        
        # Process button
        self.process_btn = QPushButton("Process File or Folder")
        self.process_btn.setEnabled(False)  # Disabled until file/folder is selected
        # Modern styling applied via global stylesheet
        self.process_btn.clicked.connect(self.process_file_or_folder)
        scroll_layout.addWidget(self.process_btn)
        
        # Connect file path changes to update button states (after buttons are created)
        self.file_path.textChanged.connect(self._update_button_states)
        # Initialize button states
        self._update_button_states()
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def _update_button_states(self):
        """Update button enabled states based on whether file/folder is selected"""
        has_path = bool(self.file_path.text() and self.file_path.text().strip())
        if hasattr(self, 'preview_btn'):
            self.preview_btn.setEnabled(has_path)
        if hasattr(self, 'process_btn'):
            self.process_btn.setEnabled(has_path)
    
    def _update_overlay_checkbox_state(self):
        """Update overlay checkbox tooltip based on folder vs file mode"""
        is_folder = self._is_folder_mode()
        # Overlay is now supported in both modes, so always enable it
        self.create_overlay.setEnabled(True)
        if is_folder:
            self.create_overlay.setToolTip(
                "Create overlay visualization showing masks overlaid on original images.\n\n"
                "Default: Enabled (checked)\n"
                "When enabled: Creates a colored overlay TIFF file for each file in the folder.\n"
                "When disabled: Skips overlay creation (faster processing).\n\n"
                "Saved as: filename_overlay.tif (in each file's output directory)"
            )
        else:
            self.create_overlay.setToolTip(
                "Create overlay visualization showing masks overlaid on original images.\n\n"
                "Default: Enabled (checked)\n"
                "When enabled: Creates a colored overlay TIFF file.\n"
                "When disabled: Skips overlay creation (faster processing).\n\n"
                "Saved as: filename_overlay.tif"
            )
    
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
                self._update_overlay_checkbox_state()
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
            self._update_overlay_checkbox_state()
    
    def handle_dropped_path_segmentation(self, dropped_path):
        """Handle dropped file or folder path for segmentation tab"""
        path_obj = Path(dropped_path)
        # Update file path
        self.file_path.setText(dropped_path)
        # Update overlay checkbox state
        self._update_overlay_checkbox_state()
        self._update_button_states()
    
    def _is_folder_mode(self):
        """Check if current path is a folder"""
        if not self.file_path.text():
            return False
        path = Path(self.file_path.text())
        return path.exists() and path.is_dir()
    
    def _get_available_models(self):
        """
        Return list of available Omnipose models.
        Based on user's confirmed list of functional models.
        Only includes models that are known to exist and work.
        """
        # User's confirmed list of available/functional models
        # These are the models that actually exist and work
        confirmed_omnipose_models = [
            'bact_phase_omni',  # Recommended for bacteria
            'bact_fluor_omni',
            'cyto2_omni',
            'worm_omni',
            'plant_omni',
            'bact_phase_affinity'
        ]
        
        # Cellpose models (always available, not Omnipose-specific)
        cellpose_models = [
            'cyto2',
            'cyto',
            'nuclei'
        ]
        
        # Combine and return
        return confirmed_omnipose_models + cellpose_models
    
    def reset_segmentation_defaults(self):
        """Reset all segmentation parameters to their default values"""
        # Set default model, fallback to first available if bact_phase_omni not available
        if 'bact_phase_omni' in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText('bact_phase_omni')
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
        self.diameter.setValue(0)
        self.flow_threshold.setValue(0.4)
        self.use_gpu.setChecked(True)
        self.cluster.setChecked(False)
        self.create_overlay.setChecked(True)
    
    def get_session_params(self):
        """Return current parameter values for session log (parameters only, no paths)."""
        return {
            "model": self.model_combo.currentText(),
            "diameter": self.diameter.value(),
            "flow_threshold": self.flow_threshold.value(),
            "use_gpu": self.use_gpu.isChecked(),
            "cluster": self.cluster.isChecked(),
            "create_overlay": self.create_overlay.isChecked(),
        }
    
    def set_session_params(self, data):
        """Apply saved parameter values (parameters only)."""
        if not isinstance(data, dict):
            return
        if "model" in data:
            idx = self.model_combo.findText(str(data["model"]))
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        for k, w in [
            ("diameter", self.diameter), ("flow_threshold", self.flow_threshold),
        ]:
            if k in data and hasattr(w, "setValue"):
                try:
                    w.setValue(float(data[k]))
                except (TypeError, ValueError):
                    pass
        for k, w in [
            ("use_gpu", self.use_gpu), ("cluster", self.cluster), ("create_overlay", self.create_overlay),
        ]:
            if k in data and hasattr(w, "setChecked"):
                w.setChecked(bool(data[k]))
    
    def preview_segmentation(self):
        """Preview segmentation on a single frame or file from folder"""
        # Note: This method is called from safe_preview_segmentation() wrapper
        # which catches all exceptions to prevent PyQt6 from aborting
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
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
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
            
            images = safe_imread(file_path)
            
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
                        
                        if model_file and os.path.exists(model_file):
                            # Model found in cache
                            status_label.setText("Loading model from cache...")
                            QApplication.processEvents()
                        else:
                            # Model not found - need to download
                            status_label.setText("Downloading model (this may take a moment)...")
                            QApplication.processEvents()
                            # Call model_path - it will download if needed
                            try:
                                model_file = model_path(current_model_type, 0, True)
                                # Verify the model was actually downloaded
                                if not model_file or not os.path.exists(model_file):
                                    # model_path returned a path but file doesn't exist - try to download explicitly
                                    status_label.setText("Model not found, downloading...")
                                    QApplication.processEvents()
                                    # Force download by calling model_path again (it should download if file doesn't exist)
                                    model_file = model_path(current_model_type, 0, True)
                                    # Wait a moment for download to complete
                                    import time
                                    time.sleep(0.5)
                                    # Verify again
                                    if not model_file or not os.path.exists(model_file):
                                        raise FileNotFoundError(f"Model file not found after download attempt: {model_file}")
                                status_label.setText("Model downloaded, loading...")
                                QApplication.processEvents()
                            except HTTPError as http_err:
                                # Handle 404 errors (model not found in repository)
                                if http_err.code == 404:
                                    # Check if model exists in cache
                                    model_file = find_model_file(current_model_type, 0, True)
                                    if model_file and os.path.exists(model_file):
                                        status_label.setText("Using cached model (not in repo)...")
                                        QApplication.processEvents()
                                    else:
                                        # Model not available - show helpful error
                                        QMessageBox.warning(
                                            preview_dialog,
                                            "Model Not Available",
                                            f"Model '{current_model_type}' is not available in the repository (404 error).\n\n"
                                            f"Available Omnipose models:\n"
                                            f"  • bact_phase_omni (recommended for bacteria)\n"
                                            f"  • bact_fluor_omni\n"
                                            f"  • cyto2_omni\n"
                                            f"  • worm_omni\n"
                                            f"  • plant_omni\n"
                                            f"  • bact_phase_affinity\n\n"
                                            f"See: https://omnipose.readthedocs.io/models.html\n\n"
                                            f"Please select a different model from the dropdown."
                                        )
                                        status_label.setText("Model not available")
                                        return False
                                else:
                                    # Other HTTP errors - check local directory one more time
                                    model_file = find_model_file(current_model_type, 0, True)
                                    if model_file and os.path.exists(model_file):
                                        status_label.setText("Download failed, using local/cached model...")
                                        QApplication.processEvents()
                                    else:
                                        # Re-raise to be handled by outer exception handler
                                        raise
                            except Exception as download_err:
                                # Check one more time if model exists (might have been downloaded in background)
                                model_file = find_model_file(current_model_type, 0, True)
                                if model_file and os.path.exists(model_file):
                                    status_label.setText("Using model from cache...")
                                    QApplication.processEvents()
                                else:
                                    # Re-raise to be handled by outer exception handler
                                    raise
                        
                        # Final verification: ensure model file exists before loading
                        if not model_file or not os.path.exists(model_file):
                            raise FileNotFoundError(
                                f"Model file not found: {model_file}\n\n"
                                f"Model '{current_model_type}' may not be available or download failed.\n"
                                f"Please check your internet connection and try again."
                            )
                        
                        status_label.setText("Loading model...")
                        QApplication.processEvents()
                        
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
                        
                        # Special handling for 404 errors (model not found)
                        if http_err.code == 404:
                            model_file = find_model_file(current_model_type, 0, True)
                            if model_file and os.path.exists(model_file):
                                # Try to use cached model even if download failed
                                try:
                                    status_label.setText("Using cached model (not in repo)...")
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
                                        "Model Not Available",
                                        f"Model '{current_model_type}' is not available in the repository (404 error).\n\n"
                                        f"Available Omnipose models:\n"
                                        f"  • bact_phase_omni (recommended for bacteria)\n"
                                        f"  • bact_fluor_omni\n"
                                        f"  • cyto2_omni\n"
                                        f"  • worm_omni\n"
                                        f"  • plant_omni\n"
                                        f"  • bact_phase_affinity\n\n"
                                        f"See: https://omnipose.readthedocs.io/models.html\n\n"
                                        f"Please select a different model from the dropdown."
                                    )
                                    status_label.setText("Model not available")
                                    return False
                            else:
                                # Model not available and not cached
                                QMessageBox.warning(
                                    preview_dialog, 
                                    "Model Not Available",
                                    f"Model '{current_model_type}' is not available in the repository (404 error).\n\n"
                                    f"Available Omnipose models:\n"
                                    f"  • bact_phase_omni (recommended for bacteria)\n"
                                    f"  • bact_fluor_omni\n"
                                    f"  • cyto2_omni\n"
                                    f"  • worm_omni\n"
                                    f"  • plant_omni\n"
                                    f"  • bact_phase_affinity\n\n"
                                    f"See: https://omnipose.readthedocs.io/models.html\n\n"
                                    f"Please select a different model from the dropdown."
                                )
                                status_label.setText("Model not available")
                                return False
                        
                        # Check for model in local directory or cache without triggering another download
                        model_file = find_model_file(current_model_type, 0, True)
                        cached_model_exists = (model_file is not None and os.path.exists(model_file))
                        
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
                        
                        # This error occurs in frozen builds due to a bug in omnipose/core.py
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
            
            # Track if we're currently updating to prevent concurrent updates
            is_updating_frame = False
            
            def go_to_frame(idx, force_recalc=False):
                nonlocal is_updating_frame
                # Prevent concurrent frame updates
                if is_updating_frame:
                    return
                
                if not (0 <= idx < num_frames):
                    # Invalid frame index - reset to current frame
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
                    return
                
                is_updating_frame = True
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
                    import traceback
                    error_msg = f"Could not display frame {idx + 1}:\n{str(e)}"
                    if hasattr(e, '__traceback__'):
                        error_msg += f"\n\n{traceback.format_exc()}"
                    QMessageBox.warning(preview_dialog, "Preview Error", error_msg)
                finally:
                    is_updating_frame = False
            
            # Use editingFinished for when user finishes editing (presses Enter or leaves field)
            # and a debounced valueChanged for when using arrow keys
            def on_frame_changed():
                # Cancel any pending timer updates
                if hasattr(frame_spin, '_change_timer'):
                    frame_spin._change_timer.stop()
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_frames:
                    go_to_frame(frame_num)
                else:
                    # Invalid frame - reset to current
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
            
            frame_spin.editingFinished.connect(on_frame_changed)
            # Debounce valueChanged to avoid triggering while typing
            frame_spin._change_timer = QTimer()
            frame_spin._change_timer.setSingleShot(True)
            frame_spin._change_timer.timeout.connect(on_frame_changed)
            # Only start timer if value is valid
            def on_value_changed(v):
                if 1 <= v <= num_frames:
                    frame_spin._change_timer.start(300)
                else:
                    # Invalid value - cancel timer and reset
                    frame_spin._change_timer.stop()
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
            frame_spin.valueChanged.connect(on_value_changed)
            
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
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(preview_dialog.close)
            layout.addWidget(close_btn)
            
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
        except BaseException as e:
            # PyQt6 is strict about exceptions in slots - catch ALL exceptions (including SystemExit, KeyboardInterrupt)
            # This outer try-except catches any exceptions not handled by inner handlers
            import traceback
            error_msg = f"Unexpected error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Preview Error", error_msg)
            # Don't re-raise - PyQt6 will abort if we do
        except BaseException as e:
            # Final catch-all for any exceptions that slip through
            # This prevents PyQt6 from calling abort()
            import traceback
            error_msg = f"Critical error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            try:
                QMessageBox.critical(self, "Preview Error", error_msg)
            except:
                # If even showing the error dialog fails, at least print to console
                print(f"CRITICAL ERROR in preview_segmentation: {error_msg}", file=sys.stderr)
            # Don't re-raise - PyQt6 will abort if we do
    
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
            input_file = Path(self.file_path.text())
            
            # Determine base directory (go up one level if input is in Tif_* folder)
            base_dir = input_file.parent
            if base_dir.name.startswith('Tif_'):
                base_dir = base_dir.parent
            
            # Create output directory at base level with _segmentation suffix
            output_dir = base_dir / f"{input_file.stem}_segmentation"
            
            cmd_args = [
                self.file_path.text(),
                '--output', str(output_dir),
                '--model', self.model_combo.currentText(),
                '--diameter', str(self.diameter.value()),
                '--flow-threshold', str(self.flow_threshold.value()),
                '--batch-size', '32',  # Increased batch size for faster GPU processing (adjust if OOM errors occur)
            ]
            
            if not self.use_gpu.isChecked():
                cmd_args.append('--no-gpu')
            if self.cluster.isChecked():
                cmd_args.append('--cluster')
            if not self.create_overlay.isChecked():
                cmd_args.append('--no-overlay')
            
            self.process_btn.setEnabled(False)
            self.process_btn.setText("Processing...")
            self.parent_window.log_output(f"Starting single file processing: {self.file_path.text()}\n")
            self.parent_window.log_output(f"Output directory: {output_dir}\n")
        else:
            # Folder mode - show file selection dialog first
            folder = Path(self.file_path.text())
            # Find TIFF files (not mask files)
            tiff_files_set = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                            set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out mask files and output files
            tiff_files = sorted([f for f in tiff_files_set 
                                if '_masks' not in f.name.lower() 
                                and '_overlay' not in f.name.lower()
                                and '_filter_visualization' not in f.name.lower()], 
                               key=lambda x: x.name.lower())
            
            if not tiff_files:
                QMessageBox.warning(self, "No TIFF Files Found", 
                                  "No TIFF files found in the selected folder.\n\n"
                                  "Please select a folder containing TIFF files to process.")
                return
            
            # Show file selection dialog
            selected_files = show_file_selection_dialog(
                self,
                tiff_files,
                title="Select Files for Segmentation",
                message=f"Select files to process (all {len(tiff_files)} files selected by default):"
            )
            
            if selected_files is None:
                # User cancelled or no files selected
                return
            
            if len(selected_files) == 0:
                QMessageBox.warning(self, "No Files Selected", 
                                  "Please select at least one file to process.")
                return
            
            # Process selected files - use batch_process_folder.py but we'll need to filter files
            # For now, let's process each file individually (simpler approach)
            script_path = str(script_dir / "batch_process_tiff_sequence.py")
            
            # Store files and current index for sequential processing
            self.folder_files = selected_files
            self.current_file_index = 0
            # Track progress within current file (for batch/frame progress)
            self.current_file_progress = 0.0  # 0.0 to 1.0
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range for percentage display
            self.parent_window.progress_bar.setValue(0)
            
            self.process_btn.setEnabled(False)
            self.process_btn.setText(f"Processing... (0/{len(selected_files)})")
            self.parent_window.log_output(f"Starting batch folder processing: {len(selected_files)} file(s) selected from {self.file_path.text()}\n")
            
            # Process first file
            self._process_next_segmentation_file(script_path)
            return  # Early return, processing continues in _process_next_segmentation_file
        
        # Show progress bar
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range for percentage display
        self.parent_window.progress_bar.setValue(0)
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self._parse_segmentation_progress)
        self.worker.start()
    
    def _process_next_segmentation_file(self, script_path):
        """Process next file in folder mode (segmentation)"""
        if self.current_file_index >= len(self.folder_files):
            # All files processed
            self.process_btn.setEnabled(True)
            is_folder = self._is_folder_mode()
            if is_folder:
                self.process_btn.setText("Process All Files in Folder")
            else:
                self.process_btn.setText("Process File")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output(f"\n✓ Segmentation complete for all {len(self.folder_files)} file(s)!\n")
            QMessageBox.information(self, "Processing Complete", 
                                  f"Successfully processed {len(self.folder_files)} file(s).")
            return
        
        tiff_file = self.folder_files[self.current_file_index]
        self.parent_window.log_output(f"\nProcessing file {self.current_file_index + 1}/{len(self.folder_files)}: {tiff_file.name}\n")
        
        # Determine base directory (go up one level if input is in Tif_* folder)
        base_dir = tiff_file.parent
        if base_dir.name.startswith('Tif_'):
            base_dir = base_dir.parent
        
        # Create output directory at base level with _segmentation suffix
        output_dir = base_dir / f"{tiff_file.stem}_segmentation"
        
        # Build command arguments for this file
        cmd_args = [
            str(tiff_file),
            '--output', str(output_dir),
            '--model', self.model_combo.currentText(),
            '--diameter', str(self.diameter.value()),
            '--flow-threshold', str(self.flow_threshold.value()),
            '--batch-size', '32',  # Increased batch size for faster GPU processing (adjust if OOM errors occur)
        ]
        
        if not self.use_gpu.isChecked():
            cmd_args.append('--no-gpu')
        if self.cluster.isChecked():
            cmd_args.append('--cluster')
        if not self.create_overlay.isChecked():
            cmd_args.append('--no-overlay')
        
        self.parent_window.log_output(f"Output directory: {output_dir}\n")
        
        # Reset current file progress when starting a new file
        self.current_file_progress = 0.0
        
        # Update progress bar to show file start
        file_progress = int((self.current_file_index / len(self.folder_files)) * 100)
        self.parent_window.progress_bar.setValue(file_progress)
        self.process_btn.setText(f"Processing... ({self.current_file_index + 1}/{len(self.folder_files)})")
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(lambda success, msg: self._on_segmentation_file_finished(success, msg, script_path))
        self.worker.progress.connect(self._parse_segmentation_progress)
        self.worker.start()
    
    def _on_segmentation_file_finished(self, success, message, script_path):
        """Handle completion of a single file in folder mode"""
        self.parent_window.log_output(message)
        if success:
            self.parent_window.log_output(f"✓ File {self.current_file_index + 1}/{len(self.folder_files)} processed successfully!\n")
            # Mark file as 100% complete
            self.current_file_progress = 1.0
        else:
            self.parent_window.log_output(f"✗ File {self.current_file_index + 1}/{len(self.folder_files)} failed!\n")
            # Still mark as complete (even if failed) for progress tracking
            self.current_file_progress = 1.0
        
        # Update progress bar to show file completion
        if hasattr(self, 'folder_files') and len(self.folder_files) > 0:
            file_portion = 100.0 / len(self.folder_files)
            file_progress_base = ((self.current_file_index + 1) / len(self.folder_files)) * 100
            self.parent_window.progress_bar.setValue(int(min(100, file_progress_base)))
        
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        
        # Move to next file
        self.current_file_index += 1
        self._process_next_segmentation_file(script_path)
    
    def _parse_segmentation_progress(self, message):
        """Parse progress messages from segmentation script and update progress bar"""
        import re
        # Log the message
        self.parent_window.log_output(message)
        
        # Update overall progress based on file, batch, and frame progress
        if hasattr(self, 'folder_files') and len(self.folder_files) > 0:
            # Initialize current_file_progress if not set
            if not hasattr(self, 'current_file_progress'):
                self.current_file_progress = 0.0
            
            # Handle model download progress (small portion of first file, ~5%)
            if 'Downloading:' in message or 'to /' in message and '.cellpose/models/' in message:
                download_match = re.search(r'(\d+)%', message)
                if download_match:
                    try:
                        download_pct = int(download_match.group(1))
                        # Model download is ~5% of first file's progress
                        if self.current_file_index == 0:
                            self.current_file_progress = (download_pct / 100.0) * 0.05
                    except:
                        pass
            
            # Try to parse batch completion: "✓ Batch 2/6 complete: 64/181 frames (35%)"
            batch_complete_match = re.search(r'Batch (\d+)/(\d+) complete', message)
            if batch_complete_match:
                try:
                    batch_num = int(batch_complete_match.group(1))
                    total_batches = int(batch_complete_match.group(2))
                    # Update progress: batch_num/total_batches (e.g., 2/6 = 0.33)
                    self.current_file_progress = batch_num / total_batches
                except:
                    pass
            
            # Try to parse batch start: "Processing batch 3/6: frames 65-96 of 181"
            batch_start_match = re.search(r'Processing batch (\d+)/(\d+):', message)
            if batch_start_match and not batch_complete_match:
                try:
                    batch_num = int(batch_start_match.group(1))
                    total_batches = int(batch_start_match.group(2))
                    # Progress at start of batch (e.g., batch 3/6 means we're at 2/6 = 0.33)
                    # But if this is the first batch and we haven't started processing yet, set to 0.05 (after model load)
                    if batch_num == 1 and self.current_file_progress < 0.05:
                        self.current_file_progress = 0.05
                    else:
                        self.current_file_progress = (batch_num - 1) / total_batches
                except:
                    pass
            
            # Try to parse frame progress within batch: "12%|#2 | 4/32 [00:19<02:18, 4.95s/it]"
            frame_progress_match = re.search(r'(\d+)%\s*\|\s*\d+/(\d+)', message)
            if frame_progress_match:
                try:
                    frame_pct = int(frame_progress_match.group(1))
                    # Find which batch we're in
                    batch_info = re.search(r'batch (\d+)/(\d+)', message)
                    if batch_info:
                        batch_num = int(batch_info.group(1))
                        total_batches = int(batch_info.group(2))
                        # Progress = (batch-1)/total + (frame_pct/100)/total
                        batch_progress = (batch_num - 1) / total_batches
                        frame_progress = (frame_pct / 100.0) / total_batches
                        self.current_file_progress = batch_progress + frame_progress
                    else:
                        # Fallback: use frame percentage directly
                        self.current_file_progress = frame_pct / 100.0
                except:
                    pass
            
            # Calculate overall progress across all files
            file_portion = 100.0 / len(self.folder_files)
            # Base progress from completed files
            file_progress_base = (self.current_file_index / len(self.folder_files)) * 100
            # Current file's progress scaled to its portion
            file_progress_current = self.current_file_progress * file_portion
            # Total progress
            total_progress = int(min(100, file_progress_base + file_progress_current))
            self.parent_window.progress_bar.setValue(total_progress)
        else:
            # Single file mode - just use percentage from message
            progress_match = re.search(r'(\d+)%', message)
            if progress_match:
                try:
                    progress_pct = int(progress_match.group(1))
                    self.parent_window.progress_bar.setValue(progress_pct)
                except:
                    pass
    
    def on_finished(self, success, message):
        self.process_btn.setEnabled(True)
        is_folder = self._is_folder_mode()
        if is_folder:
            self.process_btn.setText("Process All Files in Folder")
        else:
            self.process_btn.setText("Process File")
        self.parent_window.log_output(message)
        # Set progress to 100% on completion
        if success:
            self.parent_window.progress_bar.setValue(100)
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
        
        # File/Folder selection with drag-and-drop support
        file_group = DragDropGroupBox(
            "Mask File/Folder",
            parent=self,
            on_drop_callback=self.handle_dropped_path_analyze
        )
        file_font = QFont()
        file_font.setPointSize(12)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        # Mask File/Folder row: same layout as Segmentation tab (path + buttons, no stretch)
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a mask TIFF file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(500)
        self.file_path.setFont(file_font)
        self.file_path.setSizePolicy(size_policy)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)
        folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.file_path, 0)
        file_layout.addWidget(file_btn, 0)
        file_layout.addWidget(folder_btn, 0)
        file_group.setLayout(file_layout)
        scroll_layout.addWidget(file_group)
        
        # Combine row: same layout as Mask File/Folder (path left-aligned, buttons adjacent, no stretch)
        combine_group = QGroupBox("Combine existing summaries from folder")
        combine_layout = QHBoxLayout()
        self.combine_folder_path = QLineEdit()
        self.combine_folder_path.setPlaceholderText("Folder with *_analysis.xlsx (subfolders)...")
        self.combine_folder_path.setMinimumWidth(500)
        self.combine_folder_path.setFont(file_font)
        self.combine_folder_path.setSizePolicy(size_policy)
        combine_browse_btn = QPushButton("Browse...")
        combine_browse_btn.setFixedWidth(120)
        combine_browse_btn.clicked.connect(self._browse_combine_folder)
        combine_btn = QPushButton("Combine")
        combine_btn.setFixedWidth(130)
        combine_btn.setToolTip("Read all Summary sheets from analysis Excel files in the selected folder and its subfolders, combine into one wide-format table, and save as {folder_name}_combined_summary.xlsx in the selected folder.")
        combine_btn.clicked.connect(self._combine_summaries_from_folder)
        combine_layout.addWidget(self.combine_folder_path, 0)
        combine_layout.addWidget(combine_browse_btn, 0)
        combine_layout.addWidget(combine_btn, 0)
        combine_group.setLayout(combine_layout)
        scroll_layout.addWidget(combine_group)
        
        # Analysis parameters - side-by-side layout (compact to avoid horizontal scroll)
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_container_layout = QHBoxLayout()
        analysis_container_layout.setSpacing(8)
        analysis_container_layout.setContentsMargins(6, 6, 6, 6)
        
        # First box: Basic and Morphological filters (fused into one)
        morph_params_group = QGroupBox("Morphological Filters")
        morph_params_layout = QVBoxLayout()
        
        pixel_layout = QHBoxLayout()
        pixel_label = QLabel("Pixel Size (µm):")
        pixel_info_icon = create_info_icon(
            "Pixel size in micrometers (µm) for area calculations.\n\n"
            "Default: 0.108 µm (or from metadata if available)\n"
            "Source: Hardcoded default, or extracted from metadata\n\n"
            "This value is used to convert pixel areas to µm².\n"
            "If you loaded a file with pixel size metadata, this will be auto-filled.\n\n"
            "Common values:\n"
            "- 100x objective: ~0.108 µm/pixel\n"
            "- 60x objective: ~0.18 µm/pixel\n"
            "- 40x objective: ~0.27 µm/pixel\n\n"
            "Tip: Check your microscope settings or ND2 metadata for the correct value."
        , parent_widget=self)
        pixel_layout.addWidget(pixel_label)
        pixel_layout.addWidget(pixel_info_icon)
        # Add indicator label for auto-filled values or warning (before spin box, on the left)
        self.pixel_size_indicator = QLabel("⚠ Check value!")
        self.pixel_size_indicator.setStyleSheet("color: #FFA500; font-size: 9pt; font-weight: bold;")
        self.pixel_size_indicator.setFixedWidth(100)  # Fixed width for warning text
        self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.pixel_size_indicator.setToolTip("Default value - verify this matches your metadata!")
        pixel_layout.addWidget(self.pixel_size_indicator)
        pixel_layout.addStretch()  # Add stretch before spin box to align to right
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.001, 10.0)
        self.pixel_size.setDecimals(3)
        self.pixel_size.setSingleStep(0.001)
        # Always default to 0.108 on initialization (metadata will update it if a file is loaded)
        # Use explicit float to avoid any precision issues
        default_pixel_size = float(0.108)
        self.pixel_size.setValue(default_pixel_size)
        self.pixel_size.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        # valueChanged will only fire when user finishes editing (Enter or focus loss)
        self.pixel_size.setKeyboardTracking(False)
        # Track if metadata was loaded to show warning
        self.pixel_size_metadata_loaded = False
        # Connect value change to hide warning when user manually changes value
        self.pixel_size.valueChanged.connect(self._on_pixel_size_changed)
        # Debug: Verify the value was set correctly (can be removed later)
        # print(f"DEBUG: Pixel size set to: {self.pixel_size.value()}")
        pixel_layout.addWidget(self.pixel_size)
        morph_params_layout.addLayout(pixel_layout)
        
        # Morphological filters (now part of the same group)
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
        aspect_min_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_aspect_ratio = QDoubleSpinBox()
        self.min_aspect_ratio.setRange(1.0, 20.0)
        self.min_aspect_ratio.setValue(1.5)
        self.min_aspect_ratio.setDecimals(1)
        self.min_aspect_ratio.setSingleStep(0.1)
        self.min_aspect_ratio.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_aspect_ratio.setKeyboardTracking(False)
        aspect_min_layout.addWidget(self.min_aspect_ratio)
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
        aspect_max_layout.addStretch()  # Add stretch before spin box to align to right
        self.max_aspect_ratio = QDoubleSpinBox()
        self.max_aspect_ratio.setRange(1.0, 50.0)
        self.max_aspect_ratio.setValue(10.0)
        self.max_aspect_ratio.setDecimals(1)
        self.max_aspect_ratio.setSingleStep(0.1)
        self.max_aspect_ratio.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.max_aspect_ratio.setKeyboardTracking(False)
        aspect_max_layout.addWidget(self.max_aspect_ratio)
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
        ecc_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_eccentricity = QDoubleSpinBox()
        self.min_eccentricity.setRange(0.0, 1.0)
        self.min_eccentricity.setValue(0.5)
        self.min_eccentricity.setDecimals(2)
        self.min_eccentricity.setSingleStep(0.05)
        self.min_eccentricity.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_eccentricity.setKeyboardTracking(False)
        ecc_layout.addWidget(self.min_eccentricity)
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
        sol_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_solidity = QDoubleSpinBox()
        self.min_solidity.setRange(0.0, 1.0)
        self.min_solidity.setValue(0.7)
        self.min_solidity.setDecimals(2)
        self.min_solidity.setSingleStep(0.05)
        self.min_solidity.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_solidity.setKeyboardTracking(False)
        sol_layout.addWidget(self.min_solidity)
        morph_layout.addLayout(sol_layout)
        
        # Add morphological filters to the main layout
        morph_params_layout.addLayout(morph_layout)
        
        morph_params_group.setLayout(morph_params_layout)
        morph_params_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        morph_params_group.setContentsMargins(4, 6, 4, 6)
        analysis_container_layout.addWidget(morph_params_group)
        
        # Second box: Size filters (side-by-side)
        size_group = QGroupBox("Size Filters")
        size_layout = QVBoxLayout()
        
        min_area_layout = QHBoxLayout()
        min_area_label = QLabel("Min Area (µm², 0=no limit):")
        min_area_info_icon = create_info_icon(
            "Minimum cell area in square micrometers (µm²).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Filters out cells smaller than this value (likely debris or noise).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for bacteria: 0.5-5 µm²\n"
            "Range: 0 to 10000 µm²\n\n"
            "Tip: Use to filter out undersized objects that are likely artifacts."
        , parent_widget=self)
        min_area_layout.addWidget(min_area_label)
        min_area_layout.addWidget(min_area_info_icon)
        min_area_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_area = QDoubleSpinBox()
        self.min_area.setRange(0, 10000)
        self.min_area.setValue(0)
        self.min_area.setDecimals(1)
        self.min_area.setSingleStep(0.5)
        self.min_area.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_area.setKeyboardTracking(False)
        min_area_layout.addWidget(self.min_area)
        size_layout.addLayout(min_area_layout)
        
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
        max_area_layout.addStretch()  # Add stretch before spin box to align to right
        self.max_area = QDoubleSpinBox()
        self.max_area.setRange(0, 10000)
        self.max_area.setValue(0)
        self.max_area.setDecimals(1)
        self.max_area.setSingleStep(1.0)
        self.max_area.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.max_area.setKeyboardTracking(False)
        max_area_layout.addWidget(self.max_area)
        size_layout.addLayout(max_area_layout)
        
        min_width_layout = QHBoxLayout()
        min_width_label = QLabel("Min Width (µm, 0=no limit):")
        min_width_info_icon = create_info_icon(
            "Minimum cell width in micrometers (µm).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Width is measured as the minor axis (perpendicular to length).\n"
            "Filters out cells narrower than this value (likely debris or noise).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for rod-shaped bacteria: 0.3-0.5 µm\n"
            "Range: 0 to 100 µm\n\n"
            "Tip: Use to filter out objects that are too narrow to be single cells."
        , parent_widget=self)
        min_width_layout.addWidget(min_width_label)
        min_width_layout.addWidget(min_width_info_icon)
        min_width_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_width = QDoubleSpinBox()
        self.min_width.setRange(0, 100)
        self.min_width.setValue(0)
        self.min_width.setDecimals(1)
        self.min_width.setSingleStep(0.1)
        self.min_width.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_width.setKeyboardTracking(False)
        min_width_layout.addWidget(self.min_width)
        size_layout.addLayout(min_width_layout)
        
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
        max_width_layout.addStretch()  # Add stretch before spin box to align to right
        self.max_width = QDoubleSpinBox()
        self.max_width.setRange(0, 100)
        self.max_width.setValue(0)
        self.max_width.setDecimals(1)
        self.max_width.setSingleStep(0.1)
        self.max_width.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.max_width.setKeyboardTracking(False)
        max_width_layout.addWidget(self.max_width)
        size_layout.addLayout(max_width_layout)
        
        # Min Length filter (5th size filter to match left side)
        min_length_layout = QHBoxLayout()
        min_length_label = QLabel("Min Length (µm, 0=no limit):")
        min_length_info_icon = create_info_icon(
            "Minimum cell length in micrometers (µm).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Length is measured as the major axis (longest dimension).\n"
            "Filters out cells shorter than this value (likely debris or noise).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for rod-shaped bacteria: 1.0-3.0 µm\n"
            "Range: 0 to 100 µm\n\n"
            "Tip: Use to filter out objects that are too short to be single cells.",
            parent_widget=self
        )
        min_length_layout.addWidget(min_length_label)
        min_length_layout.addWidget(min_length_info_icon)
        min_length_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_length = QDoubleSpinBox()
        self.min_length.setRange(0, 100)
        self.min_length.setValue(0)
        self.min_length.setDecimals(1)
        self.min_length.setSingleStep(0.1)
        self.min_length.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_length.setKeyboardTracking(False)
        min_length_layout.addWidget(self.min_length)
        size_layout.addLayout(min_length_layout)
        
        size_group.setLayout(size_layout)
        size_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        size_group.setContentsMargins(4, 6, 4, 6)
        analysis_container_layout.addWidget(size_group)
        
        # Reset to defaults button for analysis parameters (at end of horizontal row)
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
        analysis_container_layout.addWidget(reset_analysis_btn)
        
        analysis_group.setLayout(analysis_container_layout)
        scroll_layout.addWidget(analysis_group)
        
        # Options and Analyze FI (side by side)
        options_container_layout = QHBoxLayout()
        
        # Left box: Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # Row 1: Include detailed per-cell data + Create filter visualization
        options_row1 = QHBoxLayout()
        detailed_layout = QHBoxLayout()
        self.detailed = QCheckBox("Include detailed per-cell data")
        self.detailed.setChecked(True)  # Default to checked
        detailed_info_icon = create_info_icon(
            "Include detailed per-cell measurements in the Excel output.\n\n"
            "Default: Enabled (checked)\n"
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
        options_row1.addLayout(detailed_layout)
        
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
        options_row1.addLayout(vis_layout)
        options_layout.addLayout(options_row1)
        
        # Row 2: Create combined summary (below Include per-cell data)
        combined_summary_layout = QHBoxLayout()
        self.create_combined_summary = QCheckBox("Create combined summary")
        self.create_combined_summary.setChecked(False)
        combined_summary_info_icon = create_info_icon(
            "When analyzing a folder of mask files, create one extra Excel that combines all Summary sheets.\n\n"
            "Output: {folder_name}_combined_summary.xlsx with one sheet:\n"
            "- 'Combined Summary (wide)': one row per frame, one set of columns per file.\n\n"
            "Use for: Downstream plotting, stats, or comparing many samples in one table.",
            parent_widget=self
        )
        combined_summary_layout.addWidget(self.create_combined_summary)
        combined_summary_layout.addWidget(combined_summary_info_icon)
        combined_summary_layout.addStretch()
        options_layout.addLayout(combined_summary_layout)
        
        options_group.setLayout(options_layout)
        options_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        options_container_layout.addWidget(options_group)
        
        # Right box: Analyze FI (with drag-and-drop support)
        fi_group = DragDropGroupBox(
            "Analyze FI",
            parent=self,
            on_drop_callback=self.handle_dropped_fi_file
        )
        fi_layout = QVBoxLayout()
        
        fi_checkbox_layout = QHBoxLayout()
        self.analyze_fi = QCheckBox("Analyze FI")
        self.analyze_fi.setChecked(False)
        fi_info_icon = create_info_icon(
            "Analyze fluorescence intensity (FI) from a separate FI channel image.\n\n"
            "When enabled: Requires selecting an FI channel TIFF file.\n"
            "The analysis will add three columns to the Excel output:\n"
            "- Mean FI intensity (average pixel intensity within each cell)\n"
            "- Contour FI intensity (average intensity at cell boundary)\n"
            "- FI ratio (contour FI / mean FI)\n\n"
            "Use when: You have a separate fluorescence channel image and want to quantify\n"
            "fluorescence intensity per cell.",
            parent_widget=self
        )
        fi_checkbox_layout.addWidget(self.analyze_fi)
        fi_checkbox_layout.addWidget(fi_info_icon)
        
        # Contour thickness parameter (on same line as Analyze FI checkbox)
        contour_thickness_label = QLabel("Contour thickness (pixels):")
        contour_thickness_info_icon = create_info_icon(
            "Thickness of the contour region used for FI intensity calculation.\n\n"
            "Default: 1 pixel (single-pixel boundary)\n"
            "Range: 1-10 pixels\n\n"
            "The contour is the region at the cell boundary.\n"
            "Increasing thickness includes more pixels near the boundary.\n\n"
            "Note: This affects both the preview visualization and the final analysis.",
            parent_widget=self
        )
        fi_checkbox_layout.addWidget(contour_thickness_label)
        fi_checkbox_layout.addWidget(contour_thickness_info_icon)
        self.fi_contour_thickness = QSpinBox()
        self.fi_contour_thickness.setRange(1, 10)
        self.fi_contour_thickness.setValue(1)  # Default: 1 pixel (current behavior)
        self.fi_contour_thickness.setEnabled(False)  # Disabled by default
        self.fi_contour_thickness.setFixedWidth(80)
        fi_checkbox_layout.addWidget(self.fi_contour_thickness)
        fi_checkbox_layout.addStretch()
        fi_layout.addLayout(fi_checkbox_layout)
        
        # FI file browse (only enabled when checkbox is checked) with drag-and-drop
        fi_file_layout = QHBoxLayout()
        self.fi_file_path = QLineEdit()
        self.fi_file_path.setPlaceholderText("Select FI channel TIFF file... (or drag and drop)")
        self.fi_file_path.setMinimumWidth(400)  # Slightly smaller since it's in a side box
        self.fi_file_path.setFont(file_font)
        self.fi_file_path.setSizePolicy(size_policy)
        self.fi_file_path.setEnabled(False)  # Disabled by default
        self.fi_file_btn = QPushButton("Browse FI File...")
        self.fi_file_btn.setFixedWidth(120)
        self.fi_file_btn.setEnabled(False)  # Disabled by default
        self.fi_file_btn.clicked.connect(self.select_fi_file)
        fi_file_layout.addWidget(self.fi_file_path, 0)
        fi_file_layout.addWidget(self.fi_file_btn, 0)
        fi_layout.addLayout(fi_file_layout)
        
        # Connect checkbox to enable/disable FI file selection and contour thickness
        self.analyze_fi.toggled.connect(self._update_fi_file_state)
        
        fi_group.setLayout(fi_layout)
        fi_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        options_container_layout.addWidget(fi_group)
        
        scroll_layout.addLayout(options_container_layout)
        
        # Preview and Analyze buttons
        # Preview button - modern styling applied via global stylesheet
        self.preview_btn = QPushButton("Preview Filter Effects")
        self.preview_btn.setToolTip("Preview how current filter parameters would affect the selected mask file")
        self.preview_btn.setEnabled(False)  # Disabled until file/folder is selected
        self.preview_btn.clicked.connect(self.preview_filter_effects)
        scroll_layout.addWidget(self.preview_btn)
        
        self.analyze_btn = QPushButton("Analyze Masks")
        self.analyze_btn.setEnabled(False)  # Disabled until file/folder is selected
        # Modern styling applied via global stylesheet
        self.analyze_btn.clicked.connect(self.analyze_masks)
        scroll_layout.addWidget(self.analyze_btn)
        
        # Connect file path changes to update button states
        self.file_path.textChanged.connect(self._update_analyze_button_state)
        # Initialize state
        self._update_analyze_button_state()
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def _update_analyze_button_state(self):
        """Update analyze and preview button enabled state based on whether file/folder is selected"""
        has_path = bool(self.file_path.text() and self.file_path.text().strip())
        if hasattr(self, 'analyze_btn'):
            self.analyze_btn.setEnabled(has_path)
        if hasattr(self, 'preview_btn'):
            self.preview_btn.setEnabled(has_path)
    
    def select_file(self):
        """Open file dialog to select a mask TIFF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask TIFF File", "", "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.file_path.setText(file_path)
            self._update_analyze_button_state()
    
    def select_folder(self):
        """Open folder dialog to select a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Mask Files", ""
        )
        if folder_path:
            self.file_path.setText(folder_path)
            self._update_analyze_button_state()
    
    def handle_dropped_path_analyze(self, dropped_path):
        """Handle dropped file or folder path for analyze masks tab"""
        path_obj = Path(dropped_path)
        # Update file path
        self.file_path.setText(dropped_path)
        # Update button states
        self._update_analyze_button_state()
    
    def select_fi_file(self):
        """Open file dialog to select an FI channel TIFF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select FI Channel TIFF File", "", "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.fi_file_path.setText(file_path)
    
    def handle_dropped_fi_file(self, dropped_path):
        """Handle dropped FI file for analyze masks tab"""
        path_obj = Path(dropped_path)
        # Check if it's a valid TIFF file
        if path_obj.is_file() and path_obj.suffix.lower() in ['.tif', '.tiff']:
            # Enable the analyze_fi checkbox if not already enabled
            if not self.analyze_fi.isChecked():
                self.analyze_fi.setChecked(True)
            # Set the file path
            self.fi_file_path.setText(dropped_path)
        else:
            QMessageBox.warning(self, "Invalid File", 
                              "Please drop a valid TIFF file (.tif or .tiff).")
    
    def _update_fi_file_state(self, checked):
        """Enable/disable FI file selection and contour thickness based on checkbox state"""
        self.fi_file_path.setEnabled(checked)
        if hasattr(self, 'fi_file_btn'):
            self.fi_file_btn.setEnabled(checked)
        if hasattr(self, 'fi_contour_thickness'):
            self.fi_contour_thickness.setEnabled(checked)
    
    def _is_folder_mode(self):
        """Check if current path is a folder"""
        if not self.file_path.text():
            return False
        path = Path(self.file_path.text())
        return path.exists() and path.is_dir()
    
    def _browse_combine_folder(self):
        """Open folder dialog for 'Combine summaries from folder' path."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Analysis Excel Files", ""
        )
        if folder_path:
            self.combine_folder_path.setText(folder_path)
    
    def _build_combined_summary_excel(self, excel_paths, output_path):
        """Build long + wide combined summary from a list of analysis Excel paths. Returns True on success."""
        import pandas as pd
        if not excel_paths:
            return False
        frames_long = []
        for p in excel_paths:
            p = Path(p)
            if not p.exists():
                continue
            try:
                df = pd.read_excel(p, sheet_name="Summary")
            except Exception:
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            df["File"] = p.stem.replace("_analysis", "").replace("_masks_analysis", "")
            frames_long.append(df)
        if not frames_long:
            return False
        long_df = pd.concat(frames_long, ignore_index=True)
        # Ensure column order: File, Frame, then rest
        cols = [c for c in ["File", "Frame"] if c in long_df.columns]
        cols += [c for c in long_df.columns if c not in cols]
        long_df = long_df[cols]
        # Wide format: one row per frame, one set of columns per file
        wide_df = None
        try:
            frames = sorted(long_df["Frame"].dropna().unique().astype(int).tolist())
            wide_data = {"Frame": frames}
            for f in long_df["File"].unique().tolist():
                sub = long_df[long_df["File"] == f].set_index("Frame")
                for col in ["Total Cell Area (µm²)", "Total Cell Area (Relative)", "Number of Cells"]:
                    if col in sub.columns:
                        wide_data[f"{f}_{col}"] = sub.reindex(frames)[col].values
            wide_df = pd.DataFrame(wide_data)
        except Exception:
            wide_df = None
        if wide_df is None or wide_df.empty:
            return False
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                wide_df.to_excel(writer, sheet_name="Combined Summary (wide)", index=False)
        except Exception:
            return False
        return True
    
    def _combine_summaries_from_folder(self):
        """Combine Summary sheets from all analysis Excel files in the folder given by combine_folder_path."""
        path_text = self.combine_folder_path.text().strip()
        if not path_text:
            QMessageBox.warning(self, "No Folder", "Please select a folder (Browse...) that contains analysis Excel files.")
            return
        folder = Path(path_text)
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "Invalid Folder", "The selected path is not an existing folder.")
            return
        # Search in folder and all subfolders (one Excel per subfolder layout)
        excel_paths = list(folder.rglob("*_analysis.xlsx")) + list(folder.rglob("*_masks_analysis.xlsx"))
        excel_paths = sorted(set(excel_paths), key=lambda p: (str(p).lower(), p.name.lower()))
        if not excel_paths:
            QMessageBox.warning(self, "No Analysis Files", 
                "No *_analysis.xlsx or *_masks_analysis.xlsx files found in the selected folder or its subfolders.")
            return
        out_path = folder / f"{folder.name}_combined_summary.xlsx"
        if self._build_combined_summary_excel([str(p) for p in excel_paths], str(out_path)):
            self.parent_window.log_output(f"Combined summary saved to: {out_path}\n")
            QMessageBox.information(self, "Combine Complete", 
                f"Combined {len(excel_paths)} Summary sheet(s) into:\n{out_path.name}\n\nSheet: 'Combined Summary (wide)'.")
        else:
            QMessageBox.warning(self, "Combine Failed", "Could not read any Summary sheets from the Excel files in the selected folder.")
    
    def _on_pixel_size_changed(self):
        """Hide warning when user manually changes pixel size (they've verified it)"""
        if not self.pixel_size_metadata_loaded:
            # User manually changed the value, hide the warning
            self.pixel_size_indicator.setText("")
            self.pixel_size_indicator.setFixedWidth(0)
            self.pixel_size_indicator.hide()
    
    def reset_analysis_defaults(self):
        """Reset all analysis parameters to their default values"""
        # Pixel size: always reset to 0.108 (default value)
        default_pixel_size = float(0.108)
        self.pixel_size.setValue(default_pixel_size)
        self.pixel_size_metadata_loaded = False  # Reset metadata flag
        self.pixel_size_indicator.setText("⚠ Check value!")  # Show warning
        self.pixel_size_indicator.setStyleSheet("color: #FFA500; font-size: 9pt; font-weight: bold;")
        self.pixel_size_indicator.setFixedWidth(100)  # Keep width for warning
        self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.pixel_size_indicator.setToolTip("Default value - verify this matches your microscope settings!\n\nIf you converted an ND2 file, pixel size should be auto-filled.\nOtherwise, check your microscope/objective settings.")
        self.pixel_size_indicator.show()
        
        # Morphological filters
        self.min_aspect_ratio.setValue(1.5)
        self.max_aspect_ratio.setValue(10.0)
        self.min_eccentricity.setValue(0.5)
        self.min_solidity.setValue(0.7)
        
        # Size filters
        self.min_area.setValue(0)
        self.max_area.setValue(0)
        self.min_width.setValue(0)
        self.max_width.setValue(0)
        self.min_length.setValue(0)
        
        # Options: "Include detailed per-cell data" stays checked by default (user must actively deselect)
        self.detailed.setChecked(True)
        self.create_vis.setChecked(True)
        if hasattr(self, "create_combined_summary"):
            self.create_combined_summary.setChecked(False)
    
    def get_session_params(self):
        """Return current parameter values for session log (parameters only, no paths)."""
        d = {
            "pixel_size": self.pixel_size.value(),
            "min_aspect_ratio": self.min_aspect_ratio.value(),
            "max_aspect_ratio": self.max_aspect_ratio.value(),
            "min_eccentricity": self.min_eccentricity.value(),
            "min_solidity": self.min_solidity.value(),
            "min_area": self.min_area.value(),
            "max_area": self.max_area.value(),
            "min_width": self.min_width.value(),
            "max_width": self.max_width.value(),
            "min_length": self.min_length.value(),
            "detailed": self.detailed.isChecked(),
            "create_vis": self.create_vis.isChecked(),
        }
        if hasattr(self, "create_combined_summary"):
            d["create_combined_summary"] = self.create_combined_summary.isChecked()
        if hasattr(self, "analyze_fi"):
            d["analyze_fi"] = self.analyze_fi.isChecked()
        if hasattr(self, "fi_contour_thickness"):
            d["fi_contour_thickness"] = self.fi_contour_thickness.value()
        return d
    
    def set_session_params(self, data):
        """Apply saved parameter values (parameters only). Ignores unknown keys."""
        if not isinstance(data, dict):
            return
        for k, v in [
            ("pixel_size", self.pixel_size), ("min_aspect_ratio", self.min_aspect_ratio),
            ("max_aspect_ratio", self.max_aspect_ratio), ("min_eccentricity", self.min_eccentricity),
            ("min_solidity", self.min_solidity), ("min_area", self.min_area),
            ("max_area", self.max_area), ("min_width", self.min_width),
            ("max_width", self.max_width), ("min_length", self.min_length),
        ]:
            if k in data and hasattr(v, "setValue"):
                try:
                    v.setValue(float(data[k]))
                except (TypeError, ValueError):
                    pass
        if "detailed" in data and hasattr(self, "detailed"):
            self.detailed.setChecked(bool(data["detailed"]))
        if "create_vis" in data and hasattr(self, "create_vis"):
            self.create_vis.setChecked(bool(data["create_vis"]))
        if "create_combined_summary" in data and hasattr(self, "create_combined_summary"):
            self.create_combined_summary.setChecked(bool(data["create_combined_summary"]))
        if "analyze_fi" in data and hasattr(self, "analyze_fi"):
            self.analyze_fi.setChecked(bool(data["analyze_fi"]))
        if "fi_contour_thickness" in data and hasattr(self, "fi_contour_thickness"):
            try:
                self.fi_contour_thickness.setValue(int(data["fi_contour_thickness"]))
            except (TypeError, ValueError):
                pass
    
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
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
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
        loading_progress = QProgressDialog("Loading mask file and initializing preview...", None, 0, 100, self)
        loading_progress.setWindowTitle("Loading")
        loading_progress.setAutoClose(False)
        loading_progress.setAutoReset(False)
        loading_progress.setCancelButton(None)
        loading_progress.setValue(0)
        loading_progress.show()
        QApplication.processEvents()
        
        try:
            loading_progress.setLabelText("Loading mask file... (20%)")
            loading_progress.setValue(20)
            QApplication.processEvents()
            
            import tifffile
            from skimage.measure import regionprops
            from skimage.morphology import remove_small_objects, binary_dilation
            from skimage.segmentation import find_boundaries
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure
            
            # Load masks
            masks = safe_imread(mask_path)
            
            loading_progress.setLabelText("Processing mask data... (50%)")
            loading_progress.setValue(50)
            QApplication.processEvents()
            if masks.ndim == 2:
                masks = masks[np.newaxis, :, :]
            num_frames = masks.shape[0]
            
            # Load FI images if FI analysis is enabled
            fi_images = None
            fi_contour_thickness = 1
            if self.analyze_fi.isChecked() and self.fi_file_path.text():
                try:
                    fi_path = Path(self.fi_file_path.text())
                    if fi_path.exists():
                        loading_progress.setLabelText("Loading FI images... (60%)")
                        loading_progress.setValue(60)
                        QApplication.processEvents()
                        fi_images = safe_imread(str(fi_path))
                        # Handle different dimensions (same logic as in analyze_cell_areas.py)
                        if fi_images.ndim == 2:
                            fi_images = fi_images[np.newaxis, :, :]
                        elif fi_images.ndim == 3 and fi_images.shape[0] != num_frames:
                            # Might be (C, Y, X) or (Y, X, C) - try to handle
                            if fi_images.shape[2] < 10:  # Likely (Y, X, C)
                                fi_images = fi_images[np.newaxis, :, :, 0]  # Take first channel
                            else:  # Likely (C, Y, X)
                                fi_images = fi_images[0, np.newaxis, :, :]  # Take first channel
                        # Ensure frame count matches
                        if fi_images.shape[0] != num_frames:
                            QMessageBox.warning(preview_dialog, "Frame Mismatch", 
                                              f"FI image has {fi_images.shape[0]} frames but mask has {num_frames} frames.")
                            fi_images = None
                        fi_contour_thickness = self.fi_contour_thickness.value()
                except Exception as e:
                    QMessageBox.warning(preview_dialog, "FI Load Error", 
                                      f"Could not load FI image:\n{str(e)}")
                    fi_images = None
            
            loading_progress.setLabelText("Initializing preview dialog... (80%)")
            loading_progress.setValue(80)
            QApplication.processEvents()
            
            # Create preview dialog (non-modal so user can access main window)
            loading_progress.setLabelText("Ready! (100%)")
            loading_progress.setValue(100)
            QApplication.processEvents()
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
            
            # Filename label
            filename_label = QLabel(f"File: {Path(mask_path).name}")
            filename_label.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 5px; color: #2C3E50;")
            layout.addWidget(filename_label)
            
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
                    'min_area': self.min_area.value() if self.min_area.value() > 0 else None,
                    'max_area': self.max_area.value() if self.max_area.value() > 0 else None,
                    'min_width': self.min_width.value() if self.min_width.value() > 0 else None,
                    'max_width': self.max_width.value() if self.max_width.value() > 0 else None,
                    'fi_contour_thickness': self.fi_contour_thickness.value() if hasattr(self, 'fi_contour_thickness') else 1,
                }
            
            # Track if we're currently showing the warning dialog to prevent duplicates
            preview_dialog._showing_warning = False
            
            # Function to update preview
            def update_preview(frame_idx, show_progress=True):
                if frame_idx < 0 or frame_idx >= num_frames:
                    return
                
                # Prevent multiple simultaneous calls
                if hasattr(preview_dialog, '_updating_preview') and preview_dialog._updating_preview:
                    return
                preview_dialog._updating_preview = True
                
                # Get current FI contour thickness (may have changed)
                nonlocal fi_contour_thickness
                if hasattr(self, 'fi_contour_thickness'):
                    fi_contour_thickness = self.fi_contour_thickness.value()
                
                # Reload FI images if FI file path changed or wasn't loaded before
                nonlocal fi_images
                if self.analyze_fi.isChecked() and self.fi_file_path.text():
                    fi_path = Path(self.fi_file_path.text())
                    if fi_path.exists() and (fi_images is None or str(fi_path) != str(Path(mask_path).parent / "fi_temp")):
                        try:
                            fi_images = safe_imread(str(fi_path))
                            # Handle different dimensions (same logic as above)
                            if fi_images.ndim == 2:
                                fi_images = fi_images[np.newaxis, :, :]
                            elif fi_images.ndim == 3 and fi_images.shape[0] != num_frames:
                                if fi_images.shape[2] < 10:
                                    fi_images = fi_images[np.newaxis, :, :, 0]
                                else:
                                    fi_images = fi_images[0, np.newaxis, :, :]
                            if fi_images.shape[0] != num_frames:
                                fi_images = None
                        except Exception:
                            fi_images = None
                elif not self.analyze_fi.isChecked():
                    fi_images = None
                
                # Show progress dialog
                progress_dialog = None
                cancelled = False
                if show_progress:
                    progress_dialog = QProgressDialog("Applying filters...", "Cancel", 0, 100, preview_dialog)
                    progress_dialog.setWindowTitle("Processing")
                    progress_dialog.setAutoClose(False)
                    progress_dialog.setAutoReset(False)
                    progress_dialog.setValue(0)
                    progress_dialog.show()
                    QApplication.processEvents()
                    
                    def check_cancelled():
                        nonlocal cancelled
                        if progress_dialog.wasCanceled():
                            cancelled = True
                    progress_dialog.canceled.connect(check_cancelled)
                
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
                        progress_dialog.setLabelText("Cleaning mask... (10%)")
                        progress_dialog.setValue(10)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
                    # Remove small objects (more aggressive pre-filtering for performance)
                    mask_cleaned = remove_small_objects(mask, min_size=10)
                    
                    # Check number of unique labels (excluding background)
                    unique_labels = np.unique(mask_cleaned)
                    num_objects = len(unique_labels) - (1 if 0 in unique_labels else 0)
                    
                    # Warn if too many objects (performance/memory concern)
                    MAX_OBJECTS_WARNING = 5000
                    MAX_OBJECTS_VIZ = 10000  # Skip visualization if more than this
                    skip_visualization = False
                    
                    if num_objects > MAX_OBJECTS_WARNING:
                        # Prevent showing multiple warning dialogs simultaneously
                        if hasattr(preview_dialog, '_showing_warning') and preview_dialog._showing_warning:
                            # Already showing a warning, skip this one
                            skip_visualization = True
                        else:
                            if progress_dialog:
                                progress_dialog.close()
                            # Mark that we're showing a warning
                            preview_dialog._showing_warning = True
                            
                            # Use modal message box to prevent multiple instances
                            msg = QMessageBox(preview_dialog)
                            msg.setWindowTitle("Many Objects Detected")
                            msg.setModal(True)  # Make it modal to prevent multiple instances
                            msg.setText(f"This frame contains {num_objects:,} objects.")
                            msg.setInformativeText(
                                f"Processing may take a long time and could use significant memory.\n\n"
                                f"Options:\n"
                                f"• 'Yes' = Continue with full processing (may be slow)\n"
                                f"• 'No' = Skip visualization, only show statistics (faster)\n"
                                f"• 'Cancel' = Skip this frame entirely"
                            )
                            yes_btn = msg.addButton("Yes", QMessageBox.ButtonRole.YesRole)
                            no_btn = msg.addButton("No (Stats Only)", QMessageBox.ButtonRole.NoRole)
                            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                            msg.setDefaultButton(no_btn)
                            msg.exec()
                            
                            # Clear the flag after dialog is closed
                            preview_dialog._showing_warning = False
                        
                        if msg.clickedButton() == cancel_btn:
                            # Re-enable buttons
                            prev_btn.setEnabled(True)
                            next_btn.setEnabled(True)
                            reapply_btn.setEnabled(True)
                            frame_spin.setEnabled(True)
                            return
                        elif msg.clickedButton() == no_btn:
                            skip_visualization = True
                        
                        # Recreate progress dialog if user continues
                        if show_progress:
                            progress_dialog = QProgressDialog("Applying filters...", "Cancel", 0, 100, preview_dialog)
                            progress_dialog.setWindowTitle("Processing")
                            progress_dialog.setAutoClose(False)
                            progress_dialog.setAutoReset(False)
                            progress_dialog.setValue(10)
                            progress_dialog.show()
                            progress_dialog.canceled.connect(check_cancelled)
                            QApplication.processEvents()
                    
                    # Auto-skip visualization for extremely large datasets
                    if num_objects > MAX_OBJECTS_VIZ:
                        skip_visualization = True
                    
                    # Get current parameters from main window
                    params = get_current_parameters()
                    
                    if progress_dialog:
                        progress_dialog.setLabelText(f"Calculating cell properties... ({num_objects:,} objects) (30%)")
                        progress_dialog.setValue(30)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
                    # Calculate properties (optimized: only request needed properties)
                    # Use a more efficient approach: request only the properties we need
                    properties = calculate_cell_properties(mask_cleaned, pixel_size=params['pixel_size'])
                    all_labels = {prop['label'] for prop in properties}
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Applying morphological filters... (50%)")
                        progress_dialog.setValue(50)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
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
                        progress_dialog.setLabelText("Applying size filters... (70%)")
                        progress_dialog.setValue(70)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
                    filtered = filter_by_size(
                        rod_shaped,
                        min_area_um2=params['min_area'],
                        max_area_um2=params['max_area'],
                        min_width_um=params['min_width'],
                        max_width_um=params['max_width']
                    )
                    accepted_labels = {cell['label'] for cell in filtered}
                    rejected_labels = all_labels - accepted_labels
                    
                    if progress_dialog:
                        if skip_visualization:
                            progress_dialog.setLabelText("Skipping visualization (too many objects)... (85%)")
                        else:
                            progress_dialog.setLabelText("Creating visualization... (85%)")
                        progress_dialog.setValue(85)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
                    # Create visualization using vectorized numpy operations (much faster)
                    # Skip visualization for very large datasets to prevent crashes
                    if skip_visualization:
                        # Show statistics only
                        fig.clear()
                        ax = fig.add_subplot(111)
                        ax.text(0.5, 0.5, 
                               f"Visualization skipped due to large number of objects ({num_objects:,})\n\n"
                               f"Statistics:\n"
                               f"• Total objects: {len(all_labels):,}\n"
                               f"• Accepted: {len(accepted_labels):,}\n"
                               f"• Rejected: {len(rejected_labels):,}\n"
                               f"• Acceptance rate: {len(accepted_labels)/len(all_labels)*100:.1f}%",
                               ha='center', va='center', fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}: Statistics Only (Visualization Skipped)")
                        ax.axis('off')
                        canvas.draw()
                    else:
                        height, width = mask.shape
                        vis = np.zeros((height, width, 3), dtype=np.uint8)
                        
                        # Convert label sets to arrays for vectorized operations
                        if accepted_labels:
                            accepted_array = np.array(list(accepted_labels))
                            # Use numpy.isin for fast membership testing
                            accepted_mask = np.isin(mask, accepted_array)
                            vis[accepted_mask, 1] = 255  # Green channel
                        
                        if rejected_labels:
                            rejected_array = np.array(list(rejected_labels))
                            rejected_mask = np.isin(mask, rejected_array)
                            vis[rejected_mask, 0] = 255  # Red channel
                        
                        # Add FI contours if FI analysis is enabled
                        if fi_images is not None and self.analyze_fi.isChecked():
                            try:
                                # Get FI image for this frame
                                fi_image = fi_images[frame_idx]
                                # Handle multi-channel FI images
                                if fi_image.ndim > 2:
                                    fi_image = fi_image[:, :, 0] if fi_image.shape[2] < 10 else fi_image[0, :, :]
                                
                                # Create mask with only accepted cells
                                accepted_mask_full = np.zeros_like(mask, dtype=bool)
                                if accepted_labels:
                                    accepted_array = np.array(list(accepted_labels))
                                    accepted_mask_full = np.isin(mask, accepted_array)
                                
                                # Find boundaries for accepted cells only
                                if np.any(accepted_mask_full):
                                    # Create a temporary mask with only accepted cells
                                    temp_mask = np.zeros_like(mask)
                                    temp_mask[accepted_mask_full] = mask[accepted_mask_full]
                                    # Find boundaries
                                    boundaries = find_boundaries(temp_mask, mode='inner')
                                    
                                    # Apply thickness using dilation
                                    if fi_contour_thickness > 1:
                                        # Dilate the boundaries to create thicker contours
                                        for _ in range(fi_contour_thickness - 1):
                                            boundaries = binary_dilation(boundaries)
                                    
                                    # Draw contours in bright yellow (red + green channels)
                                    contour_mask = boundaries & accepted_mask_full
                                    vis[contour_mask, 0] = 255  # Red channel
                                    vis[contour_mask, 1] = 255  # Green channel
                                    vis[contour_mask, 2] = 0     # Blue channel (yellow = red+green)
                            except Exception as e:
                                # If contour drawing fails, continue without contours
                                print(f"Warning: Could not draw FI contours: {str(e)}")
                        
                        if progress_dialog:
                            progress_dialog.setLabelText("Rendering display... (100%)")
                            progress_dialog.setValue(100)
                            QApplication.processEvents()
                            if cancelled:
                                return
                        
                        # Update plot
                        fig.clear()
                        ax = fig.add_subplot(111)
                        ax.imshow(vis)
                        title = f"Frame {frame_idx + 1}/{num_frames}: Green=Accepted ({len(accepted_labels)}), Red=Rejected ({len(rejected_labels)})"
                        if fi_images is not None and self.analyze_fi.isChecked():
                            title += f", Yellow=FI Contours (thickness={fi_contour_thickness})"
                        ax.set_title(title)
                        ax.axis('off')
                        canvas.draw()
                    
                    info_label.setText(f"Accepted: {len(accepted_labels)} | Rejected: {len(rejected_labels)} | Total: {len(all_labels)}")
                    
                    # Close progress dialog
                    if progress_dialog:
                        progress_dialog.setValue(100)
                        QApplication.processEvents()
                        progress_dialog.close()
                    
                    # Re-enable buttons
                    if progress_dialog:
                        prev_btn.setEnabled(True)
                        next_btn.setEnabled(True)
                        reapply_btn.setEnabled(True)
                        frame_spin.setEnabled(True)
                    
                    # Clear the updating flag
                    preview_dialog._updating_preview = False
                except Exception as e:
                    if progress_dialog:
                        progress_dialog.close()
                    # Re-enable buttons on error
                    prev_btn.setEnabled(True)
                    next_btn.setEnabled(True)
                    reapply_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    # Clear flags on error
                    if hasattr(preview_dialog, '_updating_preview'):
                        preview_dialog._updating_preview = False
                    if hasattr(preview_dialog, '_showing_warning'):
                        preview_dialog._showing_warning = False
                    QMessageBox.warning(preview_dialog, "Preview Error", f"Could not update preview:\n{str(e)}")
            
            def go_to_frame(idx):
                if 0 <= idx < num_frames:
                    # Stop any pending timer to prevent conflicts
                    if hasattr(frame_spin, '_change_timer'):
                        frame_spin._change_timer.stop()
                    preview_dialog.current_frame = idx
                    # Update spinbox without triggering signals (to avoid recursion)
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(idx + 1)  # Display is 1-based, index is 0-based
                    frame_spin.blockSignals(False)
                    try:
                        # Only show progress if it's a new frame (not cached)
                        update_preview(idx, show_progress=False)  # Filter preview is usually fast
                        # Ensure spinbox still shows correct value after update_preview
                        if preview_dialog.current_frame == idx:
                            frame_spin.blockSignals(True)
                            frame_spin.setValue(idx + 1)
                            frame_spin.blockSignals(False)
                    except Exception as e:
                        QMessageBox.warning(preview_dialog, "Preview Error", 
                                          f"Could not display frame {idx + 1}:\n{str(e)}")
            
            prev_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame - 1))
            next_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame + 1))
            
            # Use editingFinished for when user finishes editing (presses Enter or leaves field)
            # and a debounced valueChanged for when using arrow keys
            def on_frame_changed():
                # Cancel any pending timer updates to prevent multiple calls
                if hasattr(frame_spin, '_change_timer'):
                    frame_spin._change_timer.stop()
                frame_num = frame_spin.value() - 1  # Convert from 1-based display to 0-based index
                if 0 <= frame_num < num_frames:
                    # Only go to frame if it's different from current to avoid unnecessary updates
                    if frame_num != preview_dialog.current_frame:
                        go_to_frame(frame_num)
                else:
                    # Invalid frame - reset to current
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
            
            frame_spin.editingFinished.connect(on_frame_changed)
            # Debounce valueChanged to avoid triggering while typing
            frame_spin._change_timer = QTimer()
            frame_spin._change_timer.setSingleShot(True)
            frame_spin._change_timer.timeout.connect(on_frame_changed)
            # Only start timer if value is valid
            def on_value_changed(v):
                if 1 <= v <= num_frames:
                    frame_spin._change_timer.start(300)
                else:
                    # Invalid value - cancel timer and reset
                    frame_spin._change_timer.stop()
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
            frame_spin.valueChanged.connect(on_value_changed)
            
            nav_layout.addStretch()
            
            # Re-apply filters button (moved to match segmentation preview layout)
            reapply_btn = QPushButton("Re-apply Filters with Current Parameters")
            reapply_btn.setToolTip(
                "Re-apply filters on the current frame using the current parameters from the main window.\n"
                "You can change filter parameters in the main window (it stays accessible) and click this button to see the effect."
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
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(preview_dialog.close)
            layout.addWidget(close_btn)
            
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
            
            # Show file selection dialog
            selected_files = show_file_selection_dialog(
                self,
                mask_files,
                title="Select Mask Files for Analysis",
                message=f"Select mask files to analyze (all {len(mask_files)} files selected by default):"
            )
            
            if selected_files is None:
                # User cancelled or no files selected
                return
            
            if len(selected_files) == 0:
                QMessageBox.warning(self, "No Files Selected", 
                                  "Please select at least one file to analyze.")
                return
            
            # Process each file sequentially
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText(f"Analyzing... (0/{len(selected_files)})")
            self.parent_window.log_output(f"Starting mask analysis for {len(selected_files)} file(s) selected from folder: {folder}\n")
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, len(selected_files))
            self.parent_window.progress_bar.setValue(0)
            
            # Store files, current index, and output paths (for combined summary)
            self.folder_files = selected_files
            self.current_file_index = 0
            self.folder_results = []
            self.folder_output_paths = []
            
            # Process first file
            self._process_next_folder_file(script_path)
        else:
            # Single file mode
            input_file = Path(self.file_path.text())
            
            # Determine base directory (go up one level if input is in Tif_* or _segmentation folder)
            base_dir = input_file.parent
            if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name:
                base_dir = base_dir.parent
            
            # Create output path at base level with _analysis suffix
            output_path = base_dir / f"{input_file.stem}_analysis.xlsx"
            
            # Build command arguments
            cmd_args = [
                self.file_path.text(),
                '--output', str(output_path),
                '--pixel-size', str(self.pixel_size.value()),
                '--min-aspect-ratio', str(self.min_aspect_ratio.value()),
                '--max-aspect-ratio', str(self.max_aspect_ratio.value()),
                '--min-eccentricity', str(self.min_eccentricity.value()),
                '--min-solidity', str(self.min_solidity.value()),
            ]
            
            if self.min_area.value() > 0:
                cmd_args.extend(['--min-area', str(self.min_area.value())])
            if self.max_area.value() > 0:
                cmd_args.extend(['--max-area', str(self.max_area.value())])
            if self.min_width.value() > 0:
                cmd_args.extend(['--min-width', str(self.min_width.value())])
            if self.max_width.value() > 0:
                cmd_args.extend(['--max-width', str(self.max_width.value())])
            if self.min_length.value() > 0:
                cmd_args.extend(['--min-length', str(self.min_length.value())])
            if self.detailed.isChecked():
                cmd_args.append('--detailed')
            if self.create_vis.isChecked():
                cmd_args.append('--visualization')
            
            # Add FI image path if FI analysis is enabled
            if self.analyze_fi.isChecked() and self.fi_file_path.text():
                fi_path = Path(self.fi_file_path.text())
                if fi_path.exists():
                    cmd_args.extend(['--fi-image', str(fi_path)])
                    # Add contour thickness parameter
                    fi_contour_thickness = self.fi_contour_thickness.value()
                    cmd_args.extend(['--fi-contour-thickness', str(fi_contour_thickness)])
                else:
                    QMessageBox.warning(self, "FI File Not Found", 
                                      f"The selected FI file does not exist:\n{fi_path}")
                    return
            
            # Run in thread
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText("Analyzing...")
            self.parent_window.log_output(f"Starting mask analysis: {self.file_path.text()}\n")
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range
            self.parent_window.progress_bar.setValue(0)
            
            # Store reference to prevent garbage collection
            self.worker = WorkerThread(script_path, cmd_args, parent=self)
            self.worker.finished.connect(self.on_finished)
            self.worker.progress.connect(self._parse_analyze_progress)
            self.worker.start()
    
    def _parse_analyze_progress(self, message):
        """Parse progress messages from analyze script and update progress bar"""
        import re
        # Log the message
        self.parent_window.log_output(message)
        
        # Try to parse tqdm-style progress: "Creating visualization: 25%|"
        # Or percentage in message
        progress_match = re.search(r'(\d+)%', message)
        if progress_match:
            try:
                progress_pct = int(progress_match.group(1))
                # Clamp to 0-100
                progress_pct = max(0, min(100, progress_pct))
                self.parent_window.progress_bar.setValue(progress_pct)
            except:
                pass
        # Also check for "X/Y" patterns which might indicate progress
        elif 'Creating visualization' in message or 'Processing' in message:
            # If we see these messages, increment progress slightly
            current = self.parent_window.progress_bar.value()
            if current < 90:
                self.parent_window.progress_bar.setValue(min(90, current + 1))
    
    def _process_next_folder_file(self, script_path):
        """Process next file in folder mode"""
        if self.current_file_index >= len(self.folder_files):
            # All files processed
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("Analyze Masks")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output(f"\n✓ Analysis complete for all {len(self.folder_files)} file(s)!\n")
            # Create combined summary if option was enabled and we have output paths
            if getattr(self, "create_combined_summary", None) and getattr(self, "create_combined_summary").isChecked():
                paths = getattr(self, "folder_output_paths", [])
                if paths and self.folder_files:
                    folder = self.folder_files[0].parent
                    out_path = folder / f"{folder.name}_combined_summary.xlsx"
                    if self._build_combined_summary_excel(paths, str(out_path)):
                        self.parent_window.log_output(f"Combined summary saved to: {out_path}\n")
                        QMessageBox.information(self, "Analysis Complete", 
                            f"Successfully analyzed {len(self.folder_files)} mask file(s).\n\n"
                            f"Combined summary saved to:\n{out_path.name}")
                        return
            QMessageBox.information(self, "Analysis Complete", 
                                  f"Successfully analyzed {len(self.folder_files)} mask file(s).")
            return
        
        mask_file = self.folder_files[self.current_file_index]
        self.parent_window.log_output(f"\nProcessing file {self.current_file_index + 1}/{len(self.folder_files)}: {mask_file.name}\n")
        
        # Determine base directory (go up one level if input is in Tif_* or _segmentation folder)
        base_dir = mask_file.parent
        if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name:
            base_dir = base_dir.parent
        
        # Create output path at base level with _analysis suffix
        output_path = base_dir / f"{mask_file.stem}_analysis.xlsx"
        self.folder_output_paths.append(str(output_path))
        
        # Build command arguments
        cmd_args = [
            str(mask_file),
            '--output', str(output_path),
            '--pixel-size', str(self.pixel_size.value()),
            '--min-aspect-ratio', str(self.min_aspect_ratio.value()),
            '--max-aspect-ratio', str(self.max_aspect_ratio.value()),
            '--min-eccentricity', str(self.min_eccentricity.value()),
            '--min-solidity', str(self.min_solidity.value()),
        ]
        
        if self.min_area.value() > 0:
            cmd_args.extend(['--min-area', str(self.min_area.value())])
        if self.max_area.value() > 0:
            cmd_args.extend(['--max-area', str(self.max_area.value())])
        if self.min_width.value() > 0:
            cmd_args.extend(['--min-width', str(self.min_width.value())])
        if self.max_width.value() > 0:
            cmd_args.extend(['--max-width', str(self.max_width.value())])
        if self.min_length.value() > 0:
            cmd_args.extend(['--min-length', str(self.min_length.value())])
        if self.detailed.isChecked():
            cmd_args.append('--detailed')
        if self.create_vis.isChecked():
            cmd_args.append('--visualization')
        
        # Add FI image path if FI analysis is enabled
        if self.analyze_fi.isChecked() and self.fi_file_path.text():
            fi_path = Path(self.fi_file_path.text())
            if fi_path.exists():
                cmd_args.extend(['--fi-image', str(fi_path)])
                # Add contour thickness parameter
                fi_contour_thickness = self.fi_contour_thickness.value()
                cmd_args.extend(['--fi-contour-thickness', str(fi_contour_thickness)])
            else:
                QMessageBox.warning(self, "FI File Not Found", 
                                  f"The selected FI file does not exist:\n{fi_path}")
                return
        
        self.parent_window.log_output(f"Output file: {output_path}\n")
        
        # Update progress (for folder mode, show file progress)
        file_progress = int((self.current_file_index / len(self.folder_files)) * 100)
        self.parent_window.progress_bar.setValue(file_progress)
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


class TrackingTab(QWidget):
    """Tab for cell tracking using trackastra"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.device_mapping = {}  # Initialize device mapping
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Mask file selection with drag-and-drop support
        mask_group = DragDropGroupBox(
            "Mask File",
            parent=self,
            on_drop_callback=self.handle_dropped_mask_file
        )
        mask_layout = QHBoxLayout()
        self.mask_path = QLineEdit()
        self.mask_path.setPlaceholderText("Select mask TIFF file (time series)... (or drag and drop)")
        self.mask_path.setMinimumWidth(500)
        file_font = QFont()
        file_font.setPointSize(11)
        self.mask_path.setFont(file_font)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.mask_path.setSizePolicy(size_policy)
        mask_btn = QPushButton("Browse File...")
        mask_btn.setFixedWidth(120)
        mask_btn.clicked.connect(self.select_mask_file)
        mask_layout.addWidget(self.mask_path, 0)
        mask_layout.addWidget(mask_btn, 0)
        mask_group.setLayout(mask_layout)
        scroll_layout.addWidget(mask_group)
        
        # Original data file selection (optional) with drag-and-drop support
        original_group = DragDropGroupBox(
            "Original Data File (Optional)",
            parent=self,
            on_drop_callback=self.handle_dropped_original_file
        )
        original_layout = QHBoxLayout()
        self.original_path = QLineEdit()
        self.original_path.setPlaceholderText("Select original TIFF file for visualization (optional)... (or drag and drop)")
        self.original_path.setMinimumWidth(500)
        self.original_path.setFont(file_font)
        self.original_path.setSizePolicy(size_policy)
        original_btn = QPushButton("Browse File...")
        original_btn.setFixedWidth(120)
        original_btn.clicked.connect(self.select_original_file)
        original_layout.addWidget(self.original_path, 0)
        original_layout.addWidget(original_btn, 0)
        original_group.setLayout(original_layout)
        scroll_layout.addWidget(original_group)
        
        # Excel analysis file selection (for linking cell data) with drag-and-drop support
        excel_group = DragDropGroupBox(
            "Excel Analysis File (Optional)",
            parent=self,
            on_drop_callback=self.handle_dropped_excel_file
        )
        excel_layout = QHBoxLayout()
        self.excel_path = QLineEdit()
        self.excel_path.setPlaceholderText("Select Excel file with cell analysis data (e.g., *_masks_analysis.xlsx)... (or drag and drop)")
        self.excel_path.setMinimumWidth(500)
        self.excel_path.setFont(file_font)
        self.excel_path.setSizePolicy(size_policy)
        excel_info_icon = create_info_icon(
            "Link to existing Excel analysis file containing cell measurements.\n\n"
            "This file should contain per-cell data from the Analyze Masks tab.\n"
            "The tracking will link cell IDs across frames and preserve\n"
            "cell properties from the analysis file.",
            parent_widget=self
        )
        # Set size policy to prevent stretching
        excel_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        excel_btn = QPushButton("Browse File...")
        excel_btn.setFixedWidth(120)
        excel_btn.clicked.connect(self.select_excel_file)
        excel_layout.addWidget(self.excel_path, 0)
        excel_layout.addWidget(excel_info_icon, 0)  # Info icon before button
        excel_layout.addWidget(excel_btn, 0)
        excel_group.setLayout(excel_layout)
        scroll_layout.addWidget(excel_group)
        
        # Tracking parameters (side-by-side layout, no redundant group box titles)
        tracking_group = QGroupBox("Tracking Parameters")
        tracking_container_layout = QHBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['general_2d', 'general_2d_w_SAM2_features'])
        self.model_combo.setCurrentText('general_2d')
        model_info_icon = create_info_icon(
            "Pretrained Trackastra model for cell tracking.\n\n"
            "Default: 'general_2d'\n"
            "Source: Trackastra pretrained models (https://github.com/weigertlab/trackastra)\n\n"
            "Available models:\n"
            "- general_2d: General-purpose 2D tracking model (default)\n"
            "  • Fast and works well for most cell types\n"
            "  • Good balance between speed and accuracy\n"
            "  • Recommended for general use cases\n\n"
            "- general_2d_w_SAM2_features: Enhanced model with SAM2 features\n"
            "  • Uses Segment Anything Model 2 (SAM2) features for improved tracking\n"
            "  • Better performance on certain data types, especially bacteria\n"
            "  • Slower than general_2d (requires GPU for reasonable speed)\n"
            "  • Best for challenging tracking scenarios\n\n"
            "Note: Trackastra won the 7th Cell Tracking Challenge (ISBI 2024).\n"
            "The models use transformer-based architecture for cell linking.",
            parent_widget=self
        )
        model_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(model_info_icon)
        model_layout.addStretch()
        tracking_container_layout.addLayout(model_layout)
        
        # Tracking mode
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['greedy', 'greedy_nodiv', 'ilp'])
        self.mode_combo.setCurrentText('greedy')
        mode_info_icon = create_info_icon(
            "Linking mode for connecting cells across time frames.\n\n"
            "Default: 'greedy'\n"
            "Source: Trackastra linking algorithms (https://github.com/weigertlab/trackastra)\n\n"
            "Available modes:\n"
            "- greedy: Greedy linking with division support (recommended)\n"
            "  • Fast algorithm, no additional dependencies\n"
            "  • Handles cell divisions (splitting)\n"
            "  • Good for most tracking scenarios\n"
            "  • Best balance of speed and accuracy\n\n"
            "- greedy_nodiv: Greedy linking without division support\n"
            "  • Fast algorithm, no additional dependencies\n"
            "  • Cells cannot split (no division tracking)\n"
            "  • Use when cells don't divide or you want faster processing\n"
            "  • Simpler tracking logic\n\n"
            "- ilp: Integer Linear Programming (optimal linking)\n"
            "  • More accurate than greedy methods\n"
            "  • Solves global optimization problem\n"
            "  • Slower than greedy methods\n"
            "  • Requires additional dependencies (motile, Gurobi or SCIP solver)\n"
            "  • Best for challenging cases where accuracy is critical\n"
            "  • Note: Academic Gurobi licenses are free\n\n"
            "Tip: Start with 'greedy' for most cases. Use 'ilp' if you need maximum accuracy.",
            parent_widget=self
        )
        mode_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(mode_info_icon)
        mode_layout.addStretch()
        tracking_container_layout.addLayout(mode_layout)
        
        # Device selection
        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        
        # Check available devices and create mapping
        available_devices_display = ['CPU']
        self.device_mapping = {'CPU': 'cpu'}  # Map display name to actual device value
        device_info_text = "Compute device for tracking.\n\n"
        device_info_text += "Default: CPU\n"
        device_info_text += "Source: PyTorch device selection\n\n"
        device_info_text += "Options:\n"
        device_info_text += "- CPU: CPU processing (works everywhere, slower)\n"
        
        try:
            import torch
            if torch.cuda.is_available():
                available_devices_display.append('CUDA')
                self.device_mapping['CUDA'] = 'cuda'
                device_info_text += "- CUDA: NVIDIA GPU processing (faster, requires CUDA-capable GPU)\n"
            
            # Check for Apple Silicon MPS support
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                available_devices_display.append('GPU')
                self.device_mapping['GPU'] = 'mps'
                device_info_text += "- GPU: Apple Silicon GPU (M1/M2/M3) or NVIDIA GPU (faster)\n"
        except ImportError:
            pass
        
        device_info_text += "\nTip: Use GPU if available for faster processing."
        
        device_info_icon = create_info_icon(device_info_text, parent_widget=self)
        device_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.device_combo = QComboBox()
        self.device_combo.addItems(available_devices_display)
        self.device_combo.setCurrentText('CPU')
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addWidget(device_info_icon)
        device_layout.addStretch()
        tracking_container_layout.addLayout(device_layout)
        
        tracking_group.setLayout(tracking_container_layout)
        scroll_layout.addWidget(tracking_group)
        
        # Process button - modern styling applied via global stylesheet
        self.process_btn = QPushButton("Run Tracking")
        self.process_btn.setEnabled(False)  # Disabled until mask file is selected
        self.process_btn.clicked.connect(self.run_tracking)
        scroll_layout.addWidget(self.process_btn)
        
        # Connect mask path changes to update button state
        self.mask_path.textChanged.connect(self._update_tracking_button_state)
        # Initialize state
        self._update_tracking_button_state()
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def _update_tracking_button_state(self):
        """Update tracking button enabled state based on whether mask file is selected"""
        has_mask = bool(self.mask_path.text() and self.mask_path.text().strip())
        if hasattr(self, 'process_btn'):
            self.process_btn.setEnabled(has_mask)
    
    def select_mask_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask File", "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.mask_path.setText(file_path)
            self._update_tracking_button_state()
    
    def select_original_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Original Data File", "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.original_path.setText(file_path)
    
    def select_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel Analysis File", "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.excel_path.setText(file_path)
    
    def handle_dropped_mask_file(self, dropped_path):
        """Handle dropped mask file for tracking tab"""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in ['.tif', '.tiff']:
            self.mask_path.setText(dropped_path)
            self._update_tracking_button_state()
        else:
            QMessageBox.warning(self, "Invalid File", 
                              "Please drop a valid TIFF mask file.")
    
    def handle_dropped_original_file(self, dropped_path):
        """Handle dropped original data file for tracking tab"""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in ['.tif', '.tiff']:
            self.original_path.setText(dropped_path)
        else:
            QMessageBox.warning(self, "Invalid File", 
                              "Please drop a valid TIFF file.")
    
    def handle_dropped_excel_file(self, dropped_path):
        """Handle dropped Excel file for tracking tab"""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in ['.xlsx', '.xls']:
            self.excel_path.setText(dropped_path)
        else:
            QMessageBox.warning(self, "Invalid File", 
                              "Please drop a valid Excel file (.xlsx or .xls).")
    
    def run_tracking(self):
        if not self.mask_path.text():
            QMessageBox.warning(self, "Missing Input", "Please select a mask file.")
            return
        
        mask_file = Path(self.mask_path.text())
        if not mask_file.exists():
            QMessageBox.warning(self, "File Not Found", f"Mask file not found:\n{mask_file}")
            return
        
        # Get optional files
        original_file = self.original_path.text() if self.original_path.text() else None
        excel_file = self.excel_path.text() if self.excel_path.text() else None
        
        if original_file and not Path(original_file).exists():
            QMessageBox.warning(self, "File Not Found", f"Original file not found:\n{original_file}")
            return
        
        if excel_file and not Path(excel_file).exists():
            QMessageBox.warning(self, "File Not Found", f"Excel file not found:\n{excel_file}")
            return
        
        # Get parameters
        model_name = self.model_combo.currentText()
        mode = self.mode_combo.currentText()
        # Map display name to actual device value
        device_display = self.device_combo.currentText()
        device = self.device_mapping.get(device_display, 'cpu')
        
        # Check device availability with detailed debugging
        if device == 'cuda':
            debug_info = []
            debug_info.append("=== CUDA Debugging Information ===")
            
            try:
                import torch
                debug_info.append(f"✓ PyTorch version: {torch.__version__}")
                
                # Check CUDA availability
                cuda_available = torch.cuda.is_available()
                debug_info.append(f"✓ torch.cuda.is_available(): {cuda_available}")
                
                if not cuda_available:
                    debug_info.append("✗ CUDA is not available")
                    debug_info.append(f"  - CUDA version (compiled): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
                    debug_info.append(f"  - cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
                    
                    # Check if CUDA libraries are found
                    try:
                        import subprocess
                        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
                        if result.returncode == 0:
                            debug_info.append(f"  - nvcc found: {result.stdout.strip()}")
                        else:
                            debug_info.append("  - nvcc not found in PATH")
                    except:
                        pass
                    
                    error_msg = "\n".join(debug_info)
                    error_msg += "\n\nCUDA GPU is not available on this system."
                    error_msg += "\n\nPossible reasons:"
                    error_msg += "\n- No NVIDIA GPU installed"
                    error_msg += "\n- CUDA drivers not installed"
                    error_msg += "\n- PyTorch was installed without CUDA support"
                    error_msg += "\n- GPU drivers are outdated"
                    
                    reply = QMessageBox.question(
                        self, "CUDA Not Available",
                        error_msg + "\n\nWould you like to continue with CPU mode?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    device = 'cpu'
                    self.device_combo.setCurrentText('CPU')
                    self.parent_window.log_output("\n".join(debug_info) + "\n")
                else:
                    # CUDA is available, get detailed info
                    debug_info.append("✓ CUDA is available!")
                    debug_info.append(f"  - CUDA version (compiled): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
                    debug_info.append(f"  - Number of GPUs: {torch.cuda.device_count()}")
                    
                    for i in range(torch.cuda.device_count()):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_props = torch.cuda.get_device_properties(i)
                        debug_info.append(f"  - GPU {i}: {gpu_name}")
                        debug_info.append(f"    - Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
                        debug_info.append(f"    - Compute Capability: {gpu_props.major}.{gpu_props.minor}")
                    
                    # Test CUDA with a simple operation
                    debug_info.append("\nTesting CUDA functionality...")
                    try:
                        # Test basic tensor creation
                        test_tensor = torch.tensor([1.0], device='cuda')
                        debug_info.append("  ✓ Tensor creation: SUCCESS")
                        
                        # Test simple operation
                        result = test_tensor * 2.0
                        debug_info.append("  ✓ Tensor operation: SUCCESS")
                        
                        # Test memory allocation
                        test_large = torch.zeros(1000, 1000, device='cuda')
                        debug_info.append("  ✓ Memory allocation: SUCCESS")
                        
                        del test_tensor, result, test_large
                        torch.cuda.empty_cache()
                        debug_info.append("  ✓ Memory cleanup: SUCCESS")
                        
                        debug_info.append("\n✓ CUDA is working correctly!")
                        self.parent_window.log_output("\n".join(debug_info) + "\n")
                        
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        debug_info.append(f"  ✗ CUDA test failed: {str(e)}")
                        debug_info.append(f"\nError details:\n{error_details}")
                        
                        error_msg = "\n".join(debug_info)
                        error_msg += "\n\nCUDA is available but encountered an error during testing."
                        error_msg += "\nThis might indicate:"
                        error_msg += "\n- GPU driver issues"
                        error_msg += "\n- CUDA runtime problems"
                        error_msg += "\n- Memory allocation issues"
                        
                        reply = QMessageBox.question(
                            self, "CUDA Error",
                            error_msg + "\n\nWould you like to continue with CPU mode?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.No:
                            return
                        device = 'cpu'
                        self.device_combo.setCurrentText('cpu')
                        self.parent_window.log_output("\n".join(debug_info) + "\n")
                        
            except ImportError:
                debug_info.append("✗ PyTorch is not installed")
                error_msg = "\n".join(debug_info)
                error_msg += "\n\nPyTorch is required for CUDA support."
                error_msg += "\nInstall with: pip install torch torchvision"
                error_msg += "\nFor CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                
                QMessageBox.warning(
                    self, "PyTorch Not Found",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.device_combo.setCurrentText('CPU')
                self.parent_window.log_output("\n".join(debug_info) + "\n")
            except Exception as e:
                import traceback
                debug_info.append(f"✗ Unexpected error: {str(e)}")
                debug_info.append(f"\n{traceback.format_exc()}")
                error_msg = "\n".join(debug_info)
                
                QMessageBox.warning(
                    self, "CUDA Check Error",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.device_combo.setCurrentText('CPU')
                self.parent_window.log_output("\n".join(debug_info) + "\n")
        elif device == 'mps':
            # Check MPS (Apple Silicon) availability
            debug_info = []
            debug_info.append("=== GPU (Apple Silicon) Debugging Information ===")
            
            try:
                import torch
                debug_info.append(f"✓ PyTorch version: {torch.__version__}")
                
                # Check MPS availability
                if hasattr(torch.backends, 'mps'):
                    mps_available = torch.backends.mps.is_available()
                    debug_info.append(f"✓ torch.backends.mps.is_available(): {mps_available}")
                    
                    if not mps_available:
                        debug_info.append("✗ GPU is not available")
                        debug_info.append("  - This usually means you're not on Apple Silicon (M1/M2/M3)")
                        debug_info.append("  - Or PyTorch was built without GPU support")
                        
                        error_msg = "\n".join(debug_info)
                        error_msg += "\n\nMPS (Apple Silicon GPU) is not available."
                        error_msg += "\n\nPossible reasons:"
                        error_msg += "\n- Not running on Apple Silicon (M1/M2/M3 chip)"
                        error_msg += "\n- PyTorch was installed without MPS support"
                        error_msg += "\n- macOS version too old (needs macOS 12.3+)"
                        
                        reply = QMessageBox.question(
                            self, "MPS Not Available",
                            error_msg + "\n\nWould you like to continue with CPU mode?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.No:
                            return
                        device = 'cpu'
                        self.device_combo.setCurrentText('cpu')
                        self.parent_window.log_output("\n".join(debug_info) + "\n")
                    else:
                        # MPS is available, test it
                        debug_info.append("✓ GPU is available!")
                        debug_info.append("  - Device: Apple Silicon GPU")
                        
                        # Test MPS with a simple operation
                        debug_info.append("\nTesting GPU functionality...")
                        try:
                            # Test basic tensor creation
                            test_tensor = torch.tensor([1.0], device='mps')
                            debug_info.append("  ✓ Tensor creation: SUCCESS")
                            
                            # Test simple operation
                            result = test_tensor * 2.0
                            debug_info.append("  ✓ Tensor operation: SUCCESS")
                            
                            # Test memory allocation
                            test_large = torch.zeros(1000, 1000, device='mps')
                            debug_info.append("  ✓ Memory allocation: SUCCESS")
                            
                            del test_tensor, result, test_large
                            torch.mps.empty_cache()
                            debug_info.append("  ✓ Memory cleanup: SUCCESS")
                            
                            debug_info.append("\n✓ GPU is working correctly!")
                            self.parent_window.log_output("\n".join(debug_info) + "\n")
                            
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            debug_info.append(f"  ✗ MPS test failed: {str(e)}")
                            debug_info.append(f"\nError details:\n{error_details}")
                            
                            error_msg = "\n".join(debug_info)
                            error_msg += "\n\GPU is available but encountered an error during testing."
                            error_msg += "\nThis might indicate:"
                            error_msg += "\n- macOS version compatibility issues"
                            error_msg += "\n- PyTorch MPS backend problems"
                            error_msg += "\n- Memory allocation issues"
                            
                            reply = QMessageBox.question(
                                self, "MPS Error",
                                error_msg + "\n\nWould you like to continue with CPU mode?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.Yes
                            )
                            if reply == QMessageBox.StandardButton.No:
                                return
                            device = 'cpu'
                            self.device_combo.setCurrentText('cpu')
                            self.parent_window.log_output("\n".join(debug_info) + "\n")
                else:
                    debug_info.append("✗ MPS backend not available in this PyTorch version")
                    error_msg = "\n".join(debug_info)
                    error_msg += "\n\nMPS backend is not available."
                    error_msg += "\nThis PyTorch version may not support MPS."
                    
                    reply = QMessageBox.question(
                        self, "MPS Not Supported",
                        error_msg + "\n\nWould you like to continue with CPU mode?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    device = 'cpu'
                    self.device_combo.setCurrentText('CPU')
                    self.parent_window.log_output("\n".join(debug_info) + "\n")
                    
            except ImportError:
                debug_info.append("✗ PyTorch is not installed")
                error_msg = "\n".join(debug_info)
                error_msg += "\n\nPyTorch is required for MPS support."
                
                QMessageBox.warning(
                    self, "PyTorch Not Found",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.device_combo.setCurrentText('CPU')
                self.parent_window.log_output("\n".join(debug_info) + "\n")
            except Exception as e:
                import traceback
                debug_info.append(f"✗ Unexpected error: {str(e)}")
                debug_info.append(f"\n{traceback.format_exc()}")
                error_msg = "\n".join(debug_info)
                
                QMessageBox.warning(
                    self, "MPS Check Error",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.device_combo.setCurrentText('CPU')
                self.parent_window.log_output("\n".join(debug_info) + "\n")
        
        # Create worker thread
        self.worker = TrackingWorker(
            mask_file, original_file, excel_file,
            model_name, mode, device, self
        )
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.start()
        
        # Disable button during processing
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range
        self.parent_window.progress_bar.setValue(0)
        self.parent_window.log_output(f"Starting tracking for: {mask_file.name}\n")
        
        # Connect progress signal
        self.worker.progress.connect(self.on_tracking_progress)
    
    def on_tracking_progress(self, value, message):
        """Update progress bar and log message"""
        self.parent_window.progress_bar.setValue(value)
        if message:
            self.parent_window.log_output(f"[{value}%] {message}\n")
    
    def on_tracking_finished(self, success, message):
        self.parent_window.progress_bar.setValue(100)
        if success:
            self.parent_window.log_output("✓ Tracking completed successfully!\n")
            QMessageBox.information(self, "Success", "Tracking completed successfully!")
        else:
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{message}")
        # Hide progress bar after a short delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.parent_window.progress_bar.setVisible(False))
    
    def get_session_params(self):
        """Return current parameter values for session log (parameters only, no paths)."""
        return {
            "model": self.model_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "device": self.device_combo.currentText(),
        }
    
    def set_session_params(self, data):
        """Apply saved parameter values (parameters only)."""
        if not isinstance(data, dict):
            return
        if "model" in data:
            idx = self.model_combo.findText(str(data["model"]))
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if "mode" in data:
            idx = self.mode_combo.findText(str(data["mode"]))
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)
        if "device" in data:
            idx = self.device_combo.findText(str(data["device"]))
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)


class TrackingWorker(QThread):
    """Worker thread for running tracking"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)  # progress percentage, status message
    
    def __init__(self, mask_file, original_file, excel_file, model_name, mode, device, parent=None):
        super().__init__(parent)
        self.mask_file = mask_file
        self.original_file = original_file
        self.excel_file = excel_file
        self.model_name = model_name
        self.mode = mode
        self.device = device
    
    def run(self):
        # Initialize debug output list
        debug_output = []
        debug_output.append("=== Tracking Debug Information ===")
        
        try:
            from trackastra.model import Trackastra
            from trackastra.tracking import graph_to_edge_table, graph_to_ctc, graph_to_napari_tracks
            import tifffile
            import pandas as pd
            import numpy as np
            from pathlib import Path
            import networkx as nx
            
            debug_output.append("✓ All imports successful")
            self.progress.emit(5, "Loading masks...")
            
            # Load masks
            debug_output.append(f"\nLoading masks from: {self.mask_file}")
            masks = tifffile.imread(str(self.mask_file))
            debug_output.append(f"✓ Masks loaded - Shape: {masks.shape}")
            self.progress.emit(10, "Masks loaded")
            
            if len(masks.shape) != 3:
                error_msg = f"Expected 3D mask array (T, Y, X), got shape: {masks.shape}\n\n" + "\n".join(debug_output)
                self.finished.emit(False, error_msg)
                return
            
            # Load original images if provided (required for trackastra)
            if self.original_file:
                self.progress.emit(15, "Loading original images...")
                debug_output.append(f"\nLoading original images from: {self.original_file}")
                imgs = tifffile.imread(str(self.original_file))
                debug_output.append(f"✓ Images loaded - Shape: {imgs.shape}")
                self.progress.emit(20, "Images loaded")
                
                if len(imgs.shape) != 3:
                    error_msg = f"Expected 3D image array (T, Y, X), got shape: {imgs.shape}\n\n" + "\n".join(debug_output)
                    self.finished.emit(False, error_msg)
                    return
                # Ensure images and masks have same number of frames
                if imgs.shape[0] != masks.shape[0]:
                    error_msg = f"Image frames ({imgs.shape[0]}) don't match mask frames ({masks.shape[0]})\n\n" + "\n".join(debug_output)
                    self.finished.emit(False, error_msg)
                    return
            else:
                # If no original images provided, create dummy images (trackastra needs them)
                # Use masks as images (will work but not ideal)
                self.progress.emit(15, "Creating images from masks...")
                debug_output.append("\nNo original images provided - using masks as images")
                imgs = masks.astype(np.float32)
                debug_output.append("✓ Created dummy images from masks")
                self.progress.emit(20, "Images ready")
            
            # Load Excel data if provided
            cell_data = None
            if self.excel_file:
                self.progress.emit(25, "Loading Excel analysis data...")
                try:
                    excel_path = Path(self.excel_file)
                    debug_output.append(f"\nLoading Excel file: {excel_path}")
                    # Try to read the per-cell data sheet
                    excel_data = pd.read_excel(excel_path, sheet_name=None)
                    debug_output.append(f"✓ Excel file loaded with sheets: {list(excel_data.keys())}")
                    
                    # Look for sheet with per-cell data (usually named "Per-Cell Data" or similar)
                    for sheet_name in excel_data.keys():
                        if 'cell' in sheet_name.lower() or 'per' in sheet_name.lower():
                            cell_data = excel_data[sheet_name]
                            debug_output.append(f"✓ Using sheet: {sheet_name} (per-cell data)")
                            break
                    if cell_data is None:
                        # Use first sheet if no per-cell sheet found
                        cell_data = list(excel_data.values())[0]
                        debug_output.append(f"✓ Using first sheet: {list(excel_data.keys())[0]}")
                    
                    debug_output.append(f"  - Cell data shape: {cell_data.shape}")
                    debug_output.append(f"  - Cell data columns: {list(cell_data.columns)}")
                    if len(cell_data) > 0:
                        debug_output.append(f"  - Sample row: {cell_data.iloc[0].to_dict()}")
                    self.progress.emit(30, "Excel data loaded")
                except Exception as e:
                    import traceback
                    error_msg = f"Error reading Excel file: {str(e)}\n\n{traceback.format_exc()}"
                    self.finished.emit(False, error_msg)
                    return
            
            # Load pretrained model with detailed debugging
            self.progress.emit(35, f"Loading Trackastra model ({self.model_name})...")
            debug_output.append(f"\nLoading Trackastra model: {self.model_name}")
            debug_output.append(f"Device: {self.device}")
            
            try:
                import torch
                if self.device == 'cuda':
                    debug_output.append(f"CUDA available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        debug_output.append(f"GPU: {torch.cuda.get_device_name(0)}")
                        debug_output.append(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                elif self.device == 'mps':
                    debug_output.append(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        debug_output.append("GPU: Apple Silicon (MPS)")
                else:
                    debug_output.append("Using CPU")
            except:
                pass
            
            try:
                debug_output.append("Calling Trackastra.from_pretrained()...")
                model = Trackastra.from_pretrained(self.model_name, device=self.device)
                debug_output.append("✓ Model loaded successfully")
                self.progress.emit(50, "Model loaded, starting tracking...")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                debug_output.append(f"✗ Error loading model: {str(e)}")
                debug_output.append(f"\nFull traceback:\n{error_details}")
                
                error_msg = f"Error loading Trackastra model: {str(e)}\n\n"
                error_msg += "Debug information:\n" + "\n".join(debug_output)
                error_msg += "\n\nPossible issues:"
                error_msg += "\n- Model name incorrect"
                error_msg += "\n- Trackastra not properly installed"
                error_msg += "\n- CUDA/GPU issues (if using cuda device)"
                error_msg += "\n- Network issues (downloading model)"
                
                self.finished.emit(False, error_msg)
                return
            
            # Run tracking with detailed debugging
            debug_output.append(f"\nStarting tracking with mode: {self.mode}")
            debug_output.append(f"Input shapes - Images: {imgs.shape}, Masks: {masks.shape}")
            
            try:
                import torch
                if self.device == 'cuda' and torch.cuda.is_available():
                    debug_output.append(f"GPU memory before tracking: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                    debug_output.append(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    debug_output.append("MPS device ready for tracking")
            except:
                pass
            
            # Initialize trackid_to_cellid mapping (will be set if Excel file is provided)
            trackid_to_cellid = None
            
            try:
                self.progress.emit(55, "Running tracking algorithm...")
                debug_output.append("Calling model.track()...")
                track_graph, masks_tracked = model.track(imgs, masks, mode=self.mode)
                debug_output.append("✓ Tracking completed successfully")
                self.progress.emit(70, "Tracking completed, processing results...")
                
                # Map track IDs back to analysis CellIDs if Excel file is provided
                if self.excel_file and Path(self.excel_file).exists():
                    self.progress.emit(72, "Mapping track IDs to analysis CellIDs...")
                    debug_output.append("\nMapping track IDs to analysis CellIDs...")
                    try:
                        masks_tracked, trackid_to_cellid = self._map_track_ids_to_analysis_cellids(
                            masks, masks_tracked, self.excel_file
                        )
                        debug_output.append(f"✓ Mapped {len(trackid_to_cellid)} track IDs to analysis CellIDs")
                    except Exception as e:
                        import traceback
                        debug_output.append(f"⚠ Could not map track IDs to analysis CellIDs: {str(e)}")
                        debug_output.append(f"  {traceback.format_exc()}")
                        # Continue without mapping
                        trackid_to_cellid = None
                debug_output.append(f"  - Track graph type: {type(track_graph)}")
                debug_output.append(f"  - Track graph nodes: {track_graph.number_of_nodes() if hasattr(track_graph, 'number_of_nodes') else 'N/A'}")
                debug_output.append(f"  - Track graph edges: {track_graph.number_of_edges() if hasattr(track_graph, 'number_of_edges') else 'N/A'}")
                debug_output.append(f"  - Tracked masks shape: {masks_tracked.shape}")
                debug_output.append(f"  - Tracked masks dtype: {masks_tracked.dtype}")
                debug_output.append(f"  - Original masks max: {masks.max()}")
                debug_output.append(f"  - Tracked masks max: {masks_tracked.max()}")
                debug_output.append(f"  - Unique labels in first frame (original): {len(np.unique(masks[0]))}")
                debug_output.append(f"  - Unique labels in first frame (tracked): {len(np.unique(masks_tracked[0]))}")
                debug_output.append(f"  - Unique labels in last frame (original): {len(np.unique(masks[-1]))}")
                debug_output.append(f"  - Unique labels in last frame (tracked): {len(np.unique(masks_tracked[-1]))}")
                # Check if track IDs are consistent across frames
                if masks_tracked.shape[0] > 1:
                    unique_tracks = set()
                    for i in range(masks_tracked.shape[0]):
                        unique_tracks.update(np.unique(masks_tracked[i]))
                    debug_output.append(f"  - Total unique track IDs across all frames: {len(unique_tracks)}")
                    # Check if masks changed (should have consistent track IDs)
                    if np.array_equal(masks, masks_tracked):
                        debug_output.append("  ⚠ WARNING: Tracked masks are identical to input masks!")
                        debug_output.append("    This suggests track IDs were not properly assigned.")
                    else:
                        debug_output.append("  ✓ Tracked masks differ from input (track IDs assigned)")
                
                try:
                    import torch
                    if self.device == 'cuda' and torch.cuda.is_available():
                        debug_output.append(f"GPU memory after tracking: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                        torch.cuda.empty_cache()
                        debug_output.append("✓ GPU cache cleared")
                    elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        debug_output.append("✓ MPS cache cleared")
                except:
                    pass
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                debug_output.append(f"✗ Error during tracking: {str(e)}")
                debug_output.append(f"\nFull traceback:\n{error_details}")
                
                try:
                    import torch
                    if self.device == 'cuda' and torch.cuda.is_available():
                        debug_output.append(f"\nGPU memory at error: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                        debug_output.append(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                    elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        debug_output.append("\nMPS device error occurred")
                except:
                    pass
                
                error_msg = f"Error during tracking: {str(e)}\n\n"
                error_msg += "Debug information:\n" + "\n".join(debug_output)
                error_msg += "\n\nPossible issues:"
                error_msg += "\n- GPU out of memory (try CPU mode)"
                error_msg += "\n- CUDA runtime error"
                error_msg += "\n- Input data format issues"
                error_msg += "\n- Model compatibility issues"
                
                self.finished.emit(False, error_msg)
                return
            
            # Save tracking results
            # Determine base directory (go up one level if input is in Tif_*, _segmentation, or _analysis folder)
            mask_path = Path(self.mask_file)
            base_dir = mask_path.parent
            if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name or '_analysis' in base_dir.name:
                base_dir = base_dir.parent
            
            # Create output directory at base level with _tracking suffix
            output_dir = base_dir / f"{mask_path.stem}_tracking"
            output_dir.mkdir(exist_ok=True)
            
            # Extract track information from the graph
            self.progress.emit(75, "Extracting track information...")
            debug_output.append("\nExtracting track information from graph...")
            
            # Method 1: Use graph_to_napari_tracks to get proper track data
            try:
                napari_tracks, napari_graph, _ = graph_to_napari_tracks(track_graph)
                debug_output.append(f"✓ Extracted {len(napari_tracks)} track points from graph")
                debug_output.append(f"  - Napari tracks shape: {napari_tracks.shape if hasattr(napari_tracks, 'shape') else 'N/A'}")
                
                # Convert napari tracks to DataFrame
                # napari_tracks is numpy array with columns: [track_id, frame, y, x, ...]
                if len(napari_tracks) > 0 and hasattr(napari_tracks, 'shape'):
                    # napari_tracks is typically (N, 4+) array: [track_id, frame, y, x, ...]
                    num_cols = napari_tracks.shape[1] if len(napari_tracks.shape) > 1 else 4
                    col_names = ['track_id', 'frame', 'y', 'x']
                    if num_cols > 4:
                        col_names.extend([f'property_{i}' for i in range(num_cols - 4)])
                    
                    tracks_df = pd.DataFrame(napari_tracks, columns=col_names[:num_cols])
                    # Ensure track_id and frame are integers
                    if 'track_id' in tracks_df.columns:
                        tracks_df['track_id'] = tracks_df['track_id'].astype(int)
                    if 'frame' in tracks_df.columns:
                        tracks_df['frame'] = tracks_df['frame'].astype(int)
                    debug_output.append(f"✓ Created tracks DataFrame with {len(tracks_df)} rows")
                    debug_output.append(f"  - Columns: {list(tracks_df.columns)}")
                else:
                    debug_output.append("⚠ Warning: No tracks extracted from graph or invalid format")
                    tracks_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
            except Exception as e:
                debug_output.append(f"⚠ graph_to_napari_tracks failed: {str(e)}")
                # Fallback: Extract tracks manually from graph
                debug_output.append("Trying manual track extraction from graph...")
                tracks_list = []
                
                # Extract all nodes and their properties
                for node_id, node_data in track_graph.nodes(data=True):
                    # Node data typically has 'time' or 'frame' and position info
                    frame = node_data.get('time', node_data.get('frame', node_data.get('t', None)))
                    if frame is not None:
                        track_id = node_data.get('track_id', node_id)
                        y = node_data.get('y', node_data.get('centroid_y', 0))
                        x = node_data.get('x', node_data.get('centroid_x', 0))
                        tracks_list.append({
                            'track_id': track_id,
                            'frame': int(frame),
                            'y': y,
                            'x': x,
                            'node_id': node_id
                        })
                
                if tracks_list:
                    tracks_df = pd.DataFrame(tracks_list)
                    debug_output.append(f"✓ Manually extracted {len(tracks_df)} track points")
                else:
                    # Last resort: use edge table
                    debug_output.append("Trying graph_to_edge_table as fallback...")
                    try:
                        edge_df = graph_to_edge_table(track_graph)
                        debug_output.append(f"✓ Edge table has {len(edge_df)} edges")
                        # Try to extract track info from edges
                        if len(edge_df) > 0:
                            tracks_df = edge_df
                        else:
                            tracks_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
                    except Exception as e2:
                        debug_output.append(f"✗ All extraction methods failed: {str(e2)}")
                        tracks_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
            
            # Update tracks_df with analysis CellIDs if mapping was created
            if trackid_to_cellid is not None and len(trackid_to_cellid) > 0:
                if 'track_id' in tracks_df.columns:
                    tracks_df['Cell_ID'] = tracks_df['track_id'].map(trackid_to_cellid)
                    # Fill NaN with original track_id for unmapped tracks
                    tracks_df['Cell_ID'] = tracks_df['Cell_ID'].fillna(tracks_df['track_id']).astype(int)
                    debug_output.append(f"✓ Updated tracks DataFrame with analysis CellIDs ({len(trackid_to_cellid)} mappings)")
            
            # Add debug info about tracks
            debug_output.append(f"\nTrack DataFrame info:")
            debug_output.append(f"  - Shape: {tracks_df.shape}")
            debug_output.append(f"  - Columns: {list(tracks_df.columns)}")
            if len(tracks_df) > 0:
                debug_output.append(f"  - Unique track IDs: {tracks_df['track_id'].nunique() if 'track_id' in tracks_df.columns else 'N/A'}")
                if 'Cell_ID' in tracks_df.columns:
                    debug_output.append(f"  - Unique Cell IDs: {tracks_df['Cell_ID'].nunique()}")
                debug_output.append(f"  - Frame range: {tracks_df['frame'].min() if 'frame' in tracks_df.columns else 'N/A'} to {tracks_df['frame'].max() if 'frame' in tracks_df.columns else 'N/A'}")
            
            # Save tracks as CSV
            self.progress.emit(80, "Saving tracks...")
            tracks_file = output_dir / f"{Path(self.mask_file).stem}_tracks.csv"
            tracks_df.to_csv(tracks_file, index=False)
            debug_output.append(f"✓ Saved tracks to: {tracks_file}")
            
            # Save tracked masks
            self.progress.emit(85, "Saving tracked masks...")
            masks_tracked_file = output_dir / f"{Path(self.mask_file).stem}_masks_tracked.tif"
            debug_output.append(f"\nSaving tracked masks...")
            debug_output.append(f"  - Shape: {masks_tracked.shape}")
            debug_output.append(f"  - Unique labels per frame: {[len(np.unique(masks_tracked[i])) for i in range(min(5, masks_tracked.shape[0]))]}")
            tifffile.imwrite(str(masks_tracked_file), masks_tracked)
            debug_output.append(f"✓ Saved tracked masks to: {masks_tracked_file}")
            
            # Create overlay with track IDs (consistent CellIDs across all outputs)
            if self.original_file:
                self.progress.emit(87, "Creating overlay with track IDs...")
                debug_output.append("\nCreating overlay with track IDs...")
                try:
                    overlay_file = output_dir / f"{Path(self.mask_file).stem}_overlay_tracked.tif"
                    self._create_overlay_with_track_ids(imgs, masks_tracked, overlay_file)
                    debug_output.append(f"✓ Saved tracked overlay to: {overlay_file}")
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Could not create tracked overlay: {str(e)}")
                    debug_output.append(f"  {traceback.format_exc()}")
            
            # Create visualization with trajectories overlaid on original images
            if self.original_file:
                self.progress.emit(90, "Creating trajectory visualization...")
                debug_output.append("\nCreating trajectory visualization...")
                debug_output.append("  (This may take a while for large datasets - optimizing...)")
                try:
                    vis_file = output_dir / f"{Path(self.mask_file).stem}_trajectories.tif"
                    # Pass progress callback to visualization function
                    def progress_callback(pct, msg):
                        self.progress.emit(pct, msg)
                    # Create visualization (with optimizations: compressed TIFF, faster line drawing)
                    self._create_trajectory_visualization(imgs, masks_tracked, tracks_df, vis_file, progress_callback)
                    debug_output.append(f"✓ Saved trajectory visualization to: {vis_file}")
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Could not create trajectory visualization: {str(e)}")
                    debug_output.append(f"  {traceback.format_exc()}")
            
            # Save in CTC format (Cell Tracking Challenge format)
            try:
                ctc_dir = output_dir / "ctc_format"
                ctc_dir.mkdir(exist_ok=True)
                graph_to_ctc(track_graph, masks_tracked, outdir=str(ctc_dir))
            except Exception as e:
                # CTC format is optional, don't fail if it doesn't work
                pass
            
            # If Excel data was provided, merge with tracks
            if cell_data is not None:
                self.progress.emit(95, "Merging with analysis data...")
                # Merge tracking data with cell properties
                merged_data = self._merge_tracking_with_analysis(tracks_df, cell_data, masks_tracked)
                merged_file = output_dir / f"{Path(self.mask_file).stem}_tracks_with_analysis.xlsx"
                with pd.ExcelWriter(merged_file, engine='openpyxl') as writer:
                    merged_data.to_excel(writer, sheet_name='Tracks with Analysis', index=False)
                    tracks_df.to_excel(writer, sheet_name='Tracks', index=False)
                    if isinstance(cell_data, pd.DataFrame):
                        cell_data.to_excel(writer, sheet_name='Original Analysis', index=False)
                self.progress.emit(100, "Tracking completed!")
                # Add debug output to success message
                success_msg = "Tracking completed!\n\n"
                success_msg += "\n".join(debug_output)
                success_msg += f"\n\nTracks saved to: {tracks_file}\n"
                success_msg += f"Tracked masks saved to: {masks_tracked_file}\n"
                success_msg += f"Merged data saved to: {merged_file}"
                self.finished.emit(True, success_msg)
            else:
                self.progress.emit(100, "Tracking completed!")
                # Add debug output to success message
                success_msg = "Tracking completed!\n\n"
                success_msg += "\n".join(debug_output)
                success_msg += f"\n\nTracks saved to: {tracks_file}\n"
                success_msg += f"Tracked masks saved to: {masks_tracked_file}"
                self.finished.emit(True, success_msg)
                
        except ImportError as e:
            debug_output.append(f"\n✗ Import error: {str(e)}")
            error_msg = f"trackastra is not installed or import failed: {str(e)}\n\n"
            error_msg += "Debug information:\n" + "\n".join(debug_output)
            error_msg += "\n\nPlease install it with: pip install trackastra"
            self.finished.emit(False, error_msg)
        except Exception as e:
            import traceback
            debug_output.append(f"\n✗ Unexpected error: {str(e)}")
            debug_output.append(f"\nFull traceback:\n{traceback.format_exc()}")
            error_msg = f"Tracking error: {str(e)}\n\n"
            error_msg += "Debug information:\n" + "\n".join(debug_output)
            self.finished.emit(False, error_msg)
    
    def _map_track_ids_to_analysis_cellids(self, masks_original, masks_tracked, excel_file):
        """
        Map track IDs back to analysis CellIDs by matching cells between original and tracked masks.
        
        Parameters:
        -----------
        masks_original : array
            Original masks (may have analysis CellIDs if from analyzed masks)
        masks_tracked : array
            Tracked masks with track IDs
        excel_file : str
            Path to Excel file with analysis data containing Cell_ID column
            
        Returns:
        --------
        masks_tracked_remapped : array
            Tracked masks remapped to use analysis CellIDs
        trackid_to_cellid : dict
            Mapping from track_id to analysis CellID
        """
        import pandas as pd
        import numpy as np
        from skimage.measure import regionprops
        from scipy.spatial.distance import cdist
        
        # Load Excel data to get Cell_IDs
        try:
            excel_data = pd.read_excel(excel_file, sheet_name='Per-Cell Data')
        except:
            # Try other sheet names
            try:
                excel_data = pd.read_excel(excel_file, sheet_name=0)
            except:
                return masks_tracked, {}
        
        # Find Cell_ID column
        cell_id_col = None
        for col in ['Cell_ID', 'cell_id', 'Cell ID', 'CellID']:
            if col in excel_data.columns:
                cell_id_col = col
                break
        
        if cell_id_col is None:
            # No Cell_ID column found, return original tracked masks
            return masks_tracked, {}
        
        # Find frame column
        frame_col = None
        for col in ['Frame', 'frame', 't', 'time', 'Time']:
            if col in excel_data.columns:
                frame_col = col
                break
        
        if frame_col is None:
            return masks_tracked, {}
        
        # Build mapping: (frame, original_label) -> Cell_ID from Excel
        excel_mapping = {}
        for _, row in excel_data.iterrows():
            frame = int(row[frame_col]) - 1  # Convert to 0-based
            cell_id = row[cell_id_col]
            # Try to find original label (might be in 'Cell Label' column)
            if 'Cell Label' in excel_data.columns:
                original_label = row['Cell Label']
            else:
                # Can't map without original label
                continue
            if pd.notna(cell_id) and pd.notna(original_label):
                excel_mapping[(frame, int(original_label))] = int(cell_id)
        
        if not excel_mapping:
            return masks_tracked, {}
        
        # Now map track IDs to analysis CellIDs by matching cells
        # Strategy: Match cells in each frame between original and tracked masks
        trackid_to_cellid = {}
        num_frames = min(masks_original.shape[0], masks_tracked.shape[0])
        max_distance = 10.0  # pixels
        
        for frame_idx in range(num_frames):
            frame_original = masks_original[frame_idx]
            frame_tracked = masks_tracked[frame_idx]
            
            # Get centroids for both
            regions_original = regionprops(frame_original)
            regions_tracked = regionprops(frame_tracked)
            
            # Build mapping for this frame: original_label -> CellID
            orig_label_to_cellid = {}
            for region in regions_original:
                if region.label > 0:
                    if (frame_idx, region.label) in excel_mapping:
                        orig_label_to_cellid[region.label] = excel_mapping[(frame_idx, region.label)]
            
            if not orig_label_to_cellid:
                continue
            
            # Match tracked cells to original cells by position
            orig_centroids = {}
            for region in regions_original:
                if region.label > 0 and region.label in orig_label_to_cellid:
                    orig_centroids[region.label] = np.array(region.centroid)
            
            tracked_centroids = {}
            for region in regions_tracked:
                if region.label > 0:
                    tracked_centroids[region.label] = np.array(region.centroid)
            
            if len(orig_centroids) > 0 and len(tracked_centroids) > 0:
                orig_points = np.array(list(orig_centroids.values()))
                orig_labels = list(orig_centroids.keys())
                tracked_points = np.array(list(tracked_centroids.values()))
                tracked_labels = list(tracked_centroids.keys())
                
                distances = cdist(orig_points, tracked_points)
                
                # Greedy assignment
                used_orig = set()
                used_tracked = set()
                assignments = []
                for i, orig_label in enumerate(orig_labels):
                    for j, track_id in enumerate(tracked_labels):
                        if distances[i, j] <= max_distance:
                            assignments.append((distances[i, j], orig_label, track_id))
                
                assignments.sort()
                
                for dist, orig_label, track_id in assignments:
                    if orig_label not in used_orig and track_id not in used_tracked:
                        # Map track_id to analysis CellID
                        cell_id = orig_label_to_cellid[orig_label]
                        trackid_to_cellid[track_id] = cell_id
                        used_orig.add(orig_label)
                        used_tracked.add(track_id)
        
        # Remap tracked masks to use analysis CellIDs
        masks_tracked_remapped = np.zeros_like(masks_tracked)
        for frame_idx in range(num_frames):
            frame_tracked = masks_tracked[frame_idx]
            frame_remapped = masks_tracked_remapped[frame_idx]
            
            for track_id, cell_id in trackid_to_cellid.items():
                frame_remapped[frame_tracked == track_id] = cell_id
        
        return masks_tracked_remapped, trackid_to_cellid
    
    def _merge_tracking_with_analysis(self, tracks_df, cell_data, masks_tracked):
        """Merge tracking results with Excel analysis data"""
        import pandas as pd
        import numpy as np
        from skimage.measure import regionprops
        from scipy.spatial.distance import cdist
        
        # Ensure cell_data is a DataFrame
        if not isinstance(cell_data, pd.DataFrame):
            cell_data = pd.DataFrame(cell_data)
        
        # Debug: Check what columns we have
        debug_info = []
        debug_info.append(f"Tracks DataFrame columns: {list(tracks_df.columns)}")
        debug_info.append(f"Cell data DataFrame columns: {list(cell_data.columns)}")
        debug_info.append(f"Tracks shape: {tracks_df.shape}, Cell data shape: {cell_data.shape}")
        
        # Find frame column in cell_data (could be 'frame', 'Frame', 't', 'time', etc.)
        frame_col = None
        for col in ['frame', 'Frame', 't', 'time', 'Time', 'T']:
            if col in cell_data.columns:
                frame_col = col
                break
        
        if frame_col is None:
            debug_info.append("⚠ Warning: Could not find frame column in cell_data")
            # Return tracks_df with a note
            return tracks_df
        
        debug_info.append(f"Using frame column: {frame_col}")
        
        # Find cell_id column in cell_data
        cell_id_col = None
        for col in ['cell_id', 'Cell_ID', 'cell', 'Cell', 'label', 'Label']:
            if col in cell_data.columns:
                cell_id_col = col
                break
        
        # Extract track information from tracked masks
        # In tracked masks, each label is a track_id that persists across frames
        # We need to map (frame, track_id) to original cell properties from Excel
        
        merged_list = []
        
        # For each track point, find matching cell data
        matches_found = 0
        matches_by_position = 0
        matches_by_first = 0
        
        # Track which frames we've already warned about
        frames_with_no_cells = set()
        
        # Detect if Excel uses 1-based indexing by checking if frame 0 has no data but frame 1 does
        frame_0_count = len(cell_data[cell_data[frame_col].astype(int) == 0]) if frame_col else 0
        frame_1_count = len(cell_data[cell_data[frame_col].astype(int) == 1]) if frame_col else 0
        excel_is_1_indexed = (frame_0_count == 0 and frame_1_count > 0)
        if excel_is_1_indexed:
            debug_info.append(f"Detected 1-based frame indexing in Excel (Frame 0 has {frame_0_count} cells, Frame 1 has {frame_1_count} cells)")
        
        for idx, track_row in tracks_df.iterrows():
            # Get frame - try multiple column names
            frame = None
            for col in ['frame', 't', 'time', 'Frame', 'T', 'Time']:
                if col in track_row.index:
                    frame_val = track_row[col]
                    if pd.notna(frame_val):
                        frame = int(frame_val)
                        break
            
            # Get track_id
            track_id = None
            for col in ['track_id', 'track', 'Track', 'track_ID']:
                if col in track_row.index:
                    track_id_val = track_row[col]
                    if pd.notna(track_id_val):
                        track_id = int(track_id_val)
                        break
            
            if frame is None:
                debug_info.append(f"  Skipping row {idx}: frame is None")
                continue
            if track_id is None:
                debug_info.append(f"  Skipping row {idx}: track_id is None")
                continue
            
            # Get track position from tracks_df
            track_y = None
            track_x = None
            for col in ['y', 'Y', 'centroid_y', 'centroid-0']:
                if col in track_row.index and pd.notna(track_row[col]):
                    track_y = float(track_row[col])
                    break
            for col in ['x', 'X', 'centroid_x', 'centroid-1']:
                if col in track_row.index and pd.notna(track_row[col]):
                    track_x = float(track_row[col])
                    break
            
            # Start with track information
            merged_row = track_row.to_dict()
            
            # Find cells in this frame from cell_data
            # Handle 0-based vs 1-based frame indexing
            # Tracking uses 0-based (0, 1, 2...), Excel might use 1-based (1, 2, 3...)
            frame_cells = pd.DataFrame()
            
            # Use detected indexing preference
            excel_frame = frame + 1 if excel_is_1_indexed else frame
            
            # Try to find cells for this frame
            try:
                frame_cells = cell_data[cell_data[frame_col].astype(int) == excel_frame].copy()
            except:
                try:
                    frame_cells = cell_data[cell_data[frame_col] == excel_frame].copy()
                except:
                    pass
            
            # If still no match and we haven't tried the other indexing, try it
            if len(frame_cells) == 0 and not excel_is_1_indexed:
                try:
                    frame_cells = cell_data[cell_data[frame_col].astype(int) == (frame + 1)].copy()
                    if len(frame_cells) > 0:
                        excel_is_1_indexed = True  # Update detection
                        debug_info.append(f"Detected 1-based frame indexing: Frame {frame} matched to Excel Frame {frame + 1}")
                except:
                    pass
            
            if len(frame_cells) > 0:
                matched = False
                # Try to match by position if we have coordinates
                if track_y is not None and track_x is not None:
                    # Find centroid columns in cell_data
                    y_col = None
                    x_col = None
                    for col in ['centroid_y', 'Centroid_Y', 'y', 'Y', 'centroid-0', 'centroid_y-0']:
                        if col in frame_cells.columns:
                            y_col = col
                            break
                    for col in ['centroid_x', 'Centroid_X', 'x', 'X', 'centroid-1', 'centroid_x-1']:
                        if col in frame_cells.columns:
                            x_col = col
                            break
                    
                    if y_col and x_col:
                        # Calculate distances and find closest match
                        try:
                            distances = np.sqrt(
                                (frame_cells[y_col].astype(float) - track_y)**2 + 
                                (frame_cells[x_col].astype(float) - track_x)**2
                            )
                            closest_idx = distances.idxmin()
                            closest_dist = distances.loc[closest_idx]
                            
                            # If close enough (within 20 pixels), use this match
                            if closest_dist < 20:
                                matched_cell = frame_cells.loc[closest_idx]
                                # Add all cell properties to merged row
                                for col in matched_cell.index:
                                    if col not in merged_row or pd.isna(merged_row.get(col)):
                                        merged_row[col] = matched_cell[col]
                                matches_by_position += 1
                                matched = True
                        except Exception as e:
                            debug_info.append(f"  Error matching by position: {str(e)}")
                
                if not matched:
                    # No position match, use first cell in frame (or match by track_id if available)
                    if cell_id_col and 'track_id' in frame_cells.columns:
                        # Try to match by track_id
                        matching = frame_cells[frame_cells[cell_id_col] == track_id]
                        if len(matching) > 0:
                            matched_cell = matching.iloc[0]
                        else:
                            matched_cell = frame_cells.iloc[0]
                    else:
                        matched_cell = frame_cells.iloc[0]
                    
                    # Add all cell properties to merged row
                    for col in matched_cell.index:
                        if col not in merged_row or pd.isna(merged_row.get(col)):
                            merged_row[col] = matched_cell[col]
                    matches_by_first += 1
                    matched = True
                
                if matched:
                    matches_found += 1
            else:
                # Only log once per frame to reduce noise
                if frame not in frames_with_no_cells:
                    frames_with_no_cells.add(frame)
                    debug_info.append(f"  Frame {frame} (Excel Frame {excel_frame}): No cells found in cell_data")
            
            merged_list.append(merged_row)
        
        debug_info.append(f"\nMerge statistics:")
        debug_info.append(f"  - Total track points: {len(tracks_df)}")
        debug_info.append(f"  - Matches found: {matches_found}")
        debug_info.append(f"  - Matched by position: {matches_by_position}")
        debug_info.append(f"  - Matched by first cell: {matches_by_first}")
        
        if merged_list:
            merged = pd.DataFrame(merged_list)
            debug_info.append(f"✓ Merged DataFrame created with {len(merged)} rows")
        else:
            debug_info.append("⚠ Warning: No merged rows created")
            merged = tracks_df.copy()
        
        # Log debug info
        import sys
        print("\n".join(debug_info), file=sys.stderr)
        
        return merged
    
    def _create_trajectory_visualization(self, images, masks_tracked, tracks_df, output_path, progress_callback=None):
        """Create visualization showing original images with tracked cell IDs and trajectories"""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib import cm
        from scipy.ndimage import label, binary_dilation
        import tifffile
        
        # Use progress callback if provided, otherwise check for attribute
        if progress_callback is None and hasattr(self, '_progress_callback'):
            progress_callback = self._progress_callback
        
        debug_output = []
        debug_output.append("Creating trajectory visualization...")
        
        num_frames = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        
        # Normalize images for display (vectorized for speed)
        if progress_callback:
            progress_callback(91, "Normalizing images...")
        
        # Vectorized normalization across all frames
        img_min = images.min(axis=(1, 2), keepdims=True)
        img_max = images.max(axis=(1, 2), keepdims=True)
        # Avoid division by zero
        img_range = img_max - img_min
        img_range[img_range == 0] = 1  # Avoid division by zero
        images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)
        
        # Get unique track IDs
        if 'track_id' in tracks_df.columns:
            unique_tracks = tracks_df['track_id'].unique()
            num_tracks = len(unique_tracks)
            debug_output.append(f"  - Found {num_tracks} unique tracks")
        else:
            unique_tracks = np.unique(masks_tracked)
            unique_tracks = unique_tracks[unique_tracks > 0]  # Remove background
            num_tracks = len(unique_tracks)
            debug_output.append(f"  - Found {num_tracks} unique track IDs in masks")
        
        # Generate colormap for tracks
        cmap = plt.get_cmap('gist_ncar')
        track_colors = {}
        for i, track_id in enumerate(unique_tracks):
            color = cmap(i / max(num_tracks, 1))
            track_colors[int(track_id)] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        
        # Create RGB visualization
        visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        
        # Group tracks by frame for trajectory drawing
        if 'frame' in tracks_df.columns and 'track_id' in tracks_df.columns:
            tracks_by_frame = {}
            for _, row in tracks_df.iterrows():
                frame = int(row['frame'])
                track_id = int(row['track_id'])
                if frame not in tracks_by_frame:
                    tracks_by_frame[frame] = {}
                if track_id not in tracks_by_frame[frame]:
                    tracks_by_frame[frame][track_id] = []
                y = row.get('y', 0)
                x = row.get('x', 0)
                tracks_by_frame[frame][track_id].append((y, x))
        
        for frame_idx in range(num_frames):
            # Update progress if we have a callback
            if progress_callback:
                progress_pct = int(90 + (frame_idx / num_frames) * 10)  # 90-100%
                progress_callback(progress_pct, f"Creating visualization: {frame_idx + 1}/{num_frames} ({progress_pct}%)")
            
            # Start with normalized image as grayscale RGB
            img = images_norm[frame_idx]
            vis_frame = np.stack([img, img, img], axis=-1)
            
            # Overlay tracked masks with colors
            mask_frame = masks_tracked[frame_idx]
            for track_id in unique_tracks:
                track_id_int = int(track_id)
                if track_id_int in track_colors:
                    color = track_colors[track_id_int]
                    mask_region = (mask_frame == track_id_int)
                    vis_frame[mask_region, 0] = np.clip(vis_frame[mask_region, 0] * 0.5 + color[0] * 0.5, 0, 255)
                    vis_frame[mask_region, 1] = np.clip(vis_frame[mask_region, 1] * 0.5 + color[1] * 0.5, 0, 255)
                    vis_frame[mask_region, 2] = np.clip(vis_frame[mask_region, 2] * 0.5 + color[2] * 0.5, 0, 255)
            
            # Draw trajectories up to current frame (only for tracks visible in this frame)
            # Pre-filter tracks to only those present in current frame for speed
            if 'frame' in tracks_df.columns and 'track_id' in tracks_df.columns:
                # Get tracks that have points up to current frame
                tracks_in_frame = tracks_df[
                    (tracks_df['frame'] <= frame_idx) & 
                    (tracks_df['track_id'].isin(unique_tracks))
                ]['track_id'].unique()
                
                for track_id_int in tracks_in_frame:
                    track_id_int = int(track_id_int)
                    if track_id_int not in track_colors:
                        continue
                    
                    # Get all points for this track up to current frame
                    track_points = tracks_df[
                        (tracks_df['track_id'] == track_id_int) & 
                        (tracks_df['frame'] <= frame_idx)
                    ]
                    
                    if len(track_points) > 1:
                        color = track_colors[track_id_int]
                        # Draw lines connecting consecutive points using vectorized operations
                        points = track_points[['y', 'x']].values.astype(int)
                        # Clip points to valid bounds
                        points[:, 0] = np.clip(points[:, 0], 0, height - 1)
                        points[:, 1] = np.clip(points[:, 1], 0, width - 1)
                        
                        # Draw lines between consecutive points using optimized approach
                        # For speed, we'll draw lines more efficiently
                        for i in range(len(points) - 1):
                            y1, x1 = points[i]
                            y2, x2 = points[i + 1]
                            
                            # Calculate line length
                            dy = abs(y2 - y1)
                            dx = abs(x2 - x1)
                            
                            if dx == 0 and dy == 0:
                                continue
                            
                            # Optimize: Use fewer points for long lines (subsample)
                            max_steps = max(dx, dy)
                            # For very long lines, subsample more aggressively
                            if max_steps > 200:
                                step_size = max(1, max_steps // 200)  # Very aggressive subsampling
                            elif max_steps > 50:
                                step_size = max(1, max_steps // 50)  # Moderate subsampling
                            else:
                                step_size = 1  # Full resolution for short lines
                            
                            num_steps = max_steps // step_size + 1
                            
                            # Generate line points using numpy (much faster than loops)
                            if dx > dy:
                                # Horizontal line
                                x_coords = np.linspace(x1, x2, num_steps, dtype=int)
                                y_coords = np.round(np.linspace(y1, y2, num_steps)).astype(int)
                            else:
                                # Vertical line
                                y_coords = np.linspace(y1, y2, num_steps, dtype=int)
                                x_coords = np.round(np.linspace(x1, x2, num_steps)).astype(int)
                            
                            # Clip coordinates
                            y_coords = np.clip(y_coords, 1, height - 2)  # Keep 1 pixel margin for 3x3 drawing
                            x_coords = np.clip(x_coords, 1, width - 2)
                            
                            # Draw thick line using vectorized operations (batch process all points)
                            # Use advanced indexing for speed
                            for y, x in zip(y_coords, x_coords):
                                # Direct assignment is faster than clipping each time
                                y_start, y_end = y - 1, y + 2
                                x_start, x_end = x - 1, x + 2
                                vis_frame[y_start:y_end, x_start:x_end, 0] = (
                                    vis_frame[y_start:y_end, x_start:x_end, 0] * 0.3 + color[0] * 0.7
                                ).astype(np.uint8)
                                vis_frame[y_start:y_end, x_start:x_end, 1] = (
                                    vis_frame[y_start:y_end, x_start:x_end, 1] * 0.3 + color[1] * 0.7
                                ).astype(np.uint8)
                                vis_frame[y_start:y_end, x_start:x_end, 2] = (
                                    vis_frame[y_start:y_end, x_start:x_end, 2] * 0.3 + color[2] * 0.7
                                ).astype(np.uint8)
                    
                    # Draw track ID label and marker at current position
                    current_track = track_points[track_points['frame'] == frame_idx]
                    if len(current_track) > 0:
                        y = int(current_track.iloc[0]['y'])
                        x = int(current_track.iloc[0]['x'])
                        if 0 <= y < height and 0 <= x < width:
                            # Draw a more visible colored dot with white outline for contrast
                            dot_size = 5  # Larger dot for better visibility
                            outline_size = 2  # White outline thickness
                            
                            # Draw white outline first (for contrast against dark backgrounds)
                            y_outline_start = max(0, y - dot_size - outline_size)
                            y_outline_end = min(height, y + dot_size + outline_size + 1)
                            x_outline_start = max(0, x - dot_size - outline_size)
                            x_outline_end = min(width, x + dot_size + outline_size + 1)
                            
                            # Create outline mask (ring shape)
                            yy, xx = np.ogrid[y_outline_start:y_outline_end, x_outline_start:x_outline_end]
                            center_y, center_x = y - y_outline_start, x - x_outline_start
                            dist_sq = (yy - center_y)**2 + (xx - center_x)**2
                            outline_mask = (dist_sq <= (dot_size + outline_size)**2) & (dist_sq > dot_size**2)
                            
                            # Draw white outline
                            vis_frame[y_outline_start:y_outline_end, x_outline_start:x_outline_end, :][outline_mask] = [255, 255, 255]
                            
                            # Draw colored center dot
                            dot_mask = dist_sq <= dot_size**2
                            vis_frame[y_outline_start:y_outline_end, x_outline_start:x_outline_end, :][dot_mask] = color
                            
                            # Try to add text label if PIL is available
                            try:
                                from PIL import Image, ImageDraw, ImageFont
                                # Convert numpy array to PIL Image
                                pil_img = Image.fromarray(vis_frame)
                                draw = ImageDraw.Draw(pil_img)
                                
                                # Try to use a nice font, fallback to default if not available
                                try:
                                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                                except:
                                    try:
                                        font = ImageFont.truetype("arial.ttf", 10)
                                    except:
                                        font = ImageFont.load_default()
                                
                                # Draw track ID text with shadow for readability
                                track_id_str = str(track_id_int)
                                text_x, text_y = x + dot_size + outline_size + 2, y - 5
                                
                                # Draw text shadow (black)
                                draw.text((text_x + 1, text_y + 1), track_id_str, fill=(0, 0, 0), font=font)
                                # Draw text (white)
                                draw.text((text_x, text_y), track_id_str, fill=(255, 255, 255), font=font)
                                
                                # Convert back to numpy array
                                vis_frame = np.array(pil_img)
                            except ImportError:
                                # PIL not available, skip text rendering
                                pass
                            except Exception:
                                # If text rendering fails for any reason, just continue without it
                                pass
            
            visualization[frame_idx] = vis_frame
        
        # Save visualization with compression for faster I/O
        # Use compression to reduce file size and write time
        if progress_callback:
            progress_callback(99, "Saving visualization file...")
        
        try:
            # Try saving with compression (faster for large files)
            tifffile.imwrite(
                str(output_path),
                visualization,
                compression='zlib'  # Use zlib compression
            )
            debug_output.append(f"✓ Visualization saved with {num_tracks} tracks (compressed)")
        except Exception as e:
            # Fallback to uncompressed if compression fails
            debug_output.append(f"⚠ Compression failed, saving uncompressed: {str(e)}")
            tifffile.imwrite(str(output_path), visualization)
            debug_output.append(f"✓ Visualization saved with {num_tracks} tracks (uncompressed)")
    
    def _create_overlay_with_track_ids(self, images, masks_tracked, output_path):
        """Create overlay visualization with track IDs as CellIDs (consistent with tracking visualization)"""
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        import tifffile
        
        num_frames = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        
        # Normalize images
        img_min = images.min(axis=(1, 2), keepdims=True)
        img_max = images.max(axis=(1, 2), keepdims=True)
        img_range = img_max - img_min
        img_range[img_range == 0] = 1
        images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)
        
        # Get unique track IDs
        unique_tracks = np.unique(masks_tracked)
        unique_tracks = unique_tracks[unique_tracks > 0]  # Remove background
        num_tracks = len(unique_tracks)
        
        # Generate colormap for tracks
        cmap = plt.get_cmap('gist_ncar')
        colors = cmap(np.linspace(0, 1, max(num_tracks + 1, 1000)))[:num_tracks + 1]
        colors[0] = [0, 0, 0, 0]  # Background transparent
        
        # Create color mapping
        track_colors = {}
        for i, track_id in enumerate(unique_tracks):
            track_colors[int(track_id)] = colors[i + 1]  # +1 because 0 is background
        
        # Try to import PIL for text rendering
        try:
            from PIL import Image, ImageDraw, ImageFont
            from skimage.measure import regionprops
            PIL_AVAILABLE = True
        except ImportError:
            PIL_AVAILABLE = False
        
        overlays = []
        for frame_idx in tqdm(range(num_frames), desc="Creating tracked overlay"):
            img = images_norm[frame_idx]
            mask = masks_tracked[frame_idx]
            
            # Convert grayscale to RGB
            img_rgb = np.stack([img, img, img], axis=-1)
            
            # Colorize mask using track IDs
            mask_rgb = np.zeros((height, width, 3), dtype=np.float32)
            for track_id, color in track_colors.items():
                mask_rgb[mask == track_id] = color[:3]
            mask_rgb = (mask_rgb * 255).astype(np.uint8)
            
            # Blend
            overlay = (img_rgb * 0.5 + mask_rgb * 0.5).astype(np.uint8)
            
            # Add track ID labels if PIL is available
            if PIL_AVAILABLE and mask.max() > 0:
                try:
                    # Convert to PIL Image
                    pil_img = Image.fromarray(overlay)
                    draw = ImageDraw.Draw(pil_img)
                    
                    # Try to use a nice font
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                    except:
                        try:
                            font = ImageFont.truetype("arial.ttf", 10)
                        except:
                            font = ImageFont.load_default()
                    
                    # Get cell centroids and track IDs
                    regions = regionprops(mask)
                    for region in regions:
                        if region.label == 0:
                            continue
                        
                        # Get centroid
                        centroid_y, centroid_x = region.centroid
                        x, y = int(centroid_x), int(centroid_y)
                        
                        # Draw track ID text with shadow for readability
                        track_id_str = str(region.label)
                        text_x, text_y = x + 2, y - 5
                        
                        # Draw text shadow (black)
                        draw.text((text_x + 1, text_y + 1), track_id_str, fill=(0, 0, 0), font=font)
                        # Draw text (white)
                        draw.text((text_x, text_y), track_id_str, fill=(255, 255, 255), font=font)
                    
                    # Convert back to numpy array
                    overlay = np.array(pil_img)
                except Exception:
                    # If text rendering fails, continue without labels
                    pass
            
            overlays.append(overlay)
        
        overlay_array = np.stack(overlays, axis=0)
        tifffile.imwrite(str(output_path), overlay_array, compression='zlib')


class GrowthCurveVideoTab(QWidget):
    """Tab for creating growth curve videos with side-by-side graph and video"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # TIFF sequence input (file or folder) with drag-and-drop support
        video_group = DragDropGroupBox(
            "Video Input (TIFF Sequence)",
            parent=self,
            on_drop_callback=self.handle_dropped_video_path
        )
        video_layout = QVBoxLayout()
        
        # File/Folder selection
        file_folder_layout = QHBoxLayout()
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Select TIFF file or folder containing TIFF sequence... (or drag and drop)")
        self.video_path.setMinimumWidth(500)
        file_font = QFont()
        file_font.setPointSize(11)
        self.video_path.setFont(file_font)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.video_path.setSizePolicy(size_policy)
        
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_video_file)
        
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(120)
        folder_btn.clicked.connect(self.select_video_folder)
        
        file_folder_layout.addWidget(self.video_path, 0)
        file_folder_layout.addWidget(file_btn, 0)
        file_folder_layout.addWidget(folder_btn, 0)
        video_layout.addLayout(file_folder_layout)
        
        video_group.setLayout(video_layout)
        scroll_layout.addWidget(video_group)
        
        # Excel file input
        excel_group = QGroupBox("Excel Analysis File")
        excel_layout = QHBoxLayout()
        self.excel_path = QLineEdit()
        self.excel_path.setPlaceholderText("Select Excel file from Analyze Masks tab (e.g., *_masks_analysis.xlsx)...")
        self.excel_path.setMinimumWidth(500)
        self.excel_path.setFont(file_font)
        self.excel_path.setSizePolicy(size_policy)
        excel_btn = QPushButton("Browse File...")
        excel_btn.setFixedWidth(120)
        excel_btn.clicked.connect(self.select_excel_file)
        excel_info_icon = create_info_icon(
            "Excel file from the Analyze Masks tab containing growth data.\n\n"
            "Required columns:\n"
            "- Frame: Frame number (1, 2, 3, ...)\n"
            "- Total Cell Area (Relative): Normalized cell area (first frame = 1.0)",
            parent_widget=self
        )
        excel_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        excel_layout.addWidget(self.excel_path, 0)
        excel_layout.addWidget(excel_btn, 0)
        excel_layout.addWidget(excel_info_icon, 0)
        excel_group.setLayout(excel_layout)
        
        # Time interval input (next to Excel file)
        time_group = QGroupBox("Time Settings")
        time_layout = QHBoxLayout()
        time_label = QLabel("Time per frame (min.):")
        self.time_interval = QDoubleSpinBox()
        self.time_interval.setMinimum(0.0)
        self.time_interval.setMaximum(10000.0)
        self.time_interval.setValue(10.0)
        self.time_interval.setDecimals(2)
        self.time_interval.setSuffix(" min")
        time_info_icon = create_info_icon(
            "Time difference between consecutive frames.\n\n"
            "This is used to convert frame numbers to time (hours) for the x-axis of the growth curve.\n\n"
            "Example: If images were taken every 10 minutes, enter 10.0",
            parent_widget=self
        )
        time_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.time_interval.setFixedWidth(100)  # Fixed width for uniform appearance
        time_layout.addWidget(time_label)
        time_layout.addWidget(time_info_icon)
        time_layout.addStretch()  # Add stretch before spin box to align to right
        time_layout.addWidget(self.time_interval)
        time_group.setLayout(time_layout)
        
        # Excel and Time Settings side-by-side
        excel_time_container = QHBoxLayout()
        excel_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        time_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        excel_time_container.addWidget(excel_group)
        excel_time_container.addWidget(time_group)
        excel_time_widget = QWidget()
        excel_time_widget.setLayout(excel_time_container)
        scroll_layout.addWidget(excel_time_widget)
        
        # Graph and Video settings side-by-side
        settings_container_layout = QHBoxLayout()
        
        # Graph settings
        graph_group = QGroupBox("Graph Settings")
        graph_layout = QVBoxLayout()
        
        # Y-axis min/max
        yaxis_layout = QHBoxLayout()
        ymin_label = QLabel("Y-axis min:")
        self.yaxis_min = QDoubleSpinBox()
        self.yaxis_min.setMinimum(0.0)
        self.yaxis_min.setMaximum(100.0)
        self.yaxis_min.setValue(0.0)
        self.yaxis_min.setDecimals(2)
        self.yaxis_min.setSpecialValueText("Auto")
        
        ymax_label = QLabel("Y-axis max:")
        self.yaxis_max = QDoubleSpinBox()
        self.yaxis_max.setMinimum(0.0)
        self.yaxis_max.setMaximum(100.0)
        self.yaxis_max.setValue(0.0)
        self.yaxis_max.setDecimals(2)
        self.yaxis_max.setSpecialValueText("Auto")
        
        graph_info_icon = create_info_icon(
            "Configure the Y-axis range for the growth curve graph.\n\n"
            "Y-axis represents: Relative cell area (normalized to first frame = 1.0)\n\n"
            "Y-axis min/max:\n"
            "- Set to 'Auto' (0.0) to automatically scale based on data\n"
            "- Set custom values to fix the Y-axis range\n"
            "- Useful for comparing multiple videos with consistent scaling\n\n"
            "Example: Set min=0.0, max=3.0 to show growth from 1x to 3x initial area",
            parent_widget=self
        )
        graph_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        yaxis_layout.addWidget(ymin_label)
        yaxis_layout.addWidget(self.yaxis_min)
        yaxis_layout.addWidget(ymax_label)
        yaxis_layout.addWidget(self.yaxis_max)
        yaxis_layout.addWidget(graph_info_icon)
        yaxis_layout.addStretch()
        graph_layout.addLayout(yaxis_layout)
        
        graph_group.setLayout(graph_layout)
        graph_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_container_layout.addWidget(graph_group)
        
        # Video settings
        video_settings_group = QGroupBox("Video Settings")
        video_settings_layout = QHBoxLayout()
        fps_label = QLabel("Frames per second (FPS):")
        self.fps = QDoubleSpinBox()
        self.fps.setMinimum(0.1)
        self.fps.setMaximum(60.0)
        self.fps.setValue(5.0)
        self.fps.setDecimals(1)
        self.fps.setSuffix(" fps")
        fps_info_icon = create_info_icon(
            "Playback speed for the output video.\n\n"
            "Higher values = faster playback\n"
            "Lower values = slower playback\n\n"
            "Default: 5.0 fps\n"
            "Typical range: 1-10 fps for time-lapse videos",
            parent_widget=self
        )
        fps_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        video_settings_layout.addWidget(fps_label)
        video_settings_layout.addWidget(self.fps)
        video_settings_layout.addWidget(fps_info_icon)
        video_settings_layout.addStretch()
        video_settings_group.setLayout(video_settings_layout)
        video_settings_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_container_layout.addWidget(video_settings_group)
        
        # Output format selection (in same row)
        format_group = QGroupBox("Output Formats")
        format_layout = QVBoxLayout()
        
        # Format checkboxes - all in a single row with "Select All" first
        checkbox_layout = QHBoxLayout()
        
        # "Select All" checkbox first
        self.format_all = QCheckBox("Select All")
        self.format_all.stateChanged.connect(self._on_select_all_changed)
        checkbox_layout.addWidget(self.format_all)
        
        # Add a separator (spacing) between "Select All" and format options
        checkbox_layout.addSpacing(20)
        
        # Format checkboxes
        self.format_tif = QCheckBox("TIFF")
        self.format_tif.setChecked(True)  # Default enabled
        self.format_avi = QCheckBox("AVI")
        self.format_avi.setChecked(True)  # Default enabled
        self.format_gif = QCheckBox("GIF")
        self.format_mp4 = QCheckBox("MP4")
        
        # Connect format checkboxes to update "Select All" state
        self.format_tif.stateChanged.connect(self._on_format_changed)
        self.format_avi.stateChanged.connect(self._on_format_changed)
        self.format_gif.stateChanged.connect(self._on_format_changed)
        self.format_mp4.stateChanged.connect(self._on_format_changed)
        
        checkbox_layout.addWidget(self.format_tif)
        checkbox_layout.addWidget(self.format_avi)
        checkbox_layout.addWidget(self.format_gif)
        checkbox_layout.addWidget(self.format_mp4)
        checkbox_layout.addStretch()
        format_layout.addLayout(checkbox_layout)
        
        format_group.setLayout(format_layout)
        format_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_container_layout.addWidget(format_group)
        
        # Add container to scroll layout
        settings_container = QWidget()
        settings_container.setLayout(settings_container_layout)
        scroll_layout.addWidget(settings_container)
        
        # Process button
        self.process_btn = QPushButton("Create Growth Curve Video")
        self.process_btn.setEnabled(False)  # Disabled until inputs are selected
        self.process_btn.clicked.connect(self.create_video)
        scroll_layout.addWidget(self.process_btn)
        
        # Connect path changes to update button state
        self.video_path.textChanged.connect(self._update_button_state)
        self.excel_path.textChanged.connect(self._update_button_state)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
        
        # Initial button state update
        self._update_button_state()
    
    def _update_button_state(self):
        """Update process button state based on input fields"""
        has_video = bool(self.video_path.text())
        has_excel = bool(self.excel_path.text())
        has_format = (self.format_tif.isChecked() or self.format_avi.isChecked() or 
                     self.format_gif.isChecked() or self.format_mp4.isChecked())
        self.process_btn.setEnabled(has_video and has_excel and has_format)
    
    def _on_select_all_changed(self, state):
        """Handle 'Select All' checkbox change"""
        checked = state == 2  # Qt.Checked = 2
        self.format_tif.setChecked(checked)
        self.format_avi.setChecked(checked)
        self.format_gif.setChecked(checked)
        self.format_mp4.setChecked(checked)
    
    def _on_format_changed(self):
        """Handle individual format checkbox change"""
        all_checked = (self.format_tif.isChecked() and self.format_avi.isChecked() and 
                      self.format_gif.isChecked() and self.format_mp4.isChecked())
        self.format_all.blockSignals(True)
        self.format_all.setChecked(all_checked)
        self.format_all.blockSignals(False)
        self._update_button_state()
    
    def select_video_file(self):
        """Select TIFF file for video input"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TIFF File",
            "",
            "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*)"
        )
        if file_path:
            self.video_path.setText(file_path)
    
    def select_video_folder(self):
        """Select folder containing TIFF sequence"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with TIFF Sequence",
            ""
        )
        if folder_path:
            self.video_path.setText(folder_path)
    
    def handle_dropped_video_path(self, dropped_path):
        """Handle dropped file or folder path for growth curve video tab"""
        path_obj = Path(dropped_path)
        # Update video path
        self.video_path.setText(dropped_path)
    
    def select_excel_file(self):
        """Select Excel file from Analyze Masks tab"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel Analysis File",
            "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.excel_path.setText(file_path)
    
    def _is_folder_mode(self):
        """Check if video input is a folder"""
        path = Path(self.video_path.text())
        return path.exists() and path.is_dir()
    
    def _get_available_models(self):
        """
        Return list of available Omnipose models.
        Based on user's confirmed list of functional models.
        Only includes models that are known to exist and work.
        """
        # User's confirmed list of available/functional models
        # These are the models that actually exist and work
        confirmed_omnipose_models = [
            'bact_phase_omni',  # Recommended for bacteria
            'bact_fluor_omni',
            'cyto2_omni',
            'worm_omni',
            'plant_omni',
            'bact_phase_affinity'
        ]
        
        # Cellpose models (always available, not Omnipose-specific)
        cellpose_models = [
            'cyto2',
            'cyto',
            'nuclei'
        ]
        
        # Combine and return
        return confirmed_omnipose_models + cellpose_models
    
    def create_video(self):
        """Create growth curve video with side-by-side graph and video"""
        if not self.video_path.text() or not self.excel_path.text():
            QMessageBox.warning(self, "Missing Input", "Please select both video input and Excel file.")
            return
        
        video_path = Path(self.video_path.text())
        excel_path = Path(self.excel_path.text())
        
        if not video_path.exists():
            QMessageBox.warning(self, "File Not Found", f"Video input not found:\n{video_path}")
            return
        
        if not excel_path.exists():
            QMessageBox.warning(self, "File Not Found", f"Excel file not found:\n{excel_path}")
            return
        
        # Determine output directory (go up one level if input is in Tif_* or _segmentation folder)
        if video_path.is_file():
            base_dir = video_path.parent
            input_stem = video_path.stem
        else:
            base_dir = video_path
            input_stem = video_path.name
        
        # Go up one level if in segmentation or Tif_ folder (same logic as other tabs)
        if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name:
            base_dir = base_dir.parent
        
        # Create output directory at base level
        output_dir = base_dir / f"{input_stem}_growth_curve_video"
        output_dir.mkdir(exist_ok=True)
        
        # Show progress
        self.process_btn.setEnabled(False)
        self.process_btn.setText("Creating Video...")
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        
        # Get selected formats
        formats = []
        if self.format_tif.isChecked():
            formats.append('tif')
        if self.format_avi.isChecked():
            formats.append('avi')
        if self.format_gif.isChecked():
            formats.append('gif')
        if self.format_mp4.isChecked():
            formats.append('mp4')
        
        # Run in worker thread
        self.worker = GrowthCurveVideoWorker(
            video_path,
            excel_path,
            output_dir,
            self.time_interval.value(),
            self.yaxis_min.value() if self.yaxis_min.value() > 0 else None,
            self.yaxis_max.value() if self.yaxis_max.value() > 0 else None,
            self.fps.value(),
            formats,
            parent=self
        )
        self.worker.finished.connect(self.on_video_finished)
        self.worker.progress.connect(self.on_video_progress)
        self.worker.start()
    
    def on_video_progress(self, value, message):
        """Handle progress updates"""
        self.parent_window.progress_bar.setValue(value)
        if message:
            self.parent_window.log_output(message)
    
    def on_video_finished(self, success, message):
        """Handle video creation completion"""
        self.process_btn.setEnabled(True)
        self.process_btn.setText("Create Growth Curve Video")
        self.parent_window.progress_bar.setVisible(False)
        
        if success:
            # Just log the message instead of showing a popup
            self.parent_window.log_output(f"\n✓ {message}\n")
        else:
            QMessageBox.warning(self, "Error", message)
    
    def get_session_params(self):
        """Return current parameter values for session log (parameters only, no paths)."""
        return {
            "time_interval": self.time_interval.value(),
            "yaxis_min": self.yaxis_min.value(),
            "yaxis_max": self.yaxis_max.value(),
            "fps": self.fps.value(),
            "format_tif": self.format_tif.isChecked(),
            "format_avi": self.format_avi.isChecked(),
            "format_gif": self.format_gif.isChecked(),
            "format_mp4": self.format_mp4.isChecked(),
        }
    
    def set_session_params(self, data):
        """Apply saved parameter values (parameters only)."""
        if not isinstance(data, dict):
            return
        for k, w in [
            ("time_interval", self.time_interval), ("yaxis_min", self.yaxis_min),
            ("yaxis_max", self.yaxis_max), ("fps", self.fps),
        ]:
            if k in data and hasattr(w, "setValue"):
                try:
                    w.setValue(float(data[k]))
                except (TypeError, ValueError):
                    pass
        for k, w in [
            ("format_tif", self.format_tif), ("format_avi", self.format_avi),
            ("format_gif", self.format_gif), ("format_mp4", self.format_mp4),
        ]:
            if k in data and hasattr(w, "setChecked"):
                w.setChecked(bool(data[k]))


class GrowthCurveVideoWorker(QThread):
    """Worker thread for creating growth curve videos"""
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(int, str)  # value, message
    
    def __init__(self, video_path, excel_path, output_dir, time_interval, yaxis_min, yaxis_max, fps, formats, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.excel_path = excel_path
        self.output_dir = output_dir
        self.time_interval = time_interval
        self.yaxis_min = yaxis_min
        self.yaxis_max = yaxis_max
        self.fps = fps  # Frames per second for video playback
        self.formats = formats  # List of formats to save: ['tif', 'avi', 'gif', 'mp4']
    
    def run(self):
        try:
            import pandas as pd
            import numpy as np
            import tifffile
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            import cv2
            
            self.progress.emit(5, "Loading Excel data...")
            
            # Load Excel data
            df = pd.read_excel(self.excel_path, sheet_name='Summary')
            
            # Extract frame and relative area columns
            if 'Frame' not in df.columns or 'Total Cell Area (Relative)' not in df.columns:
                self.finished.emit(False, f"Excel file must contain 'Frame' and 'Total Cell Area (Relative)' columns.\n"
                                        f"Found columns: {', '.join(df.columns)}")
                return
            
            frames = df['Frame'].values
            relative_areas = df['Total Cell Area (Relative)'].values
            
            # Convert frames to time (minutes)
            time_minutes = (frames - 1) * self.time_interval
            
            self.progress.emit(10, "Loading video frames...")
            
            # Load video frames
            if self.video_path.is_file():
                # Single multi-page TIFF file
                video_frames = tifffile.imread(str(self.video_path))
                if len(video_frames.shape) == 2:
                    # Single frame, add time dimension
                    video_frames = video_frames[np.newaxis, ...]
            else:
                # Folder with TIFF sequence
                tiff_files = sorted(self.video_path.glob("*.tif")) + sorted(self.video_path.glob("*.tiff")) + \
                            sorted(self.video_path.glob("*.TIF")) + sorted(self.video_path.glob("*.TIFF"))
                if not tiff_files:
                    self.finished.emit(False, f"No TIFF files found in folder: {self.video_path}")
                    return
                
                video_frames = []
                for tiff_file in tiff_files:
                    frame = tifffile.imread(str(tiff_file))
                    if len(frame.shape) == 2:
                        video_frames.append(frame)
                    else:
                        # Multi-page, take first page
                        video_frames.append(frame[0] if len(frame.shape) == 3 else frame)
                video_frames = np.array(video_frames)
            
            num_frames = min(len(video_frames), len(frames))
            video_frames = video_frames[:num_frames]
            frames = frames[:num_frames]
            relative_areas = relative_areas[:num_frames]
            time_minutes = time_minutes[:num_frames]
            
            self.progress.emit(20, f"Processing {num_frames} frames...")
            
            # Get video dimensions
            if len(video_frames.shape) == 3:
                frame_height, frame_width = video_frames.shape[1], video_frames.shape[2]
            else:
                frame_height, frame_width = video_frames.shape[0], video_frames.shape[1]
            
            # Graph and video share 50/50 of the space
            graph_width = frame_width  # Graph gets same width as video (50/50 split)
            video_width = frame_width
            total_width = graph_width + video_width
            
            # Create output frames
            output_frames = []
            
            # Determine y-axis limits
            y_min = self.yaxis_min if self.yaxis_min is not None else max(0.0, relative_areas.min() * 0.9)
            y_max = self.yaxis_max if self.yaxis_max is not None else relative_areas.max() * 1.1
            
            for frame_idx in range(num_frames):
                if frame_idx % 10 == 0:
                    progress = 20 + int((frame_idx / num_frames) * 70)
                    self.progress.emit(progress, f"Processing frame {frame_idx + 1}/{num_frames}...")
                
                # Get current video frame
                if len(video_frames.shape) == 3:
                    video_frame = video_frames[frame_idx]
                else:
                    video_frame = video_frames
                
                # Normalize to 0-255 if needed
                if video_frame.dtype != np.uint8:
                    video_frame = ((video_frame - video_frame.min()) / (video_frame.max() - video_frame.min() + 1e-10) * 255).astype(np.uint8)
                
                # Convert to RGB if grayscale
                if len(video_frame.shape) == 2:
                    video_frame = np.stack([video_frame] * 3, axis=-1)
                elif video_frame.shape[2] == 1:
                    video_frame = np.repeat(video_frame, 3, axis=2)
                
                # Create graph
                fig = Figure(figsize=(graph_width / 100, frame_height / 100), dpi=100)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)
                
                # Plot growth curve up to current frame
                current_frames = frames[:frame_idx + 1]
                current_areas = relative_areas[:frame_idx + 1]
                current_times = time_minutes[:frame_idx + 1]
                
                ax.plot(current_times, current_areas, 'k-', linewidth=8)
                ax.scatter(current_times[-1], current_areas[-1], color='black', s=200, zorder=5, edgecolors='black', linewidths=2)
                ax.set_xlabel('Time [min]', fontsize=36, fontweight='bold')
                ax.set_ylabel('rel. cell area', fontsize=36, fontweight='bold')
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(0, time_minutes.max() * 1.05)
                ax.grid(True, alpha=0.4, linewidth=1.5)
                
                # Set white background style (matching desired output)
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                # Make tick labels bigger and bolder
                ax.tick_params(axis='both', which='major', labelsize=24, width=2, colors='black')
                ax.tick_params(axis='both', which='minor', labelsize=24, colors='black')
                # Make tick labels bold by setting font weight on the label objects
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
                ax.xaxis.label.set_color('black')
                ax.yaxis.label.set_color('black')
                # Make axis lines (spines) thicker and bolder
                for spine in ax.spines.values():
                    spine.set_color('black')
                    spine.set_linewidth(3.5)
                
                # Render graph to numpy array
                canvas.draw()
                # Get buffer from canvas (works with both old and new matplotlib)
                try:
                    # Try new matplotlib API first (buffer_rgba)
                    buf = canvas.buffer_rgba()
                    graph_image = np.asarray(buf)
                    # Convert RGBA to RGB
                    if graph_image.shape[2] == 4:
                        graph_image = graph_image[:, :, :3]
                except AttributeError:
                    # Fallback to old API (tostring_rgb)
                    try:
                        graph_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                        graph_image = graph_image.reshape(canvas.get_width_height()[::-1] + (3,))
                    except AttributeError:
                        # Alternative: use renderer buffer
                        buf = np.array(canvas.renderer.buffer_rgba())
                        graph_image = buf[:, :, :3]  # Convert RGBA to RGB
                
                # Resize graph to match video frame height
                if graph_image.shape[0] != frame_height or graph_image.shape[1] != graph_width:
                    graph_image = cv2.resize(graph_image, (graph_width, frame_height), interpolation=cv2.INTER_LINEAR)
                
                # Combine graph (left) and video (right)
                combined_frame = np.hstack([graph_image, video_frame])
                output_frames.append(combined_frame)
            
            self.progress.emit(95, "Saving output files...")
            
            output_files = []
            output_frames_array = np.array(output_frames)
            
            # Save as multi-page TIFF
            if 'tif' in self.formats:
                try:
                    output_tiff = self.output_dir / f"{self.output_dir.name}.tif"
                    # Ensure uint8 for smaller file size
                    if output_frames_array.dtype != np.uint8:
                        output_frames_array = ((output_frames_array - output_frames_array.min()) / 
                                             (output_frames_array.max() - output_frames_array.min() + 1e-10) * 255).astype(np.uint8)
                    # Use LZW compression for better compression ratio
                    tifffile.imwrite(str(output_tiff), output_frames_array, compression='lzw', photometric='rgb')
                    output_files.append(f"TIFF: {output_tiff.name}")
                except Exception as e:
                    output_files.append(f"TIFF: Error - {str(e)}")
            
            # Save as AVI
            if 'avi' in self.formats:
                try:
                    output_avi = self.output_dir / f"{self.output_dir.name}.avi"
                    # Try multiple codecs in order of preference
                    codecs = ['MJPG', 'XVID', 'H264', 'mp4v']
                    success = False
                    for codec_name in codecs:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec_name)
                            out = cv2.VideoWriter(str(output_avi), fourcc, self.fps, (total_width, frame_height))
                            if out.isOpened():
                                for frame in output_frames:
                                    out.write(frame)
                                out.release()
                                output_files.append(f"AVI: {output_avi.name} (codec: {codec_name})")
                                success = True
                                break
                        except Exception:
                            continue
                    if not success:
                        output_files.append(f"AVI: Error - No compatible codec found")
                except Exception as e:
                    output_files.append(f"AVI: Error - {str(e)}")
            
            # Save as GIF
            if 'gif' in self.formats:
                try:
                    output_gif = self.output_dir / f"{self.output_dir.name}.gif"
                    import imageio
                    from PIL import Image
                    # Convert frames to uint8 if needed and optimize
                    gif_frames = []
                    for frame in output_frames:
                        if frame.dtype != np.uint8:
                            frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255).astype(np.uint8)
                        # Convert to PIL Image for better GIF optimization
                        if len(frame.shape) == 3:
                            pil_frame = Image.fromarray(frame)
                            # Quantize to reduce colors (256 colors for better compression)
                            pil_frame = pil_frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                            gif_frames.append(pil_frame)
                        else:
                            gif_frames.append(frame)
                    # Save with optimization
                    imageio.mimsave(str(output_gif), gif_frames, fps=self.fps, loop=0, subrectangles=True)
                    output_files.append(f"GIF: {output_gif.name}")
                except Exception as e:
                    # Fallback to simple save if PIL optimization fails
                    try:
                        gif_frames = []
                        for frame in output_frames:
                            if frame.dtype != np.uint8:
                                frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255).astype(np.uint8)
                            gif_frames.append(frame)
                        imageio.mimsave(str(output_gif), gif_frames, fps=self.fps, loop=0)
                        output_files.append(f"GIF: {output_gif.name}")
                    except Exception as e2:
                        output_files.append(f"GIF: Error - {str(e2)}")
            
            # Save as MP4
            if 'mp4' in self.formats:
                try:
                    output_mp4 = self.output_dir / f"{self.output_dir.name}.mp4"
                    import imageio
                    # Convert frames to uint8 if needed
                    mp4_frames = []
                    for frame in output_frames:
                        if frame.dtype != np.uint8:
                            frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255).astype(np.uint8)
                        mp4_frames.append(frame)
                    # Try imageio first (uses ffmpeg if available)
                    try:
                        imageio.mimsave(str(output_mp4), mp4_frames, fps=self.fps, codec='libx264', quality=8)
                        output_files.append(f"MP4: {output_mp4.name}")
                    except Exception:
                        # Fallback to opencv
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(output_mp4), fourcc, self.fps, (total_width, frame_height))
                        if out.isOpened():
                            for frame in mp4_frames:
                                out.write(frame)
                            out.release()
                            output_files.append(f"MP4: {output_mp4.name}")
                        else:
                            output_files.append(f"MP4: Error - Could not create MP4 file")
                except Exception as e:
                    output_files.append(f"MP4: Error - {str(e)}")
            
            self.progress.emit(100, "Complete!")
            output_summary = "\n".join(output_files)
            self.finished.emit(True, f"Growth curve video created successfully!\n\n"
                                   f"Output directory: {self.output_dir}\n\n"
                                   f"Files created:\n{output_summary}")
        
        except Exception as e:
            import traceback
            error_msg = f"Error creating growth curve video:\n{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


class SAM2SegmentTab(QWidget):
    """Tab for SAM2 (Segment Anything Model 2) interactive segmentation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # File/Folder selection with drag-and-drop support
        file_group = DragDropGroupBox(
            "Input File/Folder",
            parent=self,
            on_drop_callback=self.handle_dropped_path_sam2
        )
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a TIFF file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(500)
        file_font = QFont()
        file_font.setPointSize(12)
        self.file_path.setFont(file_font)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.file_path.setSizePolicy(size_policy)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)
        folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.file_path, 0)
        file_layout.addWidget(file_btn, 0)
        file_layout.addWidget(folder_btn, 0)
        file_group.setLayout(file_layout)
        scroll_layout.addWidget(file_group)
        
        # SAM2 Parameters - side-by-side layout
        sam2_group = QGroupBox("SAM2 Parameters")
        sam2_container_layout = QHBoxLayout()
        
        # Left column: Model and checkpoint
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        # Model selection
        model_select_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_info_icon = create_info_icon(
            "SAM2.1 model variant for segmentation.\n\n"
            "Available models:\n"
            "- sam2.1_hiera_tiny: Fastest, smallest (38.9M params)\n"
            "- sam2.1_hiera_small: Balanced speed/accuracy (46M params)\n"
            "- sam2.1_hiera_base_plus: Better accuracy (80.8M params)\n"
            "- sam2.1_hiera_large: Best accuracy, slowest (224.4M params)\n\n"
            "Tip: Start with 'small' or 'base_plus' for good balance.",
            parent_widget=self
        )
        model_select_layout.addWidget(model_label)
        model_select_layout.addWidget(model_info_icon)
        model_select_layout.addStretch()
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "sam2.1_hiera_tiny",
            "sam2.1_hiera_small",
            "sam2.1_hiera_base_plus",
            "sam2.1_hiera_large"
        ])
        self.model_combo.setCurrentText("sam2.1_hiera_small")  # Default to balanced option
        model_select_layout.addWidget(self.model_combo)
        model_layout.addLayout(model_select_layout)
        
        # Checkpoint path
        checkpoint_layout = QHBoxLayout()
        checkpoint_label = QLabel("Checkpoint Path:")
        checkpoint_info_icon = create_info_icon(
            "Path to SAM2 model checkpoint file (.pt).\n\n"
            "If empty, will attempt to download from Hugging Face or use default location.\n\n"
            "Checkpoints can be downloaded from:\n"
            "https://github.com/facebookresearch/sam2#download-checkpoints\n\n"
            "Or use Hugging Face models (auto-downloaded if checkpoint not found).",
            parent_widget=self
        )
        checkpoint_layout.addWidget(checkpoint_label)
        checkpoint_layout.addWidget(checkpoint_info_icon)
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText("Leave empty for auto-download or default location...")
        self.checkpoint_path.setMinimumWidth(300)
        checkpoint_btn = QPushButton("Browse...")
        checkpoint_btn.setFixedWidth(100)
        checkpoint_btn.clicked.connect(self.select_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_path, 1)
        checkpoint_layout.addWidget(checkpoint_btn, 0)
        model_layout.addLayout(checkpoint_layout)
        
        model_group.setLayout(model_layout)
        model_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sam2_container_layout.addWidget(model_group)
        
        # Right column: Segmentation options
        options_group = QGroupBox("Segmentation Options")
        options_layout = QVBoxLayout()
        
        # Segmentation mode
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_info_icon = create_info_icon(
            "Segmentation mode for SAM2.\n\n"
            "- Automatic: Generate masks automatically (like SAM)\n"
            "- Interactive: Use prompts (points/boxes) for guided segmentation\n\n"
            "Note: Interactive mode requires additional UI components (future feature).",
            parent_widget=self
        )
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(mode_info_icon)
        mode_layout.addStretch()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Automatic", "Interactive (Coming Soon)"])
        self.mode_combo.setCurrentText("Automatic")
        self.mode_combo.setItemData(1, None, Qt.ItemDataRole.UserRole - 1)  # Disable interactive for now
        mode_layout.addWidget(self.mode_combo)
        options_layout.addLayout(mode_layout)
        
        # Points per side (for automatic mode)
        points_layout = QHBoxLayout()
        points_label = QLabel("Points per side:")
        points_info_icon = create_info_icon(
            "Number of points per side for automatic mask generation.\n\n"
            "Higher values = more masks generated (slower)\n"
            "Lower values = fewer masks (faster)\n\n"
            "Default: 32 (good balance)",
            parent_widget=self
        )
        points_layout.addWidget(points_label)
        points_layout.addWidget(points_info_icon)
        points_layout.addStretch()
        self.points_per_side = QSpinBox()
        self.points_per_side.setRange(8, 128)
        self.points_per_side.setValue(32)
        self.points_per_side.setFixedWidth(100)
        points_layout.addWidget(self.points_per_side)
        options_layout.addLayout(points_layout)
        
        # Stability score threshold
        stability_layout = QHBoxLayout()
        stability_label = QLabel("Stability Score:")
        stability_info_icon = create_info_icon(
            "Minimum stability score for mask filtering.\n\n"
            "Range: 0.0 to 1.0\n"
            "Higher values = more stable masks only\n"
            "Lower values = more masks (including less stable)\n\n"
            "Default: 0.95",
            parent_widget=self
        )
        stability_layout.addWidget(stability_label)
        stability_layout.addWidget(stability_info_icon)
        stability_layout.addStretch()
        self.stability_score = QDoubleSpinBox()
        self.stability_score.setRange(0.0, 1.0)
        self.stability_score.setValue(0.95)
        self.stability_score.setSingleStep(0.05)
        self.stability_score.setDecimals(2)
        self.stability_score.setFixedWidth(100)
        stability_layout.addWidget(self.stability_score)
        options_layout.addLayout(stability_layout)
        
        # GPU option
        gpu_layout = QHBoxLayout()
        self.use_gpu = QCheckBox("Use GPU (if available)")
        self.use_gpu.setChecked(True)
        gpu_info_icon = create_info_icon(
            "Use GPU acceleration for faster processing.\n\n"
            "Default: Enabled\n"
            "Automatically falls back to CPU if GPU not available.",
            parent_widget=self
        )
        gpu_layout.addWidget(self.use_gpu)
        gpu_layout.addWidget(gpu_info_icon)
        gpu_layout.addStretch()
        options_layout.addLayout(gpu_layout)
        
        # Create overlay option
        overlay_layout = QHBoxLayout()
        self.create_overlay = QCheckBox("Create overlay visualization")
        self.create_overlay.setChecked(True)
        overlay_info_icon = create_info_icon(
            "Create TIFF overlay showing masks on original images.\n\n"
            "Overlay files will be saved with '_sam2_overlay.tif' suffix.",
            parent_widget=self
        )
        overlay_layout.addWidget(self.create_overlay)
        overlay_layout.addWidget(overlay_info_icon)
        overlay_layout.addStretch()
        options_layout.addLayout(overlay_layout)
        
        options_group.setLayout(options_layout)
        options_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sam2_container_layout.addWidget(options_group)
        
        sam2_group.setLayout(sam2_container_layout)
        scroll_layout.addWidget(sam2_group)
        
        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Leave empty to save in same directory as input...")
        self.output_path.setMinimumWidth(500)
        self.output_path.setFont(file_font)
        self.output_path.setSizePolicy(size_policy)
        output_btn = QPushButton("Browse...")
        output_btn.setFixedWidth(120)
        output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_path, 0)
        output_layout.addWidget(output_btn, 0)
        output_group.setLayout(output_layout)
        scroll_layout.addWidget(output_group)
        
        # Preview button
        self.preview_btn = QPushButton("Preview File")
        self.preview_btn.setEnabled(False)
        self.preview_btn.setToolTip("Preview the selected TIFF file before processing")
        self.preview_btn.clicked.connect(self.preview_file)
        scroll_layout.addWidget(self.preview_btn)
        
        # Process button
        self.process_btn = QPushButton("Run SAM2 Segmentation")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        scroll_layout.addWidget(self.process_btn)
        
        # Connect path changes to update button state
        self.file_path.textChanged.connect(self._update_button_state)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
        
        # Initial button state update
        self._update_button_state()
    
    def _update_button_state(self):
        """Update process and preview button states based on input fields"""
        has_file = bool(self.file_path.text())
        self.process_btn.setEnabled(has_file)
        self.preview_btn.setEnabled(has_file)
    
    def select_file(self):
        """Select a TIFF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TIFF File",
            "",
            "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*)"
        )
        if file_path:
            self.file_path.setText(file_path)
    
    def select_folder(self):
        """Select a folder containing TIFF files"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with TIFF Files",
            ""
        )
        if folder_path:
            self.file_path.setText(folder_path)
    
    def handle_dropped_path_sam2(self, dropped_path):
        """Handle dropped file or folder path for SAM2 segmentation tab"""
        path_obj = Path(dropped_path)
        # Update file path
        self.file_path.setText(dropped_path)
        # Update button state
        self._update_button_state()
    
    def select_checkpoint(self):
        """Select SAM2 checkpoint file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM2 Checkpoint File",
            "",
            "PyTorch Files (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            self.checkpoint_path.setText(file_path)
    
    def select_output_dir(self):
        """Select output directory"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        if folder_path:
            self.output_path.setText(folder_path)
    
    def preview_file(self):
        """Preview the selected TIFF file"""
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a TIFF file first.")
            return
        
        file_path = Path(self.file_path.text())
        if not file_path.exists():
            QMessageBox.warning(self, "Not Found", "The selected file does not exist.")
            return
        
        if file_path.is_dir():
            QMessageBox.warning(self, "Folder Selected", "Please select a specific TIFF file for preview, not a folder.")
            return
        
        try:
            import tifffile
            import numpy as np
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure
            
            # Load TIFF file
            images = safe_imread(str(file_path))
            
            # Handle different dimensions
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
            
            num_frames = images.shape[0]
            
            # Create preview dialog
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle(f"File Preview: {file_path.name}")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)
            
            layout = QVBoxLayout()
            
            # Matplotlib figure
            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, preview_dialog)
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
            
            # Frame navigation
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
            
            preview_dialog.current_frame = 0
            
            def update_display(frame_idx):
                if frame_idx < 0 or frame_idx >= num_frames:
                    return
                
                preview_dialog.current_frame = frame_idx
                frame = images[frame_idx]
                
                # Handle multi-channel: use first channel if needed
                if frame.ndim == 3:
                    frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                
                fig.clear()
                ax = fig.add_subplot(111)
                ax.imshow(frame, cmap='gray')
                ax.set_title(f"Frame {frame_idx + 1}/{num_frames}: {file_path.name}")
                ax.axis('off')
                canvas.draw()
                
                frame_spin.blockSignals(True)
                frame_spin.setValue(frame_idx + 1)
                frame_spin.blockSignals(False)
            
            def go_to_frame(idx):
                if 0 <= idx < num_frames:
                    update_display(idx)
            
            prev_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame - 1))
            next_btn.clicked.connect(lambda: go_to_frame(preview_dialog.current_frame + 1))
            
            def on_frame_changed():
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_frames:
                    go_to_frame(frame_num)
            
            frame_spin.editingFinished.connect(on_frame_changed)
            frame_spin._change_timer = QTimer()
            frame_spin._change_timer.setSingleShot(True)
            frame_spin._change_timer.timeout.connect(on_frame_changed)
            frame_spin.valueChanged.connect(lambda v: frame_spin._change_timer.start(300))
            
            preview_dialog.setLayout(layout)
            
            # Show first frame
            update_display(0)
            
            preview_dialog.exec()
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Preview Error", 
                              f"Could not preview file:\n{str(e)}\n\n{traceback.format_exc()}")
    
    def start_processing(self):
        """Start SAM2 segmentation processing"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Processing", "Processing is already in progress.")
            return
        
        input_path = self.file_path.text()
        if not input_path:
            QMessageBox.warning(self, "No Input", "Please select an input file or folder.")
            return
        
        # Get parameters
        model_name = self.model_combo.currentText()
        checkpoint_path = self.checkpoint_path.text() if self.checkpoint_path.text() else None
        mode = self.mode_combo.currentText()
        points_per_side = self.points_per_side.value()
        stability_score = self.stability_score.value()
        use_gpu = self.use_gpu.isChecked()
        create_overlay = self.create_overlay.isChecked()
        output_dir = self.output_path.text() if self.output_path.text() else None
        
        # Disable button during processing
        self.process_btn.setEnabled(False)
        
        # Create and start worker
        self.worker = SAM2SegmentWorker(
            input_path,
            model_name,
            checkpoint_path,
            mode,
            points_per_side,
            stability_score,
            use_gpu,
            create_overlay,
            output_dir
        )
        self.worker.progress.connect(self._parse_sam2_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()
        
        # Show progress bar
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
    
    def _parse_sam2_progress(self, message):
        """Parse progress messages from SAM2 worker and update progress bar"""
        import re
        # Log the message
        self.parent_window.log_output(message)
        
        # Try to parse percentage from message
        progress_match = re.search(r'(\d+)%', message)
        if progress_match:
            try:
                progress_pct = int(progress_match.group(1))
                progress_pct = max(0, min(100, progress_pct))
                self.parent_window.progress_bar.setValue(progress_pct)
            except:
                pass
        # Check for file processing progress (e.g., "Processing file 2/5")
        file_match = re.search(r'file\s+(\d+)/(\d+)', message, re.IGNORECASE)
        if file_match:
            try:
                current_file = int(file_match.group(1))
                total_files = int(file_match.group(2))
                if total_files > 0:
                    progress_pct = int((current_file / total_files) * 100)
                    self.parent_window.progress_bar.setValue(progress_pct)
            except:
                pass
        # Increment slightly for other progress messages
        elif any(keyword in message.lower() for keyword in ['processing', 'loading', 'creating', 'saving']):
            current = self.parent_window.progress_bar.value()
            if current < 95:
                self.parent_window.progress_bar.setValue(min(95, current + 1))
    
    def on_processing_finished(self, success, message):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.parent_window.progress_bar.setValue(100)
        self.parent_window.progress_bar.setVisible(False)
        
        if success:
            self.parent_window.log_output(f"✓ SAM2 segmentation completed!\n{message}\n")
            QMessageBox.information(self, "Success", f"SAM2 segmentation completed!\n\n{message}")
        else:
            self.parent_window.log_output(f"✗ SAM2 segmentation failed:\n{message}\n")
            QMessageBox.critical(self, "Error", f"SAM2 segmentation failed:\n\n{message}")
    
    def get_session_params(self):
        """Return current parameter values for session log (parameters only, no paths)."""
        return {
            "model": self.model_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "points_per_side": self.points_per_side.value(),
            "stability_score": self.stability_score.value(),
            "use_gpu": self.use_gpu.isChecked(),
            "create_overlay": self.create_overlay.isChecked(),
        }
    
    def set_session_params(self, data):
        """Apply saved parameter values (parameters only)."""
        if not isinstance(data, dict):
            return
        if "model" in data:
            idx = self.model_combo.findText(str(data["model"]))
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if "mode" in data:
            idx = self.mode_combo.findText(str(data["mode"]))
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)
        for k, w in [
            ("points_per_side", self.points_per_side), ("stability_score", self.stability_score),
        ]:
            if k in data and hasattr(w, "setValue"):
                try:
                    w.setValue(float(data[k]))
                except (TypeError, ValueError):
                    pass
        for k, w in [("use_gpu", self.use_gpu), ("create_overlay", self.create_overlay)]:
            if k in data and hasattr(w, "setChecked"):
                w.setChecked(bool(data[k]))


class SAM2SegmentWorker(QThread):
    """Worker thread for SAM2 segmentation processing"""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, input_path, model_name, checkpoint_path, mode, points_per_side,
                 stability_score, use_gpu, create_overlay, output_dir):
        super().__init__()
        self.input_path = input_path
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.points_per_side = points_per_side
        self.stability_score = stability_score
        self.use_gpu = use_gpu
        self.create_overlay = create_overlay
        self.output_dir = output_dir
    
    def run(self):
        """Run SAM2 segmentation"""
        try:
            from pathlib import Path
            import tifffile
            import torch
            import numpy as np
            from skimage.measure import regionprops
            
            self.progress.emit("Initializing SAM2 model...")
            
            # Determine device
            if self.use_gpu and torch.cuda.is_available():
                device = "cuda"
            elif self.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
            else:
                device = "cpu"
            
            self.progress.emit(f"Using device: {device}")
            
            # Import SAM2 (optional dependency)
            try:
                from sam2.build_sam import build_sam2  # type: ignore
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
                from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
            except ImportError:
                self.finished.emit(False, "SAM2 not installed. Please install with: pip install sam2")
                return
            
            # Build model - try multiple approaches
            model = None
            predictor = None
            
            # Map model names to Hugging Face model IDs
            hf_model_map = {
                'sam2.1_hiera_tiny': 'facebook/sam2-hiera-tiny',
                'sam2.1_hiera_small': 'facebook/sam2-hiera-small',
                'sam2.1_hiera_base_plus': 'facebook/sam2-hiera-base-plus',
                'sam2.1_hiera_large': 'facebook/sam2-hiera-large',
            }
            hf_model_name = hf_model_map.get(self.model_name, 'facebook/sam2-hiera-small')
            
            # Approach 1: Try Hugging Face first (automatic download)
            if model is None:
                try:
                    self.progress.emit(f"Attempting to load from Hugging Face: {hf_model_name}")
                    self.progress.emit("This will automatically download the model if not already cached...")
                    # Try using SAM2ImagePredictor.from_pretrained (recommended method)
                    predictor = SAM2ImagePredictor.from_pretrained(hf_model_name, device=device)
                    # Build model from predictor for automatic mask generator
                    model = predictor.model
                    self.progress.emit(f"✓ Model loaded from Hugging Face: {hf_model_name}")
                except Exception as e:
                    self.progress.emit(f"Warning: Could not load from Hugging Face: {e}")
                    import traceback
                    self.progress.emit(f"Traceback: {traceback.format_exc()}")
                    # Try alternative: download checkpoint and load manually
                    try:
                        self.progress.emit("Trying alternative: downloading checkpoint from Hugging Face...")
                        try:
                            from huggingface_hub import hf_hub_download
                        except ImportError:
                            self.progress.emit("huggingface_hub not installed. Install with: pip install huggingface_hub")
                            raise
                        
                        # Map to checkpoint filename
                        checkpoint_map = {
                            'sam2.1_hiera_tiny': 'sam2_hiera_tiny.pt',
                            'sam2.1_hiera_small': 'sam2_hiera_small.pt',
                            'sam2.1_hiera_base_plus': 'sam2_hiera_base_plus.pt',
                            'sam2.1_hiera_large': 'sam2_hiera_large.pt',
                        }
                        checkpoint_filename = checkpoint_map.get(self.model_name, 'sam2_hiera_small.pt')
                        
                        # Download checkpoint
                        cache_dir = Path.home() / ".cache" / "sam2" / "checkpoints"
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_path = cache_dir / checkpoint_filename
                        
                        if not checkpoint_path.exists():
                            self.progress.emit(f"Downloading checkpoint: {checkpoint_filename}...")
                            downloaded_path = hf_hub_download(
                                repo_id=hf_model_name,
                                filename=checkpoint_filename,
                                local_dir=str(cache_dir),
                                local_dir_use_symlinks=False
                            )
                            checkpoint_path = Path(downloaded_path)
                            self.progress.emit(f"✓ Checkpoint downloaded to: {checkpoint_path}")
                        else:
                            self.progress.emit(f"Using cached checkpoint: {checkpoint_path}")
                        
                        # Load using build_sam2
                        model_name_clean = self.model_name.replace('sam2.1_', '')
                        config_name_map = {
                            'hiera_tiny': 'sam2.1_hiera_t',
                            'hiera_small': 'sam2.1_hiera_s',
                            'hiera_base_plus': 'sam2.1_hiera_b+',
                            'hiera_large': 'sam2.1_hiera_l',
                        }
                        config_name = config_name_map.get(model_name_clean, 'sam2.1_hiera_s')
                        
                        # Try to find config file
                        try:
                            import sam2  # type: ignore
                            sam2_path = Path(sam2.__file__).parent.parent
                            possible_configs = [
                                sam2_path / f"sam2/configs/{config_name}.yaml",
                                sam2_path / f"configs/{config_name}.yaml",
                                sam2_path / f"sam2/configs/sam2.1/{config_name}.yaml",
                            ]
                            config_path = None
                            for possible_config in possible_configs:
                                if possible_config.exists():
                                    config_path = str(possible_config)
                                    break
                            if config_path is None:
                                config_path = config_name
                        except:
                            config_path = config_name
                        
                        model = build_sam2(config_path, str(checkpoint_path), device=device)
                        self.progress.emit(f"✓ Model loaded from downloaded checkpoint")
                    except Exception as e2:
                        self.progress.emit(f"Alternative method also failed: {e2}")
            
            # Approach 2: Use checkpoint file if provided by user
            if model is None and self.checkpoint_path and Path(self.checkpoint_path).exists():
                try:
                    self.progress.emit(f"Loading model from user-specified checkpoint: {self.checkpoint_path}")
                    # Map model name to config path (try multiple possible locations)
                    model_name_clean = self.model_name.replace('sam2.1_', '')
                    config_name_map = {
                        'hiera_tiny': 'sam2.1_hiera_t',
                        'hiera_small': 'sam2.1_hiera_s',
                        'hiera_base_plus': 'sam2.1_hiera_b+',
                        'hiera_large': 'sam2.1_hiera_l',
                    }
                    config_name = config_name_map.get(model_name_clean, 'sam2.1_hiera_s')
                    
                    # Try different possible config paths
                    possible_configs = [
                        f"sam2/configs/{config_name}.yaml",
                        f"configs/{config_name}.yaml",
                        f"sam2/configs/sam2.1/{config_name}.yaml",
                    ]
                    
                    config_path = None
                    for possible_config in possible_configs:
                        try:
                            # Try to import and check if config exists
                            import sam2  # type: ignore
                            sam2_path = Path(sam2.__file__).parent.parent
                            full_config_path = sam2_path / possible_config
                            if full_config_path.exists():
                                config_path = str(full_config_path)
                                break
                        except:
                            pass
                    
                    if config_path is None:
                        # Use config name directly (build_sam2 may handle it)
                        config_path = config_name
                    
                    model = build_sam2(config_path, self.checkpoint_path, device=device)
                    self.progress.emit("✓ Model loaded from user checkpoint")
                except Exception as e:
                    self.progress.emit(f"Warning: Could not load from checkpoint: {e}")
                    import traceback
                    self.progress.emit(f"Traceback: {traceback.format_exc()}")
            
            # Approach 3: Try default checkpoint location (if Hugging Face failed)
            if model is None:
                try:
                    self.progress.emit("Attempting to find checkpoint in default location...")
                    # Map to checkpoint filename
                    checkpoint_map = {
                        'sam2.1_hiera_tiny': 'sam2_hiera_tiny.pt',
                        'sam2.1_hiera_small': 'sam2_hiera_small.pt',
                        'sam2.1_hiera_base_plus': 'sam2_hiera_base_plus.pt',
                        'sam2.1_hiera_large': 'sam2_hiera_large.pt',
                    }
                    checkpoint_filename = checkpoint_map.get(self.model_name, 'sam2_hiera_small.pt')
                    
                    # Common default locations
                    default_locations = [
                        Path.home() / ".cache" / "sam2" / "checkpoints" / checkpoint_filename,
                        Path.home() / ".cache" / "sam2" / "checkpoints" / f"{self.model_name}.pt",
                        Path("checkpoints") / checkpoint_filename,
                        Path("checkpoints") / f"{self.model_name}.pt",
                        Path(".") / "checkpoints" / checkpoint_filename,
                        Path(".") / "checkpoints" / f"{self.model_name}.pt",
                    ]
                    
                    checkpoint_found = None
                    for loc in default_locations:
                        if loc.exists():
                            checkpoint_found = loc
                            break
                    
                    if checkpoint_found:
                        model_name_clean = self.model_name.replace('sam2.1_', '')
                        config_name_map = {
                            'hiera_tiny': 'sam2.1_hiera_t',
                            'hiera_small': 'sam2.1_hiera_s',
                            'hiera_base_plus': 'sam2.1_hiera_b+',
                            'hiera_large': 'sam2.1_hiera_l',
                        }
                        config_name = config_name_map.get(model_name_clean, 'sam2.1_hiera_s')
                        
                        # Try to find config file
                        try:
                            import sam2  # type: ignore
                            sam2_path = Path(sam2.__file__).parent.parent
                            possible_configs = [
                                sam2_path / f"sam2/configs/{config_name}.yaml",
                                sam2_path / f"configs/{config_name}.yaml",
                                sam2_path / f"sam2/configs/sam2.1/{config_name}.yaml",
                            ]
                            config_path = None
                            for possible_config in possible_configs:
                                if possible_config.exists():
                                    config_path = str(possible_config)
                                    break
                            if config_path is None:
                                config_path = config_name
                        except:
                            config_path = config_name
                        
                        model = build_sam2(config_path, str(checkpoint_found), device=device)
                        self.progress.emit(f"✓ Model loaded from: {checkpoint_found}")
                    else:
                        raise FileNotFoundError("No checkpoint found in default locations")
                except Exception as e:
                    self.finished.emit(False, 
                        f"Could not load SAM2 model. Tried:\n\n"
                        f"1. Hugging Face automatic download (failed)\n"
                        f"2. User-specified checkpoint path (not provided or failed)\n"
                        f"3. Default checkpoint locations (not found)\n\n"
                        f"Solutions:\n"
                        f"• Install huggingface_hub: pip install huggingface_hub\n"
                        f"• Download checkpoint manually from:\n"
                        f"  https://github.com/facebookresearch/sam2#download-checkpoints\n"
                        f"• Place checkpoint in: ~/.cache/sam2/checkpoints/\n"
                        f"• Or specify checkpoint path in the UI\n\n"
                        f"Error: {str(e)}")
                    return
            
            # Create automatic mask generator
            self.progress.emit("Creating automatic mask generator...")
            mask_generator = SAM2AutomaticMaskGenerator(
                model=model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=0.8,
                stability_score_thresh=self.stability_score,
                box_nms_thresh=0.7,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Filter out very small masks
            )
            
            # Get input files
            input_path = Path(self.input_path)
            if input_path.is_file():
                tiff_files = [input_path]
            else:
                tiff_files = sorted(input_path.glob("*.tif")) + sorted(input_path.glob("*.tiff"))
            
            if not tiff_files:
                self.finished.emit(False, "No TIFF files found in input path.")
                return
            
            self.progress.emit(f"Found {len(tiff_files)} TIFF file(s). Processing...")
            
            # Process each file
            all_masks = []
            all_images = []
            processed_count = 0
            
            for file_idx, tiff_file in enumerate(tiff_files):
                self.progress.emit(f"Processing {tiff_file.name} ({file_idx + 1}/{len(tiff_files)})...")
                
                # Read image
                image_data = safe_imread(str(tiff_file))
                
                # Handle different image formats
                if image_data.ndim == 2:
                    # Single 2D image
                    images = [image_data]
                elif image_data.ndim == 3:
                    # Could be (T, Y, X) or (Y, X, C)
                    if image_data.shape[-1] < 10:
                        # Likely (T, Y, X) - time series
                        images = [image_data[i] for i in range(image_data.shape[0])]
                    else:
                        # Likely (Y, X, C) - single image with channels
                        images = [image_data[:, :, 0]]  # Take first channel
                elif image_data.ndim == 4:
                    # (T, C, Y, X) or (C, T, Y, X)
                    if image_data.shape[1] < 10:
                        # (T, C, Y, X)
                        images = [image_data[i, 0] for i in range(image_data.shape[0])]
                    else:
                        # (C, T, Y, X)
                        images = [image_data[0, i] for i in range(image_data.shape[1])]
                else:
                    self.progress.emit(f"Warning: Unsupported image format for {tiff_file.name}, skipping...")
                    continue
                
                # Process each frame
                file_masks = []
                for frame_idx, image in enumerate(images):
                    if len(images) > 1:
                        self.progress.emit(f"  Frame {frame_idx + 1}/{len(images)}...")
                    
                    # Normalize image to 0-255 uint8 for SAM2
                    img_min, img_max = image.min(), image.max()
                    if img_max > img_min:
                        image_norm = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        image_norm = image.astype(np.uint8)
                    
                    # Convert to RGB if needed (SAM2 expects RGB)
                    if len(image_norm.shape) == 2:
                        image_rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
                    elif image_norm.shape[-1] == 1:
                        image_rgb = np.repeat(image_norm, 3, axis=-1)
                    else:
                        image_rgb = image_norm[:, :, :3]  # Take first 3 channels
                    
                    # Generate masks
                    with torch.inference_mode():
                        if device == "mps":
                            # MPS doesn't support all operations, use autocast
                            with torch.autocast(device_type="cpu", dtype=torch.float32):
                                sam2_masks = mask_generator.generate(image_rgb)
                        else:
                            sam2_masks = mask_generator.generate(image_rgb)
                    
                    # Convert SAM2 masks to label mask (like Omnipose output)
                    # SAM2 returns list of dicts with 'segmentation' key
                    if not sam2_masks:
                        # No masks found
                        label_mask = np.zeros(image_norm.shape[:2], dtype=np.uint16)
                    else:
                        # Create label mask from SAM2 masks
                        label_mask = np.zeros(image_norm.shape[:2], dtype=np.uint16)
                        # Sort by area (largest first) for consistent labeling
                        sam2_masks_sorted = sorted(sam2_masks, key=lambda x: x['area'], reverse=True)
                        
                        for mask_idx, mask_dict in enumerate(sam2_masks_sorted, start=1):
                            mask = mask_dict['segmentation']
                            label_mask[mask] = mask_idx
                    
                    file_masks.append(label_mask)
                    if self.create_overlay:
                        all_images.append(image_norm)
                
                all_masks.extend(file_masks)
                processed_count += 1
            
            if not all_masks:
                self.finished.emit(False, "No masks were generated. Try adjusting parameters.")
                return
            
            # Determine output directory
            if self.output_dir:
                output_dir = Path(self.output_dir)
            else:
                output_dir = input_path.parent if input_path.is_file() else input_path
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save masks
            self.progress.emit("Saving masks...")
            output_stem = input_path.stem if input_path.is_file() else input_path.name
            masks_output = output_dir / f"{output_stem}_sam2_masks.tif"
            masks_array = np.stack(all_masks, axis=0)
            tifffile.imwrite(str(masks_output), masks_array.astype(np.uint16))
            self.progress.emit(f"Masks saved to: {masks_output}")
            
            # Create overlay if requested
            if self.create_overlay and all_images:
                self.progress.emit("Creating overlay visualization...")
                try:
                    from matplotlib import pyplot as plt
                    from PIL import Image, ImageDraw, ImageFont
                    
                    overlay_output = output_dir / f"{output_stem}_sam2_overlay.tif"
                    overlays = []
                    
                    num_frames = min(len(all_images), len(all_masks))
                    for i in range(num_frames):
                        img = all_images[i]
                        mask = all_masks[i]
                        
                        # Convert grayscale to RGB
                        if len(img.shape) == 2:
                            img_rgb = np.stack([img, img, img], axis=-1)
                        else:
                            img_rgb = img[:, :, :3]
                        
                        # Create colored overlay
                        cmap = plt.get_cmap('gist_ncar')
                        num_labels = int(mask.max())
                        if num_labels > 0:
                            colors = cmap(np.linspace(0, 1, max(num_labels + 1, 100)))[:num_labels + 1]
                            colors[0] = [0, 0, 0, 0]  # Background transparent
                            mask_rgb = colors[mask][:, :, :3] * 255
                            overlay = (img_rgb * 0.5 + mask_rgb * 0.5).astype(np.uint8)
                        else:
                            overlay = img_rgb.astype(np.uint8)
                        
                        # Add labels
                        if num_labels > 0:
                            try:
                                pil_img = Image.fromarray(overlay)
                                draw = ImageDraw.Draw(pil_img)
                                try:
                                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                                except:
                                    font = ImageFont.load_default()
                                
                                # regionprops is imported at the top of run() method
                                regions = regionprops(mask)
                                for region in regions:
                                    if region.label == 0:
                                        continue
                                    cy, cx = region.centroid
                                    x, y = int(cx), int(cy)
                                    label_str = str(region.label)
                                    draw.text((x + 1, y + 1), label_str, fill=(0, 0, 0), font=font)
                                    draw.text((x, y), label_str, fill=(255, 255, 255), font=font)
                                
                                overlay = np.array(pil_img)
                            except:
                                pass
                        
                        overlays.append(overlay)
                    
                    overlay_array = np.stack(overlays, axis=0)
                    tifffile.imwrite(str(overlay_output), overlay_array)
                    self.progress.emit(f"Overlay saved to: {overlay_output}")
                except Exception as e:
                    self.progress.emit(f"Warning: Could not create overlay: {e}")
            
            self.finished.emit(True, 
                f"SAM2 segmentation completed!\n\n"
                f"Processed: {processed_count} file(s)\n"
                f"Total frames: {len(all_masks)}\n"
                f"Masks saved to: {masks_output}\n"
                f"{f'Overlay saved to: {overlay_output}' if self.create_overlay and all_images else ''}")
        
        except Exception as e:
            import traceback
            error_msg = f"Error during SAM2 segmentation:\n{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)


# Tab label to log key mapping for session logs (order matches main window tabs)
TAB_LOG_KEYS = [
    ("nd2", "TIFF Conversion"),
    ("segmentation", "Segmentation"),
    ("sam2", "SAM2 Segment"),
    ("analyze_masks", "Analyze Masks"),
    ("tracking", "Tracking"),
    ("growth_curve_video", "Video Graph"),
]


class SessionLogsTab(QWidget):
    """Tab to list session log files, show overview of saved settings, and apply them."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._selected_path = None
        self._loaded_data = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # List of last 10 session log files
        list_group = QGroupBox("Recent session logs (last 10)")
        list_layout = QVBoxLayout()
        self.log_list = QListWidget()
        self.log_list.setMinimumHeight(120)
        self.log_list.itemSelectionChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.log_list)
        load_other_btn = QPushButton("Load other...")
        load_other_btn.setToolTip("Open a session log file that is not in the list above")
        load_other_btn.clicked.connect(self._load_other)
        list_layout.addWidget(load_other_btn)
        list_group.setLayout(list_layout)
        scroll_layout.addWidget(list_group)
        
        # Overview of what was used (params only)
        overview_group = QGroupBox("Overview of selected session")
        overview_layout = QVBoxLayout()
        self.overview_text = QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setMinimumHeight(200)
        self.overview_text.setPlaceholderText("Select a log file above or click 'Load other...' to see parameters.")
        overview_layout.addWidget(self.overview_text)
        overview_group.setLayout(overview_layout)
        scroll_layout.addWidget(overview_group)
        
        # Apply button
        self.apply_btn = QPushButton("Apply now to current session")
        self.apply_btn.setToolTip("Apply the parameters from the selected log to all tabs (parameters only; file paths are not changed).")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_now)
        scroll_layout.addWidget(self.apply_btn)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
        
        self._refresh_list()
    
    def _refresh_list(self):
        """Refresh the list of last 10 log files."""
        self.log_list.clear()
        self._selected_path = None
        self._loaded_data = None
        self.overview_text.clear()
        self.apply_btn.setEnabled(False)
        if not LOGS_DIR.exists():
            return
        files = sorted(LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:10]:
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                label = f"{mtime.strftime('%Y-%m-%d %H:%M:%S')} — {p.name}"
            except Exception:
                label = p.name
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.log_list.addItem(item)
    
    def _on_selection_changed(self):
        """When user selects an item, load that file and show overview."""
        items = self.log_list.selectedItems()
        if not items:
            self._selected_path = None
            self._loaded_data = None
            self.overview_text.clear()
            self.apply_btn.setEnabled(False)
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        self._load_and_show(path)
    
    def _load_other(self):
        """Open file dialog to load a session log file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open session log",
            str(LOGS_DIR) if LOGS_DIR.exists() else "",
            "JSON (*.json);;All files (*)"
        )
        if path:
            self._load_and_show(path)
    
    def _load_and_show(self, path):
        """Load JSON at path, store in _loaded_data, and show overview."""
        self._selected_path = path
        self._loaded_data = None
        self.overview_text.clear()
        self.apply_btn.setEnabled(False)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.overview_text.setPlainText(f"Could not load file:\n{e}")
            return
        self._loaded_data = data
        # Build overview: group by tab, param: value
        lines = []
        schema = data.get("schema_version", "?")
        ts = data.get("timestamp", "?")
        lines.append(f"Schema version: {schema}  |  Saved: {ts}\n")
        tabs_data = data.get("tabs", data)
        if not isinstance(tabs_data, dict):
            tabs_data = {}
        for key, tab_label in TAB_LOG_KEYS:
            if key not in tabs_data or not isinstance(tabs_data[key], dict):
                continue
            lines.append(f"\n——— {tab_label} ———")
            for k, v in sorted(tabs_data[key].items()):
                lines.append(f"  {k}: {v}")
        self.overview_text.setPlainText("\n".join(lines) if lines else "(No tab data)")
        self.apply_btn.setEnabled(True)
    
    def _apply_now(self):
        """Apply the currently loaded session params to the main window."""
        if not self._loaded_data or not self.parent_window:
            return
        tabs_data = self._loaded_data.get("tabs", self._loaded_data)
        if isinstance(tabs_data, dict):
            self.parent_window.apply_session_params(tabs_data)
        QMessageBox.information(self, "Applied", "Parameters from the selected session have been applied to the current session (parameters only).")
    
    def showEvent(self, event):
        """Refresh list when tab is shown."""
        super().showEvent(event)
        self._refresh_list()


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
        # Set minimum size to prevent horizontal scrollbars in tabs
        # Minimum width ensures all content fits without horizontal scrolling
        self.setMinimumSize(1200, 700)
        # Set default size (will be resizable)
        self.resize(1200, 900)
        # Center window on screen
        self._center_window()
        
        # Central widget with tabs
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Custom title bar with logo and text
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
        
        # Add stretch to center content
        title_bar_layout.addStretch()
        
        # Logo (use base_path when frozen/Nuitka so logo next to exe is found)
        logo_label = QLabel()
        logo_path = Path(base_path) / "logo2.png" if (getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')) else Path(__file__).parent / "logo2.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            # Scale logo to fit title bar (slimmer, max height 60px, maintain aspect ratio)
            if pixmap.height() > 60:
                pixmap = pixmap.scaledToHeight(60, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setStyleSheet("background: transparent;")
        else:
            logo_label.setText("BactiMate")
            logo_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #ECF0F1; background: transparent;")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        title_bar_layout.addWidget(logo_label)
        
        # Title text
        
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
        
        # Add stretch to center content
        title_bar_layout.addStretch()
        
        title_bar.setLayout(title_bar_layout)
        main_layout.addWidget(title_bar)
        
        # Create tabs
        tabs = QTabWidget()
        self.nd2_tab = ND2ConversionTab(self)
        self.processing_tab = ProcessingTab(self)
        self.analyze_tab = AnalyzeMasksTab(self)
        self.tracking_tab = TrackingTab(self)
        self.growth_curve_tab = GrowthCurveVideoTab(self)
        self.sam2_tab = SAM2SegmentTab(self)
        self.logs_tab = SessionLogsTab(self)
        
        tabs.addTab(self.nd2_tab, "TIFF Conversion")
        tabs.addTab(self.processing_tab, "Segmentation")
        tabs.addTab(self.sam2_tab, "SAM2 Segment")
        tabs.addTab(self.analyze_tab, "Analyze Masks")
        tabs.addTab(self.tracking_tab, "Tracking")
        tabs.addTab(self.growth_curve_tab, "Video Graph")
        tabs.addTab(self.logs_tab, "Session Logs")
       
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
        log_size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
    
    def closeEvent(self, event):
        """Save session log on exit (per-session, Option B)."""
        self._save_session_log()
        super().closeEvent(event)
    
    def _save_session_log(self):
        """Write current session parameters to Logs/ with timestamp in filename: YYYY-MM-DD_HH-MM-SS_session.json."""
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            now = datetime.now()
            # Filename includes date and time so every session gets a unique file
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
            pass  # avoid blocking exit on any error
    
    def gather_session_params(self):
        """Collect parameters from all tabs (no paths). Returns dict keyed by tab id."""
        out = {}
        for key, tab in [
            ("nd2", self.nd2_tab),
            ("segmentation", self.processing_tab),
            ("analyze_masks", self.analyze_tab),
            ("tracking", self.tracking_tab),
            ("growth_curve_video", self.growth_curve_tab),
            ("sam2", self.sam2_tab),
        ]:
            if hasattr(tab, "get_session_params"):
                out[key] = tab.get_session_params()
        return out
    
    def apply_session_params(self, data):
        """Apply saved parameters to tabs (parameters only; file paths unchanged)."""
        if not isinstance(data, dict):
            return
        for key, tab in [
            ("nd2", self.nd2_tab),
            ("segmentation", self.processing_tab),
            ("analyze_masks", self.analyze_tab),
            ("tracking", self.tracking_tab),
            ("growth_curve_video", self.growth_curve_tab),
            ("sam2", self.sam2_tab),
        ]:
            if key in data and hasattr(tab, "set_session_params"):
                tab.set_session_params(data[key])
    
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
        """Update all tabs with shared metadata (e.g., pixel size)"""
        pixel_size = self.shared_metadata.get('pixel_size')
        
        if pixel_size is not None:
            # Update Processing tab
            if hasattr(self, 'processing_tab') and hasattr(self.processing_tab, 'pixel_size'):
                self.processing_tab.pixel_size.setValue(pixel_size)
                # Show indicator that value was auto-filled
                if hasattr(self.processing_tab, 'pixel_size_indicator'):
                    self.processing_tab.pixel_size_indicator.setText("(from metadata)")
                self.log_output(f"✓ Updated pixel size in Processing tab: {pixel_size:.4f} µm\n")
            
            # Update Analyze Masks tab
            if hasattr(self, 'analyze_tab') and hasattr(self.analyze_tab, 'pixel_size'):
                self.analyze_tab.pixel_size.setValue(pixel_size)
                # Show indicator that value was auto-filled
                if hasattr(self.analyze_tab, 'pixel_size_indicator'):
                    self.analyze_tab.pixel_size_indicator.setText("(from metadata)")
                    self.analyze_tab.pixel_size_indicator.setStyleSheet("color: #4CAF50; font-size: 9pt;")
                    self.analyze_tab.pixel_size_indicator.setFixedWidth(100)  # Expand when content is added
                    self.analyze_tab.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.analyze_tab.pixel_size_indicator.show()
                    self.analyze_tab.pixel_size_metadata_loaded = True  # Mark that metadata was loaded
                self.log_output(f"✓ Updated pixel size in Analyze Masks tab: {pixel_size:.4f} µm\n")


def main():
    # Import sys at function level to avoid any potential scoping issues
    import sys
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
    
    # Install custom exception handler for PyQt6 to prevent abort() on exceptions in slots
    # PyQt6 calls abort() by default when exceptions occur in slots, which crashes the app
    # Note: We use the module-level sys, not a local variable
    original_excepthook = sys.excepthook
    def pyqt6_exception_handler(exc_type, exc_value, exc_traceback):
        """Custom exception handler that prevents PyQt6 from aborting"""
        # Print the exception to stderr
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        # Don't call abort() - let the exception propagate or be handled
        # This prevents PyQt6's default behavior of calling abort()
    sys.excepthook = pyqt6_exception_handler
    
    try:
        app = QApplication(sys.argv)
        
        app.setStyle('Fusion')  # Modern look
        app.setStyleSheet(get_modern_stylesheet())  # Apply modern styling
        
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
