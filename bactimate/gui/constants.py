"""
Shared constants for BactiMate GUI.
Single source of truth for tab keys, labels, script paths, and MainWindow tab widget attributes.
"""
from pathlib import Path

from bactimate.gui.paths import get_app_root, get_logs_dir, get_script_path

# Application root and logs (single source: gui/paths.py)
PROJECT_ROOT = get_app_root()
LOGS_DIR = get_logs_dir()

# Log retention: keep only the last N files so logs don't grow indefinitely
MAX_SESSION_LOGS = 100   # session JSONs (saved parameters)
MAX_GUI_LOGS = 20       # bactimate_gui_*.log (single log per run: launch + crash + Qt + debug)

# Script paths for subprocess calls
# Prefer package script (bactimate/scripts/); fallback to get_script_path for installed/frozen layouts
_pkg_convert = get_app_root() / "bactimate" / "scripts" / "convert_nd2_to_tif.py"
CONVERT_ND2_SCRIPT = _pkg_convert if _pkg_convert.exists() else get_script_path("convert_nd2_to_tif.py")
_pkg_batch_tiff = get_app_root() / "bactimate" / "scripts" / "batch_process_tiff_sequence.py"
BATCH_PROCESS_TIFF_SCRIPT = _pkg_batch_tiff if _pkg_batch_tiff.exists() else get_script_path("batch_process_tiff_sequence.py")

# Tab (key, display label) in display order. Used for session gather/apply and session log overview.
SESSION_TAB_KEYS = [
    ("nd2", "File Conversion"),
    ("segmentation", "Omnipose Segment"),
    ("unet_segment", "U-NET Segment"),
    ("analyze_masks", "Analyze Masks"),
    ("tracking", "Tracking"),
    ("growth_curve_video", "Post-processing"),
]

# MainWindow attribute name for each tab key (so we can get widget with getattr(main_window, TAB_WIDGET_ATTR[key])).
TAB_WIDGET_ATTR = {
    "nd2": "nd2_tab",
    "segmentation": "omnipose_seg_tab",
    "unet_segment": "unet_segment_tab",
    "analyze_masks": "analyze_tab",
    "tracking": "tracking_tab",
    "growth_curve_video": "post_processing_tab",
}
