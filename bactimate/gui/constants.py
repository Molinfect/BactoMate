"""
Shared constants for BactiMate GUI.
Single source of truth for tab keys, labels, script paths, and MainWindow tab widget attributes.
"""
from pathlib import Path

from bactimate.gui.paths import get_app_root, get_logs_dir, get_script_path

# Application root and logs (single source: gui/paths.py)
PROJECT_ROOT = get_app_root()
LOGS_DIR = get_logs_dir()

# Script paths for subprocess calls
CONVERT_ND2_SCRIPT = get_script_path("convert_nd2_to_tif.py")

# Tab (key, display label) in display order. Used for session gather/apply and session log overview.
# SAM2 omitted while tab is not in use (commented out in main_window).
SESSION_TAB_KEYS = [
    ("nd2", "TIFF Conversion"),
    ("segmentation", "Omnipose Segment"),
    ("analyze_masks", "Analyze Masks"),
    ("tracking", "Tracking"),
    ("growth_curve_video", "Video Graph"),
]

# MainWindow attribute name for each tab key (so we can get widget with getattr(main_window, TAB_WIDGET_ATTR[key])).
TAB_WIDGET_ATTR = {
    "nd2": "nd2_tab",
    "segmentation": "processing_tab",
    "sam2": "sam2_tab",
    "analyze_masks": "analyze_tab",
    "tracking": "tracking_tab",
    "growth_curve_video": "growth_curve_tab",
}
