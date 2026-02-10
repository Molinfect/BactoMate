"""
Shared constants for BactiMate GUI.
Single source of truth for tab keys, labels, script paths, and MainWindow tab widget attributes.
"""
from pathlib import Path

# Project root (directory containing gui/, scripts/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Session log storage
LOGS_DIR = PROJECT_ROOT / "Logs"

# Script paths for subprocess calls (relative to project root)
CONVERT_ND2_SCRIPT = PROJECT_ROOT / "scripts" / "convert_nd2_to_tif.py"

# Tab (key, display label) in display order. Used for session gather/apply and session log overview.
SESSION_TAB_KEYS = [
    ("nd2", "TIFF Conversion"),
    ("segmentation", "Omnipose Segment"),
    ("sam2", "SAM2 Segment"),
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
