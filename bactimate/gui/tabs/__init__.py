# Tab widgets for BactiMate main window.
from bactimate.gui.tabs.analyze_masks import AnalyzeMasksTab
from bactimate.gui.tabs.growth_curve_video import GrowthCurveVideoTab
from bactimate.gui.tabs.file_conversion import ND2ConversionTab
from bactimate.gui.tabs.omnipose_seg import OmniposeSegTab
from bactimate.gui.tabs.sam2 import SAM2SegmentTab
from bactimate.gui.tabs.session_logs import SessionLogsPanel, SessionLogsTab
from bactimate.gui.tabs.tracking import TrackingTab
from bactimate.gui.tabs.unet_seg import UNetSegmentTab

__all__ = [
    "AnalyzeMasksTab",
    "GrowthCurveVideoTab",
    "ND2ConversionTab",
    "OmniposeSegTab",
    "SAM2SegmentTab",
    "SessionLogsPanel",
    "SessionLogsTab",
    "TrackingTab",
    "UNetSegmentTab",
]
