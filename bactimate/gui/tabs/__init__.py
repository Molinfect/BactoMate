# Tab widgets for BactiMate main window.
from bactimate.gui.tabs.analyze_masks import AnalyzeMasksTab
from bactimate.gui.tabs.post_processing import PostProcessingTab
from bactimate.gui.tabs.file_conversion import ND2ConversionTab
from bactimate.gui.tabs.omnipose_seg import OmniposeSegTab
from bactimate.gui.tabs.session_logs import SessionLogsPanel
from bactimate.gui.tabs.tracking import TrackingTab
from bactimate.gui.tabs.unet_seg import UNetSegmentTab

__all__ = [
    "AnalyzeMasksTab",
    "PostProcessingTab",
    "ND2ConversionTab",
    "OmniposeSegTab",
    "SessionLogsPanel",
    "TrackingTab",
    "UNetSegmentTab",
]
