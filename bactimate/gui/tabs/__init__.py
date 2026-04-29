"""
Tab widgets for the BactiMate main window.

This module must remain importable in environments without Qt (e.g. some CI / packaging
checks). Import Qt-dependent widgets conditionally.
"""

try:
    import PyQt6  # noqa: F401

    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False

if QT_AVAILABLE:
    from bactimate.gui.tabs.analyze_masks import AnalyzeMasksTab, AnalyzeTab, FilteringTab
    from bactimate.gui.tabs.post_processing import PostProcessingTab
    from bactimate.gui.tabs.file_conversion import ND2ConversionTab
    from bactimate.gui.tabs.omnipose_seg import OmniposeSegTab
    from bactimate.gui.tabs.session_logs import SessionLogsPanel
    from bactimate.gui.tabs.tracking import TrackingTab
    from bactimate.gui.tabs.unet_seg import UNetSegmentTab
    from bactimate.gui.tabs.segmentation import SegmentationTab
    from bactimate.gui.tabs.training import TrainingTab

__all__ = []
if QT_AVAILABLE:
    __all__ = [
        "AnalyzeMasksTab",
        "FilteringTab",
        "AnalyzeTab",
        "PostProcessingTab",
        "ND2ConversionTab",
        "OmniposeSegTab",
        "SessionLogsPanel",
        "TrackingTab",
        "UNetSegmentTab",
        "SegmentationTab",
        "TrainingTab",
    ]
