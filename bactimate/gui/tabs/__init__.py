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
    def _startup_log(message: str) -> None:
        try:
            from bactimate.gui.log_config import write_gui_log_line
            write_gui_log_line("startup-import", message)
        except Exception:
            pass

    _startup_log("before analyze_masks")
    from bactimate.gui.tabs.analyze_masks import AnalyzeMasksTab, AnalyzeTab, FilteringTab
    _startup_log("after analyze_masks")
    _startup_log("before post_processing")
    from bactimate.gui.tabs.post_processing import PostProcessingTab
    _startup_log("after post_processing")
    _startup_log("before file_conversion")
    from bactimate.gui.tabs.file_conversion import ND2ConversionTab
    _startup_log("after file_conversion")
    _startup_log("before omnipose_seg")
    from bactimate.gui.tabs.omnipose_seg import OmniposeSegTab
    _startup_log("after omnipose_seg")
    _startup_log("before session_logs")
    from bactimate.gui.tabs.session_logs import SessionLogsPanel
    _startup_log("after session_logs")
    _startup_log("before tracking")
    from bactimate.gui.tabs.tracking import TrackingTab
    _startup_log("after tracking")
    _startup_log("before unet_seg")
    from bactimate.gui.tabs.unet_seg import UNetSegmentTab
    _startup_log("after unet_seg")
    _startup_log("before segmentation")
    from bactimate.gui.tabs.segmentation import SegmentationTab
    _startup_log("after segmentation")
    _startup_log("before training")
    from bactimate.gui.tabs.training import TrainingTab
    _startup_log("after training")

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
