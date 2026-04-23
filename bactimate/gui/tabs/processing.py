"""
Processing tab (placeholder).

The current codebase references the Omnipose patch hook used by the historical
Processing tab. Some tests assert this module exists (import discovery only).

If/when the Processing UI is reintroduced, implement `ProcessingTab` here.
"""

from __future__ import annotations

from bactimate.gui.patches import _patch_omnipose_get_masks

# Ensure the patch symbol is importable by name.
__all__ = ["_patch_omnipose_get_masks"]

try:
    import PyQt6  # noqa: F401

    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False

if QT_AVAILABLE:
    from PyQt6.QtWidgets import QWidget

    class ProcessingTab(QWidget):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            _patch_omnipose_get_masks()

    __all__.append("ProcessingTab")

