"""
Public import facade for the Filtering + Analyze stages (Sections 2–3) of the BactoMate workflow.

This module intentionally does **not** implement Segmentation (Section 1) or any live
foci-detection / Spotiflow tuning UI. Those belong to the Segmentation preview surface.

Exports:
- `AnalyzeMasksBase` (shared widget base)
- `FilteringTab` (Section 2)
- `AnalyzeTab` (Section 3)
- `AnalyzeMasksTab(parent, mode=...)` factory for backward compatibility

Legacy monolithic implementation notes are documented in:
`docs/history/analyze_masks_legacy_snapshot.md`
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import QWidget

from bactomate.gui.tabs._analyze_masks_base import AnalyzeMasksBase
from bactomate.gui.tabs._analyze_masks_filtering import FilteringTab
from bactomate.gui.tabs._analyze_masks_analyze import AnalyzeTab


def AnalyzeMasksTab(parent: Optional[QWidget] = None, *, mode: str = "analyze"):
    """Backward-compatible factory. Supported modes: 'filtering', 'analyze'."""
    m = (mode or "").strip().lower()
    if m == "filtering":
        return FilteringTab(parent)
    if m == "analyze":
        return AnalyzeTab(parent)
    raise ValueError(f"Unsupported mode: {mode!r}. Use 'filtering' or 'analyze'.")


__all__ = [
    "AnalyzeMasksBase",
    "FilteringTab",
    "AnalyzeTab",
    "AnalyzeMasksTab",
]
