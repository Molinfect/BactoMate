# Analyze Masks Legacy Snapshot

The previous monolithic `AnalyzeMasksTab(QWidget)` implementation was intentionally removed from
`bactomate/gui/tabs/analyze_masks.py` after the split into:

- `bactomate/gui/tabs/_analyze_masks_base.py`
- `bactomate/gui/tabs/_analyze_masks_filtering.py`
- `bactomate/gui/tabs/_analyze_masks_analyze.py`
- `bactomate/gui/tabs/_analyze_masks_preview_filtering.py`
- `bactomate/gui/preview/preview_window.py`

The legacy code is preserved in git history for audit/diff review and should not be restored into
runtime import modules.
