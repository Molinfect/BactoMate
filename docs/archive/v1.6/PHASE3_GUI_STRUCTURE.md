# Phase 3: GUI structure and patterns (v1.6)

**Purpose:** Map tab layout and shared UI patterns. No behavior change; Phase 1 tests stay green.

---

## 1. Tab roster (main window)

| Tab attribute | Class | Title in UI |
|---------------|--------|-------------|
| `nd2_tab` | `ND2ConversionTab` | TIFF Conversion |
| `omnipose_seg_tab` | `OmniposeSegTab` | Omnipose Segment |
| `unet_segment_tab` | `UNetSegmentTab` | U-NET Segment |
| — | *(was SAM2SegmentTab)* | Removed in Phase 5 (v1.6). |
| `analyze_tab` | `AnalyzeMasksTab` | Analyze Masks |
| `tracking_tab` | `TrackingTab` | Tracking |
| `post_processing_tab` | `PostProcessingTab` | Post-processing |

**Tab module:** Each tab lives in `bactomate/gui/tabs/<name>.py`. Exported via `bactomate/gui/tabs/__init__.py`. Main window imports from `bactomate.gui.tabs` and adds tabs in a fixed order.

**Session / status:** `SessionLogsPanel` (not a tab) is used in the main window status/session area. `SessionLogsTab` exists in code but is never instantiated (Phase 1.2 dead code).

---

## 2. Shared UI patterns

### 2.1 Parent window hooks

All tabs that run long work use the main window as parent and assume:

| Hook | Usage |
|------|--------|
| `self.parent_window` | Set in `__init__(self, parent)`; `parent` is `MainWindow`. |
| `self.parent_window.log_output(msg)` | Append text to the global log (status area). Used by: file_conversion, analyze_masks, tracking, post_processing, omnipose_seg, unet_seg. |
| `self.parent_window.progress_bar` | Single shared `QProgressBar`. Tabs set range/value/visible at start, update during work, hide at end. Used by: file_conversion, analyze_masks, tracking, post_processing, omnipose_seg. (unet_seg uses it only indirectly via script output.) |

**Pattern:** Before long run: `progress_bar.setVisible(True)`, `setRange(0, 100)`, `setValue(0)`. During: `setValue(pct)` and `log_output(...)`. On finish/cancel: `setVisible(False)` (sometimes after a short delay).

### 2.2 Session parameters (save/apply)

Tabs that participate in “Session” save/apply implement:

- `get_session_params(self) -> dict[str, Any]`
- `set_session_params(self, data: dict[str, Any]) -> None`

**Tabs with session params:** Analyze Masks, Tracking, Post-processing, Omnipose Segment, U-NET Segment (see `SESSION_TAB_KEYS` / `TAB_WIDGET_ATTR` in `constants.py`). TIFF Conversion does not expose session params.

### 2.3 Workers and progress

Tabs that run batch or long tasks typically:

1. Build a list of tasks (e.g. folder files).
2. Create a worker (subclass of `QThread`) with signals `progress` and `finished`.
3. Connect `worker.progress` to `parent_window.log_output` and/or a handler that updates `progress_bar`.
4. Connect `worker.finished` to a slot that advances to the next task or shows completion.

**Examples:** `AnalyzeMasksTab` (folder analysis), `TrackingTab` (batch tracking), `OmniposeSegTab` (folder segmentation), `ND2ConversionTab` (conversion queue), `PostProcessingTab` (video worker).

### 2.4 Layout pattern

Most tabs use: top-level `QVBoxLayout` → `QScrollArea` → inner layout with `QGroupBox` sections. File conversion, analyze_masks, tracking, post_processing, omnipose_seg, unet_seg follow this. No shared base class; structure is duplicated.

---

## 3. Duplication already noted (Phase 1.2 / Phase 4)

- **Analysis path rules** (“_analysis”, stem_analysis.xlsx, etc.): implemented in both `analyze_masks.py` and `tracking.py`. Phase 4 will extract to a single helper.
- **Pixel size / calibration:** Already centralized in `bactomate.modules.tiff_metadata`; no GUI duplication.

---

## 4. Phase 3 scope (v1.6)

| Item | Action |
|------|--------|
| Tab roster and parent hooks | Documented above. |
| Repeated UI patterns | Documented (log_output, progress_bar, session params, worker pattern). |
| Refactor shared “progress + log” helper | Not in scope for v1.6; would touch many tabs. Optional later. |
| Extract analysis path to one place | Phase 4. |

Phase 3 **done** when this doc exists and Phase 1 tests still pass (no code change in Phase 3 beyond any already done).
