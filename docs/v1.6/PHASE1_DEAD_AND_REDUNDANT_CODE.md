# Phase 1.2: Dead and redundant code

**Done when:** Lists of dead code, duplicate logic, and obsolete blocks with file/line or section.

---

## 1. Dead code (never called)

### 1.1 SessionLogsTab (entire class)

| Item | Location | Reason |
|------|----------|--------|
| `SessionLogsTab` | `bactimate/gui/tabs/session_logs.py` (class at ~line 301) | Never instantiated. Main window uses only `SessionLogsPanel` (compact widget in Status & Session logs). Docstring says "kept for backward compatibility if needed." |

**Verification:** `SessionLogsPanel` is used in `main_window.py` (`self.logs_panel = SessionLogsPanel(self)`). No code does `SessionLogsTab(...)` or `addTab(..., SessionLogsTab)`.

### 1.2 No other clearly dead functions/classes found

- All `AnalyzeMasksTab` methods checked (e.g. `_build_file_group`, `_on_pixel_size_changed`, `reset_analysis_defaults`, `_combine_summaries_from_folder`) are either called from `init_ui` / signal connections or from workflow.
- `analyze_cell_areas.py`: `assign_persistent_cell_ids`, `link_persistent_cell_ids_raw`, `_draw_foci_on_frame` are used (internally or by `assign_persistent_ids.py`).
- `get_scripts_dir()` is only used by `get_script_path()` in the same file — internal use, not dead.

---

## 2. Duplicate logic

### 2.1 “Analysis output path” and “find analysis Excel” rules

The rules “analysis outputs live under `_analysis/`” and “file is `{stem}_analysis.xlsx` or `{stem}_masks_analysis.xlsx`” appear in more than one place.

| Logic | Locations | Notes |
|-------|-----------|--------|
| **Where analysis Excel lives** (`base_dir / "_analysis"`, `stem_analysis.xlsx` naming) | **analyze_masks.py:** ~2539, 2550–2565, 2625, 1333–1334, 2744. **tracking.py:** 1046–1056 (`_resolve_analysis_excel_for_mask`). | Same convention (e.g. go up from Tif_* to get base_dir, then `base_dir / "_analysis"`), but implemented separately in each tab. |
| **Stem normalization** (`stem = path.stem.replace("_masks", "")` or `.replace("_analysis", "").replace("_masks_analysis", "")`) | **analyze_masks.py:** 1180, 1264 (and merge/folder logic). **tracking.py:** 1048 (`stem = mask_path.stem.replace("_masks", "")`). | Slight variation (analyze_masks also strips `_analysis` / `_masks_analysis` for display names). |

**Recommendation (Phase 4):** Extract to one place, e.g. “analysis path rules” helper: given mask path (and optionally “go up from Tif_*”), return `analysis_dir` and `stem`; and a small “find analysis Excel for mask” that returns path or None. Then have both Analyze Masks and Tracking call that.

### 2.2 Pixel size / calibration

- **Single source:** `bactimate.modules.tiff_metadata`: `read_calibration_from_path`, `read_metadata_from_path`, `read_calibration_from_path_or_original`.
- **Used by:** file_conversion, analyze_masks, post_processing, tracking, unet_seg, omnipose_seg, workers, convert_nd2_to_tif.
- **Conclusion:** No duplicate implementation; one module, many call sites. No change needed for duplication.

### 2.3 Merge FI / phase into one Excel

- Lived only in **analyze_masks.py** (`_merge_fi_channels_into_one_excel`, `is_fi_or_foci_col`, etc.). No duplicate elsewhere.

---

## 3. Obsolete / commented / “not in use”

### 3.1 SAM2 tab not shown

| Item | Location | Notes |
|------|----------|--------|
| SAM2 tab creation vs addTab | `main_window.py`: ~209 `self.sam2_tab = SAM2SegmentTab(self)`, ~213 `# tabs.addTab(self.sam2_tab, "SAM2 Segment")` | Tab is instantiated but not added to the tab bar. constants.py: “SAM2 omitted while tab is not in use (commented out in main_window).” |

**Conclusion:** Not dead code; tab and worker exist and can be re-enabled. No removal recommended; optional cleanup: one-line comment at `addTab` explaining why it’s commented.

### 3.2 Legacy script path

| Item | Location | Notes |
|------|----------|--------|
| `get_scripts_dir()` | `paths.py`: returns `get_app_root() / "scripts"` | Points to top-level `scripts/`, which does not exist in repo (scripts are under `bactimate/scripts/`). Only used as fallback when package script path is missing. |

**Conclusion:** Documented in Phase 1.1. Keep for fallback in installed/frozen layouts; no code removal.

### 3.3 Commented-out code blocks

- Grep for `# def`, `# class`, and similar did not find large commented-out function/class blocks.
- TODO/FIXME/deprecated: only minor hits (e.g. “deprecated area_threshold” in batch scripts). No obsolete blocks listed.

---

## 4. Summary

| Category | Count | Items |
|----------|--------|--------|
| **Dead** | 1 | `SessionLogsTab` (session_logs.py) — never instantiated |
| **Duplicate logic** | 1 | Analysis output path and “find analysis Excel” rules (analyze_masks + tracking) |
| **Obsolete / not in use** | 2 | SAM2 tab addTab commented; get_scripts_dir targets missing dir (kept as fallback) |

Phase 1.2 **done**. Proceed to Phase 1.3 (minimal tests for combined summary, merge, session apply).
