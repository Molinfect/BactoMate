# Phase 4: Structure – analysis path rules (v1.6)

**Purpose:** Single source of truth for “where analysis Excel lives” and “find analysis Excel for mask”. Phase 1 tests must stay green.

---

## 1. Shared module

**`bactomate/gui/analysis_paths.py`**

| Symbol | Purpose |
|--------|--------|
| `ANALYSIS_SUBDIR` | `"_analysis"` |
| `stem_from_mask_path(mask_path)` | Mask path stem for naming (removes `_masks`) |
| `base_dir_for_mask(mask_path)` | Base dir; one level up if mask is in `Tif_*`, `_segmentation*`, or `_analysis` |
| `analysis_dir_from_base(base_dir)` | `base_dir / _analysis` |
| `find_analysis_excel_for_mask(mask_path, base_dir)` | Returns `{stem}_analysis.xlsx` or `{stem}_masks_analysis.xlsx` in `base_dir / _analysis`, or `None` |

---

## 2. Call sites updated

| Tab | Changes |
|-----|--------|
| **Tracking** | Import `base_dir_for_mask`, `find_analysis_excel_for_mask`. `_tracking_output_dir` uses `base_dir_for_mask(mask_path)`. Single-file and batch use `base_dir_for_mask(...)` and `find_analysis_excel_for_mask(...)`. Removed static `_resolve_analysis_excel_for_mask`. |
| **Analyze Masks** | Import `base_dir_for_mask`, `stem_from_mask_path`, `analysis_dir_from_base`. Folder task loop and single-file output path use these. Combined summary output path uses `base_dir_for_mask(Path(self.folder_files[0]))` and `analysis_dir_from_base(folder)`. |

Display-name stem (Excel “File” column / sheet names: strip `_analysis`, `_masks_analysis`) remains only in analyze_masks for combined summary and merge; not duplicated in tracking.

---

## 3. Gate

Run Phase 1 tests after any further changes:

```bash
pytest tests/test_combined_summary.py tests/test_merge_fi.py tests/test_session_apply.py -v
```

All 6 must pass.
