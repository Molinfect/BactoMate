# Phase 2.1: Repeated file reads – map

**Done when:** List of flows where the same path is read more than once.

---

## 1. Analyze Masks – combined summary (`_build_combined_summary_excel`)

| Flow | What is read | Count per path |
|------|----------------|----------------|
| Build wide table | For each path in `excel_paths`: `pd.read_excel(p, sheet_name="Summary")` | 1× per path |
| Add Per-Cell sheets | For each path in `excel_paths`: `pd.read_excel(p, sheet_name="Per-Cell Data")` | 1× per path |
| **Total** | Same N paths | **2× per path** (Summary then Per-Cell in separate loops) |

**Fix (Phase 2.2):** Read both Summary and Per-Cell in one pass per path (e.g. `pd.ExcelFile(p)` and read both sheets), cache Per-Cell DataFrames, reuse when adding sheets.

---

## 2. Analyze Masks – merge (`_merge_fi_channels_into_one_excel`)

| Flow | What is read | Note |
|------|----------------|------|
| Key columns | `pd.read_excel(first_excel, sheet_name="Per-Cell Data", nrows=0)` | Once per stem group |
| Phase | `pd.read_excel(phase_path, "Per-Cell Data")` and `"Summary"` | Once per phase file |
| Each FI file | `pd.read_excel(excel_path, "Per-Cell Data")` and possibly `"Summary"` | Once per FI file |

Same Excel is not read multiple times in the same merge; no change needed here for 2.2.

---

## 3. Output file – combined summary write

| Step | Action |
|------|--------|
| 1 | `pd.ExcelWriter(output_path).to_excel(wide_df)` → creates/overwrites file |
| 2 | `load_workbook(output_path)` → reopens same file |
| 3 | Add Logs, add Per-Cell sheets, `wb.save(output_path)` |

**Fix (Phase 2.4):** Build workbook in memory (create workbook, add "Combined Summary (wide)", add each Per-Cell sheet), then `wb.save(output_path)` once. Avoids create-then-reopen.

---

## 4. ND2 / TIFF in other flows

- **analyze_cell_areas.py:** Masks and FI images read once per run (subprocess). No repeated read in one run.
- **post_processing (montage):** Each path loaded when building tiles; same path can appear in multiple datasets (e.g. ND2 with multiple channels). Load is per-tile; caching would be a larger change. Deferred.
- **TrackingWorker / GrowthCurveVideoWorker:** Read TIFF/inputs once per task. No repeated read of same path in one task.

Phase 2.1 **done.** Proceed to 2.2 (read both sheets once in combined summary) and 2.4 (single workbook write).
