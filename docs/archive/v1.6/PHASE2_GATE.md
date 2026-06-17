# Phase 2: I/O and performance – gate

**Purpose:** Reduce redundant I/O in combined-summary flow without changing behavior. Phase 1 tests must stay green.

---

## 1. Done

| Step | Status |
|------|--------|
| 2.1 Map repeated reads | ✓ `docs/v1.6/PHASE2_IO_MAP.md` |
| 2.2 Read cache (combined summary) | ✓ Each Excel opened once; Summary + Per-Cell read in one pass; Per-Cell cached and reused when adding sheets |
| 2.3 Parallel I/O | Skipped (optional; scope/disable would need design) |
| 2.4 Batch Excel write | ✓ Combined summary: build workbook in memory, single `wb.save(output_path)` (no ExcelWriter then load_workbook reopen) |
| 2.5 Gate | Run Phase 1 tests in your venv (see below) |

---

## 2. Rule

**Before considering Phase 2 complete:** Run the Phase 1 tests. All 6 must pass.

```bash
# In your project venv
pytest tests/test_combined_summary.py tests/test_merge_fi.py tests/test_session_apply.py -v
```

---

## 3. Code changes (Phase 2)

- **`bactomate/gui/tabs/analyze_masks.py` – `_build_combined_summary_excel`:**
  - First loop: open each path with `pd.ExcelFile(p)`, read Summary and Per-Cell in one pass; store Per-Cell in `per_cell_cache`.
  - Add-sheets loop: use `per_cell_cache.get(p)` instead of `pd.read_excel(p, "Per-Cell Data")`.
  - Write: build `openpyxl.Workbook()` in memory, add "Combined Summary (wide)" and all Per-Cell sheets, then `wb.save(output_path)` once (no ExcelWriter + load_workbook reopen).

Phase 2.5 **pending your run** of the 6 tests in your venv.
