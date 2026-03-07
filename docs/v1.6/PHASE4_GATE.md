# Phase 4: Structure – gate (v1.6)

**Purpose:** Extract analysis path rules to one module; both tabs use it. Phase 1 tests must stay green.

---

## 1. Done

| Step | Status |
|------|--------|
| 4.1 Shared module | ✓ `bactimate/gui/analysis_paths.py` |
| 4.2 Tracking uses module | ✓ base_dir_for_mask, find_analysis_excel_for_mask; removed _resolve_analysis_excel_for_mask |
| 4.3 Analyze Masks uses module | ✓ base_dir_for_mask, stem_from_mask_path, analysis_dir_from_base in folder loop, single-file path, combined summary path |
| 4.4 Gate | Run Phase 1 tests in your venv (see below) |

---

## 2. Rule

```bash
pytest tests/test_combined_summary.py tests/test_merge_fi.py tests/test_session_apply.py -v
```

All 6 must pass.

---

## 3. Status

**Phase 4 complete.** Next: Phase 5 (cleanup – remove SAM2 and SessionLogsTab per `PHASE5_CLEANUP.md`).
