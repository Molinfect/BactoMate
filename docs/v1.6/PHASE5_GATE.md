# Phase 5: Cleanup – gate (v1.6)

**Purpose:** Remove SAM2 (tab + worker) and dead SessionLogsTab. Phase 1 tests must pass.

---

## 1. Done

| Step | Status |
|------|--------|
| Remove SAM2 tab | ✓ Deleted `bactomate/gui/tabs/sam2.py` |
| Remove SAM2 worker | ✓ Removed `SAM2SegmentWorker` from `workers.py` |
| Main window | ✓ No SAM2 import, tab, or addTab |
| Constants | ✓ Removed `"sam2"` from `TAB_WIDGET_ATTR` |
| Tabs __init__ | ✓ No `SAM2SegmentTab` |
| Remove SessionLogsTab | ✓ Removed class from `session_logs.py`, export from `__init__` |
| Copy | ✓ `main.py`, `bactomate/__init__.py`, `session_logs.py` docstring |
| requirements.txt | ✓ Removed `sam2` package and SAM2 comment; kept `huggingface_hub` for Trackastra |
| pyproject.toml | ✓ Removed "SAM2" from description and the `[sam2]` optional extra |
| docs/README.md | ✓ Removed SAM2 extra; document CUDA extra only |
| docs/BACTOMATE_USER_MANUAL.md | ✓ Removed note about SAM2 Segment tab |
| docs/v1.6/PHASE3_GUI_STRUCTURE.md | ✓ Removed sam2 from tab/session lists |
| Gate | Run Phase 1 tests in your venv (see below) |

**Kept:** Tracking tab model "general_2d_w_SAM2_features" (Trackastra) and manual text describing it.

---

## 2. Rule

```bash
pytest tests/test_combined_summary.py tests/test_merge_fi.py tests/test_session_apply.py -v
```

All 6 must pass.

---

## 3. Status

**Phase 5 complete** after your test run. v1.6 phases 1–5 are done.
