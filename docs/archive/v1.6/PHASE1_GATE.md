# Phase 1.4: Gate

**Purpose:** Confirm Phase 1 is complete and lock the rule for Phases 2–5.

---

## 1. Gate checklist

| Check | Status |
|-------|--------|
| Phase 1.1 (entry points & dependencies) done | ✓ `docs/v1.6/PHASE1_ENTRY_POINTS_AND_DEPENDENCIES.md` |
| Phase 1.2 (dead & redundant code) done | ✓ `docs/v1.6/PHASE1_DEAD_AND_REDUNDANT_CODE.md` |
| Phase 1.3 (minimal tests) done | ✓ `tests/test_combined_summary.py`, `test_merge_fi.py`, `test_session_apply.py` |
| Tests run and pass | ✓ 6 passed (pytest in venv) |
| No “we’ll add tests later” | ✓ Tests exist before Phase 2 |

---

## 2. Rule for Phase 2–5

**Any change that touches these paths must keep the Phase 1.3 tests green:**

- **Combined summary:** `AnalyzeMasksTab._build_combined_summary_excel` and the wide table / Summary read logic.
- **Merge (phase + FI):** `AnalyzeMasksTab._merge_fi_channels_into_one_excel` and related Per-Cell merge.
- **Session apply:** `AnalyzeMasksTab.set_session_params` / `get_session_params` (and by extension `MainWindow.apply_session_params`).

Before merging or finishing work in Phase 2–5, run:

```bash
pytest tests/test_combined_summary.py tests/test_merge_fi.py tests/test_session_apply.py -v
```

All 6 tests must pass (or be skipped for the same reasons as now, e.g. no Qt).

---

## 3. Status

**Phase 1.4 gate: satisfied.** Phase 1 is complete. Proceed to Phase 2 when ready.
