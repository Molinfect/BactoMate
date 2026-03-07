# Phase 1.3: Minimal tests for critical paths

**Done when:** Combined summary, merge (phase+FI), and session apply have at least one test each; suite runs and passes (or is skipped when Qt/pytest unavailable).

---

## 1. Tests added

| Test file | What it tests | Critical path |
|-----------|----------------|---------------|
| **tests/test_combined_summary.py** | `AnalyzeMasksTab._build_combined_summary_excel` | Given 2 Excel paths with Summary sheets, builds wide table and writes "Combined Summary (wide)" sheet; empty paths return False; duplicate Frame rows per file are handled. |
| **tests/test_merge_fi.py** | `AnalyzeMasksTab._merge_fi_channels_into_one_excel` | Given one phase + one FI Excel (same stem), merge produces one Excel with Summary + Per-Cell Data and FI column. |
| **tests/test_session_apply.py** | `AnalyzeMasksTab.set_session_params` / `get_session_params` | set_session_params then get_session_params round-trip; unknown keys ignored. |

---

## 2. How to run

From project root, with pytest and dependencies installed (see requirements.txt):

```bash
pytest tests/test_combined_summary.py tests/test_merge_fi.py tests/test_session_apply.py -v
```

Or run all tests in `tests/`:

```bash
pytest tests/ -v
```

**GUI tests** (these three files) require PyQt6. If Qt or a display is not available, tests that need `AnalyzeMasksTab` skip with a message. On headless CI you may need to set:

```bash
export QT_QPA_PLATFORM=offscreen
pytest tests/ -v
```

Run as standalone scripts (no pytest):

```bash
python tests/test_combined_summary.py
python tests/test_merge_fi.py
python tests/test_session_apply.py
```

---

## 3. What is not tested (by design)

- **Full MainWindow.apply_session_params:** Only the tab’s set/get round-trip is tested to avoid heavy MainWindow construction. The same code path (set_session_params on each tab) is used by MainWindow.apply_session_params.
- **Combined summary with match_log / Logs column:** Only the wide sheet and columns are asserted; optional Logs in column G are not asserted.
- **Merge with 2+ FI channels:** Only 1 phase + 1 FI; the same merge logic applies for more channels.

---

## 4. Gate (Phase 1.4)

Any change in Phase 2–5 that touches these paths must keep these tests green (or skipped for the same reasons as before). No “we’ll add tests later” — tests exist before Phase 2.

Phase 1.3 **done**. Proceed to Phase 1.4 gate (confirm tests exist and run/skip as expected), then Phase 2.
