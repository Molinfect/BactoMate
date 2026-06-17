# Phase 5: Cleanup (v1.6)

**Purpose:** Remove dead/unused code and SAM2 (not implemented for now). Run Phase 1 tests after changes; all 6 must pass.

---

## 1. Remove SAM2 (tab + worker)

SAM2 segment tab is not in the UI (addTab commented out). We will not implement it for now; remove all SAM2-related code.

| Item | Location | Action |
|------|----------|--------|
| Tab module | `bactomate/gui/tabs/sam2.py` | Delete file. |
| Worker class | `bactomate/gui/workers.py` | Remove `SAM2SegmentWorker` (class and all its code). |
| Main window | `bactomate/gui/main_window.py` | Remove import `SAM2SegmentTab`; remove from `_get_tab_classes()`; remove `self.sam2_tab = SAM2SegmentTab(self)`; remove line `# tabs.addTab(self.sam2_tab, "SAM2 Segment")`. |
| Tabs __init__ | `bactomate/gui/tabs/__init__.py` | Remove import and export of `SAM2SegmentTab`. |
| Constants | `bactomate/gui/constants.py` | Remove `"sam2": "sam2_tab"` from `TAB_WIDGET_ATTR`; remove or shorten comment about SAM2. |
| Session/apply | Any code that uses `TAB_WIDGET_ATTR["sam2"]` or `getattr(..., "sam2_tab")` | After removing the key and the attribute, ensure no code assumes sam2_tab exists (e.g. apply_session_params already uses getattr(..., None)). |

**Docs / copy:**

- `main.py`: Remove “SAM2” from the one-line description if present.
- `docs/BACTOMATE_USER_MANUAL.md`: Remove or shorten the note about the SAM2 tab; keep or clarify Tracking’s “general_2d_w_SAM2_features” (Trackastra model name, not the SAM2 tab).

**Note:** Tracking tab’s combo option `general_2d_w_SAM2_features` is a Trackastra pretrained model that uses SAM2 features; it is independent of the SAM2 segment tab. Keep it unless you want to drop that model option.

---

## 2. Remove dead code: SessionLogsTab

| Item | Location | Action |
|------|----------|--------|
| Class | `bactomate/gui/tabs/session_logs.py` | Remove class `SessionLogsTab` (never instantiated; only `SessionLogsPanel` is used). |
| Tabs __init__ | `bactomate/gui/tabs/__init__.py` | Remove import and export of `SessionLogsTab`. |
| Main window | `main_window.py` | No change (already uses `SessionLogsPanel` only). |

---

## 3. Order of operations (suggested)

1. Remove SAM2: constants → main_window (import, _get_tab_classes, instantiation, addTab comment) → tabs/__init__ → delete `sam2.py` → remove `SAM2SegmentWorker` from `workers.py`.
2. Remove SessionLogsTab: session_logs.py (delete class) → tabs/__init__.
3. Run Phase 1 tests.
4. Optionally update user manual and main.py description.

---

## 4. Gate

After Phase 5 cleanup:

```bash
pytest tests/test_combined_summary.py tests/test_merge_fi.py tests/test_session_apply.py -v
```

All 6 tests must pass.
