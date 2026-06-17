# BactoMate — remaining work plan

Last updated: 2026-06-17 (after v2.7.10)

---

## Completed

### v2.7.8–v2.7.9
- Release hygiene, U-Net persistent Cell IDs, workers import tests, CI on `dev`, docs, slimming

### v2.7.10 (optional items + Priority 5 start)
- [x] `pytest-cov` dev extra + `[tool.coverage.*]` in `pyproject.toml`
- [x] Workspace folder renamed `BactoMate_v1.9` → `BactoMate`
- [x] `bactomate/modules/tiff_stack_io.py` — memmap for large stacks, preview page reads
- [x] `bactomate/gui/workers/tracking_io.py` — stack helpers, auto-skip viz when frames > 150
- [x] Tracking worker uses shared I/O; visualization split started (I/O module extracted)
- [x] `tests/test_pipeline_integration.py` — masks → Cell IDs → analyze → preflight → join
- [x] `tests/test_tiff_stack_io.py`

**Coverage baseline** (run locally):

```bash
pip install -e ".[dev]"
pytest tests/ --cov=bactomate --cov-report=term-missing
```

---

## Still open (Priority 5 — long term)

### 5.1 Split giant tab files

| File | Lines | Suggested split |
|------|-------|-----------------|
| `omnipose_seg.py` | ~5,500 | inference UI vs training vs batch runner |
| `post_processing.py` | ~5,400 | growth-curve video vs montage vs scale bar |
| `file_conversion.py` | ~4,100 | preview/crop vs I/O vs metadata |
| `tracking.py` (tab) | ~3,300 | preview vs batch vs lineage inspector |
| `workers/tracking.py` | ~3,200 | **next:** move `_create_*_visualization` → `tracking_viz.py` |

### 5.2 Large-stack I/O (partial)
- [x] memmap read path in `tiff_stack_io.py`
- [x] auto-skip disk visualization for stacks > 150 frames
- [ ] frame-at-a-time processing inside Trackastra loop (if memory still an issue)

### 5.3 End-to-end integration test (partial)
- [x] analyze + preflight + lineage join (no Trackastra GPU run)
- [ ] optional slow test with mocked/minimal Trackastra when feasible

---

## Release checklist

1. Bump `bactomate/__init__.py` and `pyproject.toml`
2. Update README, user manual, Windows install doc
3. Append `docs/RELEASE_NOTES.md`
4. `pytest tests/` (macOS + Windows)
5. Tag on `dev`, merge to `main`, push tag

---

## References

- Lineage join contract: `docs/LINEAGE_ANALYSIS_JOIN_CONTRACT.md`
- Scripts index: `bactomate/scripts/README.md`
- Release history: `docs/RELEASE_NOTES.md`
