# BactoMate — remaining work plan

Last updated: 2026-06-17 (after v2.7.9)

---

## Completed

### Release hygiene (v2.7.8 → v2.7.9)
- [x] Version strings synced through docs and package metadata
- [x] `RELEASE_NOTES.md` for v2.7.7, v2.7.8, and v2.7.9
- [x] Removed `scripts/repair_analysis_foci_workbook.py` and `workers/__main__.py`

### Priority 1 — U-Net persistent Cell IDs
- [x] Shared `write_persistent_cellid_mask_file()` in `cell_id_linking.py`
- [x] Omnipose refactored to use shared helper
- [x] U-Net checkbox + post-batch `_maybe_write_persistent_cellid_masks_for_output()`
- [x] Session param `assign_persistent_cellids` on U-Net tab
- [x] Tests in `test_gui_lineage_and_cellids.py`

### Priority 2 — test & CI
- [x] `tests/test_workers_imports.py`
- [x] CI on `dev` branch (`.github/workflows/build.yml`)
- [ ] Optional: `pytest-cov` baseline (deferred)

### Priority 3 — documentation
- [x] `docs/README.md` — Segmentation module (not standalone Omnipose tab)
- [x] `BACTOMATE_USER_MANUAL.md` — persistent Cell IDs on all three backends
- [x] Lineage preflight frame-index / filtered-mask reminder in Tracking tab

### Priority 4 — code slimming
- [x] Deleted `docs/archive/v1.6/`
- [x] `get_scripts_dir()` now prefers `bactomate/scripts/`
- [x] Removed deprecated `coloc_one_to_one` / `coloc_mutual_nearest` from `ColocSettings`
- [ ] Rename workspace folder `BactoMate_v1.9` (cosmetic, manual)

---

## Still open (Priority 5 — long term)

### 5.1 Split giant tab files

| File | Lines | Suggested split |
|------|-------|-----------------|
| `omnipose_seg.py` | ~5,500 | inference UI vs training vs batch runner |
| `post_processing.py` | ~5,400 | growth-curve video vs montage vs scale bar |
| `file_conversion.py` | ~4,100 | preview/crop vs I/O vs metadata |
| `tracking.py` (tab) | ~3,300 | preview vs batch vs lineage inspector |
| `workers/tracking.py` | ~3,200 | run loop vs visualization vs Excel join |

### 5.2 Large-stack I/O

- `tifffile.memmap` or page-at-a-time reads in `workers/tracking.py`
- Default batch tracking visualization off when frame count > N

### 5.3 End-to-end integration test

One `@pytest.mark.slow` test: tiny synthetic TIFF through analyze → track → lineage export.

### 5.4 Coverage baseline

```bash
pip install pytest-cov
pytest tests/ --cov=bactomate --cov-report=term-missing
```

Target blind spots: `file_conversion.py`, `omnipose_seg.py`, `workers/tracking.py`.

---

## Release checklist

1. Bump `bactomate/__init__.py` and `pyproject.toml`
2. Update README, user manual, Windows install doc
3. Append `docs/RELEASE_NOTES.md`
4. `pytest tests/` (macOS + Windows)
5. Tag `vX.Y.Z` on `dev`, merge to `main`, push tag

---

## References

- Lineage join contract: `docs/LINEAGE_ANALYSIS_JOIN_CONTRACT.md`
- Scripts index: `bactomate/scripts/README.md`
- Release history: `docs/RELEASE_NOTES.md`
