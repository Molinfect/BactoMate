# BactoMate — project plan

**Status: closed** (2026-07-05, through **v2.8.1**)

This plan tracked follow-up work after the v2.7.7–v2.7.9 refactor wave. All items that were scoped for this release cycle are **done**. What remains below is optional maintenance backlog — not blocking releases or daily use.

---

## Completed (v2.7.7 → v2.7.10)

### Release & hygiene
- Version/docs/release-notes sync for v2.7.7–v2.7.10
- Removed dead scripts (`repair_analysis_foci_workbook.py`, `workers/__main__.py`, `docs/archive/v1.6/`)
- Workspace folder renamed `BactoMate_v1.9` → `BactoMate`

### Product
- Persistent Cell IDs on **Omnipose, Cellpose-SAM, and U-Net**
- Shared `write_persistent_cellid_mask_file()` in `cell_id_linking.py`
- Lineage preflight + frame-index UX reminder in Tracking tab

### Tests & CI
- `test_workers_imports.py`, U-Net Cell ID smoke tests
- CI on `dev` branch
- `test_pipeline_integration.py` (masks → Cell IDs → analyze → preflight → lineage join)
- `test_tiff_stack_io.py`
- `pytest-cov` dev extra + coverage config in `pyproject.toml`

### Performance & structure (Priority 5 — scoped items)
- `bactomate/modules/tiff_stack_io.py` — memmap for large stacks, preview page reads
- `bactomate/gui/workers/tracking_io.py` — stack helpers, viz skip when frames > 150
- Tracking worker wired to shared I/O; `SCRIPT_LOG.propagate = False` for pytest stability

---

## Maintenance backlog (optional, when needed)

These were listed as Priority 5 “long term”. **No action required** unless you hit pain in development or user reports.

| Item | When to revisit |
|------|-----------------|
| Split `omnipose_seg.py` (~5.5k lines) | When editing segmentation UI becomes unwieldy |
| Split `post_processing.py`, `file_conversion.py` | Same |
| Split `tracking.py` tab | Same |
| Extract `tracking_viz.py` from `workers/tracking.py` | When changing trajectory/lineage rendering |
| Frame-at-a-time Trackastra loop | If 500+ frame movies OOM despite memmap |
| Full Trackastra GPU integration test | If tracking regressions become frequent |

---

## Release checklist (repeat for future v2.7.x)

1. Bump `bactomate/__init__.py` and `pyproject.toml`
2. Update README, user manual, Windows install doc
3. Append `docs/RELEASE_NOTES.md`
4. `pytest tests/` (macOS + Windows)
5. Tag on `dev`, merge to `main`, push tag

```bash
git tag vX.Y.Z
git push origin dev
git push origin vX.Y.Z
```

---

## References

- Lineage join contract: `docs/LINEAGE_ANALYSIS_JOIN_CONTRACT.md`
- Scripts index: `bactomate/scripts/README.md`
- Release history: `docs/RELEASE_NOTES.md`
