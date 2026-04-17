# Tracking tab: input resolution, outputs, and formats

This doc describes how the Tracking tab resolves inputs (like the Analyze tab), what files it writes, and the exact formats (CSV, Excel, CTC).

---

## 1. Input resolution (same idea as Analyze tab)

**Anchor: mask path.** The user selects one **mask TIFF file** or a **folder** of mask TIFFs. Everything else is resolved from that path.

| Input | How it's resolved | Shared with Analyze? |
|-------|-------------------|------------------------|
| **Mask(s)** | User selects file or folder. Folder: we collect `*_masks.tif` (and other TIFFs that are not overlay/filter_visualization). | Same pattern (mask = primary input). |
| **Base dir** | `base_dir_for_mask(mask_path)`: one level up if mask lives in `Tif_*`, `_segmentation*`, or `_analysis`. | Yes — `analysis_paths.base_dir_for_mask`. |
| **Original (phase) image** | `_resolve_original_for_mask(mask_path)`: find TIFF with stem matching mask (e.g. `stem.tif` or `base.tif`), in mask dir, parent, or subfolders. **Excludes** files with `_fi` in the name. Used for overlay and trajectory visualization only (optional). | Same idea as Analyze: “match by name/location.” Analyze does this for FI (`stem_fi.tif` etc.). |
| **Analysis Excel** | `find_analysis_excel_for_mask(mask_path, base_dir)`: look in `base_dir / _analysis` for `{stem}_analysis.xlsx` or `{stem}_masks_analysis.xlsx`. If found, we use it to map Trackastra track IDs → analysis Cell_ID and to merge tracking with per-cell analysis. | Yes — same helper. |
| **Output dir** | See §2: full runs under `_tracking/{stem}_tracking/`; preview under `_preview/{stem}_n{N}/` (no `_tracking` for preview-only outputs). | N/A (tracking-specific). |

So: **mask → base dir → original (by name), Excel (by name in _analysis).** Same “one anchor, resolve the rest” logic as Analyze (e.g. FI channel: one file for all vs match by name).

---

## 2. Run actions: preview, movement, lineage

The tab exposes **three** runs (one shared Trackastra `model.track(...)` pass per mask; movement vs lineage differ in **exports**):

| Button | Frames | Output folder | Extra outputs |
|--------|--------|----------------|---------------|
| **Preview tracking** | First **N** frames only (`N` in Tracking Parameters) | `base_dir / _preview / {stem}_n{N}` | Same file types as movement for those frames. **No `_tracking` folder** for preview (keeps full runs separate). **Caveat:** divisions after frame `N−1` are invisible. When a `*_trajectories.tif` is written, the GUI opens the trajectory viewer after the run. |
| **Run tracking (movement)** | Full stack | `base_dir / _tracking / {stem}_tracking` | Core TIFF/CSV + profile-based tabular exports (default: report + canonical `tables/`) + CTC (§2b). |
| **Run lineage tracking** | Full stack | `base_dir / _tracking / {stem}_tracking` | Same as movement, plus lineage visualizations and lineage tables. |

**Linking mode and divisions**

- **greedy** and **ilp** support **cell divisions** (lineage edges in the graph).
- **greedy_nodiv** does **not**. If you choose **Run lineage tracking** while mode is `greedy_nodiv`, the UI **switches to greedy** and shows a short notice; the worker also forces greedy if lineage is requested.
- **ilp** is a global ILP linker (often stronger on hard cases, slower; needs a solver such as Gurobi or SCIP). **greedy** is the usual default.

---

## 2b. Output profiles and files

For each mask, the worker writes under **`_tracking/{stem}_tracking/`** (movement/lineage) or
**`_preview/{stem}_n{N}/`** (preview).

### Output profile switch

`BACTIMATE_TRACKING_OUTPUT_PROFILE` controls tabular artifacts:

- `minimal` (default): simplified outputs + canonical `tables/` dataset.
- `legacy`: old CSV/XLSX artifacts (for backward compatibility).
- `both`: writes both sets.

### Core outputs (all profiles)

| File | Format | When | Description |
|------|--------|------|-------------|
| **`{stem}_tracks.csv`** | CSV | Always | One row per track point (`track_id`, `frame`, `y`, `x`, optional `Cell_ID`). Kept for GUI compatibility and quick inspection. |
| **`{stem}_masks_tracked.tif`** | TIFF (3D) | Always | Tracked label masks (persistent IDs over time). |
| **`{stem}_overlay_tracked.tif`** | TIFF (RGB) | If original image found | Overlay of tracked labels on phase image. |
| **`{stem}_trajectories.tif`** | TIFF (RGB) | If original image found | Trajectory visualization. |
| **`{stem}_lineage.tif`** | TIFF (RGB) | Lineage run + original image | Founder-colored lineage visualization with division marks. |
| **`ctc_format/`** | Folder | Always (best-effort) | CTC benchmark export (`graph_to_ctc`). |

### Minimal profile tabular outputs

| File | Format | Description |
|------|--------|-------------|
| **`{stem}_report.xlsx`** | Excel | Human-readable summary workbook (`track_summary`, `track_information`, and lineage sheets when applicable). |
| **`tables/`** | Folder | Canonical machine-readable tables for downstream analysis. |
| **`tables/metadata.json`** | JSON | Schema/version, calibration, profile, and filename map. |
| **`tables/*.parquet`** | Parquet | Preferred canonical storage. |
| **`tables/*.csv.gz`** | GZip CSV | Fallback if Parquet engine is unavailable. |

Canonical trajectory tables:
- `track_points`
- `track_summary`
- `track_information`
- `track_cells`
- `track_area`

Canonical lineage tables (lineage runs):
- `lineage_edges`
- `lineage_summary`
- `lineage_growth`
- `lineage_tracks`
- `lineage_cells`
- `lineage_points`

### Legacy profile tabular outputs

| File | Format | Description |
|------|--------|-------------|
| **`{stem}_tracks_with_analysis.xlsx`** | Excel | Merged tracking + analysis workbook. |
| **`{stem}_lineage_edges.csv`** | CSV | Parent/daughter lineage edges. |
| **`{stem}_lineage_summary.xlsx`** | Excel | Legacy lineage workbook (`Lineage edges`, `Per-track summary`). |
| **`{stem}_filtered_tracks.xlsx`** | Excel | Auto-export from GUI filter settings (only in `legacy`/`both`). |

Calibration (pixel size, time per frame) is read from original/mask TIFF and propagated to written TIFF metadata.

---

## 3. Canonical trajectory semantics (`tables/`)

- `track_points`: long table with point-level trajectory rows.
- `track_summary`: one row per track with start/end, containing cell IDs, travel/displacement/speed.
- `track_information`: one row per track with metric columns aligned to prior filtered-track conventions.
- `track_cells`: normalized mapping (`track_id`, `cell_id`).
- `track_area`: per `(track_id, frame)` area and source (`analysis:*` preferred, fallback `mask_px`).

---

## 4. Canonical lineage semantics (`tables/`, lineage runs)

- `lineage_id` is founder-based (`founder_track_id` concept).
- `lineage_edges`: directed parent/daughter links.
- `lineage_summary`: lineage-level span, containing cells/tracks, division count.
- `lineage_growth`: growth fit metrics per lineage:
  - `growth`: slope of linear fit on `log(area)` vs time.
  - `doubling_time`: `ln(2) / growth` when growth > 0.
  - `fit_r2`: goodness-of-fit of that model.
  - `metric_used`: analysis area column when available, else mask pixel area.
- `lineage_tracks`, `lineage_cells`: normalized relations.
- `lineage_points`: point-level rows with `lineage_id`.

---

## 5. `ctc_format/` folder (CTC = Cell Tracking Challenge)

- Export is generated by Trackastra via `graph_to_ctc(track_graph, masks_tracked, outdir=ctc_dir)`.
- BactiMate treats this as optional/best-effort; failures here do not fail the main tracking run.

---

## 6. Summary

- Input resolution is unchanged (mask as anchor; resolve original and analysis by name/location).
- Core TIFF + CSV artifacts remain.
- Default tabular output is now simplified (`{stem}_report.xlsx` + canonical `tables/`).
- Legacy artifacts are still available via `BACTIMATE_TRACKING_OUTPUT_PROFILE=legacy` (or `both`).
