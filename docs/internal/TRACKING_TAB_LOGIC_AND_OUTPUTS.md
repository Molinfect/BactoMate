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
| **Preview tracking** | User choice in setup dialog: **quick** (first **20** frames, default) or **full time lapse** (entire stack) | In-memory only (nothing written to disk) | Light trajectory overlay in the right-hand preview panel. Stacks are downsampled for display speed. **Not** reused by Run tracking. Full-stack preview warns when T > 80 frames. |
| **Run tracking** | Full stack (button shows frame count from mask TIFF) | `base_dir / _tracking / {stem}_tracking` | Core TIFF/CSV + report workbook (`Run info`, `raw`, `filtered`) + canonical `tables/` + CTC (§2b). Optional visualization when **Create visualization** is checked (see §2c). |
| **Run lineage tracking** | Full stack | `base_dir / _tracking / {stem}_tracking` | Same as movement, plus lineage tables; optional `*_lineage.tif` when **Create visualization** is checked. |

### Preview vs full-run visualization (why Run tracking feels slow)

- **Preview** (quick mode) tracks at most **20** frames by default; **full time lapse** mode tracks every frame (chosen in the preview setup dialog). Results stay in RAM with a **fast** overlay (`preview_fast`: no per-cell ID text, coarser lines). Nothing is written to disk.
- **Run tracking** always processes the **entire** mask stack. If **Create visualization** is enabled and a phase TIFF is found, the worker then:
  1. Builds an in-memory **overlay with track ID labels on every frame** (PIL text — expensive).
  2. Writes **`*_trajectories.tif`** using that overlay as a base (trajectory lines on top).
- Uncheck **Create visualization** in Filter tracks to skip steps 1–2 and keep CSV, tracked masks, and the Excel report only (much faster on long movies).

**Linking mode and divisions**

- **greedy** and **ilp** support **cell divisions** (lineage edges in the graph).
- **greedy_nodiv** does **not**. If you choose **Run lineage tracking** while mode is `greedy_nodiv`, the UI **switches to greedy** and shows a short notice; the worker also forces greedy if lineage is requested.
- **ilp** is a global ILP linker (often stronger on hard cases, slower; needs a solver such as Gurobi or SCIP). **greedy** is the usual default.

---

## 2b. Output profiles and files

For each mask, the worker writes under **`_tracking/{stem}_tracking/`** (movement/lineage) or
**`_preview/{stem}_n{N}/`** (preview).

### Output profile switch

`BACTOMATE_TRACKING_OUTPUT_PROFILE` controls tabular artifacts:

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
| **`{stem}_report.xlsx`** | Excel | Workbook with sheets **Run info**, **raw** (all tracks + Pass column), **filtered** (passing tracks only), plus `track_summary` (and lineage sheets when applicable). |
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

Canonical lineage tables (lineage runs, schema v2):
- `lineage_edges`
- `lineage_summary`
- `lineage_growth`
- `lineage_growth_timeseries`
- `lineage_tracks`
- `lineage_cells`
- `lineage_points`
- `lineage_events` (with `review_status`, `division_frame_1based`, area asymmetry, distances)
- `lineage_qc`
- `lineage_measurements`
- `lineage_tracks_meta`
- `lineage_tree_nodes`, `lineage_tree_edges`

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
- `lineage_edges`: directed parent/daughter links (`link_status: candidate`).
- `lineage_summary`: lineage-level span, containing cells/tracks, division count.
- `lineage_growth`: growth fit metrics per lineage:
  - `growth`: slope of linear fit on `log(area)` vs time.
  - `doubling_time`: `ln(2) / growth` when growth > 0.
  - `fit_r2`: goodness-of-fit of that model.
  - `metric_used`: analysis area column when available, else mask pixel area.
- `lineage_growth_timeseries`: per-lineage summed `total_area` by frame (for plots/Excel).
- `lineage_events`: one row per candidate division with mother/daughter area, asymmetry, distances, `review_status`.
- `lineage_qc`: suspicious links (many daughters, track gaps, far daughters, extreme asymmetry, missing analyze join).
- `lineage_tracks_meta`: generation, parent, daughters, branch depth per track.
- `lineage_tree_nodes` / `lineage_tree_edges`: layout data for in-app tree graph and PNG export.
- `lineage_tracks`, `lineage_cells`: normalized relations.
- `lineage_points`: point-level rows with `lineage_id`, joined FI/foci from Analyze when available.

### Lineage inspection (in-app)

- **Preview lineage tracking** — quick in-memory run with founder colors and division markers.
- **Open full lineage inspector** — tree graph, division review (accept/reject/flag), growth/FI/foci plots.
- **Run lineage tracking** — full export to `_tracking/{stem}_tracking/` including `{stem}_lineage_report.xlsx` and `figures/*.png`.

---

## 5. `ctc_format/` folder (CTC = Cell Tracking Challenge)

- Export is generated by Trackastra via `graph_to_ctc(track_graph, masks_tracked, outdir=ctc_dir)`.
- BactoMate treats this as optional/best-effort; failures here do not fail the main tracking run.

---

## 6. Summary

- Input resolution is unchanged (mask as anchor; resolve original and analysis by name/location).
- Core TIFF + CSV artifacts remain.
- Default tabular output is now simplified (`{stem}_report.xlsx` + canonical `tables/`).
- Legacy artifacts are still available via `BACTOMATE_TRACKING_OUTPUT_PROFILE=legacy` (or `both`).
