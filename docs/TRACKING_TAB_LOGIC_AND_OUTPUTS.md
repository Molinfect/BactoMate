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
| **Run tracking (movement)** | Full stack | `base_dir / _tracking / {stem}_tracking` | CSV, tracked masks, optional overlays/trajectories, filtered Excel, CTC (§2b). |
| **Run lineage tracking** | Full stack | `base_dir / _tracking / {stem}_tracking` | Same as movement **plus** lineage CSV/XLSX/TIFF (§2b). |

**Linking mode and divisions**

- **greedy** and **ilp** support **cell divisions** (lineage edges in the graph).
- **greedy_nodiv** does **not**. If you choose **Run lineage tracking** while mode is `greedy_nodiv`, the UI **switches to greedy** and shows a short notice; the worker also forces greedy if lineage is requested.
- **ilp** is a global ILP linker (often stronger on hard cases, slower; needs a solver such as Gurobi or SCIP). **greedy** is the usual default.

---

## 2b. Output files (movement / preview / lineage)

For each mask, the worker writes under **`_tracking/{stem}_tracking/`** (movement and lineage) or **`_preview/{stem}_n{N}/`** (preview):

| File | Format | When | Description |
|------|--------|------|-------------|
| **`{stem}_tracks.csv`** | CSV | Always | One row per (track_id, frame) with centroid; see §3. |
| **`{stem}_masks_tracked.tif`** | TIFF (3D) | Always | Tracked label masks (same shape as input masks; labels = persistent track/Cell IDs). |
| **`{stem}_overlay_tracked.tif`** | TIFF (RGB) | If original image found | Phase image with track IDs overlaid per frame. |
| **`{stem}_trajectories.tif`** | TIFF (RGB) | If original image found (and “Create visualization” on) | Trajectory visualization: frames with paths drawn. |
| **`{stem}_tracks_with_analysis.xlsx`** | Excel | If analysis Excel found | Merged tracking + analysis; see §4. |
| **`{stem}_filtered_tracks.xlsx`** | Excel | After run (when we merge “Run” with filter export) | Filtered track table; see §5. |
| **`{stem}_lineage_edges.csv`** | CSV | **Lineage** run only | Parent–daughter edges from the Napari-style graph: `parent_track_id`, `daughter_track_id`, `division_frame` (daughter’s first frame), `parent_last_frame` (optional). |
| **`{stem}_lineage_summary.xlsx`** | Excel | **Lineage** run only | Sheets **Lineage edges** (same as CSV) and **Per-track summary** (`track_id`, `founder_track_id`, `generation`, `branch_depth`, optional `Cell_ID`). |
| **`{stem}_lineage.tif`** | TIFF (RGB) | **Lineage** run, when an original/phase stack is available | Founder-family colors, trajectories, division markers, track IDs (similar style to trajectories). |
| **`ctc_format/`** | Folder | Always (best-effort) | CTC benchmark format; see §6. |

Calibration (pixel size, time per frame) is read from the original or mask TIFF and written into the output TIFF metadata where supported.

---

## 3. CSV: `{stem}_tracks.csv`

- **Columns:** `track_id`, `frame`, `y`, `x` [, `Cell_ID` if analysis Excel was used].
- **Semantics:** One row per detection: track index, frame index, and centroid (y, x). Optional `Cell_ID` aligns with Analyze Masks Cell IDs when we had an analysis Excel and could map track IDs → Cell IDs.
- **Source:** From Trackastra’s track graph via `graph_to_napari_tracks` (or fallback extraction). Saved as-is after optional Cell_ID mapping.

---

## 4. Excel: `{stem}_tracks_with_analysis.xlsx` (when analysis Excel exists)

- **Sheet “Tracks with Analysis”:** Merged table: each row = one (track_id, frame) point plus **all columns from the analysis Excel** for the matching (frame, Cell_ID). So you get trajectory positions plus morphology/FI/etc. from Analyze Masks.
- **Sheet “Tracks”:** Raw tracks table (same content as the CSV, possibly with extra columns).
- **Sheet “Original Analysis”:** Copy of the analysis Excel sheet we used for merging.

So: **tracking + analysis in one workbook**, keyed by (frame, Cell_ID) when we have analysis data.

---

## 5. Excel: `{stem}_filtered_tracks.xlsx` (filtered export)

Produced when we apply the current **Filter tracks** settings (e.g. after Run, as part of the single “Run” flow). One row per **track** (not per frame).

**Columns:**

| Column | Meaning |
|--------|--------|
| `Cell_ID` | From analysis if available, else track_id. |
| `TrackID` | Track index. |
| `trajectory_length` | Number of points in the track. |
| `duration_frames` | last_frame − first_frame + 1. |
| `duration_min` | Duration in minutes (needs calibration). |
| `first_frame`, `last_frame` | Frame range. |
| `displacement_px`, `displacement_um` | Straight-line distance first → last (px / µm). |
| `path_length_px`, `path_length_um` | Sum of step lengths (px / µm). |
| `mean_speed_px_frame`, `mean_speed_um_min`, `mean_speed_um_s` | Mean speed. |
| `straight_line_speed_*` | Displacement / duration. |
| `straightness` | Displacement / path_length (0–1). |

Only tracks that pass the current filter (length, duration, displacement, path length, speed, straightness) are included. Calibration (pixel size, time per frame) comes from the trajectory metadata or the Filter tracks widgets.

---

## 6. `ctc_format/` folder (CTC = Cell Tracking Challenge)

- **What we do:** We call Trackastra’s `graph_to_ctc(track_graph, masks_tracked, outdir=ctc_dir)`. So **contents are defined by the trackastra library**, not by BactiMate.
- **Typical CTC layout:** Benchmark submissions use a standard layout, e.g. **RES** (segmentation/track masks) and **TRA** (lineage), with fixed naming (e.g. `man_track*.tif`, `res_*.tif`) and conventions. `graph_to_ctc` is intended to write that layout so results can be submitted to the Cell Tracking Challenge.
- **In BactiMate:** We create `output_dir / "ctc_format"`, call `graph_to_ctc`, and catch errors so that if the library fails or the format changes, the main run still succeeds. So **ctc_format is optional and for benchmarking/interop**; the main outputs are the CSV, TIFFs, and Excel files above.

---

## 7. Summary

- **Input:** Mask path (file or folder) → we resolve base dir, original (phase) TIFF, and analysis Excel the same way as elsewhere in BactiMate (mask as anchor, match by name/location).
- **Runs:** **Preview** (first **N** frames → `_preview/{stem}_n{N}/`), **movement** (full → `_tracking/{stem}_tracking/`), **lineage** (full + lineage files → `_tracking/{stem}_tracking/`). Preview does not use the `_tracking` directory.
- **Outputs:**  
  - **Always:** `_tracks.csv`, `_masks_tracked.tif`, `ctc_format/` (best-effort).  
  - **If original found:** `_overlay_tracked.tif`, `_trajectories.tif` (if “Create visualization” is on).  
  - **Lineage run:** `_lineage_edges.csv`, `_lineage_summary.xlsx`, `_lineage.tif` (when original stack is available).  
  - **If analysis Excel found:** `_tracks_with_analysis.xlsx`.  
  - **After run (with filter export):** `_filtered_tracks.xlsx` using current Filter tracks settings.
- **Formats:** CSV = track_id, frame, y, x [, Cell_ID]. Excel = raw tracks, merged analysis, filtered one-row-per-track summary, and lineage summary when applicable. CTC folder = trackastra’s CTC benchmark layout.

This matches the idea: **one anchor (mask), resolve the rest; one clear set of outputs and formats.**
