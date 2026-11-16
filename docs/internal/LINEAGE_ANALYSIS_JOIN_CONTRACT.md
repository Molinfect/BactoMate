# Lineage ↔ Analyze join contract

BactoMate lineage tracking runs **after** segmentation, filtering, and analyze. This document is the stable contract for joining Trackastra outputs with analysis workbooks.

## Recommended workflow

1. Segment → optional foci CSV  
2. Filter masks (and foci)  
3. **Analyze** on the **filtered** mask (`*_mask_filtered.tif` when available)  
4. **Lineage tracking** on the **same mask file** used for analyze  
5. Read `{stem}_lineage_report.xlsx` and `tables/lineage_*` under `_tracking/{stem}_tracking/`

## Mask path

| Analyze | Lineage should use |
|---------|-------------------|
| `filtered_masks/{base}_mask_filtered.tif` | Same filtered path (preferred) |
| Raw `*_masks.tif` only | Same raw mask |

Resolution: `resolve_analysis_workbook_for_mask()` in `analysis_paths.py` tries filtered stem first, then raw.

Warning: if analyze used filtered masks but lineage uses raw `*_masks.tif`, `Cell Label` → `Cell_ID` mapping and cell counts will not match.

## Frame indexing

| Source | Convention |
|--------|------------|
| `Per-Cell Data` / `Summary` sheets | **1-based** `Frame` |
| `tracks.csv`, `lineage_points`, Trackastra | **0-based** `frame` |
| Foci CSV (segmentation) | **0-based** `frame` (shifted +1 when imported to Per-Cell) |

Join rule: `track.frame == Per-Cell.Frame - 1` when merging analysis into lineage points.

Recorded in lineage `Run Info` as `Analysis frame base` = 1, `Tracking frame base` = 0.

## Required workbook sheets

Minimum for full integration:

- **Per-Cell Data** — `Frame`, `Cell_ID`, `Cell Label` (label strongly recommended), morphology and FI/foci columns  
- **Run Info** — provenance (optional but written by analyze)

Optional:

- **Foci** — long-form foci (foci_id is workbook-local, not a longitudinal key)  
- **Colocalization {A}×{B}** — per-cell coloc metrics  

## ID spaces

| ID | Scope | Use for joins |
|----|--------|----------------|
| `track_id` | One tracking run | Primary key in `tracks.csv`, `lineage_points` |
| `Cell_ID` | One analyze run | Join via centroid/label mapping or `track_cells` |
| `Cell Label` | Per-frame mask instance | `(frame, Cell Label)` → `Cell_ID` in Excel |
| `lineage_id` | Founder track id | Family-level summaries |
| `foci_id` | One workbook | **Do not** use across time for lineage |

## Trackastra linking modes

| Mode | Divisions |
|------|-----------|
| `greedy_nodiv` | No — auto-upgraded to `greedy` for lineage runs |
| `greedy` | Yes (default for lineage) |
| `ilp` | Yes (slower; may need solver deps) |

Division edges are **candidate links** from Trackastra, not ground truth. Review them in the lineage inspector before trusting exports.

## Review status (schema v2)

Division events and edges carry `review_status`:

| Status | Meaning |
|--------|---------|
| `candidate` | Default Trackastra link; not yet reviewed |
| `accepted` | User confirmed in lineage inspector |
| `rejected` | User rejected as incorrect division |
| `flagged` | Needs manual follow-up |

## Inspection workflow

1. **Preview lineage tracking** — in-memory founder-colored overlay + side panel  
2. **Open full lineage inspector** — tree graph, division table, accept/reject/flag, growth plots  
3. **Run lineage tracking** — writes tables, Excel report, TIFF overlay, and PNG figures  

## Output layout

Under `{sample}/_tracking/{mask_stem}_tracking/`:

- `tables/lineage_edges`, `lineage_tracks`, `lineage_points`, `lineage_events`, `lineage_growth`, `lineage_growth_timeseries`, `lineage_measurements`, `lineage_qc`, `lineage_tracks_meta`, `lineage_tree_nodes`, `lineage_tree_edges` (Parquet or CSV.gz)  
- `tables/metadata.json` — `schema_version: 2` for lineage runs  
- `{stem}_lineage_report.xlsx` — human-readable integrated report (Run Info, Division Events with `division_frame_1based`, Growth Timeseries, QC Flags, …)  
- `{stem}_lineage.tif` — founder-colored overlay with division markers (full runs)  
- `figures/{stem}_lineage_tree.png`, `{stem}_lineage_growth.png`, `{stem}_division_events.png`  
- Legacy: `{stem}_lineage_edges.csv`, `{stem}_lineage_summary.xlsx`

Batch aggregate: `@summary-lineage.xlsx` at project level (when batch lineage completes).

## Implementation modules

- Python API: `bactomate.modules.lineage_integration`  
- Tree/figure helpers: `bactomate.modules.lineage_viz`  
- GUI inspector: `bactomate.gui.lineage_inspector`
