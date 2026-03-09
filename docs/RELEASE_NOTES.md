# BactiMate Release Notes

Append a new section for each release. Keep entries in release-note style (what changed, for whom).

---

## v1.6 (2026-03)

### Logs (single GUI log + session logs)

- **Session logs:** Unchanged; last 100 `*_session.json` files kept.
- **Single GUI log:** One **bactimate_gui_YYYY-MM-DD_HH-MM-SS.log** per run (from `launch_gui.sh`). Only the last 20 GUI logs are kept so they don’t grow indefinitely. On macOS, `~/Library/Application Support/BactiMate/Logs/launch.log` still gets one line per launch with the path to the full GUI log.
- **Single GUI log (continued):** Crash tracebacks, Qt messages (e.g. QThread destroyed while running), and Omnipose/model debug lines are written to the same file by Python. No separate debug log.

### Analysis

- **Per-Cell Data (Excel):** New column **Foci Mean Intensity Ratio** when Run foci is enabled and an FI channel is used. Ratio = (Foci Mean Intensity) / (Whole Cell Mean FI Intensity). Helps compare foci brightness to cell background; only present when both FI and foci analysis are run.

### Performance & I/O

- **Combined summary (Analyze Masks):** Each analysis Excel is read **once** per build (Summary + Per-Cell sheets in a single pass). Previously each file was opened twice.
- **Combined summary output:** Workbook is built in memory and written with a single save; no create-then-reopen of the output file.

### Code structure

- **Analysis paths:** Shared helpers in `bactimate.gui.analysis_paths` for “where analysis Excel lives” and “find analysis Excel for mask.” Used by Analyze Masks and Tracking; removes duplicate logic.
- **GUI:** Phase 3 documentation of tab layout and shared patterns (log_output, progress_bar, session params). No behavior change.

### Cleanup & dependencies

- **SAM2 segment tab and worker removed.** The tab was not in the UI; SAM2 is not supported in this release. The **Tracking** tab still offers the Trackastra model **general_2d_w_SAM2_features** (unchanged).
- **SessionLogsTab class removed.** Only the compact Session Logs panel in the status area is used.
- **requirements.txt:** `sam2` package and SAM2 optional extra removed. `huggingface_hub` kept for Trackastra model downloads.
- **pyproject.toml:** Description and optional extras updated (no `[sam2]` extra).

### Documentation

- **Release notes:** This file. Update it for each new version.
- **User manual:** Note about SAM2 tab removed; Tracking model option and Per-Cell columns described.
- **README:** Installation and optional extras updated (CUDA only; no SAM2).

### Testing

- Phase 1 gate tests (combined summary, merge FI, session apply) remain in place; all 6 pass after v1.6 changes.
