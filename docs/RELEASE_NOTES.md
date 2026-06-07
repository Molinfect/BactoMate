# BactoMate Release Notes

Append a new section for each release. Keep entries in release-note style (what changed, for whom).

---

## v2.2 (2026-06)

### Stabilization and workflow

- **Workflow replay:** session replay routes through the composed Segmentation tab and respects the active backend (Omnipose / Cellpose-SAM / U-Net) for output-folder resolution.
- **Project files (schema v2):** persist module paths (conversion, segmentation backend/input, filtering/analyze, tracking, post-processing, training) with migration from schema v1.
- **QThread shutdown:** cooperative stop/cancel helpers for module switching, preview replacement, app close, and growth-curve video export.
- **Startup:** lazy tab imports to reduce cold-start cost; smoke test unchanged.

### Post-processing and performance

- Growth-curve video montage: merged/separate FI layouts, graph panel composition, overlay application, and cancel during export.
- Foci overlay drawing extracted for reuse; filtering preview performance improvements.

### CI and tests

- GitHub Actions runs full `pytest tests/` with headless Qt (`QT_QPA_PLATFORM=offscreen`) on macOS and Windows.
- New smoke/integration tests: workflow replay paths, conversion, video assembly, project model migration, boundary helpers, training dry-run, Qt thread utils.

### Documentation

- Version metadata and docs updated to **2.2**.
- **Release model:** v2.2 remains **source-first for developers and researchers**; standalone installers are deferred. See root README for clone/install/launch.

---

## v2.1 (2026-06)

### Post-processing — growth-curve video montage

- **Graph appearance editor:** double-click the growth graph in preview to edit axis labels, fonts, stroke, and grid; save/load graph presets.
- **Workflow controls:** Excel Y-column picker, graph width ratio, Y-axis quick actions (Auto, Fit data, Y min = 0), and export current preview frame as PNG.
- **FI channel layouts:** choose **Merged overlay** (one colored microscopy tile) or **Separate FI tiles** (Phase, each FI channel, and Merged side by side); phase/FI roles inferred from filenames.
- **Video overlays:** mask contours and foci markers on Phase/Merged tiles; improved small-screen preview with scrollable settings.
- **Fix:** GUI startup crash when opening Post-processing before Excel path widgets were initialized.

### UI

- Session Logs and Status log areas restyled to match module UI.
- Growth graph appearance dialog buttons use BactoMate compact styling.

### Setup (Windows)

- Improved CUDA/PyTorch bundle checks and install script reliability (`setup/check_torch_cuda_bundle.py`, `install_windows.bat`).

### Documentation

- Version metadata and docs updated to **2.1**.
- **Release model:** v2.1 remains **source-first for developers and researchers**; standalone installers are deferred. See root README for clone/install/launch.

---

## v2.0 (2026-06)

### Release model

- **Source-first release:** clone from GitHub, run `setup/install_mac.sh` (macOS) or `setup/install_windows.bat` (Windows), then launch with `launch_gui.sh` or `launch_windows.bat`.
- **No DMG/EXE in this release:** binary packaging (macOS DMG, Windows zip/exe) is deferred to a follow-up release.
- **Repository renamed** from `BactiMate` to `BactoMate`; Python package/import is `bactomate`.

### Install and launch

- macOS: `setup/setup_fresh_env.sh` and `setup/install_mac.sh` now run `pip install -e .` and a post-install smoke test.
- Windows: `setup/install_windows.bat` documents VC++ Redistributable, CUDA optional install, patch application, and smoke test; see `docs/INSTALL_AND_LAUNCH_windows.md`.
- Root `README.md` provides copy-paste clone/install/launch commands for both platforms.

### UI and modules

- **Dashboard layout:** modules (Conversion, Segmentation, Analysis, Tracking, Training, Post-processing, Session Logs) use a consistent shell instead of legacy tab names.
- **Training module:** unified dataset/augmentation section with backend-specific training (Omnipose, Cellpose-SAM, U-Net) in one place.

### CI

- GitHub Actions runs source-install smoke tests on macOS and Windows (no binary build jobs until packaging scripts are restored).

### Documentation

- Updated user manual, README, and Windows install guide for v2.0 workflows.
- Release notes and version metadata bumped to 2.0.

---

## v1.6 (2026-03)

### Logs (single GUI log + session logs)

- **Session logs:** Unchanged; last 100 `*_session.json` files kept.
- **Single GUI log:** One **bactomate_gui_YYYY-MM-DD_HH-MM-SS.log** per run (from `launch_gui.sh`). Only the last 20 GUI logs are kept so they don’t grow indefinitely. On macOS, `~/Library/Application Support/BactoMate/Logs/launch.log` still gets one line per launch with the path to the full GUI log.
- **Single GUI log (continued):** Crash tracebacks, Qt messages (e.g. QThread destroyed while running), and Omnipose/model debug lines are written to the same file by Python. No separate debug log.

### Analysis

- **Per-Cell Data (Excel):** New column **Foci Mean Intensity Ratio** when Run foci is enabled and an FI channel is used. Ratio = (Foci Mean Intensity) / (Whole Cell Mean FI Intensity). Helps compare foci brightness to cell background; only present when both FI and foci analysis are run.

### Performance & I/O

- **Combined summary (Analyze Masks):** Each analysis Excel is read **once** per build (Summary + Per-Cell sheets in a single pass). Previously each file was opened twice.
- **Combined summary output:** Workbook is built in memory and written with a single save; no create-then-reopen of the output file.

### Code structure

- **Analysis paths:** Shared helpers in `bactomate.gui.analysis_paths` for “where analysis Excel lives” and “find analysis Excel for mask.” Used by Analyze Masks and Tracking; removes duplicate logic.
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
