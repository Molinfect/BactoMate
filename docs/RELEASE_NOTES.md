# BactoMate Release Notes

Append a new section for each release. Keep entries in release-note style (what changed, for whom).

---

## v2.8.2 (2026-07-05)

### Segmentation preview: Windows locale spinbox fix

- **Re-detect:** commit in-progress Diameter/Flow/Mask spinbox text (`interpretText`) before reading parameters — fixes European Windows locales where typed `10,0` was ignored until Enter/Tab.
- **Preview header:** refresh “Pending parameters” when cell-segmentation controls change, so the title line stays in sync with the spinboxes.

---

## v2.8.1 (2026-07-05)

### Windows logs, setup/build layout, and install docs

- **Windows logs:** when the project `Logs\` folder is not writable (e.g. install under `Program Files`), logs automatically go to `%APPDATA%\BactoMate\Logs`.
- **Setup reorganized:** `setup/windows/` (install, launch, constraints) and `setup/macos/` (install, launch, Qt repair); shared `apply_all_patches.py` stays at `setup/`. Legacy stub paths (`setup\install_windows.bat`, etc.) still work.
- **Build reorganized:** `build/windows/` (portable bundle) and `build/macos/` (`.app`/DMG) with README files clarifying platform scope.
- **Docs:** Windows install guide updated with install location, logs path, and admin/drag-drop troubleshooting.

---

## v2.8.0 (2026-07-05)

### Spatial metrics, pipeline QC outputs, and combined summary workbooks

- **Spatial metrics:** new `spatial_metrics.py` module adds cell geometry (centroid, orientation, circularity), fluorescence distribution stats (interior FI, CV, top-10% mean), foci-to-contour/axis/pole distances, multi-foci spread, membrane-proximal fraction, and per-Cell_ID temporal deltas in Analyze Masks.
- **Pipeline QC artifacts:** new `pipeline_outputs.py` writes standard overlays, filter visualizations, and PNG previews from TIFF stacks (always on, not gated by optional UI checkboxes).
- **Analysis workbooks:** combined Summary (long + wide), Per-Cell, and Foci sheets across folder batches; global foci IDs across frames; improved filter visualization labels.
- **GUI:** Analyze Masks refactored to use shared workbook helpers; tracking and segmentation tabs aligned with pipeline output paths.
- **Docs:** pipeline flow and folder-structure diagrams (`docs/bactomate_full_pipeline.*`, `docs/pipeline_folder_structure.*`).
- **Tests:** `test_spatial_metrics.py`; expanded combined-summary and GUI lineage tests.

---

## v2.7.10 (2026-06-17)

### I/O performance, integration test, and tooling

- **TIFF stack I/O:** `tiff_stack_io.py` uses memmap for large stacks and smarter frame counting for ImageJ-style volumes.
- **Tracking:** `tracking_io.py` extracts stack helpers; disk visualization auto-skips when mask stacks exceed 150 frames (checkbox still on for shorter movies).
- **Tests:** slow pipeline integration test (masks → Cell IDs → analyze → lineage preflight → join); `test_tiff_stack_io.py`.
- **Tooling:** `pytest-cov` dev extra and coverage config in `pyproject.toml`.

---

## v2.7.9 (2026-06-17)

### Plan execution: U-Net Cell IDs, CI, docs, cleanup

- **U-Net persistent Cell IDs:** checkbox on U-Net inference panel; writes `{stem}_cellids.tif` after batch segmentation (parity with Omnipose/Cellpose).
- **Shared helper:** `write_persistent_cellid_mask_file()` in `cell_id_linking.py` (used by Omnipose and U-Net).
- **Tracking UX:** lineage preflight dialog reminds users about 1-based analysis vs 0-based tracking frames and filtered-mask workflow.
- **CI:** GitHub Actions runs on `dev` branch pushes and PRs (in addition to `main`).
- **Tests:** `test_workers_imports.py`; U-Net Cell ID smoke tests in `test_gui_lineage_and_cellids.py`.
- **Cleanup:** removed deprecated coloc matching flags; fixed `get_scripts_dir()` to prefer `bactomate/scripts/`; deleted `docs/archive/v1.6/`.

---

## v2.7.8 (2026-06-17)

### Session schema and workers package split

- **Session schema:** new `session_schema.py` migrates legacy tab keys (e.g. `unet_segment`) to the composed Segmentation module layout; session logs panel aligned with current tab structure.
- **Workers split:** monolithic `gui/workers.py` replaced by `gui/workers/base.py`, `tracking.py`, and `growth_curve.py`; public imports unchanged via `gui/workers/__init__.py`.
- **Tests:** smoke tests for session schema migration and persistent Cell ID mask writing from the Omnipose panel.

---

## v2.7.7 (2026-06-17)

### Lineage preflight, persistent Cell IDs, and tracking performance

- **Lineage preflight:** tracking tab warns when analyze used filtered masks but lineage would use raw masks (breaks Cell_ID mapping); see `LINEAGE_ANALYSIS_JOIN_CONTRACT.md`.
- **Persistent Cell IDs:** Omnipose/Cellpose panel checkbox writes `{stem}_masks_cellids.tif` after segmentation; shared `cell_id_linking.py` module with Hungarian centroid assignment.
- **Tracking worker:** model caching, preview frame limiting, overlay-base trajectory visualization, and faster preview mode (`preview_fast`).
- **Analyze script:** refactored to use shared cell-ID linking; lineage integration strengthened.
- **Cleanup:** removed dead `processing.py` tab and `migrate_message_boxes.py`; archived v1.6 refactor notes under `docs/archive/v1.6/`; added `bactomate/scripts/README.md`.
- **Tests:** `test_lineage_preflight.py`, `test_cell_id_linking.py`.

---

## v2.7.6 (2026-06-17)

### Analyze masks: CSV export and combined folder summary

- **Also save CSV:** enabled by default; passes `--export-csv` to the analysis script so Summary, Per-Cell, and Foci tables are written beside each `*_analysis.xlsx`.
- **Create combined summary:** new Analyze-tab option (on by default) that, after a folder batch, writes `{folder}_combined_summary.xlsx` in `_analysis/` with a wide Summary sheet plus Per-Cell sheets.
- **Session defaults:** `export_csv` and `create_combined_summary` default to on for new sessions.
- **Tests:** verify `--export-csv` is included in analyze CLI args when the checkbox is checked.

---

## v2.7.5 (2026-06-17)

### Windows launcher fix

- **`bactomate/__main__.py`:** add `if __name__ == "__main__"` guard so `python -m bactomate` (used by `launch_windows.bat` and the desktop shortcut) actually starts the GUI instead of exiting silently.

---

## v2.7.4 (2026-06-17)

### Windows portable bundle build fixes

- **`build_windows_bundle.py`:** fix `_should_skip_dir()` call that crashed the bundle build; rewrite `pyvenv.cfg` with absolute runtime paths (Windows venv launcher does not accept relative paths).
- **Portable bundle launcher:** add `setup/launch_windows_bundle.bat` and `setup/patch_bundle_pyvenv.py` so `BactoMate.bat` re-patches venv paths when the bundle folder is moved.
- **`create_windows_zip.bat`:** use 7-Zip or built-in `tar` instead of PowerShell `Compress-Archive` for much faster zip creation on large bundles.

---

## v2.7.3 (2026-06-17)

### Track Windows bundle build scripts in git

- **`.gitignore`:** whitelist `build/*.bat` and `build/build_windows_bundle.py` so Windows portable bundle scripts are versioned (they were previously ignored while docs referenced them).
- **Added to repo:** `build/build_windows_bundle.bat`, `build/build_windows_bundle.py`, `build/create_windows_zip.bat`.

---

## v2.7.2 (2026-06-17)

### Windows install hardening and CI

- **Installer:** verify Python 3.11 (64-bit, non-Conda) before venv creation; install only via `venv\Scripts\python.exe`; pre-install `dbscan==0.0.10` wheel; apply `setup\constraints_windows.txt` so Omnipose does not compile native code.
- **Launchers:** strict `launch_windows.bat` clears `PYTHONHOME`/`PYTHONPATH`, sets `PYTHONNOUSERSITE=1`, and runs the venv interpreter directly (no `activate`).
- **Subprocess workers:** Windows child processes use the project venv Python and ignore stray user-site packages.
- **CI:** Windows workflow matches the new install path (verify script, constraints, explicit venv Python for smoke test and pytest).
- **Docs:** updated Windows install guide and portable bundle build instructions.

---

## v2.7.1 (2026-06-17)

### Windows install, UI polish, and TIFF loading

- **Windows installer:** fix `install_windows.bat` batch parsing after PyTorch install; pin pip below 26; reinstall PyTorch with `--no-deps` and realign NumPy `<2`; suppress pip upgrade nag.
- **Windows launcher:** root `launch_windows.bat` minimizes the console on start; generates `BactoMate.ico` and Desktop/project shortcuts during install.
- **GUI fixes:** dark dropdown menus on Windows; compact buttons no longer clip labels (dashboard workspace row, metadata dialog, segmentation combos); metadata editor opens reliably and “Use for output TIFFs” sizes correctly.
- **TIFF I/O:** plain TIFFs load via `bioio-tifffile` to avoid OME XML parse warnings; added `bioio-tifffile` dependency.
- **Dependencies:** `numpy>=1.26,<2` for TensorFlow compatibility.

---

## v2.7.0 (2026-06)

### Lineage workspace, conversion crop, and preview refactors

- **Lineage analysis workspace:** expanded in-app lineage inspector and dedicated analysis panel with plots, metrics, and export from tracking outputs.
- **File conversion crop:** optional extract-crop ROI in the conversion preview (right-drag box, handles, clear crop); drift correction runs on the full frame before cropping.
- **Preview/module refactors:** shared preview patterns across file conversion, filtering, segmentation, tracking, and post-processing modules.
- **TIFF metadata:** improved ImageJ-style metadata read/write helpers and tests.
- **Build:** macOS launcher and app-bundle script updates.
- **Tests:** conversion crop, lineage integration/viz, TIFF metadata write, and Omnipose affinity model loading.

---

## v2.6.1 (2026-06)

### Reproducible installs, CI, and Apple MPS segmentation

- **Pin PyTorch stack:** `torch==2.11.0` and `torchvision==0.26.0` in `requirements.txt` so fresh installs match tested lab/CI environments (avoids drift to torch 2.12+).
- **Apple MPS preview fix:** patch `cellpose_omni` `_extend_centers_gpu` to use `float32` on MPS instead of `float64` (fixes preview segmentation crash on Apple Silicon GPU).
- **CI green on macOS and Windows:** test fixes for Excel file-handle cleanup on Windows, path assertions, and regressions from v2.6.0 preview work.

---

## v2.6.0 (2026-06)

### Preview UX: montage LUT, foci detection, and navigation

- **Montage LUT:** histogram reads raw channel tiles (not rendered uint8), merged tiles use derived channel LUTs without global rescale oversaturation, LUT clamped to tile data range, taller histogram, and helper copy for merged tiles.
- **Montage preview:** double-click a tile to open Edit tiles; top hint and merged-LUT note in the LUT panel.
- **Foci detection toggle:** Spotiflow peak detection runs in a background thread so enabling “Detect foci” no longer freezes the UI; skips redundant TIFF reload when stacks are already in memory.
- **Filtering preview:** “Show cell overlay” toggles via full refresh so masks hide/show reliably.
- **Preview navigation:** taller frame/channel spinboxes and row spacing so controls are not clipped at the bottom.

---

## v2.5.1 (2026-06)

### CI fix and install docs

- **Fix macOS CI segfault:** lineage PNG export no longer uses `matplotlib.pyplot` (avoids native macosx backend crash in headless pytest).
- **README:** prominent copy-paste install commands for macOS and Windows; v2.5 wording and lineage feature mention.

---

## v2.5.0 (2026-06)

### In-app lineage QC and export

- **Lineage schema v2:** division events carry review status (`candidate` / `accepted` / `rejected` / `flagged`), richer metrics (generation, asymmetry, distances), lineage growth timeseries, and expanded QC flags.
- **Quick lineage QC:** preview panel in the Tracking tab with track glossary, division-aware frame summary, lineage total area plot (with honest position-only fallback), and guided 3-step workflow copy.
- **Lineage review dialog:** Divisions tab opens first for accept/reject/flag review; tree graph and growth/FI plots; Cmd+W to close.
- **Exports:** Excel lineage report, canonical tables, and PNG figures (`lineage_tree`, `lineage_growth`, division asymmetry) on full **Run lineage export**.
- **Stability:** fix preview teardown deleting shared tracking parameter widgets (`QComboBox has been deleted`).

---

## v2.4.0 (2026-06)

### Unified dark preview UI

- **Shared settings cards:** preview and module panels use centralized `build_settings_card`, compact dark theme tokens, and consistent spacing across filtering, segmentation, montage, and tracking previews.
- **Montage preview polish:** reorganized action/preset footer, full-width LUT histogram, fixed clipped buttons, and removed figure-style preset buttons; export labels spelled out (`Background`, `Format`).
- **Spinbox/combo uniformity:** one shared card-aware spinbox style across montage cards, Edit tiles dialog, and preview nav rows (no more bright-blue vs dark arrow wells).
- **Foci table highlight:** foci table uses theme tokens; double-click row shows a prominent red/white ring on the selected focus.
- **Workspace drag-and-drop:** drop a folder or file onto the dashboard workspace area to open it as the workspace root.

---

## v2.3.7 (2026-06)

### Portable native macOS launcher

- **Bundled framework link:** native launcher and bundled `Python.framework` now use `@executable_path/...` install names, so the app does not depend on Homebrew Python paths after packaging.
- **Same single-app behavior:** preserves the v2.3.6 in-process launcher fix that keeps BactoMate as the only Dock app identity.

---

## v2.3.6 (2026-06)

### macOS native launcher (no Python detour)

- **Real app process:** `BactoMate.app` now uses a compiled native Mach-O launcher linked against the bundled `Python.framework` and runs the GUI in-process via `Py_Main`, instead of shell scripts that `exec` a separate `python3` binary.
- **Why this matters:** the old launcher made macOS treat Python as a second app, causing mixed Dock icons (bundle icon, raw logo, then Python) and broken reopen behavior.

---

## v2.3.5 (2026-06)

### macOS Dock identity cleanup

- **One icon source:** runtime Dock/window icon selection now prefers the packaged `BactoMate.icns`, matching the app bundle icon instead of switching to the raw `img/logo2.png` graphic after Qt starts.
- **Explicit bundle icon:** generated `Info.plist` now names `BactoMate.icns` directly for clearer LaunchServices registration.

---

## v2.3.4 (2026-06)

### macOS bundle dependency repair

- **Build-time repair:** `build/build_app_bundle.sh` now installs missing macOS integration dependencies (`pyobjc-framework-Cocoa` for `AppKit`/`Foundation`, plus `setproctitle`) into the build venv before copying it into `BactoMate.app`.
- **Fresh setup verification:** `setup/setup_fresh_env.sh` now verifies `AppKit`/`Foundation` during macOS setup so new environments are ready for Dock icon/name integration.

---

## v2.3.3 (2026-06)

### macOS duplicate window/icon fix

- **Single running instance:** a second Finder/Dock launch now activates the existing BactoMate window instead of opening another GUI.
- **One Dock owner:** the app bundle launcher now replaces itself with the Qt process so macOS does not keep both the launcher and the GUI as separate BactoMate entries.

---

## v2.3.2 (2026-06)

### macOS Dock icon/name fix

- **Fix Python Dock identity:** macOS app builds now include the Cocoa bridge used by BactoMate to set the Dock icon and process name at runtime.
- **Build verification:** `build/build_app_bundle.sh` now fails if `setproctitle`, `AppKit`, or `Foundation` are missing from the bundled venv, preventing future app bundles that reopen as Python or lose the BactoMate icon.

---

## v2.3.1 (2026-06)

### macOS app bundle fix

- **Fix reopen failure:** the `venv/bin/BactoMate` wrapper incorrectly routed `-c` to `main.py`, so the strict bundle launcher’s Python version check corrupted the Qt plugin path on second launch (“missing Qt plugins”).
- **Launcher hardening:** strict launcher now queries Python version via `python3` directly; wrapper passes `-c`/`-m` to Python and `--version`/`--smoke-test` to `main.py`.

---

## v2.3 (2026-06)

### macOS packaging restored

- **`build/create_icons.sh`** — generate `BactoMate.icns` from `img/logo2.png` (or a fallback icon).
- **`build/build_app_bundle.sh`** — build `BactoMate.app` from source + `venv`; copy Python.framework when available; create relocatable `venv/bin/BactoMate` launcher; ad-hoc sign; verify `--version` and `--smoke-test`.
- **`build/create_dmg.sh`** — unchanged wrapper now works again; produces `dist/BactoMate.dmg` with install readme and `Open BactoMate.command`.

### Documentation

- Version metadata and docs updated to **2.3**.
- README and docs describe local macOS `.app`/DMG build steps. Windows standalone packaging remains deferred.

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
- **Release model:** v2.2 remains **source-first for developers and researchers**. See root README for clone/install/launch.

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
