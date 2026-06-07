# BactoMate Release Notes

Append a new section for each release. Keep entries in release-note style (what changed, for whom).

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
