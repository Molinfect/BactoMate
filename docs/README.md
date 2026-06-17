# BactoMate

Bacterial cell segmentation and analysis tool using Omnipose with a user-friendly GUI and batch processing capabilities.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
  - [macOS: DMG install only](#macos-dmg-install-only)
- [Graphical User Interface](#graphical-user-interface)
- [Command Line Usage](#command-line-usage)
- [Batch Processing](#batch-processing)
- [Cell Area Analysis](#cell-area-analysis)
- [Building App Bundles](#building-app-bundles)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Available Scripts](#available-scripts)
- [Release Notes](#release-notes)

## Quick Start

**v2.5 is source-first (developers & researchers).** Clone the repo, install once, then launch:

**macOS:**

```bash
git clone https://github.com/poppphil/BactoMate.git
cd BactoMate
./setup/install_mac.sh
./launch_gui.sh
```

If macOS says the scripts are not executable:

```bash
chmod +x setup/install_mac.sh setup/setup_fresh_env.sh launch_gui.sh
```

**Windows:**

```bat
git clone https://github.com/poppphil/BactoMate.git
cd BactoMate
setup\install_windows.bat
launch_windows.bat
```

Verify: `python main.py --version` (expect `BactoMate 2.7.4`) and `python main.py --smoke-test`.

macOS app/DMG packaging is restored for local builds:

```bash
./build/build_app_bundle.sh
./build/create_dmg.sh
```

The root **[README](../README.md)** has copy-paste commands. Windows details: **[INSTALL_AND_LAUNCH_windows.md](INSTALL_AND_LAUNCH_windows.md)**.

### Module shell (v2.0+)

The GUI uses a **dashboard + module shell** layout (not legacy top-level tabs):

- **File conversion** — ND2/LIF/CZI → TIFF
- **Segmentation** — Omnipose, Cellpose-SAM, or U-Net in one module; filtering and analyze stages below
- **Tracking** — Trackastra on mask time series; **lineage QC** preview, division review, and export
- **Post-processing** — image montage, growth-curve video montage, scale bar
- **Training** — shared dataset section + backend-specific training panels

Each module opens in a shell with controls on the left and an optional preview surface on the right. Workspace projects (`bactomate_project.json`, schema v2) store module paths and UI state.

### Launch the GUI (after install)

```bash
./launch_gui.sh
```

This opens the BactoMate GUI. Modules: **Conversion** (ND2/LIF/CZI → TIFF), **Segmentation** (Omnipose, U-Net, Cellpose-SAM), **Analysis** (morphology, filters, FI, foci), **Tracking** (Trackastra), **Training** (unified dataset + backend-specific training), **Post-processing** (scale bar, montage, growth curve video), **Session Logs**. For a complete reference to every control and workflow, see the **[BactoMate User Manual](BACTOMATE_USER_MANUAL.md)**. For what changed in each version, see **[Release Notes](RELEASE_NOTES.md)**.

### Process a Single File (Command Line)

```bash
source venv/bin/activate
python -m bactomate.scripts.batch_process_tiff_sequence your_sequence.tif --diameter 30
```

### Process All Files in a Folder

```bash
python -m bactomate.scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni \
    --diameter 30 \
    --pixel-size 0.108 \
    --min-aspect-ratio 2.0 \
    --max-area 50 \
    --max-width 2.0
```

## Installation

### Prerequisites

- **Python 3.11** (64-bit on Windows)
- macOS 10.15+ or Windows 10/11
- Git (optional) or ZIP download from GitHub

### Recommended: one-shot install scripts

**macOS** (from project root):

```bash
chmod +x setup/install_mac.sh launch_gui.sh
./setup/install_mac.sh
```

**Windows** (Command Prompt, project root):

```bat
setup\install_windows.bat
```

Both scripts create `venv/`, install requirements, run `pip install -e .`, apply patches, and run a smoke test.

### Manual setup (alternative)

1. Create venv: `python3 -m venv venv` (Windows: `py -3.11 -m venv venv`)
2. Activate: `source venv/bin/activate` (Windows: `venv\Scripts\activate`)
3. `pip install -r requirements.txt && pip install -e .`
4. `python setup/apply_all_patches.py`
5. `python main.py --smoke-test`

**CUDA (optional):** offered during install when an NVIDIA GPU is detected, or install manually from the [PyTorch wheel index](https://download.pytorch.org/whl/).

### Installation Details

- **Python Version**: 3.11+
- **GUI Framework**: PyQt6
- **Virtual Environment**: `venv/`
- **Package name**: `bactomate` (`pip install -e .`, CLI: `bactomate`)

### macOS: DMG install

> **v2.5 note:** macOS DMGs can be built locally with `./build/build_app_bundle.sh` followed by `./build/create_dmg.sh`. Windows portable folders: `build\build_windows_bundle.bat` then `build\create_windows_zip.bat` (see [INSTALL_AND_LAUNCH_windows.md](INSTALL_AND_LAUNCH_windows.md)).

Use this section when a **`BactoMate.dmg`** binary is available (no project folder, no Terminal, no Python install).

1. Double-click **BactoMate.dmg**. A Finder window opens; inside you will see **BactoMate** (the app), a shortcut to **Applications**, **00 - READ THIS FIRST (macOS install).txt** (same steps in plain language), and **Open BactoMate.command**.
2. Drag **BactoMate** onto **Applications** (or into your Applications folder).
3. If macOS says the app is **“damaged”** or **cannot be verified**, that is a security message for apps outside the App Store, not proof the file is corrupted. Do **one** of the following **once**:
   - Double-click **Open BactoMate.command** (works from the DMG window or after the app is in Applications), **or**
   - In **Applications**, Control-click **BactoMate** → **Open** → confirm **Open**.
4. After that, start BactoMate like any other app from Applications or the Dock.

On **organization-managed Macs**, users may need IT to allow the app or **Open Anyway** under **System Settings → Privacy & Security**.

The same content as step 1’s text file ships **inside** the DMG so recipients who never open GitHub still see instructions.

### Project structure

- **`bactomate/`** — Main package: **`gui/`** (tabs, workers, styles), **`modules/`** (bioio_reader, foci_utils, tiff_metadata, validation), **`scripts/`** (CLI entry points: analyze_cell_areas, batch_process_*, convert_nd2_to_tif, check_tiff_format, etc.).
- **`main.py`** — GUI entry point (also `python -m bactomate` via pyproject.toml).
- **`setup/`** — Environment and launch: `setup_fresh_env.sh`, `launch_gui.sh`, `launch_gui_bundle_strict.sh`, patches.
- **`build/`** — App icon and bundle: `create_icons.sh`, `build_app_bundle.sh`.
- **`docs/`** — README and User Manual.

## Graphical User Interface

Launch with `./launch_gui.sh` (or `source venv/bin/activate && python main.py`). The GUI provides:

- **TIFF Conversion** — Convert ND2, LIF, CZI (and optionally TIFF) to TIFF; channel handling, metadata, optional corrections.
- **Omnipose Segment** — Segment cells (single file or folder), preview, batch process; train custom Omnipose models; diameter, flow/mask threshold, GPU, overlay.
- **Analyze Masks** — Analyze mask TIFFs with morphological and size filters; Excel output; optional fluorescence intensity (one FI file or match FI by name; feedback list); optional **Run foci analysis** (detect bright spots in FI channel, Foci sheet in Excel); combined summary, session parameters.
- **Tracking** — Run Trackastra on mask time series; optional link to original TIFF and Excel. Preserves persistent CellIDs when the input masks already have them (e.g. from Analyze Masks or from the assign_persistent_ids script).
- **Post-processing** — Add scale bar to image; build montage from folder/files; growth curve video from TIFF sequence + Excel analysis.

**Persistent CellIDs:** So that the same cell keeps the same ID from segmentation through analysis and tracking, you can run `python -m bactomate.scripts.assign_persistent_ids path/to/masks.tif -o path/to/masks_with_cellids.tif` after segmentation; then use the output in Analyze Masks and Tracking. See the [User Manual](BACTOMATE_USER_MANUAL.md) for the full pipeline.
- **Session Logs** — List recent logs; overview shows saved values (reproducible) and marks "(default)" when unchanged; apply to current session.

For every control, tooltip text, workflow, and example scenario (e.g. ND2 with phase and two FI channels), see the **[BactoMate User Manual](BACTOMATE_USER_MANUAL.md)**.

## Command Line Usage

### Single File

```bash
python -m bactomate.scripts.batch_process_tiff_sequence input.tif --diameter 30
python -m bactomate.scripts.batch_process_tiff_sequence input.tif --model bact_phase_omni --flow-threshold 0.3 --no-gpu
```

### Folder

```bash
python -m bactomate.scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni --diameter 30 --pixel-size 0.108 \
    --min-aspect-ratio 2.0 --max-area 50 --max-width 2.0
```

**Common options:** `--model`, `--diameter`, `--flow-threshold`, `--mask-threshold`, `--cluster`, `--no-gpu`, `--batch-size`, `--output`, `--no-overlay`. See script help: `python -m bactomate.scripts.batch_process_tiff_sequence --help`.

## Batch Processing

Process all TIFFs in a folder (segment + optional analysis):

```bash
python -m bactomate.scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni --diameter 30 --pixel-size 0.108 \
    --recursive --detailed --output /path/to/output
```

Options: `--recursive`, `--skip-processing` (analyze only if masks exist), `--detailed`, `--no-visualization`, `--excel-name`.

## Cell Area Analysis

Analyze mask TIFFs (morphology filters, area in µm²). Optionally analyze fluorescence intensity (FI) and run **foci analysis** to detect bright spots in the FI channel. See **[User Manual](BACTOMATE_USER_MANUAL.md)** Section 5 and **5.6b** for full details.

```bash
python -m bactomate.scripts.analyze_cell_areas masks.tif --pixel-size 0.108 \
    --min-aspect-ratio 2.0 --max-area 30 --max-width 1.5 \
    --detailed --visualization
```

### Foci analysis and Excel export

When **Run foci analysis** is enabled (GUI: Analyze Masks → Analyze FI → Run foci analysis; CLI: `--fi-image` and `--run-foci`), the analysis detects **foci** (bright spots) in the FI channel per frame and adds foci counts plus a **Foci** sheet to the Excel.

**How foci are detected:** Detection is **cell-first**: we start from the mask and each cell (no global frame scan in Manual mode). Inside each cell we find local maxima, build the **focus blob** (connected pixels ≥ 80% of peak, same cell). Focus **area (px)** is the pixel count of this blob; **Area (µm²)** = area (px) × (pixel size)². Peaks are filtered by min/max area (µm²), min separation, min peak raw, min focus mean. In **per cell** mode, a focus is kept only if **(focus blob mean) / (mean of cell excluding focus)** ≥ your factor; the focus never contributes to that baseline. Only passing foci are exported.

**Excel sheets when Run foci is on:**

| Sheet | Description |
|-------|-------------|
| **Summary** | One row per frame; includes Total Foci and Mean Foci per Cell. |
| **Per-Cell Data** | One row per cell: morphology, FI, **Number of Foci**, **foci_ids** (comma-separated, e.g. 1,2), and aggregates (Foci Mean/Total/Max Intensity, Foci Mean/Total Area, Foci Mean Intensity Ratio). |
| **Foci** | **One row per focus.** Columns: foci_id, cell_id, frame, channel_name, y, x, intensity (peak raw), Area (px), Area (µm²), Mean intensity (focus blob), Rel intensity (peak/frame median), in_cell. Use this sheet for per-focus metrics; link back to cells via **cell_id** (filter or join). Multiple foci in one cell = multiple rows with the same cell_id. |

When analyzing a **folder** with multiple FI channels, the merged Excel includes the Foci sheet from the first channel that has one. For a complete reference to parameters, filters, and workflows, see **[User Manual](BACTOMATE_USER_MANUAL.md)** Section 5.6a and 5.6b.

## Building App Bundles

BactoMate can be packaged into a macOS `.app` bundle (venv + project files, no Nuitka) and a distributable DMG, or a Windows portable folder with embedded Python 3.11.

1. **Setup:** `./setup/setup_fresh_env.sh` (or venv already with `pip install -r requirements.txt`)
2. **Icon:** Optionally provide `img/logo2.png`, then run `./build/create_icons.sh`. If no logo is present, the script generates a simple fallback icon.
3. **Build:** `./build/build_app_bundle.sh` → `BactoMate.app` (copies source + `venv`, bundles Python.framework when available, verifies `--version` and `--smoke-test`)
4. **DMG:** `./build/create_dmg.sh` → `dist/BactoMate.dmg` (app + Applications symlink + **00 - READ THIS FIRST (macOS install).txt** + **Open BactoMate.command** for first-time open on new Macs)

**Windows bundle** (after `setup\install_windows.bat` on a build PC):

1. **Build:** `build\build_windows_bundle.bat` → `dist\BactoMate\` (source + `venv` + `runtime\Python311\`)
2. **Zip:** `build\create_windows_zip.bat` → `dist\BactoMate-win64.zip`
3. Recipients launch **`BactoMate.bat`** — no separate Python install required.

**Distribution checklist** (before shipping the DMG):

- **Icon:** In Finder, BactoMate.app shows your custom icon (not a grey placeholder). If you see a generic icon, run `./build/create_icons.sh` and rebuild.
- **Launch:** Double‑click the app (or `open BactoMate.app`) — it opens once.
- **Reopen:** Quit the app, then double‑click again (or click the Dock icon if you pinned it). It must open again (see Troubleshooting if it doesn’t).
- **Dock:** One BactoMate icon; after quit, clicking the pinned Dock icon reopens the app (pin by dragging BactoMate.app from Finder to the Dock).
- **New Macs (no Apple Developer ID):** The DMG includes **Open BactoMate.command**. Tell users: after dragging BactoMate to Applications, double‑click that script once; it clears macOS quarantine so the app opens. Then they can open BactoMate.app normally.

For structure and cross-machine notes, see the **[User Manual](BACTOMATE_USER_MANUAL.md)**.

## Troubleshooting

- **User-trained models (0 cells, different behaviour):** Use negative mask threshold (e.g. -2) for user models if needed; prefer fewer epochs (e.g. 50) and lower learning rate when fine-tuning. Run **Check pipeline** in the GUI (Omnipose Segment) or `python -m bactomate.scripts.check_omnipose_pipeline compare a.pth b.pth`. **Full details:** User Manual **Appendix B** (training troubleshooting: head drift, background as cells) and **Section 4.8** (user-trained vs built-in models).
- **Checkpoint format:** BactoMate saves checkpoints with `state_dict` + `nchan`/`nclasses`. The loader is patched via `setup/apply_all_patches.py` (run by `launch_gui.sh`). If segmentation fails, run **Check pipeline** and ensure "Loader unwraps checkpoint dict" is Yes.
- **App bundle won't launch / no logs / won't reopen:** BactoMate uses **one GUI log** plus **session logs** (under a **Logs** folder). **To see whether the launcher ran:** on macOS open **`~/Library/Application Support/BactoMate/Logs/launch.log`** — the launcher appends one line per launch with the path to the full GUI log. **Go → Go to Folder** → paste `~/Library/Application Support/BactoMate/Logs`. If the app **doesn't open again** after you quit: (1) Check that file — each launch adds a line like `2026-03-02T12:00:00 [from /path] full log: …/Logs/bactomate_gui_2026-03-02_12-00-00.log`. If there is **no new line** when you try to reopen, the launcher did not run (remove the Dock icon and always start by double‑clicking BactoMate.app). (2) **Full logs:** everything (launch, crash, Qt messages, worker/script failures, Omnipose/debug, Analyze Masks script output) goes into one **bactomate_gui_*.log** per run. Only the last **20** GUI logs are kept (capped in `log_config`); session JSONs **\*_session.json** are capped at **100**. When running from the .app these are in Application Support or inside the app’s Logs; from the project they are in `BactoMate-main-V1.0/Logs/`. (3) To see errors in the terminal when reopening, run: `"/path/to/BactoMate.app/Contents/MacOS/BactoMate"` then quit and run the same command again. **launch_chain.log** in `~/Library/Logs/BactoMate/` is a separate diagnostic (one line per start: pid, executable) used by follow_build.py; you can ignore or delete it — the main log is **bactomate_gui_*.log**.
- **App icon shows grey frame or wrong icon:** The app icon is built from **only** `img/logo2.png` (no extra background is added). Any grey or black in the icon is in that file. For an icon that is 100% your graphic (e.g. only the blue bacterium): use a `logo2.png` that contains only that graphic (e.g. on transparency, no grey or black frame). Then run `./build/create_icons.sh` and `./build/build_app_bundle.sh` again. Rebuild the DMG if needed.
- **"No bundled framework" / "Bundle requires Python" in bactomate_gui_*.log:** Rebuild with `./build/build_app_bundle.sh`. The restored builder copies the Python.framework when the venv is framework-based (for example Homebrew/python.org Python). If that framework cannot be found, the bundle falls back to the build machine’s Python and is not portable to clean Macs.
- **Dock opens Python.app or wrong app after pinning:** The app uses a **native launcher** so the main process is BactoMate, not Python. After rebuilding with `./build/build_app_bundle.sh`: remove any old BactoMate/Python icon from the Dock, then **pin BactoMate.app** by dragging it from Finder onto the Dock. "Keep in Dock" from the running app should now pin BactoMate correctly. If you still see Python.app opening: rebuild the app and re-pin from Finder.
- **Two Dock icons:** With the native launcher you should see only one BactoMate icon. If you still see two, remove the extra from the Dock and re-pin BactoMate.app from Finder.
- **"App is damaged" on another Mac:** macOS shows this when the app is not signed with a Developer ID (we use ad‑hoc signing). No payment needed: the DMG includes **Open BactoMate.command**. User drags BactoMate to Applications, then double‑clicks that script once; it clears quarantine and opens the app. After that, opening BactoMate.app works. Alternatively: Right‑click BactoMate.app → Open (first time), or in Terminal: `xattr -cr /Applications/BactoMate.app`.
- **PyQt6 (cocoa plugin):** `./setup/repair_pyqt6.sh`
- **GUI won't start:** `source venv/bin/activate`, `pip show PyQt6`, `./setup/repair_pyqt6.sh`, or `./setup/setup_fresh_env.sh`
- **Segmentation fault:** If the GUI crashes immediately after opening, try recreating the environment: `./setup/setup_fresh_env.sh`. If the crash happens when using a specific file type (e.g. CZI), the issue may be in that code path—report with file type and steps.
- **Processing / OOM:** Reduce batch size or use `--no-gpu`. Check Status & Log in the GUI.
- **Models:** Stored under `~/.cellpose/models`; if corrupted, `./setup/clear_models.sh`

## Testing

```bash
pip install pytest
pytest tests/ -v
```

- `tests/test_session_params.py` — session param get/apply
- `tests/test_safe_imread.py` — TIFF reading

See repo for full test checklist (launch, model loading, segmentation, batch, error handling).

## Available Scripts

- `main.py` — GUI entrypoint
- `launch_gui.sh` — Launch GUI (venv, patches, main.py)
- `python -m bactomate.scripts.analyze_cell_areas` — Analyze masks, Excel/CSV
- `python -m bactomate.scripts.assign_persistent_ids` — Assign persistent CellIDs to a mask stack (run after segmentation; then use output in Analyze Masks and Tracking)
- `python -m bactomate.scripts.batch_process_tiff_sequence` — Single multipage TIFF, Omnipose
- `python -m bactomate.scripts.batch_process_folder` — Folder: segment + analyze
- `python -m bactomate.scripts.convert_nd2_to_tif` — ND2/LIF/CZI → TIFF
- `python -m bactomate.scripts.check_tiff_format` — Verify/reshape TIFF
- `setup/apply_all_patches.py` — Omnipose/cellpose patches
- `setup/repair_pyqt6.sh`, `setup/setup_fresh_env.sh`, `setup/install_cuda_pytorch.sh`, `setup/clear_models.sh`
- `build/build_app_bundle.sh`, `build/create_dmg.sh`, `build/create_icons.sh`

**Documentation:** Full GUI reference, workflows, training troubleshooting, and example scenarios: **[BactoMate User Manual](BACTOMATE_USER_MANUAL.md)**. What changed in each version: **[Release Notes](RELEASE_NOTES.md)**.

## Release Notes

See **[RELEASE_NOTES.md](RELEASE_NOTES.md)** for a version-by-version list of changes. Append a new section there when you release a new version.

## License

MIT. See `../LICENSE`.

## References

- [Omnipose Documentation](https://omnipose.readthedocs.io/)
- [Cellpose Documentation](https://cellpose.readthedocs.io/)
