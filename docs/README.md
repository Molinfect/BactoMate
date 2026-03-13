# BactiMate

Bacterial cell segmentation and analysis tool using Omnipose with a user-friendly GUI and batch processing capabilities.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
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

### Launch the GUI

```bash
./launch_gui.sh
```

This opens the BactiMate GUI. Tabs: **TIFF Conversion** (ND2/LIF/CZI → TIFF), **Omnipose Segment** (segment single file or folder, train models), **Analyze Masks** (morphology, filters, optional fluorescence intensity — one FI file or match by name), **Tracking** (Trackastra), **Post-processing** (add scale bar to image, montage, growth curve video), **Session Logs**. For a complete reference to every tab, setting, and workflow, see the **[BactiMate User Manual](BACTIMATE_USER_MANUAL.md)**. For what changed in each version, see **[Release Notes](RELEASE_NOTES.md)**.

### Process a Single File (Command Line)

```bash
source venv/bin/activate
python -m bactimate.scripts.batch_process_tiff_sequence your_sequence.tif --diameter 30
```

### Process All Files in a Folder

```bash
python -m bactimate.scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni \
    --diameter 30 \
    --pixel-size 0.108 \
    --min-aspect-ratio 2.0 \
    --max-area 50 \
    --max-width 2.0
```

## Installation

### Prerequisites

- Python 3.11+
- macOS 10.13+ or Windows 10/11

### Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This installs **BioIO** and its plugins (bioio-ome-tiff, bioio-czi, bioio-lif, bioio-nd2) for robust reading of ND2, LIF, CZI, and TIFF. Legacy readers (aicsimageio, readlif, czifile) remain as fallbacks.
   
   **For NVIDIA GPU (CUDA) support (optional):**
   - **macOS/Linux:** Run `./setup/install_cuda_pytorch.sh` after installing requirements
   - Or manually: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
   - Replace `cu118` with your CUDA version (check with `nvidia-smi` or `nvcc --version`)

3. **Optional extras (pyproject.toml):**
   - CUDA PyTorch: `pip install -e .[cuda]` (or install torch from PyTorch index manually).
   - Key dependencies in `requirements.txt` are pinned for reproducible builds (e.g. PyQt6, numpy).

4. **For fresh setup (optional):**
   ```bash
   ./setup/setup_fresh_env.sh
   ```

### Installation Details

- **Python Version**: 3.11+
- **Omnipose Version**: 1.0.6
- **GUI Framework**: PyQt6
- **Virtual Environment**: `venv/`

### Project structure

- **`bactimate/`** — Main package: **`gui/`** (tabs, workers, styles), **`modules/`** (bioio_reader, foci_utils, tiff_metadata, validation), **`scripts/`** (CLI entry points: analyze_cell_areas, batch_process_*, convert_nd2_to_tif, check_tiff_format, etc.).
- **`main.py`** — GUI entry point (also `python -m bactimate` via pyproject.toml).
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

**Persistent CellIDs:** So that the same cell keeps the same ID from segmentation through analysis and tracking, you can run `python -m bactimate.scripts.assign_persistent_ids path/to/masks.tif -o path/to/masks_with_cellids.tif` after segmentation; then use the output in Analyze Masks and Tracking. See the [User Manual](BACTIMATE_USER_MANUAL.md) for the full pipeline.
- **Session Logs** — List recent logs; overview shows saved values (reproducible) and marks "(default)" when unchanged; apply to current session.

For every control, tooltip text, workflow, and example scenario (e.g. ND2 with phase and two FI channels), see the **[BactiMate User Manual](BACTIMATE_USER_MANUAL.md)**.

## Command Line Usage

### Single File

```bash
python -m bactimate.scripts.batch_process_tiff_sequence input.tif --diameter 30
python -m bactimate.scripts.batch_process_tiff_sequence input.tif --model bact_phase_omni --flow-threshold 0.3 --no-gpu
```

### Folder

```bash
python -m bactimate.scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni --diameter 30 --pixel-size 0.108 \
    --min-aspect-ratio 2.0 --max-area 50 --max-width 2.0
```

**Common options:** `--model`, `--diameter`, `--flow-threshold`, `--mask-threshold`, `--cluster`, `--no-gpu`, `--batch-size`, `--output`, `--no-overlay`. See script help: `python -m bactimate.scripts.batch_process_tiff_sequence --help`.

## Batch Processing

Process all TIFFs in a folder (segment + optional analysis):

```bash
python -m bactimate.scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni --diameter 30 --pixel-size 0.108 \
    --recursive --detailed --output /path/to/output
```

Options: `--recursive`, `--skip-processing` (analyze only if masks exist), `--detailed`, `--no-visualization`, `--excel-name`.

## Cell Area Analysis

Analyze mask TIFFs (morphology filters, area in µm²). Optionally analyze fluorescence intensity (FI) and run **foci analysis** to detect bright spots in the FI channel. See **[User Manual](BACTIMATE_USER_MANUAL.md)** Section 5 and **5.6b** for full details.

```bash
python -m bactimate.scripts.analyze_cell_areas masks.tif --pixel-size 0.108 \
    --min-aspect-ratio 2.0 --max-area 30 --max-width 1.5 \
    --detailed --visualization
```

### Foci analysis and Excel export

When **Run foci analysis** is enabled (GUI: Analyze Masks → Analyze FI → Run foci analysis; CLI: `--fi-image` and `--run-foci`), the analysis detects **foci** (bright spots) in the FI channel per frame and adds foci counts plus a **Foci** sheet to the Excel.

**How foci are detected:** Local intensity maxima (peaks) are found on a preprocessed FI image. For each peak, a **focus blob** is defined as the connected region of pixels with intensity ≥ **80% of the peak** and within the same segmented cell. Focus **area (px)** is the pixel count of this blob; **Area (µm²)** = area (px) × (pixel size)². Peaks are filtered by min/max area (µm²), min separation, min relative intensity (per frame or per cell), and optional min peak raw / min focus mean; only passing foci are exported.

**Excel sheets when Run foci is on:**

| Sheet | Description |
|-------|-------------|
| **Summary** | One row per frame; includes Total Foci and Mean Foci per Cell. |
| **Per-Cell Data** | One row per cell: morphology, FI, **Number of Foci**, **foci_ids** (comma-separated, e.g. 1,2), and aggregates (Foci Mean/Total/Max Intensity, Foci Mean/Total Area, Foci Mean Intensity Ratio). |
| **Foci** | **One row per focus.** Columns: foci_id, cell_id, frame, channel_name, y, x, intensity (peak raw), Area (px), Area (µm²), Mean intensity (focus blob), Rel intensity (peak/frame median), in_cell. Use this sheet for per-focus metrics; link back to cells via **cell_id** (filter or join). Multiple foci in one cell = multiple rows with the same cell_id. |

When analyzing a **folder** with multiple FI channels, the merged Excel includes the Foci sheet from the first channel that has one. For a complete reference to parameters, filters, and workflows, see **[User Manual](BACTIMATE_USER_MANUAL.md)** Section 5.6a and 5.6b.

## Building App Bundles

BactiMate can be packaged into a macOS `.app` bundle (venv + project files, no Nuitka) and a distributable DMG.

1. **Setup:** `./setup/setup_fresh_env.sh` (or venv already with `pip install -r requirements.txt`)
2. **Icon (required for distributable app):** Ensure `img/logo2.png` is your desired app icon, then run `./build/create_icons.sh`. The build will fail with a clear error if no valid icon ends up in the bundle.
3. **Build:** `./build/build_app_bundle.sh` → `BactiMate.app` (build verifies launcher, script, Python, and icon)
4. **DMG:** `./build/create_dmg.sh` → `dist/BactiMate.dmg` (app + Applications symlink + **Open BactiMate.command** for first-time open on new Macs)

**Distribution checklist** (before shipping the DMG):

- **Icon:** In Finder, BactiMate.app shows your custom icon (not a grey placeholder). If you see a generic icon, run `./build/create_icons.sh` and rebuild.
- **Launch:** Double‑click the app (or `open BactiMate.app`) — it opens once.
- **Reopen:** Quit the app, then double‑click again (or click the Dock icon if you pinned it). It must open again (see Troubleshooting if it doesn’t).
- **Dock:** One BactiMate icon; after quit, clicking the pinned Dock icon reopens the app (pin by dragging BactiMate.app from Finder to the Dock).
- **New Macs (no Apple Developer ID):** The DMG includes **Open BactiMate.command**. Tell users: after dragging BactiMate to Applications, double‑click that script once; it clears macOS quarantine so the app opens. Then they can open BactiMate.app normally.

For structure and cross-machine notes, see the **[User Manual](BACTIMATE_USER_MANUAL.md)**.

## Troubleshooting

- **User-trained models (0 cells, different behaviour):** Use negative mask threshold (e.g. -2) for user models if needed; prefer fewer epochs (e.g. 50) and lower learning rate when fine-tuning. Run **Check pipeline** in the GUI (Omnipose Segment) or `python -m bactimate.scripts.check_omnipose_pipeline compare a.pth b.pth`. **Full details:** User Manual **Appendix B** (training troubleshooting: head drift, background as cells) and **Section 4.8** (user-trained vs built-in models).
- **Checkpoint format:** BactiMate saves checkpoints with `state_dict` + `nchan`/`nclasses`. The loader is patched via `setup/apply_all_patches.py` (run by `launch_gui.sh`). If segmentation fails, run **Check pipeline** and ensure "Loader unwraps checkpoint dict" is Yes.
- **App bundle won't launch / no logs / won't reopen:** BactiMate uses **one GUI log** plus **session logs** (under a **Logs** folder). **To see whether the launcher ran:** on macOS open **`~/Library/Application Support/BactiMate/Logs/launch.log`** — the launcher appends one line per launch with the path to the full GUI log. **Go → Go to Folder** → paste `~/Library/Application Support/BactiMate/Logs`. If the app **doesn't open again** after you quit: (1) Check that file — each launch adds a line like `2026-03-02T12:00:00 [from /path] full log: …/Logs/bactimate_gui_2026-03-02_12-00-00.log`. If there is **no new line** when you try to reopen, the launcher did not run (remove the Dock icon and always start by double‑clicking BactiMate.app). (2) **Full logs:** everything (launch, crash, Qt messages, worker/script failures, Omnipose/debug) goes into one **bactimate_gui_*.log** per run (last 20 kept); session JSONs in **\*_session.json** (last 100). When running from the .app these are in Application Support or inside the app’s Logs; from the project they are in `BactiMate-main-V1.0/Logs/`. (3) To see errors in the terminal when reopening, run: `"/path/to/BactiMate.app/Contents/MacOS/BactiMate"` then quit and run the same command again. **launch_chain.log** in `~/Library/Logs/BactiMate/` is a separate diagnostic (one line per start: pid, executable) used by follow_build.py; you can ignore or delete it — the main log is **bactimate_gui_*.log**.
- **App icon shows grey frame or wrong icon:** The app icon is built from **only** `img/logo2.png` (no extra background is added). Any grey or black in the icon is in that file. For an icon that is 100% your graphic (e.g. only the blue bacterium): use a `logo2.png` that contains only that graphic (e.g. on transparency, no grey or black frame). Then run `./build/create_icons.sh` and `./build/build_app_bundle.sh` again. Rebuild the DMG if needed.
- **"No bundled framework" / "Bundle requires Python" in bactimate_gui_*.log:** The .app does not ship its own Python; it uses the venv, which points at your Mac’s Python (e.g. Homebrew). On the build machine that’s normal and the app should run. Only relevant when moving the .app to another Mac without Python. Not the cause of "won’t reopen" on the same machine.
- **Dock opens Python.app or wrong app after pinning:** The app uses a **native launcher** so the main process is BactiMate, not Python. After rebuilding with `./build/build_app_bundle.sh`: remove any old BactiMate/Python icon from the Dock, then **pin BactiMate.app** by dragging it from Finder onto the Dock. "Keep in Dock" from the running app should now pin BactiMate correctly. If you still see Python.app opening: rebuild the app and re-pin from Finder.
- **Two Dock icons:** With the native launcher you should see only one BactiMate icon. If you still see two, remove the extra from the Dock and re-pin BactiMate.app from Finder.
- **"App is damaged" on another Mac:** macOS shows this when the app is not signed with a Developer ID (we use ad‑hoc signing). No payment needed: the DMG includes **Open BactiMate.command**. User drags BactiMate to Applications, then double‑clicks that script once; it clears quarantine and opens the app. After that, opening BactiMate.app works. Alternatively: Right‑click BactiMate.app → Open (first time), or in Terminal: `xattr -cr /Applications/BactiMate.app`.
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
- `python -m bactimate.scripts.analyze_cell_areas` — Analyze masks, Excel/CSV
- `python -m bactimate.scripts.assign_persistent_ids` — Assign persistent CellIDs to a mask stack (run after segmentation; then use output in Analyze Masks and Tracking)
- `python -m bactimate.scripts.batch_process_tiff_sequence` — Single multipage TIFF, Omnipose
- `python -m bactimate.scripts.batch_process_folder` — Folder: segment + analyze
- `python -m bactimate.scripts.convert_nd2_to_tif` — ND2/LIF/CZI → TIFF
- `python -m bactimate.scripts.check_tiff_format` — Verify/reshape TIFF
- `setup/apply_all_patches.py` — Omnipose/cellpose patches
- `setup/repair_pyqt6.sh`, `setup/setup_fresh_env.sh`, `setup/install_cuda_pytorch.sh`, `setup/clear_models.sh`
- `build/build_app_bundle.sh`, `build/create_dmg.sh`, `build/create_icons.sh`

**Documentation:** Full GUI reference, workflows, training troubleshooting, and example scenarios: **[BactiMate User Manual](BACTIMATE_USER_MANUAL.md)**. What changed in each version: **[Release Notes](RELEASE_NOTES.md)**.

## Release Notes

See **[RELEASE_NOTES.md](RELEASE_NOTES.md)** for a version-by-version list of changes. Append a new section there when you release a new version.

## License

[Add your license information here]

## References

- [Omnipose Documentation](https://omnipose.readthedocs.io/)
- [Cellpose Documentation](https://cellpose.readthedocs.io/)
