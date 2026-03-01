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

## Quick Start

### Launch the GUI

```bash
./launch_gui.sh
```

This opens the BactiMate GUI. Tabs: **TIFF Conversion** (ND2/LIF/CZI → TIFF), **Omnipose Segment** (segment single file or folder, train models), **Analyze Masks** (morphology, filters, optional fluorescence intensity — one FI file or match by name), **Tracking** (Trackastra), **Post-processing** (add scale bar to image, montage, growth curve video), **Session Logs**. For a complete reference to every tab, setting, and workflow, see the **[BactiMate User Manual](BACTIMATE_USER_MANUAL.md)**.

### Process a Single File (Command Line)

```bash
source venv/bin/activate
python -m scripts.batch_process_tiff_sequence your_sequence.tif --diameter 30
```

### Process All Files in a Folder

```bash
python -m scripts.batch_process_folder /path/to/folder \
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
   - Install with SAM2 extra (e.g. newer torch): `pip install -e .[sam2]`
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
python -m scripts.batch_process_tiff_sequence input.tif --diameter 30
python -m scripts.batch_process_tiff_sequence input.tif --model bact_phase_omni --flow-threshold 0.3 --no-gpu
```

### Folder

```bash
python -m scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni --diameter 30 --pixel-size 0.108 \
    --min-aspect-ratio 2.0 --max-area 50 --max-width 2.0
```

**Common options:** `--model`, `--diameter`, `--flow-threshold`, `--mask-threshold`, `--cluster`, `--no-gpu`, `--batch-size`, `--output`, `--no-overlay`. See script help: `python -m scripts.batch_process_tiff_sequence --help`.

## Batch Processing

Process all TIFFs in a folder (segment + optional analysis):

```bash
python -m scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni --diameter 30 --pixel-size 0.108 \
    --recursive --detailed --output /path/to/output
```

Options: `--recursive`, `--skip-processing` (analyze only if masks exist), `--detailed`, `--no-visualization`, `--excel-name`.

## Cell Area Analysis

Analyze mask TIFFs (morphology filters, area in µm²):

```bash
python -m scripts.analyze_cell_areas masks.tif --pixel-size 0.108 \
    --min-aspect-ratio 2.0 --max-area 30 --max-width 1.5 \
    --detailed --visualization
```

See **[User Manual](BACTIMATE_USER_MANUAL.md)** Section 5 for all filters and options.

## Building App Bundles

BactiMate can be packaged into a macOS `.app` bundle (venv + project files, no Nuitka).

1. **Setup:** `./setup/setup_fresh_env.sh`
2. **Icons (first time):** `./build/create_icons.sh`
3. **Build:** `./build/build_app_bundle.sh` → `BactiMate.app`
4. **DMG (optional):** `./build/create_dmg.sh` → `dist/BactiMate.dmg`

The launcher uses `PYTHONHOME` for a relocatable venv. For structure, distribution, and cross-machine notes, see the full **[User Manual](BACTIMATE_USER_MANUAL.md)** or the original build section in the repo history.

## Troubleshooting

- **User-trained models (0 cells, different behaviour):** Use negative mask threshold (e.g. -2) for user models if needed; prefer fewer epochs (e.g. 50) and lower learning rate when fine-tuning. Run **Check pipeline** in the GUI (Omnipose Segment) or `python -m bactimate.scripts.check_omnipose_pipeline compare a.pth b.pth`. **Full details:** User Manual **Appendix B** (training troubleshooting: head drift, background as cells) and **Section 4.8** (user-trained vs built-in models).
- **Checkpoint format:** BactiMate saves checkpoints with `state_dict` + `nchan`/`nclasses`. The loader is patched via `setup/apply_all_patches.py` (run by `launch_gui.sh`). If segmentation fails, run **Check pipeline** and ensure "Loader unwraps checkpoint dict" is Yes.
- **App bundle won't launch:** Check `launch_gui.sh` is executable; check `Logs/launch.log` or `~/Library/Application Support/BactiMate/Logs/launch.log`; run `./launch_gui.sh` from inside the app bundle Resources to see errors.
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
- `python -m scripts.analyze_cell_areas` — Analyze masks, Excel/CSV
- `python -m bactimate.scripts.assign_persistent_ids` — Assign persistent CellIDs to a mask stack (run after segmentation; then use output in Analyze Masks and Tracking)
- `python -m scripts.batch_process_tiff_sequence` — Single multipage TIFF, Omnipose
- `python -m scripts.batch_process_folder` — Folder: segment + analyze
- `python -m scripts.convert_nd2_to_tif` — ND2/LIF/CZI → TIFF
- `python -m scripts.check_tiff_format` — Verify/reshape TIFF
- `setup/apply_all_patches.py` — Omnipose/cellpose patches
- `setup/repair_pyqt6.sh`, `setup/setup_fresh_env.sh`, `setup/install_cuda_pytorch.sh`, `setup/clear_models.sh`
- `build/build_app_bundle.sh`, `build/create_dmg.sh`, `build/create_icons.sh`

**Documentation:** Full GUI reference, workflows, training troubleshooting, and example scenarios: **[BactiMate User Manual](BACTIMATE_USER_MANUAL.md)**.

## License

[Add your license information here]

## References

- [Omnipose Documentation](https://omnipose.readthedocs.io/)
- [Cellpose Documentation](https://cellpose.readthedocs.io/)
