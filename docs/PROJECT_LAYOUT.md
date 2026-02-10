# BactiMate project layout

Run all commands from the **project root** (the directory containing `main.py`, `gui/`, `scripts/`, etc.).

## Folders

| Folder | Contents |
|--------|----------|
| **gui/** | GUI code: main window, tabs (nd2, processing, analyze_masks, tracking, growth_curve_video, sam2, session_logs), workers, widgets, constants, logging, session params |
| **scripts/** | Backend Python scripts (run as `python -m scripts.<name>`): analyze_cell_areas, batch_process_tiff_sequence, batch_process_folder, convert_nd2_to_tif, check_tiff_format |
| **setup/** | Setup and maintenance: launch_gui.sh (real launcher), apply_all_patches.py, setup_fresh_env.sh, repair_pyqt6.sh, install_cuda_pytorch.sh, clear_models.sh |
| **build/** | macOS build: build_app_bundle.sh, create_dmg.sh, create_icons.sh, BactiMate.spec (run from project root, e.g. `./build/build_app_bundle.sh`) |
| **img/** | Image assets: logo2.png (used for GUI and to generate BactiMate.icns via build/create_icons.sh) |
| **tests/** | Automated tests: test_session_params.py, test_safe_imread.py |
| **docs/** | Documentation (this file, IMPROVEMENTS_SUMMARY.md) |
| **Logs/** | Session log JSON files (created at runtime) |

## Root files

- **main.py** – Application entrypoint (GUI).
- **launch_gui.sh** – Stub that runs `setup/launch_gui.sh` (preferred way to start the app).
- **requirements.txt**, **pyproject.toml** – Dependencies (kept at root for pip/CI).
Runtime patches and styles live in **gui/** (gui/patches.py, gui/styles.py); loaded by main.py and main_window.

## Running scripts from the command line

From project root:

```bash
python -m scripts.analyze_cell_areas masks.tif [options]
python -m scripts.batch_process_tiff_sequence input.tif [options]
python -m scripts.batch_process_folder /path/to/folder [options]
python -m scripts.convert_nd2_to_tif file.nd2 [options]
python -m scripts.check_tiff_format file.tif [output.tif]
```

## Setup

```bash
./setup/setup_fresh_env.sh    # from project root
python3 setup/apply_all_patches.py
./setup/clear_models.sh
./setup/repair_pyqt6.sh
```
