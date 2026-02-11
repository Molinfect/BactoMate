# BactiMate project layout

Run all commands from the **project root** (the directory containing `main.py`, `bactimate/`, `setup/`, etc.).

## Folders

| Folder | Contents |
|--------|----------|
| **bactimate/** | Main package: `bactimate.gui` (main window, tabs, workers, paths, preferences), `bactimate.scripts` (backend scripts) |
| **bactimate/gui/** | GUI code: main window, tabs (nd2, processing, analyze_masks, tracking, growth_curve_video, sam2, session_logs), workers, widgets, constants, paths, logging |
| **bactimate/scripts/** | Backend scripts (shipped with package): analyze_cell_areas, batch_process_tiff_sequence, batch_process_folder, convert_nd2_to_tif, check_tiff_format |
| **scripts/** | Backend scripts at repo root (used when running from source; same content as bactimate/scripts for dev) |
| **setup/** | Setup and maintenance: launch_gui.sh (real launcher), apply_all_patches.py, setup_fresh_env.sh, repair_pyqt6.sh, install_cuda_pytorch.sh, clear_models.sh |
| **build/** | macOS build: build_app_bundle.sh, create_dmg.sh, create_icons.sh (run from project root, e.g. `./build/build_app_bundle.sh`) |
| **img/** | Image assets: logo2.png (used for GUI and to generate BactiMate.icns via build/create_icons.sh) |
| **tests/** | Automated tests: test_session_params.py, test_safe_imread.py |
| **docs/** | Documentation (README.md, INSTALL_AND_LAUNCH.md, PROJECT_LAYOUT.md, etc.) |
| **Logs/** | Session log JSON files (created at runtime) |

## Root files

- **main.py** – Application entrypoint (calls `bactimate.__main__.run()`).
- **launch_gui.sh** – Stub that runs `setup/launch_gui.sh` (preferred way to start the app from repo).
- **requirements.txt**, **pyproject.toml** – Dependencies (pip/CI).

## Running scripts from the command line

From project root (when using repo):

```bash
python -m scripts.analyze_cell_areas masks.tif [options]
python -m scripts.batch_process_tiff_sequence input.tif [options]
python -m scripts.batch_process_folder /path/to/folder [options]
python -m scripts.convert_nd2_to_tif file.nd2 [options]
```

Or via the installed package: `python -m bactimate.scripts.<name> ...`

## Setup

```bash
./setup/setup_fresh_env.sh    # from project root
python3 setup/apply_all_patches.py
./setup/clear_models.sh
./setup/repair_pyqt6.sh
```
