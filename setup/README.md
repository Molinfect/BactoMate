# BactoMate setup scripts

Platform-specific **install and launch** helpers for everyone using BactoMate from source.

**Prefer a ready-made app?** Download **BactoMate.dmg** (macOS) or **BactoMate-win64.zip** (Windows) from [GitHub Releases](https://github.com/poppphil/BactoMate/releases/latest) — you do not need `setup/` or `build/`.

**Shared** scripts live at the top level; **Windows** and **macOS** each have their own subfolder.

## Quick start

| Platform | Install | Launch |
|----------|---------|--------|
| **Windows** | `install_windows.bat` or `setup\windows\install.bat` | `launch_windows.bat` (created at project root by the installer) |
| **macOS** | `./setup/macos/install.sh` | `./launch_gui.sh` |

Legacy paths (`setup\install_windows.bat`, `./setup/install_mac.sh`, etc.) are thin stubs that forward to the folders below.

## Layout

```
setup/
  apply_all_patches.py     # Shared — Omnipose/Cellpose patches (both platforms)
  README.md                # This file
  windows/                 # Windows only — install, launch, CUDA helpers
  macos/                   # macOS only — install, launch, Qt repair, GPU test
  dev/                     # Optional developer diagnostics (not used by install)
```

### `setup/windows/` (Windows)

| File | Purpose |
|------|---------|
| `install.bat` | One-shot install: venv, pip, dbscan wheel, requirements, patches, smoke test |
| `launch_strict.bat` | Template copied to `launch_windows.bat` at project root |
| `launch_bundle.bat` | Launcher for portable `dist\BactoMate\` bundles |
| `constraints.txt` | Pip constraints (dbscan wheel pin) |
| `verify_python.py` | Rejects Conda / Store / non-3.11 Python |
| `create_launcher_assets.py` | Creates `BactoMate.lnk` and desktop shortcut |
| `minimize_console.ps1` | Minimizes console when GUI starts |
| `check_torch_cuda_bundle.py`, `get_cuda_wheel_index.py`, `torch_cuda_available.py` | Optional CUDA PyTorch during install |
| `patch_bundle_pyvenv.py` | Fixes venv paths when a portable bundle is moved |

**Install location:** use a user-writable folder, e.g. `C:\Users\<you>\BactoMate`. Avoid `Program Files`.

**Logs:** if the project `Logs\` folder is not writable, BactoMate writes to `%APPDATA%\BactoMate\Logs`.

### `setup/macos/` (macOS)

| File | Purpose |
|------|---------|
| `install.sh` | One-shot install (calls `setup_fresh_env.sh`) |
| `setup_fresh_env.sh` | Recreate venv, install requirements, editable install, patches |
| `launch_gui.sh` | Main GUI launcher |
| `launch_gui_bundle_strict.sh` | Launcher for `.app` bundle |
| `install_cuda_pytorch.sh` | Optional NVIDIA CUDA PyTorch |
| `repair_pyqt6.sh` | Fix missing/broken Qt cocoa plugin |
| `clear_models.sh` | Clear corrupted Cellpose model cache |
| `test_mps_gpu.py` | Apple Silicon MPS smoke test |

### `setup/dev/` (optional)

| File | Purpose |
|------|---------|
| `show_omnipose_path.py` | Print omnipose/peakdetect install paths |
| `fix_omnipose_core_syntax.py` | One-off fix for a broken omnipose/core.py in site-packages |

## Shared: `apply_all_patches.py`

Run manually if segmentation fails after upgrading dependencies:

```bat
venv\Scripts\python.exe setup\apply_all_patches.py
```

```bash
source venv/bin/activate && python setup/apply_all_patches.py
```

Full Windows guide: [docs/INSTALL_AND_LAUNCH_windows.md](../docs/INSTALL_AND_LAUNCH_windows.md)
