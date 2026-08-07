# BactoMate — Install and launch on Windows

This guide is for **source install** from a Git clone or ZIP download.

**Most Windows users:** download **BactoMate-win64.zip** from [GitHub Releases](https://github.com/Molinfect/BactoMate/releases/latest), extract it, and run `BactoMate.bat` — no Python install required.

For a portable folder built from source (maintainers), see **Windows bundle** below.

Windows-specific scripts live in **`setup\windows\`**. See [setup/README.md](../setup/README.md). Packaging scripts are in [build/README.md](../build/README.md) (maintainers only).

## Prerequisites

- **Windows 10 or 11 (64-bit)**
- **Python 3.11 (64-bit)** from [python.org](https://www.python.org/downloads/) — **not** Conda, **not** Microsoft Store Python
  - Enable **Add python.exe to PATH** during install
  - Use the `py` launcher: `py -3.11` (installer verifies this automatically)
- **Git** (optional) or download the repo as a ZIP from GitHub
- **Microsoft Visual C++ Redistributable (x64)** — runtime DLLs for PyTorch and scientific wheels  
  [Download from Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

You do **not** need **Microsoft C++ Build Tools** for a normal install. BactoMate pins `dbscan==0.0.10` on Windows (prebuilt wheel) so Omnipose does not compile native code from source.

**Install location:** extract or clone to a **user-writable** folder, e.g. `C:\Users\<you>\BactoMate` or `D:\LabTools\BactoMate`. Do **not** install under `Program Files` — the venv, logs, and outputs need write access. **Never** run BactoMate as Administrator (breaks drag-and-drop).

Extracting the GitHub ZIP to **Downloads** is fine, but Windows may **block** scripts until unblocked — the installer now runs `Unblock-File` automatically. If double-click still does nothing, right-click the ZIP → **Properties** → **Unblock**, then extract again.

Optional for GPU:

- NVIDIA driver + CUDA-compatible GPU

## One-shot install

Open **Command Prompt** or **PowerShell**, `cd` to the project root (folder containing `main.py`), then:

```bat
install_windows.bat
```

or:

```bat
setup\windows\install.bat
```

(`setup\install_windows.bat` is a stub to the same script.)

The script will:

1. Verify **Python 3.11, 64-bit, non-Conda** (`setup\windows\verify_python.py`)
2. Create `venv\` using only that interpreter
3. Install all packages via `venv\Scripts\python.exe` (ignores system/Homebrew/Conda Python on PATH)
4. Pre-install `dbscan==0.0.10` (Windows wheel) before Omnipose
5. Install `requirements.txt` with `setup\windows\constraints.txt`
6. Run `pip install -e .`, optional CUDA PyTorch, patches, and smoke test
7. Create strict `launch_windows.bat` (no `activate`; absolute venv Python only)

## Launch

Double-click **`launch_windows.bat`** or **`BactoMate.lnk`**, or from a terminal:

```bat
launch_windows.bat
```

The launcher clears `PYTHONHOME` / `PYTHONPATH`, sets `PYTHONNOUSERSITE=1`, and runs:

```bat
venv\Scripts\python.exe -m bactomate
```

## Verify install (any time)

```bat
venv\Scripts\python.exe main.py --version
venv\Scripts\python.exe main.py --smoke-test
```

You should see `BactoMate 2.9.3` and `Smoke test OK`.

## Logs

- **Default:** `Logs\bactomate_gui_*.log` in the project folder (when writable).
- **Protected install path** (e.g. `Program Files`): logs go to **`%APPDATA%\BactoMate\Logs`** automatically.

## Windows bundle (portable folder)

After a successful source install on a **Windows** build PC:

```bat
build\windows\build_windows_bundle.bat
build\windows\create_windows_zip.bat
```

(`build\build_windows_bundle.bat` stubs forward to the same scripts.)

This creates `dist\BactoMate\` with project files, `venv\`, and a copied **Python 3.11 runtime** (`runtime\Python311\`). Copy the folder or zip to lab PCs; launch with **`BactoMate.bat`** or **`launch_windows.bat`** inside the bundle. No separate Python install required on target machines.

**Note:** `build\macos\` is for macOS `.app` / DMG only — it does not run on Windows.

Rebuild the bundle after upgrading dependencies or BactoMate version.

## Troubleshooting

### Installer window closes immediately / nothing happens

1. Run from Command Prompt so output stays visible:
   ```bat
   cd C:\path\to\BactoMate
   install_windows.bat
   ```
2. Check `Logs\install_last.log` in the project folder for the resolved path and errors.
3. Do not install under `Program Files` or a read-only network share.
4. Unblock the ZIP (Properties → Unblock) before extracting, especially from Downloads.

### Drag-and-drop does nothing / UI feels broken

You are likely running **as Administrator**. Uncheck **Run as administrator** on the shortcut (`BactoMate.lnk` → Properties → Compatibility) and relaunch normally.

### `Microsoft Visual C++ 14.0 or greater is required` / `Failed building wheel for dbscan`

You are on an old installer or `dbscan` was upgraded to 1.0.0 (source-only on Windows).

```bat
rmdir /s /q venv
setup\windows\install.bat
```

Or manually:

```bat
venv\Scripts\python.exe -m pip install "dbscan==0.0.10" --only-binary dbscan --force-reinstall
```

### `DLL load failed` / `c10.dll` / `torch` import errors

1. Install **Visual C++ Redistributable x64** (link above) and reboot if prompted
2. Confirm **64-bit Python 3.11**: `py -3.11 -c "import struct; print(struct.calcsize('P')*8)"` → `64`
3. Recreate the venv:
   ```bat
   rmdir /s /q venv
   setup\windows\install.bat
   ```

### Conda / wrong Python version

The installer **rejects** Conda and non-3.11 interpreters. Install plain Python 3.11 from python.org and run `setup\windows\install.bat` again.

### `pip install` resolution errors

The installer uses `--use-deprecated=legacy-resolver` and `setup\windows\constraints.txt`. If it still fails:

```bat
venv\Scripts\python.exe -m pip install --use-deprecated=legacy-resolver -r requirements.txt -c setup\windows\constraints.txt
venv\Scripts\python.exe -m pip install -e .
```

### Segmentation: `pretrained_model` / Cellpose errors

Re-apply patches:

```bat
venv\Scripts\python.exe setup\apply_all_patches.py
```

### Spotiflow foci: `cannot import name 'Imports' from 'wandb.proto...'`

Spotiflow loads `wandb` at import time. An older `wandb` with a newer `protobuf` (often after unrelated `pip install -U` on the lab PC) breaks Analyze Masks → Spotiflow with no peaks and a log line like:

```text
ImportError: cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'
```

Do **not** install the latest wandb/protobuf unbounded — that pulls `protobuf` 5–7 and breaks TensorFlow / U-Net (`tensorflow-intel` needs `protobuf<5` and `tensorboard<2.17`).

Repair (does not require a full reinstall):

```bat
venv\Scripts\python.exe setup\windows\ensure_spotiflow_deps.py
venv\Scripts\python.exe main.py --smoke-test
```

Or manually (TF-safe band):

```bat
venv\Scripts\python.exe -m pip install "wandb>=0.22.0,!=0.23.1,<0.28.0" "protobuf>=4.25.0,<5.0.0" "tensorboard>=2.12.0,<2.17.0"
```

Fresh installs pin these via `requirements.txt` and `setup\windows\constraints.txt`; the installer also runs `ensure_spotiflow_deps.py` before the smoke test.

### TensorFlow: `protobuf` / `tensorboard` conflict after upgrading wandb

If pip warns that `tensorflow-intel` wants `protobuf<5` / `tensorboard<2.17` but you have protobuf 5–7 or tensorboard 2.17+, run the same repair command above. U-Net Segment will fail until those are downgraded.

### CUDA / GPU not detected

CPU mode works for most workflows. To add GPU later:

```bat
venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Use the CUDA index that matches your driver (`cu121`, `cu124`, `cu128`, etc.).

### GUI does not start (no window)

Check `Logs\bactomate_gui_*.log` in the project folder, or `%APPDATA%\BactoMate\Logs` if the project folder is not writable. Ensure PyQt6 imported during smoke test.

## Manual install (if the batch script fails)

```bat
py -3.11 setup\windows\verify_python.py
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install "pip>=24.2,<26"
venv\Scripts\python.exe -m pip install "dbscan==0.0.10" --only-binary dbscan
venv\Scripts\python.exe -m pip install --use-deprecated=legacy-resolver -r requirements.txt -c setup\windows\constraints.txt
venv\Scripts\python.exe -m pip install -e .
venv\Scripts\python.exe setup\apply_all_patches.py
venv\Scripts\python.exe setup\windows\ensure_spotiflow_deps.py
venv\Scripts\python.exe main.py --smoke-test
```
