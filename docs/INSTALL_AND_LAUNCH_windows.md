# BactoMate — Install and launch on Windows

This guide is for **source install** from a Git clone or ZIP download. For a portable folder that includes Python (no separate install on target PCs), see **Windows bundle** below.

## Prerequisites

- **Windows 10 or 11 (64-bit)**
- **Python 3.11 (64-bit)** from [python.org](https://www.python.org/downloads/) — **not** Conda, **not** Microsoft Store Python
  - Enable **Add python.exe to PATH** during install
  - Use the `py` launcher: `py -3.11` (installer verifies this automatically)
- **Git** (optional) or download the repo as a ZIP from GitHub
- **Microsoft Visual C++ Redistributable (x64)** — runtime DLLs for PyTorch and scientific wheels  
  [Download from Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

You do **not** need **Microsoft C++ Build Tools** for a normal install. BactoMate pins `dbscan==0.0.10` on Windows (prebuilt wheel) so Omnipose does not compile native code from source.

Optional for GPU:

- NVIDIA driver + CUDA-compatible GPU

## One-shot install

Open **Command Prompt** or **PowerShell**, `cd` to the project root (folder containing `main.py`), then:

```bat
setup\install_windows.bat
```

The script will:

1. Verify **Python 3.11, 64-bit, non-Conda** (`setup\verify_python_windows.py`)
2. Create `venv\` using only that interpreter
3. Install all packages via `venv\Scripts\python.exe` (ignores system/Homebrew/Conda Python on PATH)
4. Pre-install `dbscan==0.0.10` (Windows wheel) before Omnipose
5. Install `requirements.txt` with `setup\constraints_windows.txt`
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

You should see `BactoMate 2.7.4` and `Smoke test OK`.

## Windows bundle (portable folder)

After a successful source install on a build PC:

```bat
build\build_windows_bundle.bat
build\create_windows_zip.bat
```

This creates `dist\BactoMate\` with project files, `venv\`, and a copied **Python 3.11 runtime** (`runtime\Python311\`) — analogous to the macOS `.app` bundle. Copy the folder or zip to lab PCs; launch with **`BactoMate.bat`** or **`launch_windows.bat`** inside the bundle. No separate Python install required on target machines.

Rebuild the bundle after upgrading dependencies or BactoMate version.

## Troubleshooting

### `Microsoft Visual C++ 14.0 or greater is required` / `Failed building wheel for dbscan`

You are on an old installer or `dbscan` was upgraded to 1.0.0 (source-only on Windows).

```bat
rmdir /s /q venv
setup\install_windows.bat
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
   setup\install_windows.bat
   ```

### Conda / wrong Python version

The installer **rejects** Conda and non-3.11 interpreters. Install plain Python 3.11 from python.org and run `setup\install_windows.bat` again.

### `pip install` resolution errors

The installer uses `--use-deprecated=legacy-resolver` and `setup\constraints_windows.txt`. If it still fails:

```bat
venv\Scripts\python.exe -m pip install --use-deprecated=legacy-resolver -r requirements.txt -c setup\constraints_windows.txt
venv\Scripts\python.exe -m pip install -e .
```

### Segmentation: `pretrained_model` / Cellpose errors

Re-apply patches:

```bat
venv\Scripts\python.exe setup\apply_all_patches.py
```

### CUDA / GPU not detected

CPU mode works for most workflows. To add GPU later:

```bat
venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Use the CUDA index that matches your driver (`cu121`, `cu124`, `cu128`, etc.).

### GUI does not start (no window)

Check `Logs\bactomate_gui_*.log` in the project folder. Ensure PyQt6 imported during smoke test.

## Manual install (if the batch script fails)

```bat
py -3.11 setup\verify_python_windows.py
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install "pip>=24.2,<26"
venv\Scripts\python.exe -m pip install "dbscan==0.0.10" --only-binary dbscan
venv\Scripts\python.exe -m pip install --use-deprecated=legacy-resolver -r requirements.txt -c setup\constraints_windows.txt
venv\Scripts\python.exe -m pip install -e .
venv\Scripts\python.exe setup\apply_all_patches.py
venv\Scripts\python.exe main.py --smoke-test
```
