# BactoMate — Install and launch on Windows

This guide is for **source install** from a Git clone or ZIP download. v2.0 does not ship a standalone `.exe` installer yet.

## Prerequisites

- **Windows 10 or 11 (64-bit)**
- **Python 3.11 (64-bit)** from [python.org](https://www.python.org/downloads/)
  - Enable **Add python.exe to PATH** during install
  - The `py` launcher (`py -3.11`) is preferred but plain `python` works
- **Git** (optional) or download the repo as a ZIP from GitHub
- **Microsoft Visual C++ Redistributable (x64)** — required for PyTorch and many scientific wheels  
  [Download from Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

Optional for GPU:

- NVIDIA driver + CUDA-compatible GPU
- Do **not** use Conda base Python for BactoMate; use a dedicated venv in the project folder

## One-shot install

Open **Command Prompt** or **PowerShell**, `cd` to the project root (folder containing `main.py`), then:

```bat
setup\install_windows.bat
```

The script will:

1. Create `venv\`
2. Install `requirements.txt`
3. Run `pip install -e .`
4. Optionally install CUDA PyTorch if an NVIDIA GPU is detected
5. Apply compatibility patches (`setup\apply_all_patches.py`)
6. Run `python main.py --version` and `python main.py --smoke-test`
7. Create `launch_windows.bat` in the project root

## Launch

Double-click **`launch_windows.bat`**, or from a terminal:

```bat
venv\Scripts\activate
bactomate
```

Equivalent:

```bat
python main.py
```

## Verify install (any time)

```bat
venv\Scripts\activate
python main.py --version
python main.py --smoke-test
```

You should see `BactoMate 2.3.4` and `Smoke test OK`.

## Troubleshooting

### `DLL load failed` / `c10.dll` / `torch` import errors

1. Install **Visual C++ Redistributable x64** (link above) and reboot if prompted
2. Confirm **64-bit Python 3.11**: `python -c "import struct; print(struct.calcsize('P')*8)"` → `64`
3. Recreate the venv:
   ```bat
   rmdir /s /q venv
   setup\install_windows.bat
   ```

### `pip install` resolution errors

The installer uses `--use-deprecated=legacy-resolver` for compatibility. If it still fails:

```bat
venv\Scripts\activate
pip install --upgrade pip
pip install --use-deprecated=legacy-resolver -r requirements.txt
pip install -e .
```

### Segmentation: `pretrained_model` / Cellpose errors

Re-apply patches:

```bat
venv\Scripts\activate
python setup\apply_all_patches.py
```

### CUDA / GPU not detected

CPU mode works for most workflows. To add GPU later:

```bat
venv\Scripts\activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Use the CUDA index that matches your driver (`cu121`, `cu124`, `cu128`, etc.).

### GUI does not start (no window)

Check `Logs\bactomate_gui_*.log` in the project folder. Ensure PyQt6 imported during smoke test.

## Manual install (if the batch script fails)

```bat
py -3.11 -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install --use-deprecated=legacy-resolver -r requirements.txt
pip install -e .
python setup\apply_all_patches.py
python main.py --smoke-test
```
