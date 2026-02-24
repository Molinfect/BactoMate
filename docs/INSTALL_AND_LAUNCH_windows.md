# BactiMate: Install and launch on Windows

Step-by-step guide to run BactiMate on **Windows 10 or 11**, using **Python 3.10 or 3.11** (3.11 recommended) and **pip**.

---

## 1. Install Python 3.11

BactiMate needs **Python 3.10 or 3.11**. Python 3.9 is not supported (e.g. trackastra requires 3.10+).

1. Open [python.org/downloads](https://www.python.org/downloads/).
2. Download **Python 3.11.x** for Windows (e.g. **Windows installer (64-bit)**).
3. Run the installer.
4. **Important:** On the first screen, check **“Add python.exe to PATH”**, then click **“Install Now”**.
5. When finished, open **Command Prompt** or **PowerShell** and check:

   ```cmd
   py -3.11 --version
   ```

   or, if Python is on PATH:

   ```cmd
   python --version
   ```

   You should see something like `Python 3.11.x`.

---

## 2. Get the BactiMate project

- **With Git:** clone the repository, then open a terminal in the project folder:

  ```cmd
  git clone <repository-url> BactiMate
  cd BactiMate
  ```

- **From a ZIP:** unzip the archive, then open **Command Prompt** or **PowerShell** in the project folder (the one that contains `main.py`, `requirements.txt`, and the `bactimate` folder).

Check you are in the right place:

```cmd
dir main.py, requirements.txt, bactimate
```

You should see `main.py`, `requirements.txt`, and a `bactimate` directory.

---

## 3. Create a virtual environment

Use a virtual environment so BactiMate’s dependencies stay separate from other Python projects.

**Command Prompt**

```cmd
cd C:\path\to\BactiMate
py -3.11 -m venv venv
venv\Scripts\activate
```

**PowerShell**

```powershell
cd C:\path\to\BactiMate
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error in PowerShell, run once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Your prompt should start with `(venv)`.

If Python 3.11 is your default `python`, you can use `python -m venv venv` instead of `py -3.11 -m venv venv`.

---

## 4. Install dependencies and BactiMate

With the virtual environment **activated**:

**4.1 – Upgrade pip and install requirements**

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyQt6, Omnipose, NumPy, trackastra, and the rest. It can take several minutes.

If you see **“resolution-too-deep”** or dependency conflicts on a fresh install, try:

```cmd
pip install --use-deprecated=legacy-resolver -r requirements.txt
```

**4.2 – Install BactiMate in editable mode**

```cmd
pip install -e .
```

This makes the `bactimate` command (and `python -m bactimate`) use your project folder, so you can run the GUI and edit the code.

**Optional – NVIDIA GPU (CUDA) for faster segmentation and tracking**

After step 4.1, install PyTorch with CUDA, for example:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version if different (e.g. `cu121`, `cu124`). To check: run `nvidia-smi` in a terminal. Common options: 11.8 → `cu118`, 12.1 → `cu121`, 12.4 → `cu124`.

---

## 5. Launch the BactiMate GUI

With the same terminal and the virtual environment still **activated**, use any of:

```cmd
bactimate
```

or:

```cmd
python -m bactimate
```

or:

```cmd
python main.py
```

The BactiMate window should open. You can use the tabs: **TIFF Conversion**, **Omnipose Segment**, **Analyze Masks**, **Tracking**, **Video Graph**, **Session Logs**. For full documentation, see the [BactiMate User Manual](BACTIMATE_USER_MANUAL.md).

---

## 6. Next time (same machine)

Activate the environment and run the app:

**Command Prompt**

```cmd
cd C:\path\to\BactiMate
venv\Scripts\activate
bactimate
```

**PowerShell**

```powershell
cd C:\path\to\BactiMate
.\venv\Scripts\Activate.ps1
bactimate
```

You do not need to run `pip install -r requirements.txt` or `pip install -e .` again unless you update the project or change dependencies.

---

## 7. Troubleshooting

| Problem | What to try |
|--------|-------------|
| `py` or `python` not found | Reinstall Python and ensure **“Add python.exe to PATH”** is checked. Restart the terminal. Try `py -3.11` or the full path to `python.exe`. |
| `pip` not found | Use `py -3.11 -m pip` or `python -m pip` instead of `pip`. |
| PowerShell: “running scripts is disabled” | Run once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. Or use Command Prompt and `venv\Scripts\activate`. |
| `No module named 'PyQt6'` | Ensure the venv is activated (prompt starts with `(venv)`), then run `pip install -r requirements.txt` again. |
| `No module named 'bactimate'` | From the BactiMate project root (folder that contains `pyproject.toml`), run `pip install -e .`. |
| GUI does not open or crashes | Check the terminal for error messages. Try `python main.py` to see the full traceback. |
| Dependency resolution errors | Try `pip install --use-deprecated=legacy-resolver -r requirements.txt`. |
| Zeiss CZI support | After main install: `pip install aicspylibczi`. |

---

## 8. Quick reference (Windows)

| Step | Command |
|------|---------|
| Install Python | [python.org/downloads](https://www.python.org/downloads/) — tick **“Add to PATH”** |
| Go to project | `cd C:\path\to\BactiMate` |
| Create venv | `py -3.11 -m venv venv` |
| Activate venv (CMD) | `venv\Scripts\activate` |
| Activate venv (PowerShell) | `.\venv\Scripts\Activate.ps1` |
| Install dependencies | `pip install -r requirements.txt` |
| Install app | `pip install -e .` |
| Launch GUI | `bactimate` or `python main.py` |

---

**Summary:** Install Python 3.11 (with “Add to PATH”) → open Command Prompt or PowerShell in the BactiMate folder → create and activate venv → `pip install -r requirements.txt` → `pip install -e .` → run `bactimate` or `python main.py`.
