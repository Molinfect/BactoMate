# BactiMate: Install and launch (step by step)

This guide takes you from **no Python** (or the wrong version) to **BactiMate GUI running**, using the recommended **Python 3.11** and **pip install**.

---

## 1. Install Python 3.11

BactiMate requires **Python 3.10 or 3.11** (3.11 recommended). Python 3.9 is not supported (e.g. trackastra needs 3.10+).

### macOS

**Option A – Official installer**

1. Go to [python.org/downloads](https://www.python.org/downloads/).
2. Download **Python 3.11.x** for macOS.
3. Run the installer. At the end, check **“Install certificates”** if offered.
4. Open **Terminal** and check:
   ```bash
   python3.11 --version
   ```
   You should see something like `Python 3.11.x`.

**Option B – Homebrew**

```bash
brew install python@3.11
```

Then use `python3.11` (and below, use `python3.11` wherever the guide says `python3`).

### Windows

1. Go to [python.org/downloads](https://www.python.org/downloads/).
2. Download **Python 3.11.x** for Windows (e.g. “Windows installer (64-bit)”).
3. Run the installer.
4. **Important:** On the first screen, check **“Add python.exe to PATH”**, then click “Install Now”.
5. Open **Command Prompt** or **PowerShell** and check:
   ```cmd
   py -3.11 --version
   ```
   or, if that’s in PATH:
   ```cmd
   python --version
   ```
   You should see `Python 3.11.x`.

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

---

## 2. Get the BactiMate project

- **If you have Git:** clone the repo:
  ```bash
  git clone <repository-url> BactiMate
  cd BactiMate
  ```
- **If you downloaded a ZIP:** unzip it, then open a terminal in the project folder (the one that contains `main.py`, `requirements.txt`, and the `bactimate` folder).

Confirm you’re in the right place:

```bash
# macOS / Linux
ls main.py requirements.txt bactimate

# Windows (PowerShell)
dir main.py, requirements.txt, bactimate
```

You should see `main.py`, `requirements.txt`, and a `bactimate` directory.

---

## 3. Create a virtual environment

Use a **virtual environment** so BactiMate’s dependencies don’t conflict with other Python projects.

**macOS / Linux**

```bash
cd /path/to/BactiMate    # if not already there
python3.11 -m venv venv
source venv/bin/activate
```

Your prompt should start with `(venv)`.

**Windows (Command Prompt)**

```cmd
cd C:\path\to\BactiMate
py -3.11 -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell)**

```powershell
cd C:\path\to\BactiMate
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

If you only have “Python 3.11” as `python` or `python3`, you can use that instead of `python3.11` / `py -3.11` in the commands above.

---

## 4. Install dependencies and BactiMate

With the virtual environment **activated**:

**4.1 – Install required packages**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyQt6, Omnipose, NumPy, and the rest of the dependencies. It can take a few minutes.

**4.2 – Install BactiMate in “editable” mode**

```bash
pip install -e .
```

This makes the `bactimate` command (and `python -m bactimate`) use your current project folder, so you can run the GUI and still edit the code.

**Optional – GPU (CUDA) for faster segmentation**

- **NVIDIA GPU (Windows/Linux):** after step 4.1, install PyTorch with CUDA, e.g.:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
  (Replace `cu118` with your CUDA version if different.)
- **macOS (Apple Silicon):** the default PyTorch from `requirements.txt` can use MPS; no extra step needed for basic use.

---

## 5. Launch the BactiMate GUI

With the same terminal and the virtual environment still **activated**, use any of:

```bash
bactimate
```

or:

```bash
python -m bactimate
```

or:

```bash
python main.py
```

The BactiMate window should open. You can then use the tabs (TIFF Conversion, Omnipose Segment, Analyze Masks, Tracking, Video Graph, Session Logs).

---

## 6. Next time (same machine)

You only need to **activate the environment** and **run the app**:

**macOS / Linux**

```bash
cd /path/to/BactiMate
source venv/bin/activate
bactimate
```

**Windows**

```cmd
cd C:\path\to\BactiMate
venv\Scripts\activate
bactimate
```

No need to run `pip install -r requirements.txt` or `pip install -e .` again unless you pull new changes or want to update dependencies.

---

## 7. Troubleshooting

| Problem | What to try |
|--------|-------------|
| `python3.11: command not found` | Use the full path to the Python 3.11 executable, or on Windows try `py -3.11`. |
| `pip: command not found` | Run `python3.11 -m pip` (or `py -3.11 -m pip` on Windows) instead of `pip`. |
| `No module named 'PyQt6'` | Make sure the venv is activated and run `pip install -r requirements.txt` again. |
| `No module named 'bactimate'` | Run `pip install -e .` from the BactiMate project root (the folder that contains `pyproject.toml`). |
| GUI doesn’t open / crashes | Check the terminal for error messages. On macOS, if you see Qt/plugin errors, see the main README “Troubleshooting” or run `./setup/repair_pyqt6.sh` if available. |
| Script not found (e.g. conversion) | You should be in the project root when you ran `pip install -e .`; run it again from there. |

---

## 8. Quick reference

| Step | macOS / Linux | Windows |
|------|----------------|--------|
| Python 3.11 | [python.org](https://www.python.org/downloads/) or `brew install python@3.11` | [python.org](https://www.python.org/downloads/), tick “Add to PATH” |
| Go to project | `cd /path/to/BactiMate` | `cd C:\path\to\BactiMate` |
| Create venv | `python3.11 -m venv venv` | `py -3.11 -m venv venv` |
| Activate venv | `source venv/bin/activate` | `venv\Scripts\activate` |
| Install deps | `pip install -r requirements.txt` | same |
| Install app | `pip install -e .` | same |
| Launch GUI | `bactimate` or `python main.py` | same |

---

**Summary:** Install Python 3.11 → open terminal in BactiMate folder → create and activate a venv → `pip install -r requirements.txt` → `pip install -e .` → run `bactimate` (or `python main.py`).
