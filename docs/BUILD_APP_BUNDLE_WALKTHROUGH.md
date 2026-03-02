# Step-by-Step: How `build_app_bundle.sh` Builds BactiMate.app

This document walks through **what** the build script does, **how** it creates the app and icon, and **how** the app is structured and runs when you double-click it.

---

## 1. Overview

When you run from the project root:

```bash
./build/build_app_bundle.sh
```

the script:

1. Ensures a virtual environment exists and applies patches.
2. Creates the `.app` folder structure.
3. Compiles and installs the **native launcher** (the app’s main executable).
4. Copies the whole project (minus build artifacts) into the app’s Resources.
5. Prepares the **venv** inside the app (Python binary renamed to `BactiMate`, patches applied).
6. Builds the **app icon** from `img/logo2.png` → `BactiMate.icns`.
7. Writes **Info.plist** and installs the icon.
8. Signs the launcher and key binaries, and registers the app with macOS.

At the end you get **`BactiMate.app`**: a single double-clickable app that runs the BactiMate GUI using the bundled Python and dependencies.

---

## 2. Script Setup and Prerequisites (Lines 1–19)

```bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="BactiMate"
APP_DIR="$PROJECT_ROOT/$APP_NAME.app"
CONTENTS="$APP_DIR/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"
```

- **`set -e`**: Any command that fails stops the script.
- **Paths**:
  - `SCRIPT_DIR` = `.../BactiMate-main-V1.0/build`
  - `PROJECT_ROOT` = `.../BactiMate-main-V1.0`
  - `APP_DIR` = `.../BactiMate-main-V1.0/BactiMate.app`
  - `CONTENTS` = `BactiMate.app/Contents`
  - `MACOS` = `BactiMate.app/Contents/MacOS` (where the main executable goes)
  - `RESOURCES` = `BactiMate.app/Contents/Resources` (where the rest of the app lives)

Then:

```bash
[ ! -d "$PROJECT_ROOT/venv" ] && { echo "Error: venv not found..."; exit 1; }
"$PROJECT_ROOT/venv/bin/python3" "$PROJECT_ROOT/setup/apply_all_patches.py" 2>/dev/null || true
```

- **Requirement**: A `venv` must already exist (e.g. from `./setup/setup_fresh_env.sh`).
- **Patches**: Runs `apply_all_patches.py` once in the project venv (e.g. cellpose/omnipose patches). Failures are ignored so the build can continue.

```bash
rm -rf "$APP_DIR"
mkdir -p "$MACOS" "$RESOURCES"
```

- **Clean build**: Removes any existing `BactiMate.app` and creates:
  - `BactiMate.app/Contents/MacOS/`
  - `BactiMate.app/Contents/Resources/`

---

## 3. Native Launcher (Lines 20–23)

```bash
cc -o "$MACOS/$APP_NAME" "$SCRIPT_DIR/launcher.c" || { echo "Error: failed to compile..."; exit 1; }
chmod +x "$MACOS/$APP_NAME"
```

**What this does**

- Compiles **`build/launcher.c`** into **`BactiMate.app/Contents/MacOS/BactiMate`**.
- That binary is the **CFBundleExecutable**: the one process macOS starts when you double-click the app.

**Why a native launcher**

- The “app” process is this small C program, not Python.
- So the Dock, “Reopen”, and “Keep in Dock” stay tied to **BactiMate.app**, not to Python.app.

**What the launcher does (at runtime)**

1. Gets its own path (e.g. `.../BactiMate.app/Contents/MacOS/BactiMate`) via `_NSGetExecutablePath` (or `argv[0]` as fallback).
2. Derives the app bundle root (three `dirname`-style steps: MacOS → Contents → `.app`).
3. Builds the path to the shell script:  
   `.../BactiMate.app/Contents/Resources/setup/launch_gui_bundle_strict.sh`
4. **Forks**; in the **child** it runs:  
   `execl("/bin/bash", "bash", script, NULL)`  
   So the child becomes `bash` running that script.
5. In the **parent** it **waits** for the child and exits with the same status.

So the **main process** of the app is always the launcher binary; the GUI runs in a child process (bash → Python).

---

## 4. Copying the Project into the App (Lines 25–31)

```bash
rsync -a --exclude='.git' --exclude='.venv' --exclude='*.pyc' ...
  --exclude='BactiMate.app' --exclude='*.app' --exclude='build' --exclude='dist' --exclude='*.dmg' --exclude='Logs' \
  "$PROJECT_ROOT/" "$RESOURCES/"
```

**What this does**

- Copies the **entire project** into **`BactiMate.app/Contents/Resources/`**, with exclusions so you don’t ship:
  - Version control (`.git`)
  - Other venvs (`.venv`)
  - Bytecode and caches (`*.pyc`, `__pycache__`, `.gitignore`)
  - The app itself, other `.app`s, `build/`, `dist/`, DMGs, `Logs/`

**Result**

- Under `Resources/` you get the same layout as the project: `bactimate/`, `setup/`, `main.py`, `requirements.txt`, `img/`, etc.
- **Including** the project’s `venv/` (because only `.venv` is excluded). So the app has its own copy of the virtual environment.

The “Copying project…” message is kept alive by a background loop that prints “…” every 60 seconds until `rsync` finishes.

---

## 5. Python Executable Inside the App (Lines 44–48)

```bash
for c in "$RESOURCES/venv/bin/python3.11" "$RESOURCES/venv/bin/python3" "$RESOURCES/venv/bin/pythonw"; do
  [ -x "$c" ] && cp -fL "$c" "$RESOURCES/venv/bin/BactiMate" && chmod +x "$RESOURCES/venv/bin/BactiMate" && break
done
[ ! -x "$RESOURCES/venv/bin/BactiMate" ] && { echo "Error: no Python in venv"; exit 1; }
```

**What this does**

- The app will run Python via a **single** executable path: **`Resources/venv/bin/BactiMate`**.
- The script picks the first that exists and is executable among:
  - `venv/bin/python3.11`
  - `venv/bin/python3`
  - `venv/bin/pythonw`
- It **copies** that binary (following symlinks with `-L`) to **`venv/bin/BactiMate`** and makes it executable.

**Why “BactiMate”**

- The strict launcher and scripts always use `venv/bin/BactiMate`. One name works regardless of whether the venv had `python3`, `python3.11`, or `pythonw`.
- Using a copy (not a symlink) avoids the app being identified as “Python” by the system when that binary runs.

---

## 6. Patches and Fixes Inside the Bundle (Lines 50–53)

```bash
mkdir -p "$RESOURCES/Logs"
SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])")
"$PY" "$RESOURCES/setup/apply_all_patches.py" --target-site-packages "$SITE" 2>/dev/null || true
[ -f "$SITE/omnipose/core.py" ] && "$PY" "$RESOURCES/setup/fix_omnipose_core_syntax.py" "$SITE/omnipose/core.py" 2>/dev/null || true
```

**What this does**

- Creates **`Resources/Logs/`** for launch logs.
- Uses the **copied** venv’s Python (via `$PY` = `Resources/venv/bin/python3`) to find **site-packages** inside the app.
- Runs **`apply_all_patches.py --target-site-packages $SITE`** so patches (e.g. cellpose/omnipose) are applied to the **bundled** site-packages, not the project’s.
- If `omnipose/core.py` exists in that site-packages, runs **`fix_omnipose_core_syntax.py`** on it.

So the app’s venv is patched and ready to run the same code paths as when you run from the project.

---

## 7. Building the App Icon (Lines 55–58)

```bash
[ ! -s "$PROJECT_ROOT/img/logo2.png" ] && { echo "Error: img/logo2.png missing"; exit 1; }
rm -f "$SCRIPT_DIR/BactiMate.icns"
"$SCRIPT_DIR/create_icons.sh" || { echo "Error: create_icons.sh failed..."; exit 1; }
[ ! -s "$SCRIPT_DIR/BactiMate.icns" ] && { echo "Error: no BactiMate.icns produced"; exit 1; }
```

**Requirement**: **`img/logo2.png`** must exist and be non-empty.

**What `create_icons.sh` does (detailed)**

1. **Input**: `img/logo2.png`  
   **Output**: `build/BactiMate.icns` (macOS app icon file).

2. **Python/Pillow path** (if `venv/bin/python3` or `python3` has Pillow):
   - Opens `logo2.png` as RGBA.
   - Builds a **square** image: side = `max(width, height)`, **transparent** fill `(0,0,0,0)` (no black/grey).
   - Pastes the logo centered on that square.
   - Resizes to **1024×1024** with LANCZOS.
   - Saves as **ICNS** with sizes 16, 32, 64, 128, 256, 512, 1024 (one file with all resolutions).

3. **Fallback (no Pillow)**:
   - Uses macOS **sips** to generate PNGs in a **`.iconset`** folder (e.g. `icon_16x16.png`, `icon_16x16@2x.png`, …).
   - Runs **iconutil -c icns** to produce `BactiMate.icns`.

4. If the icon file is still missing or empty, the script exits with an error.

**Important**: The icon is **only** from `logo2.png` (scaled to square, transparent padding). Any grey or black you see in the icon comes from that PNG, not from the build.

After this step, **`build/BactiMate.icns`** exists and is used later when assembling the app.

---

## 8. Making Scripts Executable (Line 60)

```bash
chmod +x "$RESOURCES/launch_gui.sh" "$RESOURCES/setup/launch_gui.sh" "$RESOURCES/setup/launch_gui_bundle_strict.sh" 2>/dev/null || true
```

- Ensures the main launch scripts are executable. The app uses **`launch_gui_bundle_strict.sh`**; the others are for running from the project or older flows.

---

## 9. Info.plist (Lines 62–75)

```bash
cat > "$CONTENTS/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
...
<key>CFBundleExecutable</key><string>BactiMate</string>
<key>CFBundleIconFile</key><string>BactiMate</string>
<key>CFBundleIdentifier</key><string>org.bactimate.app</string>
...
PLIST
```

**What this does**

- Writes **`BactiMate.app/Contents/Info.plist`**, which tells macOS:
  - **CFBundleExecutable**: `BactiMate` → run **`Contents/MacOS/BactiMate`** (the native launcher).
  - **CFBundleIconFile**: `BactiMate` → use **`Contents/Resources/BactiMate.icns`** (no extension in plist).
  - **CFBundleIdentifier**: `org.bactimate.app` (unique app ID).
  - **CFBundleName**, version, **LSMinimumSystemVersion**, **NSHighResolutionCapable**, etc.

So the system knows how to start the app and which icon to show.

---

## 10. Installing the Icon and Cleanup (Lines 77–78)

```bash
cp "$SCRIPT_DIR/BactiMate.icns" "$RESOURCES/BactiMate.icns"
[ ! -s "$RESOURCES/BactiMate.icns" ] && { echo "Error: BactiMate.icns not in bundle"; exit 1; }
```

- Copies **`build/BactiMate.icns`** to **`BactiMate.app/Contents/Resources/BactiMate.icns`**.
- Verifies the file is non-empty.

---

## 11. Signing and Registration (Lines 80–88)

```bash
xattr -cr "$APP_DIR" 2>/dev/null || true
command -v codesign &>/dev/null && {
  codesign -f -s - "$MACOS/$APP_NAME" 2>/dev/null || true
  find "$RESOURCES/venv" -name "*.dylib" -type f -exec codesign -f -s - {} \; 2>/dev/null || true
  find "$RESOURCES/venv/bin" -type f \( -name "python*" -o -name "BactiMate" \) -exec codesign -f -s - {} \; 2>/dev/null || true
}
command -v .../lsregister ... && .../lsregister -f "$APP_DIR" 2>/dev/null || true
touch "$APP_DIR" 2>/dev/null || true
```

**What this does**

- **xattr -cr**: Clears extended attributes on the whole app (e.g. quarantine) so the bundle is clean.
- **codesign** (if available):
  - **Ad-hoc signs** the main executable **`MacOS/BactiMate`** (`-s -` = ad-hoc).
  - Ad-hoc signs all **`.dylib`** files under `Resources/venv/`.
  - Ad-hoc signs **`venv/bin/python*`** and **`venv/bin/BactiMate`**.
- **lsregister**: Registers the app with Launch Services so Finder/Dock recognize it and open it correctly.
- **touch**: Updates the app’s modification time.

Ad-hoc signing satisfies Gatekeeper enough for local use and keeps the app runnable without a developer certificate.

---

## 12. Final Check (Lines 90–93)

```bash
[ ! -x "$MACOS/$APP_NAME" ] || [ ! -f "$RESOURCES/setup/launch_gui_bundle_strict.sh" ] || [ ! -f "$RESOURCES/main.py" ] || [ ! -x "$RESOURCES/venv/bin/BactiMate" ] || [ ! -s "$RESOURCES/BactiMate.icns" ] && { echo "Error: bundle incomplete"; exit 1; }

echo "Created: $APP_DIR"
echo "If app does not open: BactiMate.app/Contents/Resources/Logs/launch.log"
```

- Verifies that all of these exist and (where relevant) are executable/non-empty:
  - `Contents/MacOS/BactiMate` (launcher)
  - `Contents/Resources/setup/launch_gui_bundle_strict.sh`
  - `Contents/Resources/main.py`
  - `Contents/Resources/venv/bin/BactiMate`
  - `Contents/Resources/BactiMate.icns`
- If any check fails, the script exits with “Error: bundle incomplete”.
- Otherwise it prints where the app was created and where to look for launch logs if the app doesn’t open.

---

## 13. Final Layout of BactiMate.app

After a successful build, the bundle looks like this:

```
BactiMate.app/
├── Contents/
│   ├── Info.plist              # Bundle ID, executable name, icon name, version, etc.
│   ├── MacOS/
│   │   └── BactiMate           # Native launcher binary (compiled from build/launcher.c)
│   └── Resources/
│       ├── BactiMate.icns      # App icon (from img/logo2.png)
│       ├── Logs/               # launch.log written here
│       ├── main.py             # Entry point for the GUI
│       ├── bactimate/           # Main package
│       ├── setup/
│       │   ├── launch_gui_bundle_strict.sh   # Script run by the launcher
│       │   ├── launch_gui.sh
│       │   ├── apply_all_patches.py
│       │   └── ...
│       ├── venv/
│       │   ├── bin/
│       │   │   ├── BactiMate   # Copy of python3 (used to run main.py)
│       │   │   ├── python3
│       │   │   └── ...
│       │   └── lib/pythonX.Y/site-packages/   # Patched dependencies
│       ├── img/
│       ├── requirements.txt
│       └── ... (rest of project, minus exclusions)
```

- **Double-click** runs **`MacOS/BactiMate`** (the launcher).
- The launcher runs **`Resources/setup/launch_gui_bundle_strict.sh`** as a child.
- That script runs **`Resources/venv/bin/BactiMate`** with **`Resources/main.py`**, which starts the BactiMate GUI.

---

## 14. What Happens When You Run the App (Runtime Flow)

1. **User**: Double-clicks **BactiMate.app** (or clicks its Dock icon).
2. **macOS**: Reads **Info.plist**, sees **CFBundleExecutable** = `BactiMate`, and runs:
   - **`BactiMate.app/Contents/MacOS/BactiMate`** (the native launcher).
3. **Launcher**:
   - Resolves its path (e.g. `.../BactiMate.app/Contents/MacOS/BactiMate`).
   - Computes bundle root and path to **`Contents/Resources/setup/launch_gui_bundle_strict.sh`**.
   - Forks; child runs: **`/bin/bash`** with that script.
   - Parent waits for the child and exits with its status.
4. **launch_gui_bundle_strict.sh**:
   - Checks it’s really inside a `.app` (path contains `.app/Contents/Resources`).
   - Sets **PROJECT_ROOT** = `Contents/Resources` (the app’s “project”).
   - Verifies **`venv/bin/BactiMate`** and **`main.py`** exist.
   - Detects Python version, sets **QT_PLUGIN_PATH**, **QT_FRAMEWORK_PATH**, **DYLD_FRAMEWORK_PATH**, **PATH**, etc.
   - Runs **`venv/bin/BactiMate main.py`** (as a **child** process, no `exec`).
   - When Python exits, the script exits with the same code.
5. **Python**:
   - **main.py** optionally sets the process title to “BactiMate” (if setproctitle is installed), then calls **bactimate.__main__.run()**, which starts the PyQt6 GUI.
6. **GUI**: User sees the BactiMate window; when they quit, Python exits → script exits → launcher exits → app is closed.

So the **only** process that is the “app” from macOS’s point of view is the small launcher binary; everything else runs under it. That’s why the Dock and “Reopen” stay correct for BactiMate.app.

---

## 15. Summary Table

| Step in script          | What it creates / does |
|-------------------------|------------------------|
| Setup                   | Paths, check venv, apply patches (project) |
| Clean + dirs            | Empty `BactiMate.app/Contents/{MacOS,Resources}` |
| Launcher                | Compile `launcher.c` → `MacOS/BactiMate` |
| rsync                   | Full project (incl. venv) → `Resources/` |
| Python binary           | Copy `venv/bin/python3` → `venv/bin/BactiMate` |
| Patches in bundle       | `apply_all_patches.py`, `fix_omnipose_core_syntax.py` on bundle venv |
| Icons                   | `create_icons.sh`: logo2.png → `build/BactiMate.icns` |
| chmod                   | Launch scripts executable |
| Info.plist              | Bundle metadata (executable, icon, ID, version) |
| Icon in bundle          | Copy `BactiMate.icns` → `Resources/` |
| Signing + registration  | Ad-hoc sign launcher + venv binaries; lsregister; touch |
| Verification            | Check launcher, script, main.py, BactiMate, icns |

Together, this produces a single **BactiMate.app** that looks and runs like a normal macOS app, with one main executable (the launcher), a clear icon, and correct Dock/Reopen behaviour.
