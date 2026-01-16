# Building BactiMate App Bundle

This document describes how to build a macOS `.app` bundle for BactiMate, similar to how [ilastik](https://github.com/ilastik/ilastik) packages their application.

## Overview

Instead of compiling with Nuitka (which has PyQt6 compatibility issues on macOS), we bundle the entire project folder including the Python virtual environment into a macOS `.app` bundle. This approach:

- ✅ Avoids Nuitka compilation issues
- ✅ Works with PyQt6 without modifications
- ✅ Keeps the environment small and self-contained
- ✅ Easy to maintain and update
- ✅ Similar to how ilastik distributes their application

## Building the App Bundle

### Prerequisites

- macOS (for building macOS app bundles)
- Python 3.9+ with virtual environment set up
- All dependencies installed in `venv/`

### Build Steps

1. **Set up the environment** (if not already done):
   ```bash
   ./setup_fresh_env.sh
   ```

2. **Build the app bundle**:
   ```bash
   ./build_app_bundle.sh
   ```

   This will create `BactiMate.app` in the project root directory.

3. **Test the app bundle**:
   ```bash
   open BactiMate.app
   ```

   Or double-click `BactiMate.app` in Finder.

4. **Create a DMG for distribution** (optional):
   ```bash
   ./create_dmg.sh
   ```

   This creates `dist/BactiMate.dmg` that users can mount and drag to Applications.

## How It Works

The app bundle structure:

```
BactiMate.app/
├── Contents/
│   ├── MacOS/
│   │   └── BactiMate          # Launcher executable (runs launch_gui.sh)
│   ├── Resources/             # All project files
│   │   ├── venv/              # Python virtual environment
│   │   ├── omnipose_batch_gui.py
│   │   ├── launch_gui.sh      # Main launch script
│   │   └── ... (all other project files)
│   └── Info.plist             # macOS app metadata
```

The launcher (`Contents/MacOS/BactiMate`) is a shell script that:
1. Finds the project directory inside the app bundle
2. Changes to that directory
3. Executes `launch_gui.sh` which activates the venv and runs the Python script

## Distribution

### For Users

Users can:
1. Download the `.tar.bz2` file (if you create one)
2. Extract it by double-clicking
3. Copy `BactiMate.app` to their Applications folder
4. Control-click and choose "Open" the first time (to bypass Gatekeeper)
5. Double-click to launch thereafter

### Creating a Distribution Archive

To create a compressed archive for distribution:

```bash
# Create a tar.bz2 archive (similar to ilastik)
tar -cjf BactiMate-macOS.tar.bz2 BactiMate.app
```

Or use the DMG method:

```bash
./create_dmg.sh
```

## Advantages Over Nuitka

1. **No compilation required** - Just bundle existing files
2. **Faster builds** - No long compilation process
3. **Easier debugging** - Can inspect/modify files inside the bundle
4. **PyQt6 compatible** - No need to switch to PySide6
5. **Smaller learning curve** - Standard macOS app bundle structure

## File Size

The app bundle includes:
- Python virtual environment (~500MB-1GB depending on dependencies)
- All project files
- Models and data files

Total size is typically 500MB-1.5GB, which is reasonable for a scientific application with ML dependencies.

## Troubleshooting

### App won't launch

1. Check that `launch_gui.sh` is executable:
   ```bash
   chmod +x BactiMate.app/Contents/Resources/launch_gui.sh
   ```

2. Check console logs:
   ```bash
   open Console.app
   # Then try launching BactiMate.app and look for errors
   ```

3. Test the launcher manually:
   ```bash
   cd BactiMate.app/Contents/Resources
   ./launch_gui.sh
   ```

### Code signing issues

The build script applies ad-hoc code signatures. For distribution outside the App Store, you may need:
- A Developer ID certificate from Apple
- Proper code signing with `codesign --sign "Developer ID Application: Your Name"`

### Gatekeeper warnings

Users may see a warning the first time they open the app. They need to:
1. Control-click the app
2. Choose "Open"
3. Click "Open" in the security dialog

After the first launch, macOS will remember the app is safe.

## Comparison with ilastik

This approach mirrors how ilastik distributes their application:
- Self-contained app bundle with Python environment
- Simple launcher script
- No compilation step
- Easy to update (just replace files in Resources/)

See: https://github.com/ilastik/ilastik
