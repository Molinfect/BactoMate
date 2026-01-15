#!/bin/bash
# Build script for macOS using Nuitka
# This creates a standalone application that should avoid PyInstaller issues

set -e  # Exit on error

echo "=========================================="
echo "Building BactiMate for macOS with Nuitka"
echo "=========================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: This script must be run on macOS"
    exit 1
fi

# Check if we're in a venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: Not in a virtual environment. Activating venv..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "ERROR: No venv found. Please run setup first."
        exit 1
    fi
fi

# Check if Nuitka is installed
if ! python3 -c "import nuitka" 2>/dev/null; then
    echo "Installing Nuitka..."
    pip install nuitka
fi

# Apply all patches before building
echo "Applying compatibility patches..."
if [ -f "apply_all_patches.py" ]; then
    python3 apply_all_patches.py
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build dist BactiMate.app BactiMate.dist

# Create output directory
mkdir -p dist

# Build with Nuitka
echo ""
echo "Building with Nuitka (this may take a while)..."
echo ""

python3 -m nuitka \
    --standalone \
    --onefile \
    --macos-create-app-bundle \
    --macos-app-icon=BactiMate.icns \
    --output-dir=dist \
    --output-filename=BactiMate \
    --follow-imports \
    --include-package=omnipose \
    --include-package=cellpose_omni \
    --include-package=PyQt5 \
    --include-package=numpy \
    --include-package=scipy \
    --include-package=skimage \
    --include-package=torch \
    --include-package=matplotlib \
    --include-package-data=omnipose \
    --include-package-data=cellpose_omni \
    --include-package-data=PyQt5 \
    --enable-plugin=pyqt5 \
    --macos-app-protected-resource=NSMicrophoneUsageDescription="BactiMate needs microphone access" \
    --macos-app-protected-resource=NSCameraUsageDescription="BactiMate needs camera access" \
    --macos-app-version=1.0.0 \
    --macos-app-name=BactiMate \
    --macos-bundle-identifier=com.bactimate.app \
    --nofollow-import-to=tkinter \
    --nofollow-import-to=matplotlib.tests \
    --nofollow-import-to=numpy.tests \
    --nofollow-import-to=scipy.tests \
    --nofollow-import-to=pandas.tests \
    omnipose_batch_gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Nuitka build failed!"
    exit 1
fi

# Check if .app was created
if [ -d "dist/BactiMate.app" ]; then
    echo ""
    echo "=========================================="
    echo "Build complete!"
    echo "=========================================="
    echo ""
    echo "The application bundle is located in: dist/BactiMate.app"
    echo ""
    echo "To create a DMG file, run: ./create_dmg.sh"
    echo ""
    echo "NOTE: Nuitka builds may require testing to ensure all dependencies"
    echo "      are properly included. If you encounter issues, you may need"
    echo "      to add additional --include-package flags."
    echo ""
else
    echo ""
    echo "WARNING: BactiMate.app was not found in expected location."
    echo "Checking dist directory..."
    ls -la dist/
fi
