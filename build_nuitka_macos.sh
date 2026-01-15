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

# Check Python version - Nuitka requires CPython (not Apple Python)
PYTHON_CMD="python3"
if command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
    # Prefer Homebrew Python 3.11 if available
    PYTHON_CMD="/opt/homebrew/bin/python3.11"
    echo "Using Homebrew Python 3.11 for Nuitka build"
elif command -v /usr/local/bin/python3 &> /dev/null; then
    # Try /usr/local/bin/python3 (Homebrew on Intel Mac)
    PYTHON_CMD="/usr/local/bin/python3"
    echo "Using /usr/local/bin/python3 for Nuitka build"
else
    # Check if current Python is Apple Python
    if python3 -c "import sys; sys.exit(0 if 'CommandLineTools' in sys.executable or 'Apple' in sys.version else 1)" 2>/dev/null; then
        echo "ERROR: Apple Python detected. Nuitka requires CPython from python.org or Homebrew."
        echo "Please install Python from: https://www.python.org/downloads/macos/"
        echo "Or install via Homebrew: brew install python@3.11"
        exit 1
    fi
fi

# Check if we're in a venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: Not in a virtual environment. Activating venv..."
    if [ -d "venv" ]; then
        source venv/bin/activate
        # Verify venv Python is not Apple Python
        if python3 -c "import sys; sys.exit(0 if 'CommandLineTools' in sys.executable or 'Apple' in sys.version else 1)" 2>/dev/null; then
            echo "WARNING: venv uses Apple Python. Creating new venv with $PYTHON_CMD..."
            mv venv venv_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
            $PYTHON_CMD -m venv venv
            source venv/bin/activate
            echo "Installing requirements in new venv..."
            pip install --upgrade pip
            pip install -r requirements.txt
            python3 apply_all_patches.py
        fi
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
echo "NOTE: QML warnings about Qt Quick 3D files are safe to ignore -"
echo "      we use PyQt5 widgets, not QML, so these files aren't needed."
echo ""

python3 -m nuitka \
    --standalone \
    --static-libpython=no \
    --macos-create-app-bundle \
    --macos-app-icon=BactiMate.icns \
    --macos-app-name=BactiMate \
    --macos-app-version=1.0.0 \
    --macos-app-protected-resource=NSMicrophoneUsageDescription:"BactiMate needs microphone access" \
    --macos-app-protected-resource=NSCameraUsageDescription:"BactiMate needs camera access" \
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
    --nofollow-import-to=tkinter \
    --nofollow-import-to=matplotlib.tests \
    --nofollow-import-to=numpy.tests \
    --nofollow-import-to=scipy.tests \
    --nofollow-import-to=pandas.tests \
    --nofollow-import-to=PyQt5.QtQml \
    --nofollow-import-to=PyQt5.QtQuick \
    --nofollow-import-to=PyQt5.QtQuick3D \
    omnipose_batch_gui.py 2>&1 | grep -v "WARNING: pyqt5: Including QML file" | tee /tmp/nuitka_build.log

BUILD_EXIT_CODE=${PIPESTATUS[0]}

if [ $BUILD_EXIT_CODE -ne 0 ]; then
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
