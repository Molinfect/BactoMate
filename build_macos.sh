#!/bin/bash
# Build script for macOS application bundle
# This creates a standalone .app bundle that can run on any macOS machine

set -e  # Exit on error

echo "=========================================="
echo "Building BactiMate for macOS"
echo "=========================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: This script must be run on macOS"
    exit 1
fi

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.app

# Build the application bundle
echo ""
echo "Building application bundle..."
python3 -m PyInstaller BactiMate.spec

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed!"
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
else
    echo ""
    echo "ERROR: BactiMate.app was not created!"
    exit 1
fi
