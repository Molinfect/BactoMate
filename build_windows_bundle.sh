#!/bin/bash
# Build script for Windows bundle (can be run on macOS/Linux to prepare for Windows)
# This creates a folder structure that can be transferred to Windows
# Similar to the macOS .app bundle approach - no compilation needed

set -e  # Exit on error

echo "=========================================="
echo "Building BactiMate Windows Bundle"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf "BactiMate-Windows"
rm -rf "dist/BactiMate-Windows"

# Create bundle directory
echo "Creating bundle structure..."
BUNDLE_DIR="BactiMate-Windows"
mkdir -p "$BUNDLE_DIR"

# Create launcher batch file for Windows
echo "Creating Windows launcher..."
cat > "$BUNDLE_DIR/BactiMate.bat" << 'BAT_EOF'
@echo off
REM BactiMate launcher for Windows

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found!
    echo Please ensure venv directory exists in the bundle.
    pause
    exit /b 1
)

REM Activate venv and run the GUI
echo Launching BactiMate...
call "venv\Scripts\activate.bat"
python omnipose_batch_gui.py

REM If there's an error, keep window open
if errorlevel 1 (
    echo.
    echo ERROR: BactiMate failed to launch.
    echo Check the error messages above.
    pause
)
BAT_EOF

# Copy all project files (excluding build artifacts)
echo "Copying project files to bundle..."
echo "  (this may take several minutes for large venv...)"

# Files/directories to exclude
EXCLUDE_PATTERNS=(
    ".git"
    ".gitignore"
    "__pycache__"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    ".DS_Store"
    "dist"
    "build"
    "*.app"
    "*.dmg"
    "*.log"
    "nuitka-crash-report.xml"
    "venv_backup"
    ".venv"
    "BactiMate-Windows"
)

# Build rsync exclude arguments
RSYNC_EXCLUDES=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    RSYNC_EXCLUDES+=("--exclude=$pattern")
done

# Use rsync to copy files
if ! rsync -av --progress "${RSYNC_EXCLUDES[@]}" \
    "$SCRIPT_DIR/" "$BUNDLE_DIR/"; then
    echo "ERROR: Failed to copy files to bundle!"
    exit 1
fi

# Verify critical files were copied
if [ ! -d "$BUNDLE_DIR/venv" ]; then
    echo "ERROR: venv directory not found in bundle!"
    exit 1
fi

if [ ! -f "$BUNDLE_DIR/omnipose_batch_gui.py" ]; then
    echo "ERROR: omnipose_batch_gui.py not found in bundle!"
    exit 1
fi

# Show final size
BUNDLE_SIZE=$(du -sh "$BUNDLE_DIR" | cut -f1)
echo "  Windows bundle created: $BUNDLE_SIZE"

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "The Windows bundle is located at: $BUNDLE_DIR"
echo ""
echo "To distribute:"
echo "  1. Transfer the entire '$BUNDLE_DIR' folder to a Windows machine"
echo "  2. Users can run 'BactiMate.bat' to launch the application"
echo ""
echo "Or create a zip file:"
echo "  zip -r BactiMate-Windows.zip $BUNDLE_DIR"
echo ""
