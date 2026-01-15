#!/bin/bash
# Setup a completely fresh virtual environment for BactiMate

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Setting up fresh BactiMate environment"
echo "=========================================="
echo ""

# Step 1: Remove old virtual environment
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
    echo "✓ Old venv removed"
else
    echo "No existing venv found (that's okay)"
fi

# Step 2: Create new virtual environment
echo ""
echo "Creating new virtual environment..."
python3 -m venv venv
echo "✓ New venv created"

# Step 3: Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Step 4: Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ Pip upgraded"

# Step 5: Install requirements
echo ""
echo "Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Requirements installed"
    
    # Apply all patches automatically
    echo ""
    echo "Applying compatibility patches..."
    if [ -f "apply_all_patches.py" ]; then
        python3 apply_all_patches.py || echo "⚠ Warning: Some patches may have failed"
    else
        echo "⚠ Warning: apply_all_patches.py not found, trying individual patches..."
        # Fallback to individual patches
        [ -f "patch_peakdetect.py" ] && python3 patch_peakdetect.py || true
        [ -f "patch_cellpose_models.py" ] && python3 patch_cellpose_models.py || true
        [ -f "patch_omnipose_model_url.py" ] && python3 patch_omnipose_model_url.py || true
    fi
else
    echo "WARNING: requirements.txt not found!"
fi

# Step 6: Verify PyQt5 installation
echo ""
echo "Verifying PyQt5 installation..."
python3 -c "import PyQt5; print('✓ PyQt5 imported successfully')" || {
    echo "✗ PyQt5 import failed. Installing PyQt5..."
    pip install "PyQt5==5.15.10"
    python3 -c "import PyQt5; print('✓ PyQt5 installed and imported')"
}

# Step 7: Check for cocoa plugin
echo ""
echo "Checking for Qt cocoa plugin..."
PLUGIN_PATH="venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins/platforms/libqcocoa.dylib"
if [ -f "$PLUGIN_PATH" ]; then
    echo "✓ Cocoa plugin found at: $PLUGIN_PATH"
else
    echo "✗ Cocoa plugin not found. Reinstalling PyQt5..."
    pip uninstall -y PyQt5 PyQt5-Qt5 PyQt5-sip 2>/dev/null || true
    pip install --no-cache-dir "PyQt5==5.15.10"
    if [ -f "$PLUGIN_PATH" ]; then
        echo "✓ Cocoa plugin now found"
    else
        echo "✗ WARNING: Cocoa plugin still not found after reinstall"
    fi
fi

# Step 8: Test PyQt5
echo ""
echo "Testing PyQt5 (this may take a moment)..."
TEST_OUTPUT=$(python3 -c "
import sys, os
script_dir = '$SCRIPT_DIR'
os.environ['QT_PLUGIN_PATH'] = script_dir + '/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins'
from PyQt5.QtCore import QCoreApplication
QCoreApplication.setLibraryPaths([os.environ['QT_PLUGIN_PATH']])
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)
print('SUCCESS')
" 2>&1)

if echo "$TEST_OUTPUT" | grep -q "SUCCESS"; then
    echo "✓ PyQt5 test passed!"
else
    echo "✗ PyQt5 test failed:"
    echo "$TEST_OUTPUT" | head -5
    echo ""
    echo "Running repair script..."
    if [ -f "repair_pyqt5.sh" ]; then
        ./repair_pyqt5.sh
    fi
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "You can now run: ./launch_gui.sh"
echo ""
