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
    
    # Check for NVIDIA GPU and offer CUDA installation
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "NVIDIA GPU detected!"
        read -p "Would you like to install PyTorch with CUDA support for GPU acceleration? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [ -f "install_cuda_pytorch.sh" ]; then
                echo "Installing PyTorch with CUDA..."
                bash install_cuda_pytorch.sh
            else
                echo "Installing PyTorch with CUDA (default: cu118)..."
                pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                echo "✓ CUDA-enabled PyTorch installed"
            fi
        fi
    fi
    
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

# Step 6: Verify PyQt6 installation
echo ""
echo "Verifying PyQt6 installation..."
python3 -c "import PyQt6; print('✓ PyQt6 imported successfully')" || {
    echo "✗ PyQt6 import failed. Installing PyQt6..."
    pip install "PyQt6>=6.6.0"
    python3 -c "import PyQt6; print('✓ PyQt6 installed and imported')"
}

# Step 7: Check for cocoa plugin
echo ""
echo "Checking for Qt cocoa plugin..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "python3.11")
PLUGIN_PATH="venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins/platforms/libqcocoa.dylib"
if [ -f "$PLUGIN_PATH" ]; then
    echo "✓ Cocoa plugin found at: $PLUGIN_PATH"
else
    echo "✗ Cocoa plugin not found. Reinstalling PyQt6..."
    pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip 2>/dev/null || true
    pip install --no-cache-dir "PyQt6>=6.6.0"
    if [ -f "$PLUGIN_PATH" ]; then
        echo "✓ Cocoa plugin now found"
    else
        echo "✗ WARNING: Cocoa plugin still not found after reinstall"
    fi
fi

# Step 8: Test PyQt6
echo ""
echo "Testing PyQt6 (this may take a moment)..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "python3.11")
TEST_OUTPUT=$(python3 -c "
import sys, os
script_dir = '$SCRIPT_DIR'
os.environ['QT_PLUGIN_PATH'] = script_dir + '/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins'
from PyQt6.QtCore import QCoreApplication
QCoreApplication.setLibraryPaths([os.environ['QT_PLUGIN_PATH']])
from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)
print('SUCCESS')
" 2>&1)

if echo "$TEST_OUTPUT" | grep -q "SUCCESS"; then
    echo "✓ PyQt6 test passed!"
else
    echo "✗ PyQt6 test failed:"
    echo "$TEST_OUTPUT" | head -5
    echo ""
    echo "Running repair script..."
    if [ -f "repair_pyqt6.sh" ]; then
        ./repair_pyqt6.sh
    fi
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "You can now run: ./launch_gui.sh"
echo ""
