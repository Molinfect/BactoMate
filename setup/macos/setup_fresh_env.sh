#!/bin/bash
# Setup a completely fresh virtual environment for BactoMate

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Setting up fresh BactoMate environment"
echo "=========================================="
echo ""

# Step 1: Remove old virtual environment
# rm -rf on a large venv can take 10+ min. Rename then delete in background with nohup so it survives script exit.
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    if [ -d "venv.old" ]; then
        nohup rm -rf venv.old >/dev/null 2>&1 &
        disown 2>/dev/null || true
    fi
    mv venv venv.old
    nohup rm -rf venv.old >/dev/null 2>&1 &
    disown 2>/dev/null || true
    echo "✓ Old venv moved to venv.old (deleting in background; if venv.old remains, run: rm -rf venv.old)"
else
    echo "No existing venv found (that's okay)"
fi

# Step 2: Create new virtual environment (need Python 3.10+ for trackastra and others)
echo ""
echo "Creating new virtual environment..."
PYTHON_CMD=""
for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>/dev/null) || continue
        major=$(echo "$ver" | cut -d' ' -f1)
        minor=$(echo "$ver" | cut -d' ' -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$candidate"
            echo "Using: $candidate ($($candidate --version 2>&1))"
            break
        fi
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "✗ Error: BactoMate requires Python 3.10 or 3.11."
    echo "  trackastra and other dependencies do not support Python 3.9."
    echo "  Install Python 3.11 from https://www.python.org/downloads/ or run: brew install python@3.11"
    echo "  Then run this script again (or use: python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt)"
    exit 1
fi
"$PYTHON_CMD" -m venv venv
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
    python3 -c "import torch, torchvision; print(f'  PyTorch stack: torch {torch.__version__}, torchvision {torchvision.__version__}')"

    echo ""
    echo "Installing BactoMate in editable mode..."
    pip install -e .
    echo "✓ Editable install complete (bactomate CLI available in venv)"
    
    # Check for NVIDIA GPU and offer CUDA installation
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "NVIDIA GPU detected!"
        read -p "Would you like to install PyTorch with CUDA support for GPU acceleration? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [ -f "$SCRIPT_DIR/install_cuda_pytorch.sh" ]; then
                echo "Installing PyTorch with CUDA..."
                bash "$SCRIPT_DIR/install_cuda_pytorch.sh"
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
    if [ -f "$SCRIPT_DIR/apply_all_patches.py" ]; then
        python3 "$SCRIPT_DIR/apply_all_patches.py" || echo "⚠ Warning: Some patches may have failed"
    else
        echo "⚠ Warning: apply_all_patches.py not found in setup/"
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

# Step 7: macOS Dock/menu-bar integration (AppKit/Foundation via pyobjc)
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo ""
    echo "Checking for macOS app integration dependencies..."
    if ! python3 -c "import AppKit, Foundation" 2>/dev/null; then
        echo "Installing pyobjc-framework-Cocoa..."
        python3 -m pip install "pyobjc-framework-Cocoa"
    fi
    python3 -c "import AppKit, Foundation; print('✓ AppKit/Foundation available')" || {
        echo "✗ AppKit/Foundation import failed after install."
        exit 1
    }
fi

# Step 8: Check for Qt cocoa plugin
echo ""
echo "Checking for Qt cocoa plugin..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "python3.11")
PLUGIN_PATH="$PROJECT_ROOT/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins/platforms/libqcocoa.dylib"
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

# Step 9: Test PyQt6
echo ""
echo "Testing PyQt6 (this may take a moment)..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "python3.11")
TEST_OUTPUT=$(python3 -c "
import sys, os
project_root = '$PROJECT_ROOT'
os.environ['QT_PLUGIN_PATH'] = project_root + '/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins'
from PyQt6.QtCore import QCoreApplication
QCoreApplication.setLibraryPaths([os.environ.get('QT_PLUGIN_PATH', '')])
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
    if [ -f "$SCRIPT_DIR/repair_pyqt6.sh" ]; then
        bash "$SCRIPT_DIR/repair_pyqt6.sh"
    fi
fi

echo ""
echo "Running install verification..."
source "$PROJECT_ROOT/venv/bin/activate"
python main.py --version
python main.py --smoke-test
echo "✓ Smoke test passed"

mkdir -p Logs
: > Logs/.gitkeep 2>/dev/null || touch Logs/.gitkeep

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Verify anytime:  source venv/bin/activate && python main.py --smoke-test"
echo "Launch GUI:      ./launch_gui.sh   (from project root)"
echo ""
