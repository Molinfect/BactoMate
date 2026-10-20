#!/bin/bash
# Setup a completely fresh virtual environment for BactoMate (macOS source install).
#
# Requires plain CPython 3.11 (not Conda). requirements.txt and the editable install
# take numpy, scipy, and scikit-image from wheels only, so they never need a compiler.
# An existing venv is kept as venv.previous until the smoke test passes.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Re-run ourselves with the whole transcript (stdout + stderr) copied to Logs/install.log.
# A pipe (not process substitution) keeps the log ordered, including the final REASON line.
if [ -z "${BACTOMATE_INSTALL_LOGGED:-}" ]; then
    mkdir -p Logs
    export BACTOMATE_INSTALL_LOGGED=1
    set +e
    # tee -i survives Ctrl-C so the inner script's failure summary still reaches screen and log.
    bash "$SCRIPT_DIR/setup_fresh_env.sh" "$@" 2>&1 | tee -i Logs/install.log
    exit "${PIPESTATUS[0]}"
fi

VERIFY_PY="$SCRIPT_DIR/verify_python.py"
WHEEL_ONLY_ARGS=(--only-binary numpy,scipy,scikit-image)
WHEEL_REASON="pip could not install a wheel for numpy, scipy, or scikit-image"
RESTORE_CMD="rm -rf venv && mv venv.previous venv"

REASON=""
VENV_MOVED=0
INSTALL_OK=0

on_exit() {
    local status=$?
    trap - EXIT
    if [ "$INSTALL_OK" -eq 1 ] || [ "$status" -eq 0 ]; then
        exit "$status"
    fi
    echo ""
    echo "=========================================="
    echo "✗ BactoMate install FAILED"
    echo "=========================================="
    if [ "$VENV_MOVED" -eq 1 ]; then
        echo "Your previous environment is still on disk as venv.previous."
        echo "To restore it, run from $PROJECT_ROOT:"
        echo "  $RESTORE_CMD"
    fi
    echo "REASON: ${REASON:-install step failed (see the transcript above)}"
    exit "$status"
}
trap on_exit EXIT
trap 'REASON="interrupted"; exit 130' INT TERM

echo "=========================================="
echo "Setting up fresh BactoMate environment"
echo "=========================================="
echo ""

# Step 1: Find an acceptable Python (3.11, not Conda). No fallback to other versions.
echo "Looking for Python 3.11..."
PYTHON_CMD=""
CHECKED_PATHS=()
FIRST_VERIFY_ERROR=""
ALL_VERIFY_ERRORS=""

try_candidate() {
    local candidate_path="$1" output
    if output=$("$candidate_path" "$VERIFY_PY" 2>&1); then
        PYTHON_CMD="$output"
        return 0
    fi
    [ -n "$FIRST_VERIFY_ERROR" ] || FIRST_VERIFY_ERROR="$output"
    ALL_VERIFY_ERRORS+="[$candidate_path]"$'\n'"$output"$'\n'
    return 1
}

already_checked() {
    local candidate_path="$1" checked
    for checked in ${CHECKED_PATHS[@]+"${CHECKED_PATHS[@]}"}; do
        [ "$candidate_path" -ef "$checked" ] && return 0
    done
    return 1
}

# PATH first (respects the user's choice), then standard install locations so an
# active Conda env on PATH does not hide a python.org or Homebrew 3.11.
CANDIDATES=(
    "$(command -v python3.11 2>/dev/null || true)"
    "$(command -v python3 2>/dev/null || true)"
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
    /opt/homebrew/bin/python3.11
    /usr/local/bin/python3.11
)
for candidate in "${CANDIDATES[@]}"; do
    [ -n "$PYTHON_CMD" ] && break
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    already_checked "$candidate" && continue
    CHECKED_PATHS+=("$candidate")
    try_candidate "$candidate" || true
done

if [ -z "$PYTHON_CMD" ]; then
    echo "✗ No acceptable Python found."
    if [ -n "$ALL_VERIFY_ERRORS" ]; then
        printf '%s' "$ALL_VERIFY_ERRORS"
        REASON="$(printf '%s\n' "$FIRST_VERIFY_ERROR" | head -n 1) (checked: ${CHECKED_PATHS[*]})"
    else
        echo "Python 3.11 was not found on PATH or in the python.org / Homebrew install locations."
        echo "  Fix: Install plain Python 3.11 from https://www.python.org/downloads/"
        echo "       or run: brew install python@3.11"
        echo "  Lab users can instead download BactoMate.dmg from the GitHub release."
        REASON="Python 3.11 was not found"
    fi
    exit 1
fi

PYTHON_VERSION_STR="$("$PYTHON_CMD" --version 2>&1)"
echo "Using interpreter: $PYTHON_CMD"
echo "Interpreter version: $PYTHON_VERSION_STR"
echo "✓ Python accepted"

# Step 2: Protect the previous environment. Never delete a leftover venv.previous.
if [ -d "venv.previous" ]; then
    REASON="venv.previous already exists from an earlier run"
    echo ""
    echo "✗ A directory named venv.previous already exists (left over from an earlier run)."
    echo "  It may be your last working environment, so it will not be deleted automatically."
    echo "  Either restore it:       $RESTORE_CMD"
    echo "  or remove it yourself:   rm -rf venv.previous"
    echo "  Then run this script again."
    exit 1
fi

# Step 3: Keep the existing venv as venv.previous until the smoke test passes.
echo ""
if [ -d "venv" ]; then
    mv venv venv.previous
    VENV_MOVED=1
    echo "✓ Existing venv moved to venv.previous (kept until the smoke test passes)"
else
    echo "No existing venv found (that's okay)"
fi

# Step 4: Create new virtual environment
echo ""
echo "Creating new virtual environment..."
REASON="could not create the virtual environment with $PYTHON_CMD"
"$PYTHON_CMD" -m venv venv
echo "✓ New venv created"

# Step 5: Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

# Step 6: Pin pip (matches the Windows installer)
echo ""
echo "Pinning pip (>=24.2,<26)..."
REASON="could not install pip>=24.2,<26, setuptools, and wheel"
"$VENV_PYTHON" -m pip install "pip>=24.2,<26"
"$VENV_PYTHON" -m pip install setuptools wheel
echo "✓ pip pinned"

# Step 7: Install requirements (wheels only for the packages that compiled in lab logs)
echo ""
echo "Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    REASON="$WHEEL_REASON"
    if ! "$VENV_PYTHON" -m pip install "${WHEEL_ONLY_ARGS[@]}" -r requirements.txt; then
        echo ""
        echo "✗ A wheel was required for numpy, scipy, or scikit-image and none could be installed."
        echo "  Source builds are not attempted (they need a compiler and fail on newer Python)."
        echo "  Use plain Python 3.11 from https://www.python.org/downloads/ (or: brew install python@3.11),"
        echo "  or lab users can download BactoMate.dmg from the GitHub release."
        exit 1
    fi
    echo "✓ Requirements installed"
    REASON="requirements were installed but a post-install step failed"
    "$VENV_PYTHON" -c "import torch, torchvision; print(f'  PyTorch stack: torch {torch.__version__}, torchvision {torchvision.__version__}')"

    echo ""
    echo "Installing BactoMate in editable mode..."
    REASON="pip install -e . failed"
    if ! "$VENV_PYTHON" -m pip install "${WHEEL_ONLY_ARGS[@]}" -e .; then
        echo "✗ Editable install failed (wheel-only for numpy, scipy, scikit-image)."
        exit 1
    fi
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
    
    # Apply all patches automatically (shared script lives in setup/, not setup/macos/)
    echo ""
    echo "Applying compatibility patches..."
    PATCHES_PY="$PROJECT_ROOT/setup/apply_all_patches.py"
    if [ -f "$PATCHES_PY" ]; then
        python3 "$PATCHES_PY" || echo "⚠ Warning: Some patches may have failed"
    else
        echo "⚠ Warning: apply_all_patches.py not found at $PATCHES_PY"
    fi
else
    echo "WARNING: requirements.txt not found!"
fi

# Step 8: Verify PyQt6 installation
REASON="PyQt6 or macOS app integration packages could not be installed"
echo ""
echo "Verifying PyQt6 installation..."
python3 -c "import PyQt6; print('✓ PyQt6 imported successfully')" || {
    echo "✗ PyQt6 import failed. Installing PyQt6..."
    pip install "PyQt6>=6.6.0"
    python3 -c "import PyQt6; print('✓ PyQt6 installed and imported')"
}

# Step 9: macOS Dock/menu-bar integration (AppKit/Foundation via pyobjc)
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

# Step 10: Check for Qt cocoa plugin
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

# Step 11: Test PyQt6
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
REASON="smoke test failed (python main.py --smoke-test)"
source "$PROJECT_ROOT/venv/bin/activate"
python main.py --version
python main.py --smoke-test
echo "✓ Smoke test passed"
INSTALL_OK=1

# Smoke test passed: the previous environment is no longer needed.
# rm -rf on a large venv can take 10+ min, so delete in the background with nohup
# (survives script exit; output detached so the log pipe can close).
# Rename first: a half-deleted venv.previous would look like a restorable backup to a rerun.
if [ -d "venv.previous" ]; then
    TRASH_DIR="venv.deleting.$$"
    mv venv.previous "$TRASH_DIR"
    nohup rm -rf "$TRASH_DIR" >/dev/null 2>&1 </dev/null &
    disown 2>/dev/null || true
    echo "✓ Previous venv deleting in background (if $TRASH_DIR remains, run: rm -rf $TRASH_DIR)"
fi

mkdir -p Logs
: > Logs/.gitkeep 2>/dev/null || touch Logs/.gitkeep

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Verify anytime:  source venv/bin/activate && python main.py --smoke-test"
echo "Launch GUI:      ./launch_gui.sh   (from project root)"
echo "Install log:     Logs/install.log"
echo ""
