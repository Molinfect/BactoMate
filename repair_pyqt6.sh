#!/bin/bash
# Repair PyQt6 installation if cocoa plugin is missing or broken
# NOTE: This is a LAST RESORT repair mechanism. If the bundle is built correctly
# (Qt libs present, signed, unquarantined, plugin paths correct), this should
# never need to run on user machines.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Use explicit venv Python and pip paths (deterministic, like launcher)
# Fallback to python if python3 doesn't exist (shouldn't happen in venv)
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
if [ ! -x "$VENV_PYTHON" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
fi

VENV_PIP="$SCRIPT_DIR/venv/bin/pip"
if [ ! -x "$VENV_PIP" ]; then
    VENV_PIP="$SCRIPT_DIR/venv/bin/pip3"
fi

# Verify venv Python exists
if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: Python interpreter not found in venv: $VENV_PYTHON"
    exit 1
fi

# Detect Python version dynamically (will recompute after reinstall)
detect_python_version() {
    "$VENV_PYTHON" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "python3.11"
}

PYTHON_VERSION=$(detect_python_version)
PLUGIN_PATH="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins/platforms/libqcocoa.dylib"

echo "Checking PyQt6 installation..."

# First check if plugin file exists
if [ ! -f "$PLUGIN_PATH" ]; then
    echo "Cocoa plugin file not found. Reinstalling PyQt6..."
    NEEDS_REINSTALL=1
else
    # Test if PyQt6 can actually load the plugin (quick test)
    TEST_OUTPUT=$("$VENV_PYTHON" -c "
import sys, os
sys.path.insert(0, '.')
os.environ['QT_PLUGIN_PATH'] = '$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins'
from PyQt6.QtCore import QCoreApplication
QCoreApplication.setLibraryPaths(['$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins'])
from PyQt6.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
print('SUCCESS')
" 2>&1)
    
    if echo "$TEST_OUTPUT" | grep -q "SUCCESS"; then
        echo "✓ PyQt6 is working correctly"
        exit 0
    fi
    
    if echo "$TEST_OUTPUT" | grep -q "Could not find the Qt platform plugin"; then
        echo "PyQt6 plugin loading issue detected. Reinstalling..."
        NEEDS_REINSTALL=1
    else
        echo "PyQt6 test failed. Reinstalling..."
        NEEDS_REINSTALL=1
    fi
fi

if [ "${NEEDS_REINSTALL:-0}" = "1" ]; then
    # Check network availability before attempting install
    echo "WARNING: About to perform network installation of PyQt6."
    echo "This requires internet connectivity and may take several minutes."
    
    # Quick network check (try to reach PyPI)
    if ! "$VENV_PYTHON" -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=3)" 2>/dev/null; then
        echo "ERROR: Network connection unavailable or blocked."
        echo "Cannot repair PyQt6 without network access."
        echo "Please check your internet connection and try again."
        exit 1
    fi
    
    echo "Reinstalling PyQt6 (this may take a minute)..."
    
    # Reinstall PyQt6 - use latest stable version
    # Force reinstall to ensure clean installation
    if ! "$VENV_PIP" uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip 2>/dev/null; then
        echo "WARNING: Failed to uninstall existing PyQt6 packages (may not exist)"
    fi
    
    # Use --no-cache-dir to avoid corrupted cache issues
    if ! "$VENV_PIP" install --no-cache-dir "PyQt6>=6.6.0" 2>&1; then
        echo "WARNING: Failed to install PyQt6>=6.6.0, trying latest version..."
        if ! "$VENV_PIP" install --no-cache-dir PyQt6 2>&1; then
            echo "ERROR: Failed to install PyQt6. Network may be unavailable or PyPI may be unreachable."
            exit 1
        fi
    fi
    
    # CRITICAL: Recompute Python version after reinstall (may have changed)
    PYTHON_VERSION=$(detect_python_version)
    PLUGIN_PATH="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins/platforms/libqcocoa.dylib"
    
    echo "Testing PyQt6 after repair..."
    
    # Test again
    TEST_OUTPUT=$("$VENV_PYTHON" -c "
import sys, os
sys.path.insert(0, '.')
os.environ['QT_PLUGIN_PATH'] = '$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins'
from PyQt6.QtCore import QCoreApplication
QCoreApplication.setLibraryPaths(['$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins'])
from PyQt6.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
print('SUCCESS')
" 2>&1)
    
    if echo "$TEST_OUTPUT" | grep -q "SUCCESS"; then
        echo "✓ PyQt6 repaired successfully!"
        exit 0
    else
        echo "✗ Repair failed. Error output:"
        echo "$TEST_OUTPUT" | grep -E "Error|Could not|failed" | head -3
        exit 1
    fi
fi
