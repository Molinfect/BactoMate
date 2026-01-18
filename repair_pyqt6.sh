#!/bin/bash
# Repair PyQt6 installation if cocoa plugin is missing or broken

cd "$(dirname "$0")"
source venv/bin/activate

SCRIPT_DIR="$(pwd)"
# Detect Python version dynamically
PYTHON_VERSION=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "python3.11")
PLUGIN_PATH="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins/platforms/libqcocoa.dylib"

echo "Checking PyQt6 installation..."

# First check if plugin file exists
if [ ! -f "$PLUGIN_PATH" ]; then
    echo "Cocoa plugin file not found. Reinstalling PyQt6..."
else
    # Test if PyQt6 can actually load the plugin (quick test)
    TEST_OUTPUT=$(python3 -c "
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
    else
        echo "PyQt6 test failed. Reinstalling..."
    fi
fi

echo "Reinstalling PyQt6 (this may take a minute)..."

# Reinstall PyQt6 - use latest stable version
# Force reinstall to ensure clean installation
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip 2>/dev/null
# Use --no-cache-dir to avoid corrupted cache issues
pip install --no-cache-dir "PyQt6>=6.6.0" >/dev/null 2>&1 || pip install --no-cache-dir PyQt6 >/dev/null 2>&1

echo "Testing PyQt6 after repair..."

# Test again
TEST_OUTPUT=$(python3 -c "
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
