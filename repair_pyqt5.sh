#!/bin/bash
# Repair PyQt5 installation if cocoa plugin is missing or broken

cd "$(dirname "$0")"
source venv/bin/activate

SCRIPT_DIR="$(pwd)"
PLUGIN_PATH="$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins/platforms/libqcocoa.dylib"

echo "Checking PyQt5 installation..."

# First check if plugin file exists
if [ ! -f "$PLUGIN_PATH" ]; then
    echo "Cocoa plugin file not found. Reinstalling PyQt5..."
else
    # Test if PyQt5 can actually load the plugin (quick test)
    TEST_OUTPUT=$(python3 -c "
import sys, os
sys.path.insert(0, '.')
os.environ['QT_PLUGIN_PATH'] = '$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins'
from PyQt5.QtCore import QCoreApplication
QCoreApplication.setLibraryPaths(['$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins'])
from PyQt5.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
print('SUCCESS')
" 2>&1)
    
    if echo "$TEST_OUTPUT" | grep -q "SUCCESS"; then
        echo "✓ PyQt5 is working correctly"
        exit 0
    fi
    
    if echo "$TEST_OUTPUT" | grep -q "Could not find the Qt platform plugin"; then
        echo "PyQt5 plugin loading issue detected. Reinstalling..."
    else
        echo "PyQt5 test failed. Reinstalling..."
    fi
fi

echo "Reinstalling PyQt5 (this may take a minute)..."

# Reinstall PyQt5 - pin to a stable version that works on macOS
# Force reinstall to ensure clean installation
pip uninstall -y PyQt5 PyQt5-Qt5 PyQt5-sip 2>/dev/null
# Use a specific version that's known to work well on macOS
# Use --no-cache-dir to avoid corrupted cache issues
pip install --no-cache-dir "PyQt5==5.15.10" >/dev/null 2>&1 || pip install --no-cache-dir PyQt5 >/dev/null 2>&1

echo "Testing PyQt5 after repair..."

# Test again
TEST_OUTPUT=$(python3 -c "
import sys, os
sys.path.insert(0, '.')
os.environ['QT_PLUGIN_PATH'] = '$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins'
from PyQt5.QtCore import QCoreApplication
QCoreApplication.setLibraryPaths(['$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins'])
from PyQt5.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
print('SUCCESS')
" 2>&1)

if echo "$TEST_OUTPUT" | grep -q "SUCCESS"; then
    echo "✓ PyQt5 repaired successfully!"
    exit 0
else
    echo "✗ Repair failed. Error output:"
    echo "$TEST_OUTPUT" | grep -E "Error|Could not|failed" | head -3
    exit 1
fi
