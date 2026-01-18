#!/bin/bash
# Launch the Omnipose Batch Processing GUI

# Redirect stderr to log file for debugging
LOG_DIR="$HOME/Library/Logs/BactiMate"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch.log"
exec 2>> "$LOG_FILE"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Log script directory for debugging
echo "$(date): Launch script directory: $SCRIPT_DIR" >> "$LOG_FILE"
echo "$(date): Looking for Python in: $SCRIPT_DIR/venv/bin/" >> "$LOG_FILE"

# Use venv Python directly instead of activation (more reliable across machines)
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
VENV_PYTHONW="$SCRIPT_DIR/venv/bin/pythonw"

# Debug: List what's in venv/bin
if [ -d "$SCRIPT_DIR/venv/bin" ]; then
    echo "$(date): Contents of venv/bin:" >> "$LOG_FILE"
    ls -la "$SCRIPT_DIR/venv/bin" | head -20 >> "$LOG_FILE" 2>&1 || true
else
    echo "$(date): ERROR: venv/bin directory does not exist: $SCRIPT_DIR/venv/bin" >> "$LOG_FILE"
fi

# Verify venv Python exists and is executable
if [ ! -f "$VENV_PYTHON" ] && [ ! -f "$VENV_PYTHONW" ]; then
    echo "$(date): ERROR: Python interpreter not found in venv: $SCRIPT_DIR/venv/bin/" >> "$LOG_FILE"
    echo "$(date): VENV_PYTHON=$VENV_PYTHON" >> "$LOG_FILE"
    echo "$(date): VENV_PYTHONW=$VENV_PYTHONW" >> "$LOG_FILE"
    osascript -e 'display dialog "BactiMate Error: Python environment not found. The application may be corrupted. Check ~/Library/Logs/BactiMate/launch.log for details." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
fi

# Use pythonw if available, otherwise python3
if [ -f "$VENV_PYTHONW" ] && [ -x "$VENV_PYTHONW" ]; then
    PYTHON_CMD="$VENV_PYTHONW"
elif [ -f "$VENV_PYTHON" ] && [ -x "$VENV_PYTHON" ]; then
    PYTHON_CMD="$VENV_PYTHON"
else
    echo "$(date): ERROR: Python interpreter is not executable: $VENV_PYTHON" >> "$LOG_FILE"
    osascript -e 'display dialog "BactiMate Error: Python interpreter is not executable. The application may be corrupted." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
fi

# Test that Python actually works
echo "$(date): Testing Python executable: $PYTHON_CMD" >> "$LOG_FILE"
echo "$(date): Python file info:" >> "$LOG_FILE"
ls -la "$PYTHON_CMD" >> "$LOG_FILE" 2>&1 || true
file "$PYTHON_CMD" >> "$LOG_FILE" 2>&1 || true

# Try to run Python and capture both stdout and stderr
PYTHON_TEST_OUTPUT=$("$PYTHON_CMD" --version 2>&1)
PYTHON_TEST_EXIT=$?

if [ $PYTHON_TEST_EXIT -ne 0 ]; then
    echo "$(date): ERROR: Python interpreter failed to run: $PYTHON_CMD" >> "$LOG_FILE"
    echo "$(date): Exit code: $PYTHON_TEST_EXIT" >> "$LOG_FILE"
    echo "$(date): Error output: $PYTHON_TEST_OUTPUT" >> "$LOG_FILE"
    
    # Try to get more diagnostic info
    echo "$(date): Attempting to diagnose Python issue..." >> "$LOG_FILE"
    echo "$(date): Checking Python architecture..." >> "$LOG_FILE"
    arch >> "$LOG_FILE" 2>&1 || true
    echo "$(date): System architecture:" >> "$LOG_FILE"
    uname -m >> "$LOG_FILE" 2>&1 || true
    
    # Check if it's a code signing issue
    if command -v codesign >/dev/null 2>&1; then
        echo "$(date): Checking code signature..." >> "$LOG_FILE"
        codesign -dv --verbose=4 "$PYTHON_CMD" >> "$LOG_FILE" 2>&1 || true
    fi
    
    # Check for missing libraries
    if command -v otool >/dev/null 2>&1; then
        echo "$(date): Checking linked libraries..." >> "$LOG_FILE"
        otool -L "$PYTHON_CMD" | head -10 >> "$LOG_FILE" 2>&1 || true
    fi
    
    osascript -e 'display dialog "BactiMate Error: Python interpreter failed to start. The application may be corrupted or incompatible with this system. Check ~/Library/Logs/BactiMate/launch.log for details." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
else
    echo "$(date): Python test successful: $PYTHON_TEST_OUTPUT" >> "$LOG_FILE"
fi

# Set PATH to prioritize venv binaries
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

# Only set PYTHONHOME for app bundles (when running from .app bundle)
# For regular venv usage, don't set PYTHONHOME - let Python auto-detect the base installation
# This is needed for relocatable app bundles but breaks regular venv usage
if [[ "$SCRIPT_DIR" == *".app/Contents/Resources"* ]]; then
    export PYTHONHOME="$SCRIPT_DIR/venv"
else
    # Don't set PYTHONHOME for regular venv - Python will find base installation automatically
    unset PYTHONHOME 2>/dev/null || true
fi

# Verify patches are applied (quick check, auto-apply if needed)
# Note: SCRIPT_DIR is already set above
if [ -f "$SCRIPT_DIR/apply_all_patches.py" ]; then
    # Quick check: verify cellpose_omni has the model_type fix
    "$PYTHON_CMD" -c "
import sys
try:
    from cellpose_omni import models
    import inspect
    src = inspect.getsourcefile(models)
    content = open(src).read()
    has_fix = 'model_type=model_type' in content and 'self.cp = CellposeModel' in content
    has_github = 'raw.githubusercontent.com/kevinjohncutler/omnipose-models/main' in content
    if not has_fix or not has_github:
        sys.exit(1)
except Exception:
    sys.exit(1)
" 2>> "$LOG_FILE" || {
        echo "$(date): Applying compatibility patches..." >> "$LOG_FILE"
        "$PYTHON_CMD" "$SCRIPT_DIR/apply_all_patches.py" >> "$LOG_FILE" 2>&1 || true
    }
fi

# Set Qt environment variables for macOS
export QT_MAC_WANTS_LAYER=1
export QT_AUTO_SCREEN_SCALE_FACTOR=1

# Set framework path for Qt libraries (needed for plugins to find Qt frameworks)
# SCRIPT_DIR is already set above

# Detect Python version dynamically using venv Python (works with 3.10, 3.11, etc.)
PYTHON_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>> "$LOG_FILE" || echo "python3.11")

QT_LIB_DIR="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/lib"
if [ -d "$QT_LIB_DIR" ]; then
    export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR:$DYLD_FRAMEWORK_PATH"
    export QT_FRAMEWORK_PATH="$QT_LIB_DIR"
fi

# Find Qt platform plugins in venv - use absolute path with detected Python version
VENV_QT_PLUGINS="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
if [ -d "$VENV_QT_PLUGINS" ]; then
    export QT_PLUGIN_PATH="$VENV_QT_PLUGINS"
    # Also set QML2_IMPORT_PATH if needed
    export QML2_IMPORT_PATH="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/qml"
    # Verify the cocoa plugin exists
    if [ ! -f "$VENV_QT_PLUGINS/platforms/libqcocoa.dylib" ]; then
        echo "$(date): WARNING: Cocoa plugin not found. Running repair script..." >> "$LOG_FILE"
        "$SCRIPT_DIR/repair_pyqt6.sh" >> "$LOG_FILE" 2>&1 || true
        # Re-export after potential repair
        export QT_PLUGIN_PATH="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
    fi
else
    echo "$(date): WARNING: Qt plugins directory not found. Running repair script..." >> "$LOG_FILE"
    "$SCRIPT_DIR/repair_pyqt6.sh" >> "$LOG_FILE" 2>&1 || true
    # Re-export after potential repair
    export QT_PLUGIN_PATH="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
fi

# Verify PyQt6 is installed and plugin exists
# Check both import and plugin file existence
PYQT6_OK=false
if "$PYTHON_CMD" -c "import PyQt6" 2>> "$LOG_FILE"; then
    if [ -f "$VENV_QT_PLUGINS/platforms/libqcocoa.dylib" ]; then
        PYQT6_OK=true
    fi
fi

if [ "$PYQT6_OK" = false ]; then
    echo "$(date): PyQt6 issue detected. Running automatic repair..." >> "$LOG_FILE"
    if "$SCRIPT_DIR/repair_pyqt6.sh" >> "$LOG_FILE" 2>&1; then
        # Re-export after repair
        export QT_PLUGIN_PATH="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
        export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR:$DYLD_FRAMEWORK_PATH"
        echo "$(date): Repair completed successfully." >> "$LOG_FILE"
    else
        echo "$(date): Automatic repair failed." >> "$LOG_FILE"
        osascript -e 'display dialog "BactiMate Error: PyQt6 installation is corrupted and cannot be repaired automatically. Please reinstall the application." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
        exit 1
    fi
fi

# Ensure all environment variables are set before launching
# Re-export to make absolutely sure they're set (using detected Python version)
export QT_PLUGIN_PATH="$SCRIPT_DIR/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
if [ -d "$QT_LIB_DIR" ]; then
    # Remove any system Qt paths from DYLD_FRAMEWORK_PATH to avoid conflicts
    if [ -n "$DYLD_FRAMEWORK_PATH" ]; then
        # Filter out system Qt paths (Homebrew/conda)
        CLEANED_DYLD=$(echo "$DYLD_FRAMEWORK_PATH" | tr ':' '\n' | grep -v -E '(homebrew|anaconda|miniconda|/usr/local)' | tr '\n' ':' | sed 's/:$//')
        if [ -n "$CLEANED_DYLD" ]; then
            export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR:$CLEANED_DYLD"
        else
            export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR"
        fi
    else
        export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR"
    fi
    export QT_FRAMEWORK_PATH="$QT_LIB_DIR"
fi
export QT_MAC_WANTS_LAYER=1
export QT_AUTO_SCREEN_SCALE_FACTOR=1

# Clear any conflicting Qt environment variables
unset QT_QPA_PLATFORM_PLUGIN_PATH 2>/dev/null || true

# On macOS, try to use pythonw if available, otherwise use python3
# Pass environment explicitly to ensure Qt finds plugins
# Log the launch attempt
echo "$(date): Starting Python with QT_PLUGIN_PATH=$QT_PLUGIN_PATH" >> "$LOG_FILE"

# Launch the application
echo "$(date): Using Python: $PYTHON_CMD" >> "$LOG_FILE"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Only pass PYTHONHOME if it's set (for app bundles)
    if [ -n "$PYTHONHOME" ]; then
        exec env PYTHONHOME="$PYTHONHOME" PATH="$PATH" QT_PLUGIN_PATH="$QT_PLUGIN_PATH" DYLD_FRAMEWORK_PATH="$DYLD_FRAMEWORK_PATH" QT_FRAMEWORK_PATH="$QT_FRAMEWORK_PATH" QT_MAC_WANTS_LAYER=1 QT_AUTO_SCREEN_SCALE_FACTOR=1 "$PYTHON_CMD" "$SCRIPT_DIR/omnipose_batch_gui.py"
    else
        exec env PATH="$PATH" QT_PLUGIN_PATH="$QT_PLUGIN_PATH" DYLD_FRAMEWORK_PATH="$DYLD_FRAMEWORK_PATH" QT_FRAMEWORK_PATH="$QT_FRAMEWORK_PATH" QT_MAC_WANTS_LAYER=1 QT_AUTO_SCREEN_SCALE_FACTOR=1 "$PYTHON_CMD" "$SCRIPT_DIR/omnipose_batch_gui.py"
    fi
else
    # Only pass PYTHONHOME if it's set (for app bundles)
    if [ -n "$PYTHONHOME" ]; then
        exec env PYTHONHOME="$PYTHONHOME" PATH="$PATH" "$PYTHON_CMD" "$SCRIPT_DIR/omnipose_batch_gui.py"
    else
        exec env PATH="$PATH" "$PYTHON_CMD" "$SCRIPT_DIR/omnipose_batch_gui.py"
    fi
fi
