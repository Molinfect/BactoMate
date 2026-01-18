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

# Use venv Python directly instead of activation (more reliable across machines)
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
VENV_PYTHONW="$SCRIPT_DIR/venv/bin/pythonw"

# Verify venv Python exists and is executable
if [ ! -f "$VENV_PYTHON" ] && [ ! -f "$VENV_PYTHONW" ]; then
    echo "$(date): ERROR: Python interpreter not found in venv: $SCRIPT_DIR/venv/bin/" >> "$LOG_FILE"
    osascript -e 'display dialog "BactiMate Error: Python environment not found. The application may be corrupted." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
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
if ! "$PYTHON_CMD" --version >> "$LOG_FILE" 2>&1; then
    echo "$(date): ERROR: Python interpreter failed to run: $PYTHON_CMD" >> "$LOG_FILE"
    osascript -e 'display dialog "BactiMate Error: Python interpreter failed to start. The application may be corrupted or incompatible with this system." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
fi

# Set PYTHONHOME to venv to ensure Python uses the bundled environment
export PYTHONHOME="$SCRIPT_DIR/venv"

# Set PATH to prioritize venv binaries
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

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
    exec env PYTHONHOME="$PYTHONHOME" PATH="$PATH" QT_PLUGIN_PATH="$QT_PLUGIN_PATH" DYLD_FRAMEWORK_PATH="$DYLD_FRAMEWORK_PATH" QT_FRAMEWORK_PATH="$QT_FRAMEWORK_PATH" QT_MAC_WANTS_LAYER=1 QT_AUTO_SCREEN_SCALE_FACTOR=1 "$PYTHON_CMD" "$SCRIPT_DIR/omnipose_batch_gui.py"
else
    exec env PYTHONHOME="$PYTHONHOME" PATH="$PATH" "$PYTHON_CMD" "$SCRIPT_DIR/omnipose_batch_gui.py"
fi
