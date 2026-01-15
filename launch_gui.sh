#!/bin/bash
# Launch the Omnipose Batch Processing GUI

cd "$(dirname "$0")"
source venv/bin/activate

# Verify patches are applied (quick check, auto-apply if needed)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/apply_all_patches.py" ]; then
    # Quick check: verify cellpose_omni has the model_type fix
    python3 -c "
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
" 2>/dev/null || {
        echo "Applying compatibility patches..." >&2
        python3 "$SCRIPT_DIR/apply_all_patches.py" >/dev/null 2>&1 || true
    }
fi

# Set Qt environment variables for macOS
export QT_MAC_WANTS_LAYER=1
export QT_AUTO_SCREEN_SCALE_FACTOR=1

# Set framework path for Qt libraries (needed for plugins to find Qt frameworks)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QT_LIB_DIR="$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/lib"
if [ -d "$QT_LIB_DIR" ]; then
    export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR:$DYLD_FRAMEWORK_PATH"
    export QT_FRAMEWORK_PATH="$QT_LIB_DIR"
fi

# Find Qt platform plugins in venv - use absolute path
VENV_QT_PLUGINS="$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins"
if [ -d "$VENV_QT_PLUGINS" ]; then
    export QT_PLUGIN_PATH="$VENV_QT_PLUGINS"
    # Also set QML2_IMPORT_PATH if needed
    export QML2_IMPORT_PATH="$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/qml"
    # Verify the cocoa plugin exists
    if [ ! -f "$VENV_QT_PLUGINS/platforms/libqcocoa.dylib" ]; then
        echo "WARNING: Cocoa plugin not found. Running repair script..." >&2
        "$SCRIPT_DIR/repair_pyqt5.sh"
        # Re-export after potential repair
        export QT_PLUGIN_PATH="$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins"
    fi
else
    echo "WARNING: Qt plugins directory not found. Running repair script..." >&2
    "$SCRIPT_DIR/repair_pyqt5.sh"
    # Re-export after potential repair
    export QT_PLUGIN_PATH="$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins"
fi

# Verify PyQt5 is installed and plugin exists
# Check both import and plugin file existence
PYQT5_OK=false
if python3 -c "import PyQt5" 2>/dev/null; then
    if [ -f "$VENV_QT_PLUGINS/platforms/libqcocoa.dylib" ]; then
        PYQT5_OK=true
    fi
fi

if [ "$PYQT5_OK" = false ]; then
    echo "PyQt5 issue detected. Running automatic repair..." >&2
    if "$SCRIPT_DIR/repair_pyqt5.sh" >/dev/null 2>&1; then
        # Re-export after repair
        export QT_PLUGIN_PATH="$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins"
        export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR:$DYLD_FRAMEWORK_PATH"
        echo "Repair completed successfully." >&2
    else
        echo "Automatic repair failed. Please run: ./repair_pyqt5.sh manually" >&2
        exit 1
    fi
fi

# Ensure all environment variables are set before launching
# Re-export to make absolutely sure they're set
export QT_PLUGIN_PATH="$SCRIPT_DIR/venv/lib/python3.9/site-packages/PyQt5/Qt5/plugins"
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
if [[ "$OSTYPE" == "darwin"* ]]; then
    PYTHONW="$SCRIPT_DIR/venv/bin/pythonw"
    PYTHON3="$SCRIPT_DIR/venv/bin/python3"
    if [ -f "$PYTHONW" ]; then
        exec env QT_PLUGIN_PATH="$QT_PLUGIN_PATH" DYLD_FRAMEWORK_PATH="$DYLD_FRAMEWORK_PATH" QT_FRAMEWORK_PATH="$QT_FRAMEWORK_PATH" QT_MAC_WANTS_LAYER=1 QT_AUTO_SCREEN_SCALE_FACTOR=1 "$PYTHONW" "$SCRIPT_DIR/omnipose_batch_gui.py"
    else
        # Fallback to python3 with proper environment
        exec env QT_PLUGIN_PATH="$QT_PLUGIN_PATH" DYLD_FRAMEWORK_PATH="$DYLD_FRAMEWORK_PATH" QT_FRAMEWORK_PATH="$QT_FRAMEWORK_PATH" QT_MAC_WANTS_LAYER=1 QT_AUTO_SCREEN_SCALE_FACTOR=1 "$PYTHON3" "$SCRIPT_DIR/omnipose_batch_gui.py"
    fi
else
    exec "$SCRIPT_DIR/venv/bin/python3" "$SCRIPT_DIR/omnipose_batch_gui.py"
fi
