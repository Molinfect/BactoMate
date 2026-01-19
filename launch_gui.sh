#!/bin/bash
# Launch the Omnipose Batch Processing GUI

# CRITICAL: Unset PYTHONHOME immediately - it breaks venv usage
# Venvs don't contain the full standard library, so PYTHONHOME must point to base Python
# We'll only set it later if we're in an app bundle with a bundled framework
# Log initial PYTHONHOME value before unsetting (for debugging)
INITIAL_PYTHONHOME="${PYTHONHOME:-<not set>}"

# Redirect stderr to log file for debugging
LOG_DIR="$HOME/Library/Logs/BactiMate"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch.log"
exec 2>> "$LOG_FILE"

# Now unset PYTHONHOME and log it
unset PYTHONHOME 2>/dev/null || true
echo "$(date): [DEBUG] Initial PYTHONHOME was: $INITIAL_PYTHONHOME" >> "$LOG_FILE"
echo "$(date): [DEBUG] PYTHONHOME after unset: ${PYTHONHOME:-<unset>}" >> "$LOG_FILE"

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
    
    # Log Python system information for debugging cross-machine issues
    echo "$(date): Python system information:" >> "$LOG_FILE"
    "$PYTHON_CMD" -c "import sys; import platform; print(f'  executable: {sys.executable}'); print(f'  prefix: {sys.prefix}'); print(f'  base_prefix: {sys.base_prefix}'); print(f'  version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); print(f'  platform: {platform.platform()}')" >> "$LOG_FILE" 2>&1 || true
fi

# Set PATH to prioritize venv binaries
export PATH="$SCRIPT_DIR/venv/bin:$PATH"

# Detect Python version from venv (for matching framework version)
PYTHON_MAJOR_MINOR=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.11")

# Only set PYTHONHOME for app bundles (when running from .app bundle)
# Note: PYTHONHOME is already unset at the start of this script
# For regular venv usage, don't set PYTHONHOME - let Python auto-detect the base installation
# This is needed for relocatable app bundles but breaks regular venv usage
if [[ "$SCRIPT_DIR" == *".app/Contents/Resources"* ]]; then
    # Add bundled Python framework to library search path for self-contained bundles
    APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    BUNDLED_FRAMEWORK="$APP_DIR/Contents/Frameworks/Python.framework"
    if [ -d "$BUNDLED_FRAMEWORK" ]; then
        # Framework is bundled - set PYTHONHOME to the framework's Python installation
        # Find the Python version in the framework that matches the venv Python version
        # Prefer Versions/Current if present, otherwise match major.minor version
        FRAMEWORK_PYTHON_DIR=""
        if [ -L "$BUNDLED_FRAMEWORK/Versions/Current" ]; then
            # Use Current symlink if available
            FRAMEWORK_PYTHON_DIR="$BUNDLED_FRAMEWORK/Versions/Current"
            echo "$(date): Found Versions/Current symlink in framework" >> "$LOG_FILE"
        else
            # Try to match the detected Python version (e.g., 3.11)
            POSSIBLE_VERSION_DIR="$BUNDLED_FRAMEWORK/Versions/${PYTHON_MAJOR_MINOR}"
            if [ -d "$POSSIBLE_VERSION_DIR" ]; then
                FRAMEWORK_PYTHON_DIR="$POSSIBLE_VERSION_DIR"
                echo "$(date): Found matching framework version: ${PYTHON_MAJOR_MINOR}" >> "$LOG_FILE"
            else
                # Fallback: try to find the best matching version
                # Prefer versions that match major version, then closest minor version
                AVAILABLE_VERSIONS=$(ls "$BUNDLED_FRAMEWORK/Versions" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+$' | sort -V || echo "")
                if [ -n "$AVAILABLE_VERSIONS" ]; then
                    # Extract major version from detected Python (e.g., "3" from "3.11")
                    DETECTED_MAJOR=$(echo "$PYTHON_MAJOR_MINOR" | cut -d'.' -f1)
                    # Try to find version matching major version first
                    MATCHING_MAJOR=$(echo "$AVAILABLE_VERSIONS" | grep "^${DETECTED_MAJOR}\." | sort -V | tail -1 || echo "")
                    if [ -n "$MATCHING_MAJOR" ]; then
                        FRAMEWORK_PYTHON_DIR="$BUNDLED_FRAMEWORK/Versions/${MATCHING_MAJOR}"
                        echo "$(date): Using framework version with matching major version: ${MATCHING_MAJOR} (venv is ${PYTHON_MAJOR_MINOR})" >> "$LOG_FILE"
                    else
                        # Last resort: use the highest available version
                        HIGHEST_VERSION=$(echo "$AVAILABLE_VERSIONS" | sort -V | tail -1)
                        FRAMEWORK_PYTHON_DIR="$BUNDLED_FRAMEWORK/Versions/${HIGHEST_VERSION}"
                        echo "$(date): WARNING: No matching major version found, using highest available: ${HIGHEST_VERSION} (venv is ${PYTHON_MAJOR_MINOR})" >> "$LOG_FILE"
                        echo "$(date): This may cause compatibility issues if versions differ significantly" >> "$LOG_FILE"
                    fi
                fi
            fi
        fi
        
        if [ -n "$FRAMEWORK_PYTHON_DIR" ] && [ -d "$FRAMEWORK_PYTHON_DIR" ]; then
            # CRITICAL: For standalone bundles, venvs are NOT self-contained - they depend on base Python.
            # This only works because:
            # 1. We bundle the full Python framework (contains complete stdlib)
            # 2. We set PYTHONHOME to point at the bundled framework (not the venv)
            # 3. We set DYLD_FRAMEWORK_PATH so Python can find the bundled framework libraries
            # 4. The venv Python executable is linked to use @rpath (fixed during build)
            
            # Set PYTHONHOME to the framework's Python installation (contains standard library)
            export PYTHONHOME="$FRAMEWORK_PYTHON_DIR"
            # CRITICAL: Add framework to DYLD_FRAMEWORK_PATH FIRST so Python can find it
            # This must be set before Python runs, otherwise it will look for the original hardcoded paths
            if [ -n "$DYLD_FRAMEWORK_PATH" ]; then
                export DYLD_FRAMEWORK_PATH="$BUNDLED_FRAMEWORK:$DYLD_FRAMEWORK_PATH"
            else
                export DYLD_FRAMEWORK_PATH="$BUNDLED_FRAMEWORK"
            fi
            echo "$(date): Using bundled Python framework: $BUNDLED_FRAMEWORK" >> "$LOG_FILE"
            echo "$(date): PYTHONHOME set to framework: $PYTHONHOME" >> "$LOG_FILE"
            echo "$(date): DYLD_FRAMEWORK_PATH set to: $DYLD_FRAMEWORK_PATH" >> "$LOG_FILE"
            
            # Verify the framework has the standard library
            # Framework structure varies:
            # - python.org: Versions/X.Y/lib/pythonX.Y/ (e.g., Versions/3.11/lib/python3.11/)
            # - Homebrew: Versions/X.Y/lib/pythonX.Y/ (same structure)
            # FRAMEWORK_PYTHON_DIR already points to Versions/X.Y, so check lib/pythonX.Y
            # Extract version from FRAMEWORK_PYTHON_DIR to ensure consistency
            FRAMEWORK_VERSION=$(basename "$FRAMEWORK_PYTHON_DIR")
            STDlib_PATH="$FRAMEWORK_PYTHON_DIR/lib/python${FRAMEWORK_VERSION}"
            
            # Also try with detected venv version in case of mismatch
            ALT_STDlib_PATH="$FRAMEWORK_PYTHON_DIR/lib/python${PYTHON_MAJOR_MINOR}"
            
            if [ -d "$STDlib_PATH" ]; then
                # Verify it actually contains Python modules (not just empty directory)
                if [ -f "$STDlib_PATH/encodings/__init__.py" ] || [ -d "$STDlib_PATH/encodings" ]; then
                    echo "$(date): ✓ Framework contains standard library at: $STDlib_PATH" >> "$LOG_FILE"
                else
                    echo "$(date): WARNING: Framework lib directory exists but may not contain standard library" >> "$LOG_FILE"
                    echo "$(date):   Checked: $STDlib_PATH" >> "$LOG_FILE"
                fi
            elif [ -d "$ALT_STDlib_PATH" ] && [ "$FRAMEWORK_VERSION" != "$PYTHON_MAJOR_MINOR" ]; then
                # Version mismatch - try with venv version
                if [ -f "$ALT_STDlib_PATH/encodings/__init__.py" ] || [ -d "$ALT_STDlib_PATH/encodings" ]; then
                    echo "$(date): ✓ Framework contains standard library at: $ALT_STDlib_PATH (version mismatch: framework=$FRAMEWORK_VERSION, venv=$PYTHON_MAJOR_MINOR)" >> "$LOG_FILE"
                else
                    echo "$(date): WARNING: Framework standard library not found at either expected location" >> "$LOG_FILE"
                    echo "$(date):   Checked: $STDlib_PATH and $ALT_STDlib_PATH" >> "$LOG_FILE"
                fi
            else
                echo "$(date): WARNING: Framework standard library not found at expected location" >> "$LOG_FILE"
                echo "$(date):   Expected: $STDlib_PATH (framework version: $FRAMEWORK_VERSION)" >> "$LOG_FILE"
                if [ "$FRAMEWORK_VERSION" != "$PYTHON_MAJOR_MINOR" ]; then
                    echo "$(date):   Also checked: $ALT_STDlib_PATH (venv version: $PYTHON_MAJOR_MINOR)" >> "$LOG_FILE"
                fi
                # Try to find stdlib in alternative locations
                POSSIBLE_STDLIB=$(find "$FRAMEWORK_PYTHON_DIR/lib" -name "encodings" -type d 2>/dev/null | head -1)
                if [ -n "$POSSIBLE_STDLIB" ]; then
                    echo "$(date):   Found encodings at: $POSSIBLE_STDLIB" >> "$LOG_FILE"
                    echo "$(date):   Framework structure may differ from expected layout" >> "$LOG_FILE"
                fi
            fi
        else
            # Fallback: Don't set PYTHONHOME to venv - it doesn't have full stdlib
            # Let Python auto-detect the base installation instead
            unset PYTHONHOME 2>/dev/null || true
            export DYLD_FRAMEWORK_PATH="$BUNDLED_FRAMEWORK:$DYLD_FRAMEWORK_PATH"
            echo "$(date): WARNING: Framework Python directory not found at expected location" >> "$LOG_FILE"
            echo "$(date): PYTHONHOME left unset - Python will auto-detect base installation" >> "$LOG_FILE"
            echo "$(date): Bundle may not work on machines without Python installed" >> "$LOG_FILE"
        fi
    else
        # No framework bundled - don't set PYTHONHOME to venv (it lacks full stdlib)
        # Let Python auto-detect the base installation instead
        unset PYTHONHOME 2>/dev/null || true
        echo "$(date): No bundled framework found" >> "$LOG_FILE"
        echo "$(date): PYTHONHOME left unset - Python will auto-detect base installation" >> "$LOG_FILE"
        echo "$(date): WARNING: Bundle requires Python to be installed on target machine" >> "$LOG_FILE"
    fi
else
    # Don't set PYTHONHOME for regular venv - Python will find base installation automatically
    # This is already unset above, but ensure it stays unset
    unset PYTHONHOME 2>/dev/null || true
    echo "$(date): Running from regular venv - PYTHONHOME is unset (Python will auto-detect base installation)" >> "$LOG_FILE"
    echo "$(date): [DEBUG] PYTHONHOME check before exec: ${PYTHONHOME:-<unset>}" >> "$LOG_FILE"
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
# NOTE: Qt plugin loading failures can come from multiple sources:
# - Missing Qt runtime dependencies inside the bundle
# - Code signing issues on libqcocoa.dylib and other Qt libraries
# - Quarantine attributes on .dylib files (removed during build)
# - Mismatched Qt minor versions between PyQt6 components
# - System Qt paths conflicting with bundled Qt
# We address these by: code signing Qt libs, removing quarantine, setting paths correctly,
# and filtering system Qt paths. The repair_pyqt6.sh script handles runtime fixes.
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
echo "$(date): [DEBUG] Final PYTHONHOME before exec: ${PYTHONHOME:-<unset>}" >> "$LOG_FILE"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Only pass PYTHONHOME if it's set (for app bundles)
    if [ -n "$PYTHONHOME" ]; then
        echo "$(date): [DEBUG] Passing PYTHONHOME=$PYTHONHOME to Python (app bundle mode)" >> "$LOG_FILE"
        exec env PYTHONHOME="$PYTHONHOME" PATH="$PATH" QT_PLUGIN_PATH="$QT_PLUGIN_PATH" DYLD_FRAMEWORK_PATH="$DYLD_FRAMEWORK_PATH" QT_FRAMEWORK_PATH="$QT_FRAMEWORK_PATH" QT_MAC_WANTS_LAYER=1 QT_AUTO_SCREEN_SCALE_FACTOR=1 "$PYTHON_CMD" "$SCRIPT_DIR/omnipose_batch_gui.py"
    else
        echo "$(date): [DEBUG] NOT passing PYTHONHOME to Python (regular venv mode)" >> "$LOG_FILE"
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
