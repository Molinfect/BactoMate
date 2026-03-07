#!/bin/bash
# Launch the Omnipose Batch Processing GUI

# CRITICAL: Unset PYTHONHOME immediately - it breaks venv usage
# Venvs don't contain the full standard library, so PYTHONHOME must point to base Python
# We'll only set it later if we're in an app bundle with a bundled framework
# Log initial PYTHONHOME value before unsetting (for debugging)
INITIAL_PYTHONHOME="${PYTHONHOME:-<not set>}"

# Get script directory and project root. When run from .app, script is in Resources (venv lives here).
# When run from project, script is in setup/ so project root is SCRIPT_DIR/..
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/venv/bin" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

# Use BactiMate Logs folder. When running from .app: try Logs inside the app first so logs appear in Contents/Resources/Logs
if [[ "$PROJECT_ROOT" == *".app/Contents/Resources"* ]] || [[ "$PROJECT_ROOT" == *"/Contents/Resources"* ]]; then
    APP_LOGS="$PROJECT_ROOT/Logs"
    mkdir -p "$APP_LOGS" 2>/dev/null || true
    if [[ -w "$APP_LOGS" ]] 2>/dev/null; then
        LOG_DIR="$APP_LOGS"
    else
        if [[ "$OSTYPE" == "darwin"* ]]; then
            LOG_DIR="$HOME/Library/Application Support/BactiMate/Logs"
        elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
            LOG_DIR="${APPDATA:-$HOME/AppData/Roaming}/BactiMate/Logs"
        else
            LOG_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/BactiMate/Logs"
        fi
    fi
else
    LOG_DIR="$PROJECT_ROOT/Logs"
fi
mkdir -p "$LOG_DIR"
# Timestamped launch log (one per run); keep last 20 so logs don't grow indefinitely (matches MAX_LAUNCH_LOGS in constants.py)
LAUNCH_KEEP=20
for f in $(ls -t "$LOG_DIR"/launch_*.log 2>/dev/null | tail -n +$((LAUNCH_KEEP))); do
    rm -f "$f" 2>/dev/null || true
done
LOG_FILE="$LOG_DIR/launch_$(date '+%Y-%m-%d_%H-%M-%S').log"
# On macOS also ensure Application Support exists and write one line there so "reopen didn't run" can be diagnosed
CANONICAL_LOG=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    CANONICAL_LOG="$HOME/Library/Application Support/BactiMate/Logs/launch.log"
    mkdir -p "$(dirname "$CANONICAL_LOG")"
    echo "$(date '+%Y-%m-%dT%H:%M:%S') [from ${PROJECT_ROOT:-?}] full log: $LOG_FILE" >> "$CANONICAL_LOG" 2>/dev/null || true
    # Dock/reopen diagnostics: use same LOG_DIR (inside app or Application Support)
    if [[ "$PROJECT_ROOT" == *".app/Contents/Resources"* ]]; then
        DOCK_DEBUG="$LOG_DIR/dock_debug.txt"
        echo "" >> "$DOCK_DEBUG" 2>/dev/null || true
        echo "=== $(date '+%Y-%m-%dT%H:%M:%S') ===" >> "$DOCK_DEBUG" 2>/dev/null || true
        echo "LAUNCH_SCRIPT pid=$$ ppid=$PPID  LAUNCHER_PID=${BACTIMATE_LAUNCHER_PID:-unknown}  (ppid should equal LAUNCHER_PID; if not, Reopen may fail)" >> "$DOCK_DEBUG" 2>/dev/null || true
    fi
fi
echo "$(date '+%Y-%m-%dT%H:%M:%S'): === BactiMate launch attempt ===" >> "$LOG_FILE"

# If running interactively (from terminal), also show errors in terminal
# Check if stderr is a terminal (not redirected)
if [ -t 2 ]; then
    # Running in terminal - tee errors to both log and terminal
    exec 2> >(tee -a "$LOG_FILE" >&2)
else
    # Running from app bundle or redirected - just log
    exec 2>> "$LOG_FILE"
fi

# Now unset PYTHONHOME and log it
unset PYTHONHOME 2>/dev/null || true
echo "$(date): [DEBUG] Initial PYTHONHOME was: $INITIAL_PYTHONHOME" >> "$LOG_FILE"
echo "$(date): [DEBUG] PYTHONHOME after unset: ${PYTHONHOME:-<unset>}" >> "$LOG_FILE"
echo "$(date): Log file: $LOG_FILE" >> "$LOG_FILE"

# Log for debugging
echo "$(date): Launch script directory (setup): $SCRIPT_DIR" >> "$LOG_FILE"
echo "$(date): Project root: $PROJECT_ROOT" >> "$LOG_FILE"
echo "$(date): Looking for Python in: $PROJECT_ROOT/venv/bin/" >> "$LOG_FILE"

# Use venv Python directly instead of activation (more reliable across machines).
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python3"
VENV_PYTHONW="$PROJECT_ROOT/venv/bin/pythonw"
# In .app mode prefer a BactiMate-named launcher binary so Dock identity stays on BactiMate.
VENV_BACTIMATE="$PROJECT_ROOT/venv/bin/BactiMate"

# Debug: List what's in venv/bin
if [ -d "$PROJECT_ROOT/venv/bin" ]; then
    echo "$(date): Contents of venv/bin:" >> "$LOG_FILE"
    ls -la "$PROJECT_ROOT/venv/bin" | head -20 >> "$LOG_FILE" 2>&1 || true
else
    echo "$(date): ERROR: venv/bin directory does not exist: $PROJECT_ROOT/venv/bin" >> "$LOG_FILE"
fi

# Verify venv Python exists and is executable
if [ ! -f "$VENV_PYTHON" ] && [ ! -f "$VENV_PYTHONW" ] && [ ! -f "$VENV_BACTIMATE" ]; then
    echo "$(date): ERROR: Python interpreter not found in venv: $PROJECT_ROOT/venv/bin/" >> "$LOG_FILE"
    echo "$(date): VENV_PYTHON=$VENV_PYTHON" >> "$LOG_FILE"
    echo "$(date): VENV_PYTHONW=$VENV_PYTHONW" >> "$LOG_FILE"
    echo "$(date): VENV_BACTIMATE=$VENV_BACTIMATE" >> "$LOG_FILE"
    if [[ "$PROJECT_ROOT" != *".app/Contents/Resources"* ]]; then
        MSG="BactiMate Error: Python environment not found. From the project folder run: ./setup/setup_fresh_env.sh then try again. Logs: $LOG_DIR"
    else
        MSG="BactiMate Error: Python environment not found. The application may be corrupted. Logs: $LOG_DIR"
    fi
    osascript -e "display dialog \"$MSG\" buttons {\"OK\"} default button \"OK\" with icon stop" 2>/dev/null || true
    exit 1
fi

# Deterministic startup path:
# - In .app mode we only allow the bundled BactiMate launcher binary.
# - In project mode we use pythonw/python3 from the local venv.
if [[ "$PROJECT_ROOT" == *".app/Contents/Resources"* ]]; then
    if [ -f "$VENV_BACTIMATE" ] && [ -x "$VENV_BACTIMATE" ]; then
        PYTHON_CMD="$VENV_BACTIMATE"
        echo "$(date): Using bundled BactiMate binary: $PYTHON_CMD" >> "$LOG_FILE"
    else
        echo "$(date): ERROR: Required bundled launcher missing or not executable: $VENV_BACTIMATE" >> "$LOG_FILE"
        osascript -e 'display dialog "BactiMate Error: App bundle is incomplete (missing venv/bin/BactiMate). Rebuild the app bundle." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
        exit 1
    fi
elif [ -f "$VENV_PYTHONW" ] && [ -x "$VENV_PYTHONW" ]; then
    PYTHON_CMD="$VENV_PYTHONW"
elif [ -f "$VENV_PYTHON" ] && [ -x "$VENV_PYTHON" ]; then
    PYTHON_CMD="$VENV_PYTHON"
else
    echo "$(date): ERROR: Python interpreter is not executable: $VENV_PYTHON" >> "$LOG_FILE"
    osascript -e 'display dialog "BactiMate Error: Python interpreter is not executable. The environment may be corrupted." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
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

    # No fallback to system Python in .app mode: fail fast with one clear path.
    echo "$(date): Attempting to diagnose Python issue..." >> "$LOG_FILE"
    echo "$(date): Checking Python architecture..." >> "$LOG_FILE"
    arch >> "$LOG_FILE" 2>&1 || true
    echo "$(date): System architecture:" >> "$LOG_FILE"
    uname -m >> "$LOG_FILE" 2>&1 || true

    if command -v codesign >/dev/null 2>&1; then
        echo "$(date): Checking code signature..." >> "$LOG_FILE"
        codesign -dv --verbose=4 "$PYTHON_CMD" >> "$LOG_FILE" 2>&1 || true
    fi
    if command -v otool >/dev/null 2>&1; then
        echo "$(date): Checking linked libraries..." >> "$LOG_FILE"
        otool -L "$PYTHON_CMD" | head -10 >> "$LOG_FILE" 2>&1 || true
    fi

    osascript -e "display dialog \"BactiMate Error: Python launcher failed to start. Check logs: $LOG_DIR\" buttons {\"OK\"} default button \"OK\" with icon stop" 2>/dev/null || true
    exit 1
fi

if [ $PYTHON_TEST_EXIT -eq 0 ]; then
    echo "$(date): Python test successful: $PYTHON_TEST_OUTPUT" >> "$LOG_FILE"
    
    # Log Python version prominently (venv can be 3.11 or 3.9 depending on how it was created)
    PYTHON_VERSION_FULL=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null || echo "unknown")
    echo "$(date): Using Python $PYTHON_VERSION_FULL from venv: $PYTHON_CMD" >> "$LOG_FILE"
    
    # Log Python system information for debugging cross-machine issues
    echo "$(date): Python system information:" >> "$LOG_FILE"
    "$PYTHON_CMD" -c "import sys; import platform; print(f'  executable: {sys.executable}'); print(f'  prefix: {sys.prefix}'); print(f'  base_prefix: {sys.base_prefix}'); print(f'  version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); print(f'  platform: {platform.platform()}')" >> "$LOG_FILE" 2>&1 || true
fi

# Set PATH to prioritize venv binaries
export PATH="$PROJECT_ROOT/venv/bin:$PATH"

# Detect Python version from venv in one call (for framework and Qt paths)
PYINFO=$("$PYTHON_CMD" -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>/dev/null || echo "3 11")
PYTHON_MAJOR_MINOR=$(echo "$PYINFO" | tr ' ' '.')
PYTHON_VERSION="python${PYTHON_MAJOR_MINOR}"

# Only set PYTHONHOME for app bundles (when running from .app bundle)
# Note: PYTHONHOME is already unset at the start of this script
# For regular venv usage, don't set PYTHONHOME - let Python auto-detect the base installation
# This is needed for relocatable app bundles but breaks regular venv usage
if [[ "$PROJECT_ROOT" == *".app/Contents/Resources"* ]]; then
    # Add bundled Python framework to library search path for self-contained bundles
    APP_DIR="$(cd "$PROJECT_ROOT/../.." && pwd)"
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
            # CRITICAL: Clean DYLD_FRAMEWORK_PATH first, then add bundled framework
            # This ensures our bundled paths can't be filtered out by accident
            if [ -n "$DYLD_FRAMEWORK_PATH" ]; then
                # Clean existing DYLD_FRAMEWORK_PATH (remove system paths, keep app bundles, dedupe)
                CLEANED_DYLD=$(
                    echo "$DYLD_FRAMEWORK_PATH" | tr ':' '\n' | while IFS= read -r path; do
                        # Skip empty entries
                        [ -z "$path" ] && continue
                        # Always keep bundled frameworks inside an app
                        if [[ "$path" == *".app/Contents/Frameworks"* ]]; then
                            echo "$path"
                        elif [[ "$path" == */opt/homebrew/* ]] || [[ "$path" == */usr/local/* ]] || [[ "$path" == */anaconda* ]] || [[ "$path" == */miniconda* ]]; then
                            : # drop system paths
                        else
                            echo "$path"
                        fi
                    done | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//'
                )
                if [ -n "$CLEANED_DYLD" ]; then
                    export DYLD_FRAMEWORK_PATH="$BUNDLED_FRAMEWORK:$CLEANED_DYLD"
                else
                    export DYLD_FRAMEWORK_PATH="$BUNDLED_FRAMEWORK"
                fi
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
            # Clean DYLD_FRAMEWORK_PATH first, then add bundled framework
            if [ -n "$DYLD_FRAMEWORK_PATH" ]; then
                CLEANED_DYLD=$(
                    echo "$DYLD_FRAMEWORK_PATH" | tr ':' '\n' | while IFS= read -r path; do
                        [ -z "$path" ] && continue
                        if [[ "$path" == *".app/Contents/Frameworks"* ]]; then
                            echo "$path"
                        elif [[ "$path" == */opt/homebrew/* ]] || [[ "$path" == */usr/local/* ]] || [[ "$path" == */anaconda* ]] || [[ "$path" == */miniconda* ]]; then
                            : # drop system paths
                        else
                            echo "$path"
                        fi
                    done | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//'
                )
                if [ -n "$CLEANED_DYLD" ]; then
                    export DYLD_FRAMEWORK_PATH="$BUNDLED_FRAMEWORK:$CLEANED_DYLD"
                else
                    export DYLD_FRAMEWORK_PATH="$BUNDLED_FRAMEWORK"
                fi
            else
                export DYLD_FRAMEWORK_PATH="$BUNDLED_FRAMEWORK"
            fi
            echo "$(date): WARNING: Framework Python directory not found at expected location" >> "$LOG_FILE"
            echo "$(date): PYTHONHOME left unset - Python will auto-detect base installation" >> "$LOG_FILE"
            echo "$(date): Bundle may not work on machines without Python installed" >> "$LOG_FILE"
        fi
    else
        # No framework bundled: use venv's Python (it will find base via symlink, e.g. Homebrew).
        # On the build machine this works. For distribution to other machines, bundle a Python.framework.
        unset PYTHONHOME 2>/dev/null || true
        echo "$(date): No bundled Python framework - using venv (finds base Python on this machine)" >> "$LOG_FILE"
        echo "$(date): PYTHONHOME unset (Python auto-detects). For portable .app, add Contents/Frameworks/Python.framework" >> "$LOG_FILE"
    fi
else
    # Don't set PYTHONHOME for regular venv - Python will find base installation automatically
    # This is already unset above, but ensure it stays unset
    unset PYTHONHOME 2>/dev/null || true
    echo "$(date): Running from regular venv - PYTHONHOME is unset (Python will auto-detect base installation)" >> "$LOG_FILE"
    echo "$(date): [DEBUG] PYTHONHOME check before exec: ${PYTHONHOME:-<unset>}" >> "$LOG_FILE"
fi

# Verify patches are applied (fast file check, no Python import of cellpose/torch)
if [ -f "$SCRIPT_DIR/apply_all_patches.py" ]; then
    PATCH_FILE=""
    for f in "$PROJECT_ROOT/venv/lib/python"*"/site-packages/cellpose_omni/models.py"; do
        [ -f "$f" ] && PATCH_FILE="$f" && break
    done 2>/dev/null
    NEED_PATCH=false
    if [ -n "$PATCH_FILE" ]; then
        grep -q 'model_type=model_type' "$PATCH_FILE" 2>/dev/null && grep -q 'raw.githubusercontent.com/kevinjohncutler/omnipose-models/main' "$PATCH_FILE" 2>/dev/null || NEED_PATCH=true
    else
        NEED_PATCH=true
    fi
    if [ "$NEED_PATCH" = true ]; then
        echo "$(date): Applying compatibility patches..." >> "$LOG_FILE"
        "$PYTHON_CMD" "$SCRIPT_DIR/apply_all_patches.py" >> "$LOG_FILE" 2>&1 || true
    fi
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
# PYTHON_VERSION was set earlier (one call with PYTHON_MAJOR_MINOR)

QT_LIB_DIR="$PROJECT_ROOT/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/lib"
if [ -d "$QT_LIB_DIR" ]; then
    # Just set QT_FRAMEWORK_PATH here - DYLD_FRAMEWORK_PATH will be handled in final cleanup
    export QT_FRAMEWORK_PATH="$QT_LIB_DIR"
fi

# Find Qt platform plugins in venv - use absolute path with detected Python version
VENV_QT_PLUGINS="$PROJECT_ROOT/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
if [ -d "$VENV_QT_PLUGINS" ]; then
    export QT_PLUGIN_PATH="$VENV_QT_PLUGINS"
    # Also set QML2_IMPORT_PATH if needed
    export QML2_IMPORT_PATH="$PROJECT_ROOT/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/qml"
    # Verify the cocoa plugin exists
    if [ ! -f "$VENV_QT_PLUGINS/platforms/libqcocoa.dylib" ]; then
        echo "$(date): WARNING: Cocoa plugin not found. Running repair script..." >> "$LOG_FILE"
        "$SCRIPT_DIR/repair_pyqt6.sh" >> "$LOG_FILE" 2>&1 || true
        # Re-export after potential repair
        export QT_PLUGIN_PATH="$PROJECT_ROOT/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
    fi
else
    echo "$(date): WARNING: Qt plugins directory not found. Running repair script..." >> "$LOG_FILE"
    "$SCRIPT_DIR/repair_pyqt6.sh" >> "$LOG_FILE" 2>&1 || true
    # Re-export after potential repair
    export QT_PLUGIN_PATH="$PROJECT_ROOT/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
fi

# Verify PyQt6 is installed and plugin exists
# Check both import and plugin file existence
PYQT6_OK=false
if "$PYTHON_CMD" -c "import PyQt6" 2>> "$LOG_FILE"; then
    if [ -f "$VENV_QT_PLUGINS/platforms/libqcocoa.dylib" ]; then
        PYQT6_OK=true
    fi
fi

# When running from .app and check failed: clear quarantine and re-sign (same as build), then re-check
if [ "$PYQT6_OK" = false ] && [[ "$PROJECT_ROOT" == *".app/Contents/Resources"* ]]; then
    echo "$(date): PyQt6 check failed in app bundle. Clearing quarantine and re-signing venv..." >> "$LOG_FILE"
    APP_BUNDLE="$(cd "$PROJECT_ROOT/../.." && pwd)"
    xattr -cr "$APP_BUNDLE" 2>> "$LOG_FILE" || true
    if command -v codesign &>/dev/null 2>&1; then
        find "$PROJECT_ROOT/venv" -name "*.dylib" -type f -exec codesign -f -s - {} \; 2>> "$LOG_FILE" || true
        find "$PROJECT_ROOT/venv/bin" -type f \( -name "python*" -o -name "pythonw*" \) -exec codesign -f -s - {} \; 2>> "$LOG_FILE" || true
    fi
    PYQT6_OK=false
    if "$PYTHON_CMD" -c "import PyQt6" 2>> "$LOG_FILE"; then
        if [ -f "$VENV_QT_PLUGINS/platforms/libqcocoa.dylib" ]; then
            PYQT6_OK=true
            echo "$(date): PyQt6 OK after clearing quarantine." >> "$LOG_FILE"
        fi
    fi
fi

if [ "$PYQT6_OK" = false ]; then
    echo "$(date): PyQt6 issue detected. Running automatic repair..." >> "$LOG_FILE"
    if "$SCRIPT_DIR/repair_pyqt6.sh" >> "$LOG_FILE" 2>&1; then
        # Re-export after repair (final cleanup block will handle DYLD_FRAMEWORK_PATH)
        export QT_PLUGIN_PATH="$PROJECT_ROOT/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
        echo "$(date): Repair completed successfully." >> "$LOG_FILE"
    else
        echo "$(date): Automatic repair failed." >> "$LOG_FILE"
        # When from app: try launching anyway so user gets real Qt error or app may still work
        if [[ "$PROJECT_ROOT" == *".app/Contents/Resources"* ]]; then
            echo "$(date): Continuing to launch (from app); Python may show the real error." >> "$LOG_FILE"
        else
            osascript -e "display dialog \"BactiMate Error: PyQt6 could not be loaded. Often fixed by rebuilding the app (clears quarantine).\n\nDetails in: $LOG_FILE\" buttons {\"OK\"} default button \"OK\" with icon stop" 2>/dev/null || true
            exit 1
        fi
    fi
fi

# Ensure all environment variables are set before launching
# Re-export to make absolutely sure they're set (using detected Python version)
export QT_PLUGIN_PATH="$PROJECT_ROOT/venv/lib/${PYTHON_VERSION}/site-packages/PyQt6/Qt6/plugins"
if [ -d "$QT_LIB_DIR" ]; then
    # Final cleanup and ensure QT_LIB_DIR is present (avoid duplicates)
    if [ -n "$DYLD_FRAMEWORK_PATH" ]; then
        # Clean DYLD_FRAMEWORK_PATH (remove system paths, keep app bundles, dedupe)
        CLEANED_DYLD=$(
            echo "$DYLD_FRAMEWORK_PATH" | tr ':' '\n' | while IFS= read -r path; do
                [ -z "$path" ] && continue
                if [[ "$path" == *".app/Contents/Frameworks"* ]]; then
                    echo "$path"
                elif [[ "$path" == */opt/homebrew/* ]] || [[ "$path" == */usr/local/* ]] || [[ "$path" == */anaconda* ]] || [[ "$path" == */miniconda* ]]; then
                    : # drop system paths
                else
                    echo "$path"
                fi
            done | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//'
        )
        # Ensure QT_LIB_DIR is at the front (avoid duplicates)
        case ":$CLEANED_DYLD:" in
            *":$QT_LIB_DIR:"*)
                # Already present, remove it and prepend to ensure it's first
                CLEANED_DYLD=$(echo "$CLEANED_DYLD" | tr ':' '\n' | grep -v "^$QT_LIB_DIR" | tr '\n' ':' | sed 's/:$//')
                export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR${CLEANED_DYLD:+:$CLEANED_DYLD}"
                ;;
            *)
                # Not present, prepend it
                if [ -n "$CLEANED_DYLD" ]; then
                    export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR:$CLEANED_DYLD"
                else
                    export DYLD_FRAMEWORK_PATH="$QT_LIB_DIR"
                fi
                ;;
        esac
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

# If running from terminal, show a message
if [ -t 0 ]; then
    echo "Launching BactiMate GUI..." >&2
    echo "Log file: $LOG_FILE" >&2
fi

# Keep process name stable on macOS so Dock/menu bar stay on BactiMate.
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -n "${PYTHONHOME:-}" ]; then
        echo "$(date): [DEBUG] Passing PYTHONHOME to Python (app bundle mode)" >> "$LOG_FILE"
    else
        echo "$(date): [DEBUG] NOT passing PYTHONHOME (regular venv mode)" >> "$LOG_FILE"
    fi
    export PYTHONHOME PATH QT_PLUGIN_PATH DYLD_FRAMEWORK_PATH QT_FRAMEWORK_PATH
    export QT_MAC_WANTS_LAYER=1 QT_AUTO_SCREEN_SCALE_FACTOR=1
    exec -a BactiMate "$PYTHON_CMD" "$PROJECT_ROOT/main.py"
else
    if [ -n "${PYTHONHOME:-}" ]; then
        exec "$PYTHON_CMD" "$PROJECT_ROOT/main.py"
    else
        exec "$PYTHON_CMD" "$PROJECT_ROOT/main.py"
    fi
fi
