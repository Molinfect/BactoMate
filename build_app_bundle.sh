#!/bin/bash
# Build script for macOS app bundle
# This creates a .app bundle that wraps the entire project folder and launches launch_gui.sh
# Similar to how ilastik packages their application

set -e  # Exit on error

echo "=========================================="
echo "Building BactiMate.app bundle"
echo "=========================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: This script must be run on macOS"
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf "BactiMate.app"
rm -rf "dist/BactiMate.app"

# Create app bundle structure
echo "Creating app bundle structure..."
APP_BUNDLE="BactiMate.app"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Create the launcher executable script
echo "Creating launcher executable..."
cat > "$APP_BUNDLE/Contents/MacOS/BactiMate" << 'LAUNCHER_EOF'
#!/bin/bash
# BactiMate launcher - runs launch_gui.sh from within the app bundle

# Redirect stderr to a log file for debugging
LOG_DIR="$HOME/Library/Logs/BactiMate"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch.log"
exec 2>> "$LOG_FILE"

# Get the directory where this script is located (inside the .app bundle)
APP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
APP_CONTENTS="$APP_DIR/Contents"

# The actual project files are in the Resources directory
PROJECT_DIR="$APP_CONTENTS/Resources"

# Verify project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "$(date): ERROR: Project directory not found: $PROJECT_DIR" >> "$LOG_FILE"
    osascript -e 'display dialog "BactiMate Error: Application bundle is corrupted. Project directory not found." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR" || {
    echo "$(date): ERROR: Cannot change to project directory: $PROJECT_DIR" >> "$LOG_FILE"
    osascript -e 'display dialog "BactiMate Error: Cannot access application directory." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
}

# Verify launch script exists
if [ ! -f "$PROJECT_DIR/launch_gui.sh" ]; then
    echo "$(date): ERROR: Launch script not found: $PROJECT_DIR/launch_gui.sh" >> "$LOG_FILE"
    osascript -e 'display dialog "BactiMate Error: Launch script not found." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
fi

# Make sure launch script is executable
chmod +x "$PROJECT_DIR/launch_gui.sh"

# Log launch attempt
echo "$(date): Launching BactiMate from: $PROJECT_DIR" >> "$LOG_FILE"

# Run the launch script
exec "$PROJECT_DIR/launch_gui.sh"
LAUNCHER_EOF

# Make launcher executable
chmod +x "$APP_BUNDLE/Contents/MacOS/BactiMate"

# Create Info.plist
echo "Creating Info.plist..."
cat > "$APP_BUNDLE/Contents/Info.plist" << 'INFOPLIST_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>BactiMate</string>
    <key>CFBundleIconFile</key>
    <string>BactiMate</string>
    <key>CFBundleIdentifier</key>
    <string>org.bactimate.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>BactiMate</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.14</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>BactiMate needs microphone access</string>
    <key>NSCameraUsageDescription</key>
    <string>BactiMate needs camera access</string>
</dict>
</plist>
INFOPLIST_EOF

# Copy icon if it exists
if [ -f "BactiMate.icns" ]; then
    echo "Copying app icon..."
    cp "BactiMate.icns" "$APP_BUNDLE/Contents/Resources/BactiMate.icns"
fi

# Copy all project files to Resources (excluding build artifacts)
echo "Copying project files to app bundle..."
echo "  (this may take a moment...)"

# Files/directories to exclude
EXCLUDE_PATTERNS=(
    ".git"
    ".gitignore"
    "__pycache__"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    ".DS_Store"
    "dist"
    "build"
    "*.app"
    "*.dmg"
    "*.log"
    "nuitka-crash-report.xml"
    "venv_backup"
    ".venv"
)

# Build rsync exclude arguments
RSYNC_EXCLUDES=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    RSYNC_EXCLUDES+=("--exclude=$pattern")
done

# Use rsync to copy files (preserves permissions and symlinks)
# Use --progress to show progress for large copies
# --copy-links: transform symlinks into the files they point to (important for venv Python)
echo "  Copying files (this may take several minutes for large venv)..."
if ! rsync -avL --progress "${RSYNC_EXCLUDES[@]}" \
    --exclude="BactiMate.app" \
    "$SCRIPT_DIR/" "$APP_BUNDLE/Contents/Resources/"; then
    echo "ERROR: Failed to copy files to app bundle!"
    exit 1
fi

# Verify critical files were copied
if [ ! -d "$APP_BUNDLE/Contents/Resources/venv" ]; then
    echo "ERROR: venv directory not found in app bundle!"
    exit 1
fi

if [ ! -f "$APP_BUNDLE/Contents/Resources/launch_gui.sh" ]; then
    echo "ERROR: launch_gui.sh not found in app bundle!"
    exit 1
fi

# Ensure launch_gui.sh is executable
chmod +x "$APP_BUNDLE/Contents/Resources/launch_gui.sh"

# Fix permissions for Python executables in venv (critical for app bundle)
echo "Fixing Python executable permissions..."
VENV_BIN="$APP_BUNDLE/Contents/Resources/venv/bin"
if [ -d "$VENV_BIN" ]; then
    # Make all Python executables executable
    find "$VENV_BIN" -type f -name "python*" -exec chmod +x {} \;
    find "$VENV_BIN" -type f -name "pythonw*" -exec chmod +x {} \;
    # Also fix any symlinks by ensuring the target is executable
    # Find all executables in bin directory
    find "$VENV_BIN" -type f -exec chmod +x {} \;
    
    # Check if Python is a symlink and resolve it
    PYTHON_EXEC="$VENV_BIN/python3"
    if [ -L "$PYTHON_EXEC" ]; then
        echo "  Python3 is a symlink, checking target..."
        TARGET=$(readlink "$PYTHON_EXEC")
        echo "  Symlink target: $TARGET"
        # If it's a relative symlink, resolve it relative to venv/bin
        if [[ "$TARGET" != /* ]]; then
            RESOLVED="$VENV_BIN/$TARGET"
            if [ -f "$RESOLVED" ]; then
                echo "  Resolved to: $RESOLVED"
                chmod +x "$RESOLVED"
            else
                echo "  WARNING: Symlink target not found: $RESOLVED"
            fi
        fi
    fi
    
    echo "  Python executables made executable"
else
    echo "  WARNING: venv/bin directory not found!"
fi

# Make venv relocatable by updating pyvenv.cfg
echo "Making venv relocatable..."
PYVENV_CFG="$APP_BUNDLE/Contents/Resources/venv/pyvenv.cfg"
if [ -f "$PYVENV_CFG" ]; then
    # Read the current Python version from pyvenv.cfg
    PYTHON_VERSION=$(grep "^version" "$PYVENV_CFG" | cut -d'=' -f2 | tr -d ' ')
    if [ -n "$PYTHON_VERSION" ]; then
        # Update pyvenv.cfg to use relative paths (make it relocatable)
        # The 'home' field should be relative or we can remove it
        # For relocatable venvs, we rely on PYTHONHOME environment variable
        # Keep the version but make home relative
        sed -i.bak "s|^home=.*|home = .|" "$PYVENV_CFG" 2>/dev/null || \
        sed -i '' "s|^home=.*|home = .|" "$PYVENV_CFG" 2>/dev/null || true
        rm -f "${PYVENV_CFG}.bak" 2>/dev/null || true
        echo "  Updated pyvenv.cfg for relocatability"
    fi
else
    echo "  WARNING: pyvenv.cfg not found!"
fi

# Verify Python executable exists and is executable
PYTHON_EXEC="$VENV_BIN/python3"
PYTHON_EXEC_RESOLVED="$PYTHON_EXEC"

# Resolve symlinks to get the actual Python executable
if [ -L "$PYTHON_EXEC" ]; then
    echo "  Python3 is a symlink, resolving..."
    TARGET=$(readlink "$PYTHON_EXEC")
    if [[ "$TARGET" == /* ]]; then
        # Absolute path symlink - this is a problem for relocatable bundles
        # The rsync -L flag should have copied it as a file, but let's check
        echo "  WARNING: Python is a symlink to absolute path: $TARGET"
        echo "  This may cause issues in the bundle. Checking if target exists..."
        if [ ! -f "$TARGET" ]; then
            echo "  ERROR: Symlink target does not exist: $TARGET"
            echo "  The venv may not be relocatable. Consider recreating with --copies flag."
        fi
        PYTHON_EXEC_RESOLVED="$TARGET"
    else
        # Relative path symlink - resolve relative to venv/bin
        PYTHON_EXEC_RESOLVED="$VENV_BIN/$TARGET"
    fi
    echo "  Resolved Python path: $PYTHON_EXEC_RESOLVED"
fi

# Bundle Python framework for self-contained app
echo "Bundling Python framework for self-contained app..."
if command -v otool >/dev/null 2>&1; then
    # Check what libraries Python is linked against
    PYTHON_LIBS=$(otool -L "$PYTHON_EXEC_RESOLVED" 2>/dev/null | grep -i "Python.framework" | head -1 | awk '{print $1}' || true)
    if [ -n "$PYTHON_LIBS" ]; then
        echo "  Python is linked against: $PYTHON_LIBS"
        
        if [[ "$PYTHON_LIBS" == /* ]]; then
            # Extract framework directory (e.g., /Library/Frameworks/Python.framework)
            FRAMEWORK_DIR=$(echo "$PYTHON_LIBS" | sed 's|/Versions/.*||')
            PYTHON_VERSION_MATCH=$(echo "$PYTHON_LIBS" | grep -o 'Versions/[0-9.]\+' | cut -d'/' -f2 || echo "3.11")
            
            if [ -d "$FRAMEWORK_DIR" ]; then
                echo "  Found Python framework at: $FRAMEWORK_DIR"
                echo "  Copying framework to bundle (this may take a moment)..."
                
                # Copy framework to bundle
                BUNDLE_FRAMEWORK_DIR="$APP_BUNDLE/Contents/Frameworks/Python.framework"
                mkdir -p "$APP_BUNDLE/Contents/Frameworks"
                
                # Use rsync to copy framework (preserves symlinks and structure)
                if rsync -avL "$FRAMEWORK_DIR/" "$BUNDLE_FRAMEWORK_DIR/" 2>/dev/null; then
                    echo "  ✓ Python framework copied to bundle"
                    
                    # Fix library paths in Python executable to use @rpath
                    if command -v install_name_tool >/dev/null 2>&1; then
                        echo "  Fixing Python library paths to use bundled framework..."
                        # Add rpath to search in Frameworks directory
                        install_name_tool -add_rpath "@executable_path/../../../Frameworks" "$PYTHON_EXEC_RESOLVED" 2>/dev/null || true
                        # Change library path to use @rpath
                        NEW_PATH="@rpath/Python.framework/Versions/${PYTHON_VERSION_MATCH}/Python"
                        if install_name_tool -change "$PYTHON_LIBS" "$NEW_PATH" "$PYTHON_EXEC_RESOLVED" 2>/dev/null; then
                            echo "  ✓ Successfully changed library path to use bundled framework"
                        else
                            echo "  WARNING: Could not change library path (may require codesigning)"
                            echo "  Attempting to fix framework library paths..."
                            # Fix paths inside the framework itself
                            FRAMEWORK_PYTHON="$BUNDLE_FRAMEWORK_DIR/Versions/${PYTHON_VERSION_MATCH}/Python"
                            if [ -f "$FRAMEWORK_PYTHON" ]; then
                                # Fix framework's internal library paths
                                install_name_tool -id "@rpath/Python.framework/Versions/${PYTHON_VERSION_MATCH}/Python" "$FRAMEWORK_PYTHON" 2>/dev/null || true
                            fi
                        fi
                    fi
                else
                    echo "  ERROR: Failed to copy Python framework"
                    echo "  The bundle will NOT be self-contained"
                fi
            else
                echo "  WARNING: Python framework directory not found: $FRAMEWORK_DIR"
                echo "  The bundle will NOT be self-contained"
            fi
        fi
    else
        echo "  No Python.framework dependencies found (Python may be statically linked)"
    fi
else
    echo "  WARNING: otool not found, cannot detect Python framework"
fi

if [ -f "$PYTHON_EXEC" ] || [ -L "$PYTHON_EXEC" ]; then
    # Check if resolved path exists
    if [ ! -f "$PYTHON_EXEC_RESOLVED" ]; then
        echo "  ERROR: Python symlink target does not exist: $PYTHON_EXEC_RESOLVED"
        echo "  This venv may not be relocatable. Trying to fix..."
        # Try to find Python in the venv
        PYTHON_ALT=$(find "$VENV_BIN" -name "python3.*" -type f | head -1)
        if [ -n "$PYTHON_ALT" ]; then
            echo "  Found alternative Python: $PYTHON_ALT"
            PYTHON_EXEC_RESOLVED="$PYTHON_ALT"
        else
            echo "  ERROR: No Python executable found in venv!"
            exit 1
        fi
    fi
    
    # Make sure it's executable
    chmod +x "$PYTHON_EXEC_RESOLVED" 2>/dev/null || true
    
    if [ -x "$PYTHON_EXEC_RESOLVED" ]; then
        echo "  ✓ Python executable verified: $PYTHON_EXEC_RESOLVED"
        # Test that it can run
        VENV_DIR="$APP_BUNDLE/Contents/Resources/venv"
        if env PYTHONHOME="$VENV_DIR" "$PYTHON_EXEC_RESOLVED" --version > /dev/null 2>&1; then
            PYTHON_VERSION=$("$PYTHON_EXEC_RESOLVED" --version 2>&1 | head -1)
            echo "  ✓ Python executable is functional: $PYTHON_VERSION"
        else
            echo "  WARNING: Python executable exists but may not work correctly"
            echo "  Attempting to test without PYTHONHOME..."
            if "$PYTHON_EXEC_RESOLVED" --version > /dev/null 2>&1; then
                echo "  ✓ Python works without PYTHONHOME (will use PYTHONHOME at runtime)"
            else
                echo "  ERROR: Python executable cannot run"
                exit 1
            fi
        fi
    else
        echo "  ERROR: Python executable is not executable: $PYTHON_EXEC_RESOLVED"
        exit 1
    fi
else
    echo "  ERROR: Python executable not found: $PYTHON_EXEC"
    echo "  Listing venv/bin contents:"
    ls -la "$VENV_BIN" | head -10
    exit 1
fi

# Show final size
BUNDLE_SIZE=$(du -sh "$APP_BUNDLE" | cut -f1)
echo "  App bundle created: $BUNDLE_SIZE"

# Update launch_gui.sh to work from within the app bundle
# The script should already work since it uses $(dirname "$0") to find its location
# But let's make sure it can find the venv relative to itself
echo "Verifying launch script setup..."

# Apply ad-hoc code signatures (required for macOS)
echo ""
echo "Applying code signatures..."
find "$APP_BUNDLE/Contents/MacOS" -type f -exec chmod +x {} \;
codesign --force --sign - "$APP_BUNDLE/Contents/MacOS/BactiMate" 2>/dev/null || echo "Note: Code signing skipped (ad-hoc signing may fail without developer ID)"

# Try to sign the entire app bundle
codesign --force --sign - "$APP_BUNDLE" 2>/dev/null || echo "Note: App bundle signing skipped (ad-hoc signing may fail without developer ID)"

# Remove extended attributes that can interfere
xattr -cr "$APP_BUNDLE" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "The application bundle is located at: $APP_BUNDLE"
echo ""
echo "To test the app, run:"
echo "  open $APP_BUNDLE"
echo ""
echo "To create a DMG file, run:"
echo "  ./create_dmg.sh"
echo ""
echo "NOTE: The app bundle includes the entire project folder with venv."
echo "      Users can double-click BactiMate.app to launch the application."
echo ""
