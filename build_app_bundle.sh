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

# Get the directory where this script is located (inside the .app bundle)
APP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
APP_CONTENTS="$APP_DIR/Contents"

# The actual project files are in the Resources directory
PROJECT_DIR="$APP_CONTENTS/Resources"

# Change to project directory and run launch script
cd "$PROJECT_DIR"

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
echo "  Copying files (this may take several minutes for large venv)..."
if ! rsync -av --progress "${RSYNC_EXCLUDES[@]}" \
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
