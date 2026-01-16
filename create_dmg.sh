#!/bin/bash
# Create a DMG file for macOS distribution
# This creates a disk image that users can mount and drag the app to Applications

set -e  # Exit on error

APP_NAME="BactiMate"
# Check for app bundle in root first, then dist (for backwards compatibility)
if [ -d "${APP_NAME}.app" ]; then
    APP_PATH="${APP_NAME}.app"
elif [ -d "dist/${APP_NAME}.app" ]; then
    APP_PATH="dist/${APP_NAME}.app"
else
    echo "ERROR: ${APP_NAME}.app not found!"
    echo "Please run ./build_app_bundle.sh first"
    exit 1
fi

DMG_NAME="${APP_NAME}.dmg"
DMG_PATH="dist/${DMG_NAME}"
VOLUME_NAME="${APP_NAME}"

echo "=========================================="
echo "Creating DMG for macOS"
echo "=========================================="
echo ""
echo "Using app bundle: $APP_PATH"
echo ""

# Check if create-dmg is installed, otherwise use built-in hdiutil
if ! command -v create-dmg &> /dev/null; then
    echo "create-dmg not found. Using built-in hdiutil instead..."
    echo ""
    
    # Basic hdiutil method (merged from create_dmg_simple.sh)
    if [ -d "dist/${APP_NAME}_dmg" ]; then
        rm -rf "dist/${APP_NAME}_dmg"
    fi
    mkdir -p "dist/${APP_NAME}_dmg"
    cp -R "$APP_PATH" "dist/${APP_NAME}_dmg/"
    ln -s /Applications "dist/${APP_NAME}_dmg/Applications"
    hdiutil create -volname "$VOLUME_NAME" \
        -srcfolder "dist/${APP_NAME}_dmg" \
        -ov -format UDZO \
        "$DMG_PATH"
    rm -rf "dist/${APP_NAME}_dmg"
    
    if [ -f "$DMG_PATH" ]; then
        echo "DMG created: $DMG_PATH"
        open -R "$DMG_PATH"
    else
        echo "ERROR: DMG creation failed!"
        exit 1
    fi
    exit 0
fi

# Clean up old DMG
if [ -f "$DMG_PATH" ]; then
    echo "Removing old DMG..."
    rm "$DMG_PATH"
fi

# Create DMG
echo "Creating DMG file..."
create-dmg \
    --volname "$VOLUME_NAME" \
    --volicon "$APP_PATH/Contents/Resources/icon.icns" 2>/dev/null || true \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "$APP_NAME.app" 175 190 \
    --hide-extension "$APP_NAME.app" \
    --app-drop-link 425 190 \
    --hdiutil-quiet \
    "$DMG_PATH" \
    "$APP_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "DMG created successfully!"
    echo "=========================================="
    echo ""
    echo "DMG file: $DMG_PATH"
    echo ""
    echo "Users can double-click the DMG to mount it,"
    echo "then drag the app to their Applications folder."
    echo ""
    
    # Optional: Open the DMG location in Finder
    open -R "$DMG_PATH"
else
    echo ""
    echo "ERROR: DMG creation failed!"
    exit 1
fi
