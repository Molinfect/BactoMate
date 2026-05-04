#!/bin/bash
# Create a distributable DMG containing BactiMate.app and an Applications symlink.
# Run from project root: ./build/create_dmg.sh
# Prerequisite: BactiMate.app must exist (run ./build/build_app_bundle.sh first).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="BactiMate"
APP_DIR="$PROJECT_ROOT/$APP_NAME.app"
DMG_NAME="BactiMate"
DIST_DIR="$PROJECT_ROOT/dist"
DMG_PATH="$DIST_DIR/${DMG_NAME}.dmg"

if [ ! -d "$APP_DIR" ]; then
    echo "Error: $APP_DIR not found. Run ./build/build_app_bundle.sh first."
    exit 1
fi

mkdir -p "$DIST_DIR"

# Temporary layout directory for DMG contents
DMG_LAYOUT=$(mktemp -d "${TMPDIR:-/tmp}/bactimate_dmg.XXXXXX")
trap "rm -rf '$DMG_LAYOUT'" EXIT

echo "Preparing DMG contents..."
cp -R "$APP_DIR" "$DMG_LAYOUT/"
ln -s /Applications "$DMG_LAYOUT/Applications"

# Plain-language steps for people who only receive BactiMate.dmg (no repo, no Terminal required).
DMG_README="$DMG_LAYOUT/00 - READ THIS FIRST (macOS install).txt"
cat > "$DMG_README" << 'READMEEOF'
BactiMate — install from this disk image (macOS)
==============================================

You only need this file (BactiMate.dmg). You do not need Python, Git, or the command line.

1) Double-click BactiMate.dmg. A window opens with BactiMate, an Applications shortcut, this note, and "Open BactiMate.command".

2) Drag the BactiMate icon onto the Applications folder (or drag BactiMate into your Applications folder in Finder).

3) The first time you open BactiMate, macOS may say the app is "damaged" or that it cannot be verified. That wording is misleading — the download is usually fine; the Mac is blocking apps that were not purchased from the App Store. Use step 4 or 5 once; after that BactiMate opens normally.

4) Easiest fix: double-click "Open BactiMate.command" once. It clears that block and starts BactiMate. You can run it from this window or after you copied BactiMate to Applications (the script looks in both places).

5) Alternative: open Finder → Applications → Control-click (right-click) BactiMate → Open → click Open in the dialog. You only need to do this the first time.

6) If your Mac is managed by IT and nothing works, ask them to allow BactiMate or use "Open Anyway" in System Settings → Privacy & Security after you try to open the app once.

Later updates: replace BactiMate in Applications with the newer app from a new DMG, then run step 4 or 5 again if macOS blocks the new copy.
READMEEOF

# Helper script: clears macOS quarantine so the app opens on fresh Macs (no Apple Developer ID needed).
# Users run this once after copying the app from the DMG; then they can open BactiMate normally.
OPEN_SCRIPT="$DMG_LAYOUT/Open BactiMate.command"
cat > "$OPEN_SCRIPT" << 'OPENEOF'
#!/bin/bash
set -e
APPLICATIONS_APP="/Applications/BactiMate.app"
DMG_APP="$(dirname "$0")/BactiMate.app"
if [ -d "$APPLICATIONS_APP" ]; then
  xattr -cr "$APPLICATIONS_APP" 2>/dev/null || true
  open "$APPLICATIONS_APP"
elif [ -d "$DMG_APP" ]; then
  xattr -cr "$DMG_APP" 2>/dev/null || true
  open "$DMG_APP"
else
  echo "BactiMate.app not found. Drag it to Applications first, then run this again."
  echo "Press Enter to close."
  read -r
fi
OPENEOF
chmod +x "$OPEN_SCRIPT"

echo "Creating disk image..."
# Remove existing DMG so hdiutil create doesn't prompt
rm -f "$DMG_PATH"
hdiutil create -volname "$DMG_NAME" -srcfolder "$DMG_LAYOUT" -ov -format UDZO "$DMG_PATH"

echo "Created: $DMG_PATH"
echo "  DMG-only users: see '00 - READ THIS FIRST (macOS install).txt' inside the image; drag app to Applications, then run 'Open BactiMate.command' once if macOS blocks the app."
