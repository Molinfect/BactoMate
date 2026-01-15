#!/bin/bash
# Convert logo2.png to .icns (macOS) and .ico (Windows) formats

set -e

SOURCE="logo2.png"
ICONSET_DIR="BactiMate.iconset"
ICNS_FILE="BactiMate.icns"
ICO_FILE="BactiMate.ico"

echo "=========================================="
echo "Creating application icons from logo2.png"
echo "=========================================="
echo ""

# Check if source exists
if [ ! -f "$SOURCE" ]; then
    echo "ERROR: $SOURCE not found!"
    exit 1
fi

# Clean up old files
rm -rf "$ICONSET_DIR"
rm -f "$ICNS_FILE" "$ICO_FILE"

echo "Creating macOS icon set (.icns)..."
mkdir -p "$ICONSET_DIR"

# Create all required icon sizes for macOS
# macOS requires multiple sizes in an iconset
sips -z 16 16     "$SOURCE" --out "${ICONSET_DIR}/icon_16x16.png" > /dev/null 2>&1
sips -z 32 32     "$SOURCE" --out "${ICONSET_DIR}/icon_16x16@2x.png" > /dev/null 2>&1
sips -z 32 32     "$SOURCE" --out "${ICONSET_DIR}/icon_32x32.png" > /dev/null 2>&1
sips -z 64 64     "$SOURCE" --out "${ICONSET_DIR}/icon_32x32@2x.png" > /dev/null 2>&1
sips -z 128 128   "$SOURCE" --out "${ICONSET_DIR}/icon_128x128.png" > /dev/null 2>&1
sips -z 256 256   "$SOURCE" --out "${ICONSET_DIR}/icon_128x128@2x.png" > /dev/null 2>&1
sips -z 256 256   "$SOURCE" --out "${ICONSET_DIR}/icon_256x256.png" > /dev/null 2>&1
sips -z 512 512   "$SOURCE" --out "${ICONSET_DIR}/icon_256x256@2x.png" > /dev/null 2>&1
sips -z 512 512   "$SOURCE" --out "${ICONSET_DIR}/icon_512x512.png" > /dev/null 2>&1
sips -z 1024 1024 "$SOURCE" --out "${ICONSET_DIR}/icon_512x512@2x.png" > /dev/null 2>&1

# Convert iconset to .icns
iconutil -c icns "$ICONSET_DIR" -o "$ICNS_FILE"

# Clean up iconset directory
rm -rf "$ICONSET_DIR"

if [ -f "$ICNS_FILE" ]; then
    echo "✓ Created: $ICNS_FILE"
else
    echo "✗ Failed to create $ICNS_FILE"
    exit 1
fi

echo ""
echo "Creating Windows icon (.ico)..."
# For Windows, create a multi-resolution ICO file
# ICO files can contain multiple sizes
# We'll create a simple ICO with common sizes using Python/PIL if available

if python3 -c "from PIL import Image" 2>/dev/null; then
    python3 << EOF
from PIL import Image
import sys

# Open source image
img = Image.open('$SOURCE')

# Create ICO file with multiple sizes
sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
images = []

for size in sizes:
    resized = img.resize(size, Image.Resampling.LANCZOS)
    images.append(resized)

# Save as ICO
images[0].save('$ICO_FILE', format='ICO', sizes=[(s[0], s[1]) for s in sizes])
print(f"✓ Created: $ICO_FILE")
EOF
else
    echo "⚠ PIL/Pillow not available. Creating basic ICO..."
    # Fallback: just copy and rename (won't work perfectly but better than nothing)
    # Actually, let's use sips to create a 256x256 PNG and note that user needs to convert manually
    sips -z 256 256 "$SOURCE" --out "${ICO_FILE%.ico}.png" > /dev/null 2>&1
    echo "⚠ Created ${ICO_FILE%.ico}.png - you may need to convert this to .ico manually"
    echo "   Or install Pillow: pip install Pillow"
    ICO_FILE="${ICO_FILE%.ico}.png"
fi

echo ""
echo "=========================================="
echo "Icon creation complete!"
echo "=========================================="
echo ""
echo "Created files:"
echo "  - $ICNS_FILE (macOS)"
if [ -f "$ICO_FILE" ]; then
    echo "  - $ICO_FILE (Windows)"
fi
echo ""
echo "These will be used automatically when you rebuild the app."
