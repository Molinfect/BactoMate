#!/bin/bash
# Generate the macOS app icon used by build_app_bundle.sh.
# Run from project root: ./build/macos/create_icons.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ICONSET="$SCRIPT_DIR/BactoMate.iconset"
ICNS="$SCRIPT_DIR/BactoMate.icns"
SOURCE_ICON="${1:-}"

if [ -z "$SOURCE_ICON" ]; then
    for candidate in \
        "$PROJECT_ROOT/img/logo2.png" \
        "$PROJECT_ROOT/img/logo.png" \
        "$PROJECT_ROOT/assets/logo2.png" \
        "$PROJECT_ROOT/assets/logo.png"
    do
        if [ -f "$candidate" ]; then
            SOURCE_ICON="$candidate"
            break
        fi
    done
fi

rm -rf "$ICONSET"
mkdir -p "$ICONSET"

if [ -n "$SOURCE_ICON" ] && [ -f "$SOURCE_ICON" ] && command -v sips >/dev/null 2>&1; then
    echo "Generating iconset from $SOURCE_ICON"
    sips -z 16 16 "$SOURCE_ICON" --out "$ICONSET/icon_16x16.png" >/dev/null
    sips -z 32 32 "$SOURCE_ICON" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
    sips -z 32 32 "$SOURCE_ICON" --out "$ICONSET/icon_32x32.png" >/dev/null
    sips -z 64 64 "$SOURCE_ICON" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
    sips -z 128 128 "$SOURCE_ICON" --out "$ICONSET/icon_128x128.png" >/dev/null
    sips -z 256 256 "$SOURCE_ICON" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
    sips -z 256 256 "$SOURCE_ICON" --out "$ICONSET/icon_256x256.png" >/dev/null
    sips -z 512 512 "$SOURCE_ICON" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
    sips -z 512 512 "$SOURCE_ICON" --out "$ICONSET/icon_512x512.png" >/dev/null
    sips -z 1024 1024 "$SOURCE_ICON" --out "$ICONSET/icon_512x512@2x.png" >/dev/null
else
    echo "No source logo found; generating fallback BactoMate icon."
    python3 - "$ICONSET" <<'PY'
from __future__ import annotations

import math
import struct
import sys
import zlib
from pathlib import Path


def write_png(path: Path, size: int) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = []
    cx = cy = (size - 1) / 2.0
    radius = size * 0.39
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            dx = x - cx
            dy = y - cy
            d = math.hypot(dx, dy)
            if d <= radius:
                # Soft blue bacterium/cell shape.
                shade = max(0.0, min(1.0, 1.0 - d / radius))
                r = int(24 + 36 * shade)
                g = int(120 + 70 * shade)
                b = int(210 + 35 * shade)
                a = 255
            else:
                r = g = b = a = 0
            row.extend((r, g, b, a))
        rows.append(bytes(row))
    raw = b"".join(rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


iconset = Path(sys.argv[1])
mapping = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}
for name, size in mapping.items():
    write_png(iconset / name, size)
PY
fi

if command -v iconutil >/dev/null 2>&1; then
    rm -f "$ICNS"
    iconutil -c icns "$ICONSET" -o "$ICNS"
    echo "Created: $ICNS"
else
    echo "Warning: iconutil not found; leaving iconset at $ICONSET"
fi
