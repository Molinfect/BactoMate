#!/bin/bash
# Build BactoMate.app from the current source tree and venv.
# Run from project root after ./setup/install_mac.sh:
#   ./build/build_app_bundle.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="BactoMate"
APP_DIR="$PROJECT_ROOT/${APP_NAME}.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"
VENV_DIR="$PROJECT_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: macOS app bundles can only be built on macOS."
    exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Error: $VENV_PYTHON not found."
    echo "Run ./setup/install_mac.sh first, then rerun this script."
    exit 1
fi

echo "Checking macOS app integration dependencies..."
MISSING_MACOS_DEPS="$("$VENV_PYTHON" - <<'PY'
import importlib.util
missing = [
    name
    for name in ("setproctitle", "AppKit", "Foundation")
    if importlib.util.find_spec(name) is None
]
print(",".join(missing))
PY
)"
if [ -n "$MISSING_MACOS_DEPS" ]; then
    echo "Missing in build venv: ${MISSING_MACOS_DEPS//,/, }"
    if [[ "$MISSING_MACOS_DEPS" == *"AppKit"* || "$MISSING_MACOS_DEPS" == *"Foundation"* ]]; then
        echo "Installing pyobjc-framework-Cocoa into the build venv..."
        "$VENV_PYTHON" -m pip install "pyobjc-framework-Cocoa"
    fi
    if [[ "$MISSING_MACOS_DEPS" == *"setproctitle"* ]]; then
        echo "Installing setproctitle into the build venv..."
        "$VENV_PYTHON" -m pip install "setproctitle>=1.3.0,<2.0.0"
    fi
    MISSING_MACOS_DEPS="$("$VENV_PYTHON" - <<'PY'
import importlib.util
missing = [
    name
    for name in ("setproctitle", "AppKit", "Foundation")
    if importlib.util.find_spec(name) is None
]
print(",".join(missing))
PY
)"
    if [ -n "$MISSING_MACOS_DEPS" ]; then
        echo "Error: macOS app integration dependencies still missing: ${MISSING_MACOS_DEPS//,/, }"
        echo "Run: $VENV_PYTHON -m pip install -r requirements.txt"
        echo "Or rerun: ./setup/install_mac.sh"
        exit 1
    fi
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "Error: rsync is required to build the app bundle."
    exit 1
fi

PY_INFO="$("$VENV_PYTHON" - <<'PY'
import sys
from pathlib import Path
print(f"{sys.version_info.major}.{sys.version_info.minor}")
print(Path(sys.base_prefix).resolve())
print(Path(sys.executable).resolve())
PY
)"
PY_MM="$(echo "$PY_INFO" | sed -n '1p')"
PY_BASE_PREFIX="$(echo "$PY_INFO" | sed -n '2p')"
PY_EXECUTABLE="$(echo "$PY_INFO" | sed -n '3p')"

echo "Building $APP_NAME.app"
echo "  Python: $PY_EXECUTABLE"
echo "  Python version: $PY_MM"
echo "  Base prefix: $PY_BASE_PREFIX"

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$FRAMEWORKS_DIR"

echo "Copying project files..."
rsync -a \
    --exclude ".git/" \
    --exclude ".pytest_cache/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "*.pyo" \
    --exclude ".DS_Store" \
    --exclude "BactoMate.app/" \
    --exclude "dist/" \
    --exclude "build/BactoMate.iconset/" \
    --exclude "build/BactoMate.icns" \
    --exclude "Logs/" \
    --exclude "venv/" \
    "$PROJECT_ROOT/" "$RESOURCES_DIR/"

echo "Copying virtual environment..."
rsync -a "$VENV_DIR/" "$RESOURCES_DIR/venv/"

FRAMEWORK_ROOT=""
if [[ "$PY_BASE_PREFIX" == *"/Python.framework/Versions/"* ]]; then
    FRAMEWORK_ROOT="${PY_BASE_PREFIX%/Versions/*}"
elif [ -d "$PY_BASE_PREFIX/../../Python.framework" ]; then
    FRAMEWORK_ROOT="$(cd "$PY_BASE_PREFIX/../../Python.framework" && pwd)"
fi

if [ -n "$FRAMEWORK_ROOT" ] && [ -d "$FRAMEWORK_ROOT" ]; then
    echo "Copying Python.framework..."
    rsync -a "$FRAMEWORK_ROOT/" "$FRAMEWORKS_DIR/Python.framework/"
    (
        cd "$FRAMEWORKS_DIR/Python.framework"
        ln -sfn "$PY_MM" Versions/Current
        ln -sfn "Versions/Current/Python" Python
        ln -sfn "Versions/Current/Headers" Headers
        ln -sfn "Versions/Current/Resources" Resources
    )
else
    echo "Warning: Python.framework not found. Bundle will only work where this venv's base Python exists."
fi

echo "Relinking bundled venv Python..."
(
    cd "$RESOURCES_DIR/venv/bin"
    rm -f "python" "python3" "python$PY_MM"
    if [ -d "$FRAMEWORKS_DIR/Python.framework/Versions/$PY_MM/bin" ]; then
        ln -s "../../../Frameworks/Python.framework/Versions/$PY_MM/bin/python$PY_MM" "python$PY_MM"
    else
        # Same-machine fallback when no framework was copied.
        ln -s "$PY_EXECUTABLE" "python$PY_MM"
    fi
    ln -s "python$PY_MM" python3
    ln -s "python$PY_MM" python
)

cat > "$RESOURCES_DIR/venv/bin/BactoMate" <<'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$SCRIPT_DIR/python3"
MAIN="$SCRIPT_DIR/../../main.py"
case "${1:-}" in
  -c|-m)
    exec "$PY" "$@"
    ;;
  --version|--smoke-test|-V|-v)
    exec "$PY" "$MAIN" "$@"
    ;;
esac
if [ "$#" -gt 0 ] && [ -f "$1" ]; then
    exec "$PY" "$@"
fi
exec "$PY" "$MAIN" "$@"
EOF
chmod +x "$RESOURCES_DIR/venv/bin/BactoMate"

echo "Creating native app launcher..."
LAUNCHER_SRC="$SCRIPT_DIR/macos_launcher.c"
LAUNCHER_BIN="$MACOS_DIR/$APP_NAME"
PY_FRAMEWORK_DIR="$FRAMEWORKS_DIR/Python.framework/Versions/$PY_MM"
PY_INCLUDE_DIR="$PY_FRAMEWORK_DIR/include/python${PY_MM}"
LINK_FRAMEWORK_DIR="$FRAMEWORKS_DIR/Python.framework/Versions/$PY_MM"
LINK_INCLUDE_DIR="$LINK_FRAMEWORK_DIR/include/python${PY_MM}"
LAUNCHER_CFLAGS=(-O2 -Wall -Wextra -DBACTOMATE_PY_MM=\"$PY_MM\")
LAUNCHER_LDFLAGS=(
    -Wl,-rpath,@executable_path/../Frameworks/Python.framework/Versions/$PY_MM
)

if [ -d "$LINK_FRAMEWORK_DIR" ] && [ -f "$LINK_INCLUDE_DIR/Python.h" ] && [ -f "$LINK_FRAMEWORK_DIR/Python" ]; then
    echo "  Linking launcher against Python.framework ($PY_MM)"
    clang "${LAUNCHER_CFLAGS[@]}" -DBACTOMATE_LINK_PYTHON=1 \
        -I"$LINK_INCLUDE_DIR" \
        "$LINK_FRAMEWORK_DIR/Python" \
        "${LAUNCHER_LDFLAGS[@]}" \
        "$LAUNCHER_SRC" \
        -o "$LAUNCHER_BIN"
else
    echo "  Warning: Python.framework headers not found; building exec fallback launcher."
    clang "${LAUNCHER_CFLAGS[@]}" \
        "${LAUNCHER_LDFLAGS[@]}" \
        "$LAUNCHER_SRC" \
        -o "$LAUNCHER_BIN"
fi
chmod +x "$LAUNCHER_BIN"
chmod +x "$RESOURCES_DIR/setup/"*.sh 2>/dev/null || true

echo "Creating Info.plist..."
APP_VERSION="$("$VENV_PYTHON" - <<'PY'
from bactomate import __version__
print(__version__)
PY
)"
cat > "$CONTENTS_DIR/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>BactoMate</string>
    <key>CFBundleExecutable</key>
    <string>BactoMate</string>
    <key>CFBundleIconFile</key>
    <string>BactoMate.icns</string>
    <key>CFBundleIdentifier</key>
    <string>de.hu-berlin.bactomate</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>BactoMate</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${APP_VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${APP_VERSION}</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
EOF
echo "APPL????" > "$CONTENTS_DIR/PkgInfo"

echo "Generating/copying app icon..."
if [ ! -f "$SCRIPT_DIR/BactoMate.icns" ]; then
    "$SCRIPT_DIR/create_icons.sh"
fi
if [ -f "$SCRIPT_DIR/BactoMate.icns" ]; then
    cp "$SCRIPT_DIR/BactoMate.icns" "$RESOURCES_DIR/BactoMate.icns"
fi

echo "Removing quarantine attributes..."
xattr -cr "$APP_DIR" 2>/dev/null || true

if command -v codesign >/dev/null 2>&1; then
    echo "Ad-hoc signing native binaries and app bundle..."
    find "$APP_DIR" \( -name "*.dylib" -o -name "*.so" \) -type f -print0 \
        | xargs -0 -n 1 codesign -f -s - 2>/dev/null || true
    find "$RESOURCES_DIR/venv/bin" -type f -perm -111 -print0 \
        | xargs -0 -n 1 codesign -f -s - 2>/dev/null || true
    codesign -f -s - --deep "$APP_DIR" >/dev/null 2>&1 || true
fi

echo "Verifying bundled Python..."
"$RESOURCES_DIR/venv/bin/BactoMate" --version
"$RESOURCES_DIR/venv/bin/BactoMate" --smoke-test
"$RESOURCES_DIR/venv/bin/python3" - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name in ("setproctitle", "AppKit", "Foundation")
    if importlib.util.find_spec(name) is None
]
if missing:
    print(
        "Error: bundled macOS app integration dependencies are missing: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    print(
        "Run ./setup/install_mac.sh or pip install -r requirements.txt, then rebuild.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

echo "Created: $APP_DIR"
echo "Next: ./build/create_dmg.sh"
