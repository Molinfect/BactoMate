#!/bin/bash
# Build BactoMate.app from the current source tree and venv.
# Run from project root after ./setup/macos/install.sh:
#   ./build/macos/build_app_bundle.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
    echo "Run ./setup/macos/install.sh first, then rerun this script."
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
        echo "Or rerun: ./setup/macos/install.sh"
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

# Relink copied Python.framework so subprocesses (venv/bin/python3) do not depend on Homebrew paths.
relink_bundled_python_framework() {
    local fw_ver="$1"
    local py_mm="$2"
    local launcher_bin="${3:-}"

    if ! command -v install_name_tool >/dev/null 2>&1; then
        echo "Warning: install_name_tool not available; bundled subprocess Python may not be portable."
        return 0
    fi

    local py_dylib="$fw_ver/Python"
    local py_bin="$fw_ver/bin/python$py_mm"
    if [ ! -f "$py_dylib" ] || [ ! -f "$py_bin" ]; then
        return 0
    fi

    local framework_bin_py_ref="@loader_path/../Python"
    local launcher_py_ref="@executable_path/../Frameworks/Python.framework/Versions/$py_mm/Python"
    local dylib_id="@rpath/Python"
    local launcher_rpath="@executable_path/../Frameworks/Python.framework/Versions/$py_mm"

    install_name_tool -id "$dylib_id" "$py_dylib" 2>/dev/null || true

    fix_macho_python_ref() {
        local target="$1"
        local new_ref="$2"
        local rpath="${3:-}"
        [ -f "$target" ] || return 0
        if ! file "$target" | grep -q "Mach-O"; then
            return 0
        fi
        local ref
        while IFS= read -r ref; do
            case "$ref" in
                *Python.framework*/Python|*Cellar/python@*|@rpath/Python)
                    install_name_tool -change "$ref" "$new_ref" "$target" 2>/dev/null || true
                    ;;
            esac
        done < <(otool -L "$target" 2>/dev/null | awk 'NR>1 {print $1}' | grep -E 'Python$|python@|@rpath/Python' || true)
        if [ -n "$rpath" ]; then
            install_name_tool -add_rpath "$rpath" "$target" 2>/dev/null || true
        fi
    }

    fix_macho_python_ref "$py_bin" "$framework_bin_py_ref"
    if [ -d "$fw_ver/bin" ]; then
        for bin in "$fw_ver/bin"/*; do
            fix_macho_python_ref "$bin" "$framework_bin_py_ref"
        done
    fi
    fix_macho_python_ref \
        "$fw_ver/Resources/Python.app/Contents/MacOS/Python" \
        "@loader_path/../../../../Python"
    if [ -n "$launcher_bin" ] && [ -f "$launcher_bin" ]; then
        fix_macho_python_ref "$launcher_bin" "$launcher_py_ref" "$launcher_rpath"
    fi

    if otool -L "$py_bin" 2>/dev/null | grep -q "Cellar/python"; then
        echo "Error: failed to relink $py_bin away from Homebrew Python."
        exit 1
    fi
    local python_app="$fw_ver/Resources/Python.app/Contents/MacOS/Python"
    if [ -f "$python_app" ] && otool -L "$python_app" 2>/dev/null | grep -q "Cellar/python"; then
        echo "Error: failed to relink $py_bin away from Homebrew Python."
        exit 1
    fi
}

if [ -d "$FRAMEWORKS_DIR/Python.framework/Versions/$PY_MM" ]; then
    echo "Relinking bundled Python.framework for portable subprocesses..."
    relink_bundled_python_framework "$FRAMEWORKS_DIR/Python.framework/Versions/$PY_MM" "$PY_MM"
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

if [ -d "$FRAMEWORKS_DIR/Python.framework/Versions/$PY_MM" ]; then
    PY_MICRO="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
    FRAMEWORK_BIN="$FRAMEWORKS_DIR/Python.framework/Versions/$PY_MM/bin"
    cat > "$RESOURCES_DIR/venv/pyvenv.cfg" <<EOF
home = $FRAMEWORK_BIN
include-system-site-packages = false
version = $PY_MICRO
executable = $FRAMEWORK_BIN/python$PY_MM
command = $PY_EXECUTABLE -m venv $VENV_DIR
EOF
fi

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
    relink_bundled_python_framework "$LINK_FRAMEWORK_DIR" "$PY_MM" "$LAUNCHER_BIN"
else
    echo "  Warning: Python.framework headers not found; building exec fallback launcher."
    clang "${LAUNCHER_CFLAGS[@]}" \
        "${LAUNCHER_LDFLAGS[@]}" \
        "$LAUNCHER_SRC" \
        -o "$LAUNCHER_BIN"
fi
chmod +x "$LAUNCHER_BIN"
chmod +x "$RESOURCES_DIR/setup/"*.sh "$RESOURCES_DIR/setup/macos/"*.sh 2>/dev/null || true

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
    if [ -d "$FRAMEWORKS_DIR/Python.framework" ]; then
        find "$FRAMEWORKS_DIR/Python.framework" -type f \( -perm -111 -o -name "Python" -o -name "*.dylib" \) -print0 \
            | xargs -0 -n 1 codesign -f -s - 2>/dev/null || true
    fi
    find "$APP_DIR" \( -name "*.dylib" -o -name "*.so" \) -type f -print0 \
        | xargs -0 -n 1 codesign -f -s - 2>/dev/null || true
    find "$RESOURCES_DIR/venv/bin" -type f -perm -111 -print0 \
        | xargs -0 -n 1 codesign -f -s - 2>/dev/null || true
    codesign -f -s - "$LAUNCHER_BIN" 2>/dev/null || true
    codesign -f -s - --deep "$APP_DIR" >/dev/null 2>&1 || true
fi

echo "Verifying bundled Python..."
"$LAUNCHER_BIN" --version
"$LAUNCHER_BIN" --smoke-test
"$RESOURCES_DIR/venv/bin/BactoMate" --version
"$RESOURCES_DIR/venv/bin/BactoMate" --smoke-test
FW_PY="$FRAMEWORKS_DIR/Python.framework/Versions/$PY_MM/bin/python$PY_MM"
if otool -L "$FW_PY" 2>/dev/null | grep -q "Cellar/python"; then
    echo "Error: bundled framework python still references Homebrew."
    exit 1
fi
"$RESOURCES_DIR/venv/bin/python3" --version
"$RESOURCES_DIR/venv/bin/python3" -c "import numpy; print('subprocess numpy ok')"
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
        "Run ./setup/macos/install.sh or pip install -r requirements.txt, then rebuild.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

RELEASE_TAG_SH="$SCRIPT_DIR/release_tag.sh"
if [[ -f "$RELEASE_TAG_SH" ]]; then
    # shellcheck source=release_tag.sh
    source "$RELEASE_TAG_SH"
    bactomate_tag_release_artifacts "$PROJECT_ROOT" "$APP_DIR"
fi

echo "Created: $APP_DIR"
echo "Next: ./build/macos/create_dmg.sh"
