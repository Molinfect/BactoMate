#!/bin/bash
# Strict launcher for macOS app bundles.
# One deterministic path, no runtime fallbacks.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/venv/bin" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Diagnostic for follow_build.py: who started this script (PPID = launcher or something else?)
CHAIN_LOG="$HOME/Library/Logs/BactoMate/launch_chain.log"
mkdir -p "$(dirname "$CHAIN_LOG")" 2>/dev/null || true
echo "STRICT_LAUNCHER pid=$$ ppid=$PPID script=$0 root=$PROJECT_ROOT" >> "$CHAIN_LOG" 2>/dev/null || true

LOG_DIR="$PROJECT_ROOT/Logs"
mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG_FILE="$LOG_DIR/launch.log"
exec >>"$LOG_FILE" 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S'): === Strict bundle launch attempt ==="
echo "$(date): Script dir: $SCRIPT_DIR"
echo "$(date): Project root: $PROJECT_ROOT"

USER_LOG="$HOME/Library/Logs/BactoMate/launch.log"
_log_user() { mkdir -p "$(dirname "$USER_LOG")" 2>/dev/null; echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$USER_LOG" 2>/dev/null; }

if [[ "$PROJECT_ROOT" != *".app/Contents/Resources"* ]]; then
    echo "$(date): ERROR: strict bundle launcher was executed outside a .app bundle."
    _log_user "Launch failed (not run from .app). See: $LOG_FILE"
    osascript -e 'display dialog "BactoMate Error: strict bundle launcher must run from BactoMate.app." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
fi

PYTHON_CMD="$PROJECT_ROOT/venv/bin/BactoMate"
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python3"
MAIN_PY="$PROJECT_ROOT/main.py"

if [ ! -x "$PYTHON_CMD" ]; then
    echo "$(date): ERROR: Missing executable: $PYTHON_CMD"
    _log_user "Launch failed: missing venv/bin/BactoMate. Full log: $LOG_FILE"
    osascript -e 'display dialog "BactoMate Error: missing venv/bin/BactoMate in app bundle. Rebuild the app." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
fi

if [ ! -f "$MAIN_PY" ]; then
    echo "$(date): ERROR: Missing main.py: $MAIN_PY"
    _log_user "Launch failed: missing main.py. Full log: $LOG_FILE"
    osascript -e 'display dialog "BactoMate Error: missing main.py in app bundle. Rebuild the app." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
    exit 1
fi

PY_MM=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
if [ -z "${PY_MM:-}" ]; then
    echo "$(date): ERROR: Could not query Python version from $PYTHON_BIN"
    _log_user "Launch failed: Python not runnable. Full log: $LOG_FILE"
    osascript -e "display dialog \"BactoMate Error: bundled launcher failed. Check: $LOG_FILE\" buttons {\"OK\"} default button \"OK\" with icon stop" 2>/dev/null || true
    exit 1
fi

QT_BASE="$PROJECT_ROOT/venv/lib/python${PY_MM}/site-packages/PyQt6/Qt6"
QT_PLUGIN_PATH="$QT_BASE/plugins"
QT_FRAMEWORK_PATH="$QT_BASE/lib"

if [ ! -d "$QT_PLUGIN_PATH" ]; then
    echo "$(date): ERROR: Missing Qt plugin path: $QT_PLUGIN_PATH"
    _log_user "Launch failed: missing Qt plugins. Full log: $LOG_FILE"
    osascript -e "display dialog \"BactoMate Error: missing Qt plugins. Check: $LOG_FILE\" buttons {\"OK\"} default button \"OK\" with icon stop" 2>/dev/null || true
    exit 1
fi

unset PYTHONHOME 2>/dev/null || true
unset QT_QPA_PLATFORM_PLUGIN_PATH 2>/dev/null || true

export PATH="$PROJECT_ROOT/venv/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export QT_PLUGIN_PATH="$QT_PLUGIN_PATH"
export QT_FRAMEWORK_PATH="$QT_FRAMEWORK_PATH"
export QT_MAC_WANTS_LAYER=1
export QT_AUTO_SCREEN_SCALE_FACTOR=1
if [ -d "$QT_FRAMEWORK_PATH" ]; then
    export DYLD_FRAMEWORK_PATH="$QT_FRAMEWORK_PATH"
fi

echo "$(date): Launching strict path:"
echo "$(date):   PYTHON_CMD=$PYTHON_CMD"
echo "$(date):   MAIN_PY=$MAIN_PY"
echo "$(date):   QT_PLUGIN_PATH=$QT_PLUGIN_PATH"
echo "STRICT_ABOUT_TO_RUN python_cmd=$PYTHON_CMD main_py=$MAIN_PY" >> "$CHAIN_LOG" 2>/dev/null || true

# Run Python as child; do not exec. So the app's main process stays the bundle's launcher script,
# and macOS never associates this app with Python.app (avoids reopen/redirect to Homebrew Python.app).
"$PYTHON_CMD" "$MAIN_PY"
exit $?
