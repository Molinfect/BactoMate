#!/bin/bash
# Back-compat stub — use setup/macos/launch_gui.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/macos/launch_gui.sh"
