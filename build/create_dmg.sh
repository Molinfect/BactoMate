#!/bin/bash
# Back-compat stub — use build/macos/create_dmg.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/macos/create_dmg.sh"
