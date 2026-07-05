#!/bin/bash
# Back-compat stub — use build/macos/build_app_bundle.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/macos/build_app_bundle.sh"
