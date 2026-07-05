#!/bin/bash
# Back-compat stub — use setup/macos/setup_fresh_env.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/macos/setup_fresh_env.sh"
