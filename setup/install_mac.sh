#!/bin/bash
# BactoMate macOS one-shot install. Run from anywhere inside the repo.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/setup_fresh_env.sh"
