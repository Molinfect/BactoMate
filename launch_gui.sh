#!/bin/bash
# Stub: run the real launcher from setup/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/setup/launch_gui.sh"
