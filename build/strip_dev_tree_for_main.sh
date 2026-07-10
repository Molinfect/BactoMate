#!/usr/bin/env bash
# Remove developer-only files from the working tree before updating public main.
#
# Usage (on dev, after you are ready to release):
#   See docs/internal/RELEASE_WORKFLOW.md — strip on main AFTER merging dev, not on dev.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "Stripping developer-only paths under $ROOT ..."

rm -rf docs/internal

for name in BactoMate_Mns_NatureMethods_v2.docx; do
    if [ -e "$name" ]; then
        rm -f "$name"
        echo "  removed $name"
    fi
done

mkdir -p Logs
find Logs -mindepth 1 ! -name '.gitkeep' -print -delete 2>/dev/null || true
touch Logs/.gitkeep

echo "Done. Review: git status"
echo "Public trees should still include user docs under docs/ and an empty Logs/."
