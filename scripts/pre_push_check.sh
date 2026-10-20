#!/usr/bin/env bash
# Local smoke + pytest gate before push (mirrors CI headless Qt steps).
# Usage: ./scripts/pre_push_check.sh
# Skip:  SKIP=1 git push

set -euo pipefail

if [[ "${SKIP:-}" == "1" ]]; then
  echo "pre_push_check: SKIP=1 — skipped."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

VENV_PY="$ROOT/venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "pre_push_check: missing venv at $ROOT/venv" >&2
  echo "Create one first, e.g. ./setup/macos/install.sh or python3 -m venv venv && pip install -r requirements.txt && pip install -e ." >&2
  exit 1
fi

export QT_QPA_PLATFORM=offscreen

echo "pre_push_check: smoke test"
"$VENV_PY" main.py --smoke-test

echo "pre_push_check: pytest (headless Qt)"
"$VENV_PY" -m pytest tests/ -q --tb=short

echo "pre_push_check: OK"
