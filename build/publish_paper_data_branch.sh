#!/usr/bin/env bash
# Publish paper raw microscopy data to orphan branch ``paper-data`` on GitHub.
#
# Data lives on ``paper-data`` only — never on ``main`` / ``dev``. That keeps
# release bundles, CI, and source zips free of multi-GB raw TIFFs while still
# giving journals a stable public URL.
#
# Usage:
#   1. Copy raw files into ./paper_data/ (gitignored on dev/main).
#   2. ./build/publish_paper_data_branch.sh
#      or: ./build/publish_paper_data_branch.sh /path/to/your/raw/folder
#
# After push, cite for reviewers:
#   https://github.com/Molinfect/BactoMate/tree/paper-data
#   https://github.com/Molinfect/BactoMate/archive/refs/heads/paper-data.zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_SRC="${1:-$ROOT/paper_data}"
BRANCH="paper-data"
WORKTREE="$ROOT/../.bactomate-paper-data-worktree"

if [[ ! -d "$DATA_SRC" ]]; then
    echo "Error: data folder not found: $DATA_SRC" >&2
    echo "Copy your raw files into $ROOT/paper_data/ or pass a path as the first argument." >&2
    exit 1
fi

if [[ -z "$(find "$DATA_SRC" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Error: data folder is empty: $DATA_SRC" >&2
    exit 1
fi

cd "$ROOT"
ORIG_BRANCH="$(git branch --show-current 2>/dev/null || true)"
if [[ -n "$ORIG_BRANCH" ]] && [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: commit or stash working-tree changes before publishing paper data." >&2
    exit 1
fi

cleanup() {
    cd "$ROOT" 2>/dev/null || true
    if git worktree list 2>/dev/null | grep -q "$WORKTREE"; then
        git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
    fi
    if [[ -n "$ORIG_BRANCH" ]]; then
        git checkout "$ORIG_BRANCH" 2>/dev/null || true
    fi
}
trap cleanup EXIT

rm -rf "$WORKTREE"
mkdir -p "$(dirname "$WORKTREE")"
git worktree add --detach "$WORKTREE" HEAD

cd "$WORKTREE"
git checkout --orphan "$BRANCH"
git reset --hard
git clean -fdx

rsync -a --delete \
    --exclude ".DS_Store" \
    --exclude "._*" \
    "$DATA_SRC/" ./

cat > README.md <<'EOF'
# BactoMate — paper raw data

Raw microscopy files used in:

> Hallenga, L., Fornoff, S., Pesch, M., Kohlheyer, D., Ahmada, S., Hör, J., Erhardt, M., Popp, P. F. (2026).
> BactoMate: an integrated platform for reproducible bacterial microscopy analysis. *bioRxiv*.

Software source code and releases: [Molinfect/BactoMate](https://github.com/Molinfect/BactoMate) (`main` branch).

This branch is **data only**. It is not merged into `main` / `dev` and is not included in
BactoMate app bundles or GitHub release zips.

## Download

- Browse: https://github.com/Molinfect/BactoMate/tree/paper-data
- ZIP: https://github.com/Molinfect/BactoMate/archive/refs/heads/paper-data.zip
- Clone: `git clone --branch paper-data --depth 1 https://github.com/Molinfect/BactoMate.git BactoMate-paper-data`
EOF

git add -A
git commit -m "Paper raw microscopy data"

echo ""
echo "Pushing branch $BRANCH to origin ..."
git push -u origin "$BRANCH" --force
echo ""
echo "Public URLs for journal / reviewers:"
echo "  https://github.com/Molinfect/BactoMate/tree/paper-data"
echo "  https://github.com/Molinfect/BactoMate/archive/refs/heads/paper-data.zip"
