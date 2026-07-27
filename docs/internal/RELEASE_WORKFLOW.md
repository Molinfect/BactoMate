# Release workflow (maintainers)

Use this checklist when shipping a new version (e.g. **v2.8.8**).  
**Audience:** you + Cursor (“follow `docs/internal/RELEASE_WORKFLOW.md` and release v2.8.8”).

**Canonical GitHub repo:** [Molinfect/BactoMate](https://github.com/Molinfect/BactoMate)  
Releases: [https://github.com/Molinfect/BactoMate/releases](https://github.com/Molinfect/BactoMate/releases)

```bash
# Point origin at the lab repo (once per clone)
git remote set-url origin https://github.com/Molinfect/BactoMate.git
git remote -v
```

## Branch roles

| Branch | Purpose |
|--------|---------|
| **`dev`** | Day-to-day work. Keeps `docs/internal/`, local logs, draft files. |
| **`main`** | Public GitHub + release tags. No `docs/internal/`. Empty `Logs/` only. |

**Rule:** run `./build/strip_dev_tree_for_main.sh` on **`main` after merging `dev`**, not on `dev`.  
If you strip and commit on `dev`, internal docs are gone from `dev` too.

---

## Pre-release (on `dev`)

1. **Finish features** and run tests:
   ```bash
   git checkout dev
   git pull origin dev
   pytest tests/ -q
   python main.py --smoke-test
   ```

2. **Bump version** (example `2.8.4` → `2.8.5`):
   - `bactomate/__init__.py` → `__version__`
   - `pyproject.toml` → `[project].version`
   - `docs/RELEASE_NOTES.md` → new section at top (date + bullets)
   - `docs/BACTOMATE_USER_MANUAL.md` → header **Version**
   - `README.md`, `docs/README.md`, `docs/INSTALL_AND_LAUNCH_windows.md` → version strings in install/verify examples (if present)

3. **Commit on `dev`** and push:
   ```bash
   git add -A
   git commit -m "Release v2.8.5: <short summary>"
   git push origin dev
   ```

---

## Publish to `main`

4. **Merge `dev` into `main`** (internal docs come along temporarily — that is OK):
   ```bash
   git checkout main
   git pull origin main
   git merge dev
   ```

5. **Strip developer-only tree on `main` only:**
   ```bash
   ./build/strip_dev_tree_for_main.sh
   git add -A
   git status   # confirm docs/internal/ deleted, Logs/ only .gitkeep
   git commit -m "chore: strip internal tree for public main (v2.8.5)"
   ```

6. **Push `main`:**
   ```bash
   git push origin main
   ```

7. **Return to `dev`** (unchanged — still has `docs/internal/`):
   ```bash
   git checkout dev
   ```

---

## Tag and CI release

8. **Tag on `main`** (must be `v*` for `.github/workflows/release.yml`):
   ```bash
   git checkout main
   git pull origin main
   git tag -a v2.8.5 -m "BactoMate v2.8.5"
   git push origin v2.8.5
   ```

9. **Watch GitHub Actions** → workflow **Create GitHub Release**:
   - Builds macOS DMG + Windows zip (both exclude `docs/internal/`)
   - Publishes release with assets attached

10. **Smoke-check** the [latest release](https://github.com/Molinfect/BactoMate/releases/latest):
    - Download zip / DMG if attached
    - Or clone `main` at tag from `https://github.com/Molinfect/BactoMate.git` and run install + `--smoke-test`

---

## Quick command block (copy-paste)

Replace `2.8.5` and edit the commit message.

```bash
# --- on dev ---
git checkout dev && git pull origin dev
pytest tests/ -q && python main.py --smoke-test
# (bump version + RELEASE_NOTES, then:)
git add -A && git commit -m "Release v2.8.5: <summary>" && git push origin dev

# --- merge to main, strip, push ---
git checkout main && git pull origin main && git merge dev
./build/strip_dev_tree_for_main.sh
git add -A && git commit -m "chore: strip internal tree for public main (v2.8.5)"
git push origin main

# --- tag (triggers CI) ---
git tag -a v2.8.5 -m "BactoMate v2.8.5"
git push origin v2.8.5

git checkout dev
```

---

## What CI builds automatically

Pushing tag `v*` runs `.github/workflows/release.yml`:

- **macOS:** `BactoMate.dmg` (may be skipped if &gt; 2 GiB)
- **Windows:** `BactoMate-win64.zip`

Bundles use `build/public_packaging.py` — no `docs/internal/`, empty `Logs/`.

Local builds (optional): see `build/README.md`.

---

## Cursor prompt (save this)

When you want the agent to run the process:

> Follow `docs/internal/RELEASE_WORKFLOW.md` and prepare release **v2.8.5**: bump version, update RELEASE_NOTES with [your bullets], commit on `dev`, merge to `main`, strip, tag, and push. Do not commit unless I confirm the RELEASE_NOTES text.

Adjust version and summary each time.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `docs/internal/` missing on `dev` after a bad strip | `git checkout dev -- docs/internal` from an earlier commit, or restore from `main~1` before strip |
| Tag pushed but CI failed | Fix on `main`, new tag `v2.8.5.1` or delete tag and re-push (only if release not published) |
| Version mismatch in GUI | `bactomate/__init__.py` and `pyproject.toml` must match; reinstall editable: `pip install -e .` |

---

## Optional later

- CI job on `main` that fails if `docs/internal/` exists
- `build/bump_version.py` to update all version strings in one step
