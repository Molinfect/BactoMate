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
| **`dev`** | Day-to-day development, including publishable developer documentation. |
| **`main`** | Released code and documentation, with version tags. |

Both branches and their history are public. Keep publishable `docs/internal/` files on both branches; the directory name describes its developer audience, not access restrictions. Keep confidential notes, manuscripts, credentials, and sensitive data outside this repository in private storage. Deleting a committed file does not remove it from history.

Release builds exclude developer documentation, the known local manuscript draft, and runtime logs when copying files into bundles. Do not delete these paths from the source tree for releases. GitHub source archives include the tracked source tree, including developer documentation.

---

## Pre-release (on `dev`)

1. **Finish features** and run tests:
   ```bash
   git checkout dev
   git pull origin dev
   ./scripts/pre_push_check.sh
   ```
   (`SKIP=1` skips the script; CI still runs on code pushes. Docs-only pushes skip CI via `paths-ignore` in `build.yml`.)

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

4. **Update `main` and merge `dev`, keeping developer documentation:**
   ```bash
   git checkout main
   git pull --ff-only origin main
   ```

   **First release after retiring the old cleanup process:** earlier commits deleted `docs/internal/` on `main`. Git may retain those deletions or report modify/delete conflicts. Restore the publishable documentation from `dev` as part of this merge:
   ```bash
   git merge --no-commit dev
   # Documentation conflicts from the old cleanup are expected here.
   git restore --source=dev --staged --worktree -- docs/internal
   git status
   # Resolve any remaining conflicts and review the staged changes before committing.
   git commit
   ```

   **Subsequent releases:** merge normally:
   ```bash
   git merge dev
   ```

5. **Push `main`:**
   ```bash
   git push origin main
   ```

---

## Tag and CI release

6. **Tag on `main`** (must be `v*` for `.github/workflows/release.yml`):
   ```bash
   git tag -a v2.8.5 -m "BactoMate v2.8.5"
   git push origin v2.8.5
   ```

7. **Watch GitHub Actions** → workflow **Create GitHub Release**:
   - Builds macOS DMG + Windows zip (both exclude `docs/internal/`)
   - Publishes release with assets attached

8. **Smoke-check** the [latest release](https://github.com/Molinfect/BactoMate/releases/latest):
    - Download zip / DMG if attached
    - Or clone `main` at tag from `https://github.com/Molinfect/BactoMate.git` and run install + `--smoke-test`

9. **Sync `dev` with the released `main`** so release merge commits do not accumulate as missing history on `dev`:
   ```bash
   git checkout dev
   git pull --ff-only origin dev
   git merge main
   git push origin dev
   ```

---

## Quick command block (copy-paste)

Replace `2.8.5` and edit the commit message. For the first release after retiring cleanup commits, use the transition commands in **Publish to `main`** instead of `git merge dev` below.

```bash
# --- on dev ---
git checkout dev && git pull origin dev
pytest tests/ -q && python main.py --smoke-test
# or:
./scripts/pre_push_check.sh
# (bump version + RELEASE_NOTES, then:)
git add -A && git commit -m "Release v2.8.5: <summary>" && git push origin dev

# --- merge to main and push ---
git checkout main && git pull --ff-only origin main && git merge dev
git push origin main

# --- tag (triggers CI) ---
git tag -a v2.8.5 -m "BactoMate v2.8.5"
git push origin v2.8.5

# --- sync released history back to dev ---
git checkout dev && git pull --ff-only origin dev && git merge main
git push origin dev
```

---

## What CI builds automatically

Pushing tag `v*` runs `.github/workflows/release.yml`:

- **macOS:** `BactoMate.dmg` (LZMA-compressed; the release fails instead of publishing if it exceeds 2 GiB)
- **Windows:** `BactoMate-win64.zip`

Windows bundles use `build/public_packaging.py`; the macOS build applies exclusions through `rsync`. Both omit `docs/internal/` and the known local manuscript draft, and create an empty `Logs/` directory without changing the source checkout.

Local builds (optional): see `build/README.md`.

---

## Cursor prompt (save this)

When you want the agent to run the process:

> Follow `docs/internal/RELEASE_WORKFLOW.md` and prepare release **v2.8.5**: bump version, update RELEASE_NOTES with [your bullets], commit on `dev`, merge to `main` keeping developer documentation, tag, push, and sync `main` back into `dev`. Do not commit unless I confirm the RELEASE_NOTES text.

Adjust version and summary each time.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `docs/internal/` missing or in conflict during the first merge | On `main`, use the transition commands above to restore it from `dev` before committing the merge |
| Tag pushed but CI failed | Fix on `main`, new tag `v2.8.5.1` or delete tag and re-push (only if release not published) |
| Version mismatch in GUI | `bactomate/__init__.py` and `pyproject.toml` must match; reinstall editable: `pip install -e .` |

---

## Optional later

- `build/bump_version.py` to update all version strings in one step
