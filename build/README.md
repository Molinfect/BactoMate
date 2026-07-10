# BactoMate packaging (maintainers only)

> **End users:** you do not need this folder. Install from source via [`setup/`](../setup/README.md), or download **BactoMate.dmg** / **BactoMate-win64.zip** from [GitHub Releases](https://github.com/poppphil/BactoMate/releases/latest).

Packaging helpers for **distributable builds** after a normal source install. Scripts are split by platform — there is no single cross-platform build.

Tagged releases (`v*`) build these assets automatically in GitHub Actions and attach them to the release. To build locally, use the commands below.

## Quick reference

| Platform | What you get | Build (from project root) |
|----------|--------------|---------------------------|
| **macOS** | `BactoMate.app` + optional `.dmg` | `./build/macos/build_app_bundle.sh` then `./build/macos/create_dmg.sh` |
| **Windows** | Portable folder `dist\BactoMate\` + optional zip | `build\windows\build_windows_bundle.bat` then `build\windows\create_windows_zip.bat` |

**Prerequisite:** run the platform install script first (`setup/macos/install.sh` or `setup\windows\install.bat`).

## Layout

```
build/
  README.md           # This file
  macos/              # macOS .app and DMG (run on a Mac only)
  windows/            # Windows portable bundle + zip (run on Windows only)
```

### `build/macos/` — macOS only

| File | Purpose |
|------|---------|
| `build_app_bundle.sh` | Build `BactoMate.app` from source + venv |
| `create_dmg.sh` | Create `dist/BactoMate.dmg` |
| `create_icons.sh` | Generate `BactoMate.icns` for the app |
| `macos_launcher.c` | Native launcher binary for the `.app` |

These scripts **cannot** run on Windows or Linux.

### `build/windows/` — Windows only

| File | Purpose |
|------|---------|
| `build_windows_bundle.bat` | Entry point: builds `dist\BactoMate\` |
| `build_windows_bundle.py` | Copies project, venv, and Python 3.11 runtime |
| `create_windows_zip.bat` | Zips the bundle for distribution |

The portable bundle includes its own Python runtime — target PCs do **not** need a separate Python install.

These scripts **cannot** run on macOS.

## Typical workflows

### macOS (on a Mac)

```bash
./setup/macos/install.sh
./build/macos/build_app_bundle.sh
./build/macos/create_dmg.sh
```

Output: `dist/BactoMate.dmg` (Finder tag `v<version>` from `bactomate.__version__`)

### Windows (on a Windows PC)

```bat
setup\windows\install.bat
build\windows\build_windows_bundle.bat
build\windows\create_windows_zip.bat
```

Output: `dist\BactoMate-win64.zip`

Copy the zip to lab machines; launch with `BactoMate.bat` inside the extracted folder.

## Release checklist

Before uploading a manual release (if not using CI):

1. On `dev`, finish features and tests.
2. Run `./build/strip_dev_tree_for_main.sh`, commit, and merge into `main` (or run the script on `main` after merge).
3. Tag on `main` (`v*`).
4. Build DMG (macOS) and/or Windows zip on the target OS.
5. Attach `dist/BactoMate.dmg` and `dist/BactoMate-win64.zip` to the GitHub release.
6. Smoke-test: open the app, run `--version` and `--smoke-test` from the bundle.

### dev vs main

- **`dev`** — full tree, including `docs/internal/` (plans, join contracts, pipeline diagrams).
- **`main`** — public-facing clone; run `build/strip_dev_tree_for_main.sh` so internal docs and draft manuscripts are not published.

Bundles (DMG / zip) always exclude `docs/internal/` and include an empty `Logs/` directory regardless of branch.

Step-by-step release (version bump, merge, strip, tag): **`docs/internal/RELEASE_WORKFLOW.md`** on the `dev` branch.
