# 03 — Source zip and release page point at the packaged apps

**What to build:** Someone who downloads GitHub’s automatic “Source code (zip)” sees, in the extracted folder, that a lab install is the DMG or the Windows zip. The release description leads with those files. It does not tell macOS users to clone the repo and run the source installer when the DMG is missing. User-facing docs name Python 3.11, not “3.11+”.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] A plain-text install note at the repository root says the source archive is for developers, and lab users should download `BactoMate.dmg` or `BactoMate-win64.zip` from the GitHub release.
- [x] The project README points source-zip readers at that note.
- [x] The release workflow body lists the DMG and the Windows zip first, says the source archive is not the app, and does not give `./setup/macos/install.sh` as the fallback when no DMG was uploaded.
- [x] User-facing install docs require Python 3.11 specifically.
