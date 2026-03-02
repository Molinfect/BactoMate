# Follow Build Report Summary

## Build
- See `build_log.txt` for full build output.
- See `snapshot_after_build.txt` for app layout and key binaries.

## Launch chain (who started what)

### Launch 1

```
=== FOLLOW_BUILD LAUNCH 1 at 2026-03-02T23:58:50.969999 ===
LAUNCHER pid=33856 exe=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/MacOS/BactiMate bundle=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app
STRICT_LAUNCHER pid=33857 ppid=33856 script=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/setup/launch_gui_bundle_strict.sh root=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources
STRICT_ABOUT_TO_RUN python_cmd=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/venv/bin/BactiMate main_py=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/main.py
MAIN_PY pid=33874 ppid=33857 executable=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/venv/bin/BactiMate argv0=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/main.py
```

**Interpretation:**
- LAUNCHER ran with pid=33856 (this should be the main process of the app).
- Launcher exe path: /Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/MacOS/BactiMate (inside BactiMate.app: True)
- STRICT_LAUNCHER ppid=33856 matches LAUNCHER pid: script was started by the launcher (correct).
- MAIN_PY ppid=33857 matches STRICT_LAUNCHER pid: Python was started by the script (correct).
- Python executable: /Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/venv/bin/BactiMate (if this is Homebrew Python, the Dock may show Python.app)

### Launch 2

```
=== FOLLOW_BUILD LAUNCH 2 at 2026-03-02T23:59:42.544130 ===
LAUNCHER pid=33898 exe=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/MacOS/BactiMate bundle=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app
STRICT_LAUNCHER pid=33899 ppid=33898 script=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/setup/launch_gui_bundle_strict.sh root=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources
STRICT_ABOUT_TO_RUN python_cmd=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/venv/bin/BactiMate main_py=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/main.py
MAIN_PY pid=33916 ppid=33899 executable=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/venv/bin/BactiMate argv0=/Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/main.py
```

**Interpretation:**
- LAUNCHER ran with pid=33898 (this should be the main process of the app).
- Launcher exe path: /Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/MacOS/BactiMate (inside BactiMate.app: True)
- STRICT_LAUNCHER ppid=33898 matches LAUNCHER pid: script was started by the launcher (correct).
- MAIN_PY ppid=33899 matches STRICT_LAUNCHER pid: Python was started by the script (correct).
- Python executable: /Users/philipppopp/Desktop/BactiMate-main-V1.0/BactiMate.app/Contents/Resources/venv/bin/BactiMate (if this is Homebrew Python, the Dock may show Python.app)

## Why Dock / Reopen might open Python.app

If the **LAUNCHER** line is missing on either launch, the app's main process is not the native launcher.
macOS then associates the running app with whatever process has the menu bar (usually Python).
Pinning that in the Dock or using Reopen will open that app (Python.app) instead of BactiMate.app.

**Entry points matter:**
- Double-clicking **BactiMate.app** runs `Contents/MacOS/BactiMate` (the native launcher) → correct.
- Running `python main.py` or `./setup/launch_gui.sh` from the project folder does **not** use the launcher;
  the main process is Python, so Dock/Reopen will show Python.app. Always start the app via the .app bundle.

Fix: ensure the build produced `BactiMate.app/Contents/MacOS/BactiMate` as a **compiled binary** (see snapshot),
and that you start BactiMate only by double-clicking BactiMate.app (or its Dock icon after pinning the .app).

## Icon switch (not captured by this diagnostic)

This report does **not** record Dock or window icons over time. If you see:
- **One icon briefly** (e.g. BactiMate.app icon or a placeholder), then it **disappears or changes**,
- and the app then shows with a **different icon** (e.g. logo2.png in the window, or Python’s icon in the Dock),
that is **not** visible in the launch chain or process tree. Likely cause: macOS shows the **launcher’s** icon at first (the .app icon); when the **Python** process becomes the one with the menu bar / front window, the Dock can switch to the icon of that process. If the system still associates `venv/bin/BactiMate` (a copy of the Python binary) with **Python.app**, it will show Python’s icon. So you get: BactiMate icon → then Python icon. To fix that, the app would need to stop the Dock from switching to the child process’s identity (e.g. different executable or bundle association). If you observe this, note it when interpreting the report.
