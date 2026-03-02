#!/usr/bin/env python3
"""
Follow Build & Launch – monitor every step of the BactiMate.app build and the first
(and optionally second) launches to produce a report that explains what was built,
how, and why relaunch/Dock pinning may redirect to Python.app.

Usage (from project root):
  python follow_build.py                    # build + 2 launches, full report
  python follow_build.py --no-build        # only run 2 launches (use existing app)
  python follow_build.py --build-only      # only build + snapshot, no launches
  python follow_build.py --launches 1      # build + 1 launch only

Output: follow_build_report/YYYY-MM-DD_HH-MM-SS/ with:
  - build_log.txt          (raw build output)
  - snapshot_after_build.txt (files in app, key binaries, Info.plist)
  - launch_chain_1.txt    (launch_chain.log excerpt for first launch)
  - process_tree_1.txt    (ps snapshot during first launch)
  - launch_chain_2.txt    (same for second launch)
  - process_tree_2.txt
  - SUMMARY.md            (what ran, parent chain, conclusions)
"""

from __future__ import annotations

import argparse
import os
import plistlib
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "BactiMate.app"
CHAIN_LOG = Path(os.path.expanduser("~/Library/Logs/BactiMate/launch_chain.log"))
REPORT_DIR = PROJECT_ROOT / "follow_build_report"


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, timeout: int | None = 300) -> tuple[int, str, str]:
    """Run command; return (returncode, stdout, stderr)."""
    cwd = cwd or PROJECT_ROOT
    env = env or os.environ
    try:
        r = subprocess.run(
            cmd,
            cwd=str(cwd),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def ensure_report_dir() -> Path:
    """Create follow_build_report/YYYY-MM-DD_HH-MM-SS/ and return it."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def do_build(out: Path) -> bool:
    """Run build_app_bundle.sh; save log. Return True if success."""
    build_script = PROJECT_ROOT / "build" / "build_app_bundle.sh"
    if not build_script.exists():
        with open(out / "build_log.txt", "w") as f:
            f.write("Error: build/build_app_bundle.sh not found\n")
        return False
    code, stdout, stderr = run(["bash", "-x", str(build_script)], timeout=600)
    log_path = out / "build_log.txt"
    with open(log_path, "w") as f:
        f.write("=== stdout ===\n")
        f.write(stdout)
        f.write("\n=== stderr ===\n")
        f.write(stderr)
        f.write("\n=== exit code ===\n")
        f.write(str(code))
    return code == 0


def snapshot_after_build(out: Path) -> None:
    """Record app contents, launcher binary, venv/BactiMate, Info.plist."""
    lines = []
    app = Path(APP_PATH)
    if not app.exists():
        lines.append("BactiMate.app not found.")
        with open(out / "snapshot_after_build.txt", "w") as f:
            f.write("\n".join(lines))
        return

    lines.append("=== BactiMate.app structure (key paths) ===")
    for name in ["Contents/MacOS/BactiMate", "Contents/Resources/main.py",
                 "Contents/Resources/setup/launch_gui_bundle_strict.sh",
                 "Contents/Resources/venv/bin/BactiMate", "Contents/Resources/BactiMate.icns",
                 "Contents/Info.plist"]:
        p = app / name
        if p.exists():
            st = p.stat()
            lines.append(f"  {name}: size={st.st_size} executable={os.access(p, os.X_OK)}")
        else:
            lines.append(f"  {name}: MISSING")

    lines.append("\n=== Contents/MacOS/BactiMate (file type) ===")
    launcher = app / "Contents" / "MacOS" / "BactiMate"
    if launcher.exists():
        code, so, se = run(["file", str(launcher)])
        lines.append(so.strip() or se.strip())
        code2, so2, se2 = run(["codesign", "-dv", str(launcher)])
        lines.append("codesign -dv: " + (so2.strip() or se2.strip() or "(none)"))
    else:
        lines.append("(missing)")

    lines.append("\n=== Info.plist (CFBundleExecutable, CFBundleIconFile) ===")
    plist_path = app / "Contents" / "Info.plist"
    if plist_path.exists():
        try:
            with open(plist_path, "rb") as f:
                pl = plistlib.load(f)
            for k in ["CFBundleExecutable", "CFBundleIconFile", "CFBundleIdentifier"]:
                lines.append(f"  {k}: {pl.get(k, '')}")
        except Exception as e:
            lines.append(f"  (parse error: {e})")
    else:
        lines.append("  (missing)")

    lines.append("\n=== venv/bin/BactiMate (file type) ===")
    vb = app / "Contents" / "Resources" / "venv" / "bin" / "BactiMate"
    if vb.exists():
        code, so, se = run(["file", str(vb)])
        lines.append(so.strip() or se.strip())
    else:
        lines.append("(missing)")

    with open(out / "snapshot_after_build.txt", "w") as f:
        f.write("\n".join(lines))


def clear_chain_log_marker(out: Path, launch_num: int) -> None:
    """Append a marker to launch_chain.log so we can split launches."""
    try:
        CHAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CHAIN_LOG, "a") as f:
            f.write("\n=== FOLLOW_BUILD LAUNCH %d at %s ===\n" % (launch_num, datetime.now().isoformat()))
    except Exception:
        pass


def read_chain_log_after_marker(launch_num: int) -> str:
    """Read launch_chain.log (bundle + user) and return the block for this launch (after marker)."""
    raw_parts = []
    bundle_log = APP_PATH / "Contents" / "Resources" / "Logs" / "launch_chain.log"
    if bundle_log.exists():
        try:
            raw_parts.append(bundle_log.read_text())
        except Exception:
            pass
    try:
        raw_parts.append(CHAIN_LOG.read_text())
    except Exception:
        pass
    raw = "\n".join(raw_parts) if raw_parts else ""
    if not raw:
        return "(could not read launch_chain.log from bundle or ~/Library/Logs/BactiMate)"
    # Use last occurrence of this launch's marker (current run), not an older run's
    marker = "=== FOLLOW_BUILD LAUNCH %d " % launch_num
    idx = -1
    at = 0
    while True:
        found = raw.find(marker, at)
        if found == -1:
            break
        idx = found
        at = found + 1
    if idx == -1:
        return raw
    end = raw.find("=== FOLLOW_BUILD LAUNCH ", idx + len(marker))
    if end == -1:
        return raw[idx:]
    return raw[idx:end]


def capture_process_tree() -> str:
    """Capture ps -ef and a few ps variants to see process hierarchy."""
    out = []
    for title, cmd in [
        ("ps -ef (full)", ["ps", "-ef"]),
        ("ps -eo pid,ppid,comm,args (trunc)", ["ps", "-eo", "pid,ppid,comm,args"]),
    ]:
        code, so, se = run(cmd, timeout=5)
        out.append("--- " + title + " ---\n" + (so or se))
    return "\n".join(out)


def wait_until_bactimate_gone(timeout_sec: int = 20) -> bool:
    """Wait until no BactiMate.app process is running (so next launch starts fresh). Returns True if gone."""
    app_path_str = str(APP_PATH)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        code, so, _ = run(
            ["pgrep", "-f", app_path_str],
            timeout=5,
        )
        if code != 0 or not so.strip():
            return True
        time.sleep(1)
    return False


def launch_app_and_capture(out: Path, launch_num: int, wait_sec: int = 30, quit_for_next: bool = True) -> None:
    """Open BactiMate.app, wait, capture chain log and process tree, then quit so next launch is clean."""
    clear_chain_log_marker(out, launch_num)
    # Use 'open' so we simulate real user double-click (LaunchServices)
    run(["open", "-a", str(APP_PATH)], timeout=10)
    time.sleep(wait_sec)
    chain = read_chain_log_after_marker(launch_num)
    tree = capture_process_tree()
    (out / ("launch_chain_%d.txt" % launch_num)).write_text(chain)
    (out / ("process_tree_%d.txt" % launch_num)).write_text(tree)
    if not quit_for_next:
        return
    # Quit the app so the next launch starts a new instance (and writes to chain log)
    run(["osascript", "-e", 'quit app "BactiMate"'], timeout=5)
    time.sleep(3)
    gone = wait_until_bactimate_gone(timeout_sec=15)
    if not gone:
        print("  (App did not quit gracefully; force-killing BactiMate.app processes...)")
        run(["pkill", "-f", str(APP_PATH)], timeout=5)
        time.sleep(2)
        if not wait_until_bactimate_gone(timeout_sec=5):
            print("  (Warning: some BactiMate processes may still be running)")
    time.sleep(1)


def write_summary(out: Path, num_launches: int) -> None:
    """Write SUMMARY.md: interpret chain logs and process trees."""
    lines = [
        "# Follow Build Report Summary",
        "",
        "## Build",
        "- See `build_log.txt` for full build output.",
        "- See `snapshot_after_build.txt` for app layout and key binaries.",
        "",
        "## Launch chain (who started what)",
        ""
    ]
    for i in range(1, num_launches + 1):
        chain_path = out / ("launch_chain_%d.txt" % i)
        if not chain_path.exists():
            continue
        chain = chain_path.read_text()
        lines.append("### Launch %d" % i)
        lines.append("")
        lines.append("```")
        lines.append(chain.strip())
        lines.append("```")
        lines.append("")
        # Parse and interpret
        launcher_pid = None
        launcher_exe = None
        strict_pid = None
        strict_ppid = None
        main_pid = None
        main_ppid = None
        main_executable = None
        for line in chain.splitlines():
            if line.startswith("LAUNCHER pid="):
                m = re.search(r"pid=(\d+)\s+exe=(\S+)\s+bundle=(\S+)", line)
                if m:
                    launcher_pid, launcher_exe, _ = m.group(1), m.group(2), m.group(3)
                else:
                    m = re.search(r"pid=(\d+)", line)
                    if m:
                        launcher_pid = m.group(1)
            elif line.startswith("STRICT_LAUNCHER pid="):
                m = re.search(r"pid=(\d+)\s+ppid=(\d+)", line)
                if m:
                    strict_pid, strict_ppid = m.group(1), m.group(2)
            elif line.startswith("MAIN_PY pid="):
                m = re.search(r"executable=(\S+)", line)
                if m:
                    main_executable = m.group(1)
                m = re.search(r"pid=(\d+)\s+ppid=(\d+)", line)
                if m:
                    main_pid, main_ppid = m.group(1), m.group(2)
        lines.append("**Interpretation:**")
        if launcher_pid:
            lines.append("- LAUNCHER ran with pid=%s (this should be the main process of the app)." % launcher_pid)
            if launcher_exe:
                inside = "BactiMate.app" in (launcher_exe or "")
                lines.append("- Launcher exe path: %s (inside BactiMate.app: %s)" % (launcher_exe, inside))
        else:
            lines.append("- **LAUNCHER line missing**: the native launcher did not run. The app may be using an old script or a different entry point (e.g. launch_gui.sh). Dock/Reopen will then associate with Python.")
        if strict_ppid and launcher_pid and strict_ppid == launcher_pid:
            lines.append("- STRICT_LAUNCHER ppid=%s matches LAUNCHER pid: script was started by the launcher (correct)." % strict_ppid)
        elif strict_ppid and launcher_pid:
            lines.append("- STRICT_LAUNCHER ppid=%s != LAUNCHER pid %s: script was NOT started by our launcher (wrong parent)." % (strict_ppid, launcher_pid))
        if main_ppid and strict_pid and main_ppid == strict_pid:
            lines.append("- MAIN_PY ppid=%s matches STRICT_LAUNCHER pid: Python was started by the script (correct)." % main_ppid)
        elif main_ppid and strict_pid:
            lines.append("- MAIN_PY ppid=%s != STRICT_LAUNCHER pid %s: Python was not started by our script." % (main_ppid, strict_pid))
        if main_executable:
            lines.append("- Python executable: %s (if this is Homebrew Python, the Dock may show Python.app)" % main_executable)
        lines.append("")
    lines.append("## Why Dock / Reopen might open Python.app")
    lines.append("")
    lines.append("The app uses an **embedded-Python launcher**: the GUI runs in the same process as the native launcher (no separate Python process), so the Dock and pin always show BactiMate.app. The Python framework is copied into the app, so the DMG works on any Mac.")
    lines.append("")
    lines.append("If the **LAUNCHER** line is missing on either launch, the app's main process is not the native launcher.")
    lines.append("macOS then associates the running app with whatever process has the menu bar (usually Python).")
    lines.append("Pinning that in the Dock or using Reopen will open that app (Python.app) instead of BactiMate.app.")
    lines.append("")
    lines.append("**Entry points matter:**")
    lines.append("- Double-clicking **BactiMate.app** runs `Contents/MacOS/BactiMate` (the native launcher) → correct.")
    lines.append("- Running `python main.py` or `./setup/launch_gui.sh` from the project folder does **not** use the launcher;")
    lines.append("  the main process is Python, so Dock/Reopen will show Python.app. Always start the app via the .app bundle.")
    lines.append("")
    lines.append("Fix: ensure the build produced `BactiMate.app/Contents/MacOS/BactiMate` as a **compiled binary** (see snapshot),")
    lines.append("and that you start BactiMate only by double-clicking BactiMate.app (or its Dock icon after pinning the .app).")
    lines.append("")
    lines.append("## Icon switch (not captured by this diagnostic)")
    lines.append("")
    lines.append("This report does **not** record Dock or window icons over time. If you see:")
    lines.append("- **One icon briefly** (e.g. BactiMate.app icon or a placeholder), then it **disappears or changes**,")
    lines.append("- and the app then shows with a **different icon** (e.g. logo2.png in the window, or Python’s icon in the Dock),")
    lines.append("that is **not** visible in the launch chain or process tree. Likely cause: macOS shows the **launcher’s** icon at first (the .app icon); when the **Python** process becomes the one with the menu bar / front window, the Dock can switch to the icon of that process. If the system still associates `venv/bin/BactiMate` (a copy of the Python binary) with **Python.app**, it will show Python’s icon. So you get: BactiMate icon → then Python icon. To fix that, the app would need to stop the Dock from switching to the child process’s identity (e.g. different executable or bundle association). If you observe this, note it when interpreting the report.")
    lines.append("")
    (out / "SUMMARY.md").write_text("\n".join(lines))


def main() -> int:
    global APP_PATH
    parser = argparse.ArgumentParser(description="Monitor BactiMate build and launches; produce report.")
    parser.add_argument("--no-build", action="store_true", help="Skip build; use existing BactiMate.app")
    parser.add_argument("--build-only", action="store_true", help="Only build and snapshot; no launches")
    parser.add_argument("--launches", type=int, default=2, metavar="N", help="Number of launches to capture (default 2)")
    parser.add_argument("--app-path", type=Path, default=APP_PATH, help="Path to BactiMate.app")
    args = parser.parse_args()
    APP_PATH = args.app_path

    out = ensure_report_dir()
    print("Report directory:", out)

    if not args.no_build:
        print("Running build (build_app_bundle.sh with -x)...")
        ok = do_build(out)
        print("Build exit:", "OK" if ok else "FAILED")
        snapshot_after_build(out)
        print("Snapshot written.")
        if args.build_only:
            print("Build-only: skipping launches.")
            return 0 if ok else 1
        if not ok:
            print("Build failed; aborting launches.")
            return 1
    else:
        if not APP_PATH.exists():
            print("Error: --no-build but BactiMate.app not found at", APP_PATH)
            return 1
        snapshot_after_build(out)

    num_launches = max(0, min(args.launches, 5))
    for i in range(1, num_launches + 1):
        print("Launch %d: opening app, waiting 30s, capturing chain log and process tree..." % i)
        launch_app_and_capture(out, i, quit_for_next=(i < num_launches))
        if i < num_launches:
            print("App quit; starting next launch...")

    write_summary(out, num_launches)
    print("Summary written to", out / "SUMMARY.md")
    print("Done. Inspect", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
