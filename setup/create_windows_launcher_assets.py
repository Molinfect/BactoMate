"""
Create Windows launcher assets: img/BactoMate.ico and BactoMate.lnk shortcut.

.bat files cannot carry a custom icon in Explorer; the shortcut uses the logo so
double-click looks like launching an app. Run from install_windows.bat step 7.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_ico(root: Path) -> Path | None:
    ico = root / "img" / "BactoMate.ico"
    png = root / "img" / "logo2.png"
    if ico.is_file():
        return ico
    if not png.is_file():
        print(f"[WARN] Logo not found: {png}", file=sys.stderr)
        return None
    try:
        from PIL import Image
    except ImportError:
        print("[WARN] Pillow not installed; skipping BactoMate.ico creation.", file=sys.stderr)
        return None
    img = Image.open(png).convert("RGBA")
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    img.save(ico, format="ICO", sizes=sizes)
    return ico


def ensure_shortcut(
    root: Path,
    ico: Path | None,
    *,
    lnk_path: Path | None = None,
    description: str = "Launch BactoMate",
) -> Path | None:
    bat = root / "launch_windows.bat"
    lnk = lnk_path if lnk_path is not None else root / "BactoMate.lnk"
    if not bat.is_file():
        print(f"[WARN] Launcher not found: {bat}", file=sys.stderr)
        return None
    try:
        import win32com.client  # type: ignore[import-untyped]
    except ImportError:
        return _ensure_shortcut_powershell(root, bat, lnk, ico, description=description)

    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(str(lnk))
    shortcut.TargetPath = str(bat.resolve())
    shortcut.WorkingDirectory = str(root.resolve())
    shortcut.WindowStyle = 7  # Minimized (if shell uses shortcut to open console)
    shortcut.Description = description
    if ico is not None and ico.is_file():
        shortcut.IconLocation = f"{ico.resolve()},0"
    shortcut.Save()
    return lnk if lnk.is_file() else None


def _desktop_directory() -> Path | None:
    try:
        import win32com.client  # type: ignore[import-untyped]

        shell = win32com.client.Dispatch("WScript.Shell")
        desktop = shell.SpecialFolders("Desktop")
        if desktop:
            return Path(str(desktop))
    except Exception:
        pass
    candidate = Path.home() / "Desktop"
    return candidate if candidate.is_dir() else None


def ensure_desktop_shortcut(root: Path, ico: Path | None) -> Path | None:
    desktop = _desktop_directory()
    if desktop is None:
        print("[WARN] Desktop folder not found; skipping desktop shortcut.", file=sys.stderr)
        return None
    return ensure_shortcut(
        root,
        ico,
        lnk_path=desktop / "BactoMate.lnk",
        description="Launch BactoMate",
    )


def _ensure_shortcut_powershell(
    root: Path,
    bat: Path,
    lnk: Path,
    ico: Path | None,
    *,
    description: str = "Launch BactoMate",
) -> Path | None:
    import subprocess

    icon_line = ""
    if ico is not None and ico.is_file():
        icon_line = f'$s.IconLocation = "{ico.resolve()},0"'
    ps = f"""
$w = New-Object -ComObject WScript.Shell
$s = $w.CreateShortcut('{lnk.resolve()}')
$s.TargetPath = '{bat.resolve()}'
$s.WorkingDirectory = '{root.resolve()}'
$s.Description = '{description}'
$s.WindowStyle = 7
{icon_line}
$s.Save()
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[WARN] Could not create BactoMate.lnk: {exc}", file=sys.stderr)
        return None
    return lnk if lnk.is_file() else None


def main() -> int:
    root = _project_root()
    ico = ensure_ico(root)
    created: list[Path] = []
    lnk = ensure_shortcut(root, ico)
    if lnk is not None:
        created.append(lnk)
        print(f"Created shortcut: {lnk}")
    desktop_lnk = ensure_desktop_shortcut(root, ico)
    if desktop_lnk is not None:
        created.append(desktop_lnk)
        print(f"Created desktop shortcut: {desktop_lnk}")
    if not created and ico is not None:
        print(f"Icon ready: {ico}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
