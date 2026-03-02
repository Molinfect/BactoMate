#!/usr/bin/env python3
"""
Check that BactiMate's required and optional dependencies are available.
Run from CLI: python -m bactimate.scripts.check_environment
Or from GUI: Help → Check environment
"""
from __future__ import annotations

import sys


def check_environment() -> str:
    """Check imports and return a multi-line report (OK or missing/error per component)."""
    lines = ["BactiMate environment check", "=" * 40]
    all_ok = True

    # Core (required for GUI and basic use)
    deps = [
        ("numpy", "NumPy"),
        ("tifffile", "tifffile"),
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl (Excel)"),
        ("skimage", "scikit-image"),
        ("PyQt6", "PyQt6"),
    ]
    for module, label in deps:
        try:
            __import__(module)
            lines.append(f"  OK   {label}")
        except ImportError as e:
            lines.append(f"  MISS {label}: {e}")
            all_ok = False

    # Microscopy: BioIO (primary) and legacy readers
    optional = [
        ("bioio", "BioIO (microscopy import)"),
        ("nd2", "ND2 (Nikon)"),
        ("PIL", "Pillow (image labels)"),
    ]
    for module, label in optional:
        try:
            __import__(module)
            lines.append(f"  OK   {label}")
        except ImportError:
            lines.append(f"  ---  {label} (optional, not installed)")

    # BioIO plugins (optional; report which formats are available)
    try:
        from bactimate.modules.bioio_reader import is_available as bioio_available
        if bioio_available():
            try:
                from bactimate.modules.bioio_reader import get_available_plugins
                plugins = get_available_plugins()
                if plugins:
                    lines.append(f"  OK   BioIO plugins: {', '.join(plugins)}")
                else:
                    lines.append("  OK   BioIO (plugins: install bioio-ome-tiff, bioio-czi, bioio-lif, bioio-nd2 as needed)")
            except Exception:
                lines.append("  OK   BioIO (plugin list skipped)")
        else:
            lines.append("  ---  BioIO not available")
    except Exception:
        lines.append("  ---  BioIO check skipped")

    lines.append("")
    if all_ok:
        lines.append("Result: All required dependencies are available.")
    else:
        lines.append("Result: Some required dependencies are missing. Install with: pip install -r requirements.txt")
    return "\n".join(lines)


def main() -> int:
    report = check_environment()
    print(report)
    return 0 if "All required" in report else 1


if __name__ == "__main__":
    sys.exit(main())
