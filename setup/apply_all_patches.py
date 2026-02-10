#!/usr/bin/env python3
"""
Apply all BactiMate compatibility patches in one place.
Patches applied:
1. peakdetect - scipy ifft import fix (scipy >= 1.10)
2. cellpose_omni - model_type parameter fix (architecture detection)
3. cellpose_omni - model URL fix (GitHub instead of Cellpose server)
"""

import sys
from pathlib import Path
from typing import Optional


def _find_in_syspath(relative_path: str) -> Optional[Path]:
    """Return first existing path under sys.path, or None."""
    for p in sys.path:
        candidate = Path(p) / relative_path
        if candidate.exists():
            return candidate
    return None


def patch_peakdetect() -> bool:
    """Patch peakdetect/peakdetect.py to fix scipy ifft import (scipy >= 1.10)."""
    peakdetect_file = _find_in_syspath("peakdetect/peakdetect.py")
    if not peakdetect_file:
        print("WARNING: Could not find peakdetect/peakdetect.py (package may not be installed yet)")
        return False

    content = peakdetect_file.read_text()
    if "except ImportError:" in content and "scipy.fft" in content:
        print("  ✓ peakdetect is already patched")
        return True

    old_import = "from scipy import fft, ifft"
    new_import = """try:
    from scipy import fft, ifft
except ImportError:
    try:
        from scipy.fft import fft, ifft
    except ImportError:
        from numpy.fft import fft, ifft"""

    if old_import not in content:
        print("  WARNING: Could not find expected import in peakdetect.py")
        return False

    backup = peakdetect_file.with_suffix(".py.backup")
    if not backup.exists():
        backup.write_text(content)
        print(f"  Created backup at: {backup}")
    content = content.replace(old_import, new_import)
    peakdetect_file.write_text(content)
    print("  ✓ Patched peakdetect (scipy compatibility)")
    return True


def patch_cellpose_omni() -> bool:
    """Patch cellpose_omni/models.py: model_type fix and model URL fix."""
    models_file = _find_in_syspath("cellpose_omni/models.py")
    if not models_file:
        print("  WARNING: Could not find cellpose_omni/models.py")
        return False

    content = models_file.read_text()
    original = content
    changed = False

    # 1) model_type fix: pass model_type to CellposeModel for architecture detection
    old_line = """        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                pretrained_model=self.pretrained_model,
                                diam_mean=self.diam_mean, use_torch=self.torch, 
                                dim=self.dim, omni=self.omni)"""
    new_line = """        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                pretrained_model=self.pretrained_model,
                                model_type=model_type,  # FIX: Pass model_type for architecture detection
                                diam_mean=self.diam_mean, use_torch=self.torch, 
                                dim=self.dim, omni=self.omni)"""
    if old_line in content and new_line not in content:
        content = content.replace(old_line, new_line)
        changed = True

    # 2) Model URL fix: use Omnipose GitHub instead of Cellpose server
    old_url = "_MODEL_URL = 'https://www.cellpose.org/models'"
    new_url = "_MODEL_URL = 'https://raw.githubusercontent.com/kevinjohncutler/omnipose-models/main'"
    if old_url in content:
        content = content.replace(old_url, new_url)
        changed = True
    elif "raw.githubusercontent.com/kevinjohncutler/omnipose-models/main" in content:
        pass  # already patched
    else:
        print("  WARNING: Could not find _MODEL_URL line in cellpose_omni/models.py")

    if not changed:
        if "model_type=model_type" in content and "raw.githubusercontent.com" in content:
            print("  ✓ cellpose_omni is already patched")
            return True
        print("  WARNING: No changes applied to cellpose_omni/models.py")
        return False

    backup = Path(str(models_file) + ".backup")
    if not backup.exists():
        backup.write_text(original)
        print(f"  Created backup at: {backup}")
    models_file.write_text(content)
    print("  ✓ Patched cellpose_omni (model_type + model URL)")
    return True


def main() -> bool:
    print("=" * 50)
    print("Applying BactiMate compatibility patches")
    print("=" * 50)

    ok_peak = False
    ok_cellpose = False

    print("\nPatching peakdetect (scipy compatibility)...")
    try:
        ok_peak = patch_peakdetect()
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\nPatching cellpose_omni (model_type + model URL)...")
    try:
        ok_cellpose = patch_cellpose_omni()
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 50)
    print(f"Done. peakdetect: {'OK' if ok_peak else 'skipped/failed'}; cellpose_omni: {'OK' if ok_cellpose else 'skipped/failed'}")
    print("=" * 50)
    return ok_peak or ok_cellpose


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
