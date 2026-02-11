#!/usr/bin/env python3
"""
Apply all BactiMate compatibility patches in one place.
Patches applied:
1. peakdetect - scipy ifft import fix (scipy >= 1.10)
2. cellpose_omni - model_type parameter fix (architecture detection)
3. cellpose_omni - model URL fix (GitHub instead of Cellpose server)
4. cellpose_omni - register bact_phase_affinity (checkpoint is 1-channel; do not set nchan=2)
5. cellpose_omni transforms - trim to nchan after reshape (so 1-channel models get 1-channel input)
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

    # 3) Register bact_phase_affinity so it is not replaced with 'cyto' (path would be overwritten)
    if "'bact_phase_affinity'" not in content:
        if "MODEL_NAMES = MCHN_MODEL_NAMES + MONO_OMNI_MODELS" in content:
            content = content.replace(
                "MODEL_NAMES = MCHN_MODEL_NAMES + MONO_OMNI_MODELS",
                "MODEL_NAMES = MCHN_MODEL_NAMES + MONO_OMNI_MODELS + ['bact_phase_affinity']",
                1,
            )
            changed = True
        elif "MODEL_NAMES = ['cyto','nuclei','cyto2']+OMNI_MODELS" in content:
            content = content.replace(
                "MODEL_NAMES = ['cyto','nuclei','cyto2']+OMNI_MODELS",
                "MODEL_NAMES = ['cyto','nuclei','cyto2']+OMNI_MODELS + ['bact_phase_affinity']",
                1,
            )
            changed = True
        elif "MODEL_NAMES = C1_MODELS + C2_BD_MODELS + C1_BD_MODELS + C2_MODELS + CP_MODELS" in content:
            content = content.replace(
                "MODEL_NAMES = C1_MODELS + C2_BD_MODELS + C1_BD_MODELS + C2_MODELS + CP_MODELS",
                "MODEL_NAMES = C1_MODELS + C2_BD_MODELS + C1_BD_MODELS + C2_MODELS + CP_MODELS + ['bact_phase_affinity']",
                1,
            )
            changed = True

    # 4) Affinity (bact_phase_affinity): checkpoint in omnipose-models is 1-channel. Ensure we do NOT
    #    set nchan=2 for affinity (revert if a previous patch added it), so the model stays 1-channel.
    if "model_type in C2_MODEL_NAMES or (model_type and 'affinity' in model_type)" in content:
        content = content.replace(
            "if model_type in C2_MODEL_NAMES or (model_type and 'affinity' in model_type):\n                self.nchan = 2",
            "if model_type in C2_MODEL_NAMES:\n                self.nchan = 2",
            1,
        )
        changed = True

    if not changed:
        # Consider already patched: URL + bact_phase_affinity registered, and no wrong nchan=2 for affinity
        already_ok = (
            "raw.githubusercontent.com" in content
            and "bact_phase_affinity" in content
            and "model_type in C2_MODEL_NAMES or (model_type and 'affinity'" not in content
        )
        if already_ok:
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


def patch_cellpose_omni_transforms() -> bool:
    """Patch cellpose_omni/transforms.py: after reshape(), trim to nchan so 1-channel models get 1-channel input."""
    transforms_file = _find_in_syspath("cellpose_omni/transforms.py")
    if not transforms_file:
        print("  WARNING: Could not find cellpose_omni/transforms.py")
        return False

    content = transforms_file.read_text()
    old = "        x = reshape(x, channels=channels, channel_axis=channel_axis)\n        # print('AAA',x.shape,channels)\n    else:"
    new = "        x = reshape(x, channels=channels, channel_axis=channel_axis)\n        if x.shape[-1] > nchan and x.ndim > dim:\n            x = x[..., :nchan]\n        # print('AAA',x.shape,channels)\n    else:"

    if new in content:
        print("  ✓ cellpose_omni transforms (nchan trim) already patched")
        return True
    if old not in content:
        print("  WARNING: Could not find expected block in cellpose_omni/transforms.py")
        return False

    content = content.replace(old, new)
    transforms_file.write_text(content)
    print("  ✓ Patched cellpose_omni transforms (nchan trim)")
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

    ok_transforms = False
    print("\nPatching cellpose_omni transforms (nchan trim)...")
    try:
        ok_transforms = patch_cellpose_omni_transforms()
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 50)
    print(f"Done. peakdetect: {'OK' if ok_peak else 'skipped/failed'}; cellpose_omni: {'OK' if ok_cellpose else 'skipped/failed'}; transforms: {'OK' if ok_transforms else 'skipped/failed'}")
    print("=" * 50)
    return ok_peak or ok_cellpose


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
