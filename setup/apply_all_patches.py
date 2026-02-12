#!/usr/bin/env python3
"""
Apply all BactiMate compatibility patches in one place.
Patches applied:
1. peakdetect - scipy ifft import fix (scipy >= 1.10)
2. cellpose_omni - model_type parameter fix (architecture detection)
3. cellpose_omni - model URL fix (GitHub instead of Cellpose server)
4. cellpose_omni - register bact_phase_affinity (checkpoint is 1-channel; do not set nchan=2)
5. cellpose_omni transforms - trim to nchan after reshape (so 1-channel models get 1-channel input)
6. cellpose_omni io - NumPy 2.0 safe TIFF read (newbyteorder removed in NumPy 2.0)
7. cellpose_omni models - Cellpose.__init__ accepts pretrained_model (path) for user-trained models
8. cellpose_omni __main__ - pass channels to model.train() when nchan==2 so grayscale is duplicated to 2 channels
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


def patch_cellpose_omni_io() -> bool:
    """Patch cellpose_omni/io.py: NumPy 2.0 safe TIFF read (tifffile uses newbyteorder() removed in NumPy 2.0)."""
    io_file = _find_in_syspath("cellpose_omni/io.py")
    if not io_file:
        print("  WARNING: Could not find cellpose_omni/io.py")
        return False

    content = io_file.read_text()
    # Patch the .tif/.tiff branch to catch newbyteorder AttributeError and fall back to PIL
    old_block = """    if ext== '.tif' or ext=='.tiff':
        img = tifffile.imread(filename)
        return img"""
    new_block = """    if ext== '.tif' or ext=='.tiff':
        try:
            img = tifffile.imread(filename)
            return img
        except (AttributeError, TypeError) as e:
            if 'newbyteorder' in str(e):
                try:
                    from PIL import Image
                    img = Image.open(filename)
                    frames = []
                    try:
                        while True:
                            frames.append(np.array(img))
                            img.seek(img.tell() + 1)
                    except EOFError:
                        pass
                    if len(frames) == 1:
                        return frames[0]
                    elif len(frames) > 1:
                        return np.stack(frames, axis=0)
                    img.seek(0)
                    return np.array(img)
                except Exception:
                    raise e
            raise"""
    if new_block in content:
        print("  ✓ cellpose_omni io (NumPy 2.0 TIFF) already patched")
        return True
    if old_block not in content:
        print("  WARNING: Could not find expected imread block in cellpose_omni/io.py")
        return False

    content = content.replace(old_block, new_block)
    backup = Path(str(io_file) + ".backup")
    if not backup.exists():
        backup.write_text(io_file.read_text())
        print(f"  Created backup at: {backup}")
    io_file.write_text(content)
    print("  ✓ Patched cellpose_omni io (NumPy 2.0 TIFF read)")
    return True


def patch_cellpose_omni_cellpose_pretrained() -> bool:
    """Patch Cellpose.__init__ to accept pretrained_model=path for user-trained models (BactiMate)."""
    models_file = _find_in_syspath("cellpose_omni/models.py")
    if not models_file:
        print("  WARNING: Could not find cellpose_omni/models.py")
        return False
    content = models_file.read_text()
    # 1) Add pretrained_model=None to Cellpose.__init__ signature
    old_sig = """    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None, 
                 use_torch=True, model_dir=None, dim=2, omni=None):"""
    new_sig = """    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None, 
                 use_torch=True, model_dir=None, dim=2, omni=None, pretrained_model=None):"""
    if new_sig in content:
        pass  # already patched
    elif old_sig not in content:
        print("  WARNING: Could not find Cellpose.__init__ signature in cellpose_omni/models.py")
        return False
    else:
        content = content.replace(old_sig, new_sig)

    # 2) After "self.dim = dim # 2D vs 3D", branch on pretrained_model path
    old_block = """        # for now, omni models cannot do net_avg 
        if self.omni:
            net_avg = False
        model_range = range(4) if net_avg else range(1)
        self.pretrained_model = [model_path(model_type, j, torch) for j in model_range]
        
        self.diam_mean = 30. #default for any cyto model 
        nuclear = 'nuclei' in model_type
        bacterial = ('bact' in model_type) or ('worm' in model_type) 
        plant = 'plant' in model_type
        
        if nuclear:
            self.diam_mean = 17. 
        elif bacterial or plant:
            #self.diam_mean = 0.
            net_avg = False # No bacterial, plant, or omni models have additional models
        
        if not net_avg:
            self.pretrained_model = self.pretrained_model[0]

        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                pretrained_model=self.pretrained_model,
                                model_type=model_type,  # FIX: Pass model_type for architecture detection
                                diam_mean=self.diam_mean, use_torch=self.torch, 
                                dim=self.dim, omni=self.omni)
        self.cp.model_type = model_type

        # size model not used for bacterial model
        if not bacterial:
            self.pretrained_size = size_model_path(model_type, torch)
            self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                                cp_model=self.cp)
            self.sz.model_type = model_type
        else:
            self.pretrained_size = None"""
    new_block = """        # BactiMate: load user-trained model by path when pretrained_model is a path
        if pretrained_model and isinstance(pretrained_model, str) and os.path.exists(pretrained_model):
            self.omni = True if omni is None else omni
            self.pretrained_model = pretrained_model
            self.diam_mean = 30.
            # Infer nchan from checkpoint so architecture matches (avoid size mismatch when loading state_dict)
            user_nchan = 1
            try:
                import torch as _torch
                _ckpt = _torch.load(self.pretrained_model, map_location='cpu')
                if isinstance(_ckpt, dict):
                    for _k in sorted(_ckpt.keys()):
                        _v = _ckpt[_k]
                        if hasattr(_v, 'shape') and len(_v.shape) == 4 and 'weight' in _k and 'conv' in _k:
                            user_nchan = int(_v.shape[1])
                            break
            except Exception:
                pass
            self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                    pretrained_model=self.pretrained_model,
                                    model_type=None,
                                    diam_mean=self.diam_mean, use_torch=self.torch,
                                    dim=self.dim, omni=self.omni,
                                    nchan=user_nchan,
                                    nclasses=3)
            self.cp.model_type = 'bact_phase_omni'
            self.pretrained_size = None
        else:
            # for now, omni models cannot do net_avg 
            if self.omni:
                net_avg = False
            model_range = range(4) if net_avg else range(1)
            self.pretrained_model = [model_path(model_type, j, torch) for j in model_range]
            
            self.diam_mean = 30. #default for any cyto model 
            nuclear = 'nuclei' in model_type
            bacterial = ('bact' in model_type) or ('worm' in model_type) 
            plant = 'plant' in model_type
            
            if nuclear:
                self.diam_mean = 17. 
            elif bacterial or plant:
                #self.diam_mean = 0.
                net_avg = False # No bacterial, plant, or omni models have additional models
            
            if not net_avg:
                self.pretrained_model = self.pretrained_model[0]

            self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                    pretrained_model=self.pretrained_model,
                                    model_type=model_type,  # FIX: Pass model_type for architecture detection
                                    diam_mean=self.diam_mean, use_torch=self.torch, 
                                    dim=self.dim, omni=self.omni)
            self.cp.model_type = model_type

            # size model not used for bacterial model
            if not bacterial:
                self.pretrained_size = size_model_path(model_type, torch)
                self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                                    cp_model=self.cp)
                self.sz.model_type = model_type
            else:
                self.pretrained_size = None"""
    # Already patched with nchan inference + nclasses=3
    if new_block in content:
        print("  ✓ cellpose_omni Cellpose (pretrained_model path) already patched")
        return True
    # Upgrade: had nclasses=3 but no nchan inference (causes size mismatch for 2-channel checkpoints)
    old_patch_no_nchan = """            self.diam_mean = 30.
            self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                    pretrained_model=self.pretrained_model,
                                    model_type=None,
                                    diam_mean=self.diam_mean, use_torch=self.torch,
                                    dim=self.dim, omni=self.omni,
                                    nclasses=3)
            self.cp.model_type = 'bact_phase_omni'
            self.pretrained_size = None
        else:"""
    if old_patch_no_nchan in content:
        content = content.replace(old_patch_no_nchan, new_block[:new_block.index("        else:") + len("        else:")])
        models_file.write_text(content)
        print("  ✓ Updated cellpose_omni Cellpose patch (added nchan inference from checkpoint)")
        return True
    # Upgrade previous patch (had pretrained_model but missing nclasses=3) to avoid NoneType + int
    old_patch_no_nclasses = """            self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                    pretrained_model=self.pretrained_model,
                                    model_type=None,
                                    diam_mean=self.diam_mean, use_torch=self.torch,
                                    dim=self.dim, omni=self.omni)
            self.cp.model_type = 'bact_phase_omni'"""
    if old_patch_no_nclasses in content:
        content = content.replace(
            old_patch_no_nclasses,
            """            self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                    pretrained_model=self.pretrained_model,
                                    model_type=None,
                                    diam_mean=self.diam_mean, use_torch=self.torch,
                                    dim=self.dim, omni=self.omni,
                                    nclasses=3)
            self.cp.model_type = 'bact_phase_omni'""",
        )
        models_file.write_text(content)
        print("  ✓ Updated cellpose_omni Cellpose patch (added nclasses=3)")
        return True
    if old_block not in content:
        print("  WARNING: Could not find Cellpose init block in cellpose_omni/models.py")
        return False
    content = content.replace(old_block, new_block)
    backup = Path(str(models_file) + ".backup")
    if not backup.exists():
        backup.write_text(models_file.read_text())
        print(f"  Created backup at: {backup}")
    models_file.write_text(content)
    print("  ✓ Patched cellpose_omni Cellpose (pretrained_model path for user models)")
    return True


def patch_cellpose_omni_main_train_channels() -> bool:
    """Patch cellpose_omni/__main__.py: pass channels to model.train() when nchan==2 so grayscale is duplicated to 2 channels (avoids IndexError in random_crop_warp)."""
    main_file = _find_in_syspath("cellpose_omni/__main__.py")
    if not main_file:
        print("  WARNING: Could not find cellpose_omni/__main__.py")
        return False
    content = main_file.read_text()
    old = "                channels=None,  # or channels if you prefer"
    new = "                channels=[args.chan, args.chan2] if args.nchan == 2 else None,  # BactiMate: duplicate grayscale to 2 channels when nchan==2"
    if new in content:
        print("  ✓ cellpose_omni __main__ (train channels) already patched")
        return True
    if old not in content:
        print("  WARNING: Could not find expected 'channels=None' in cellpose_omni/__main__.py")
        return False
    content = content.replace(old, new)
    main_file.write_text(content)
    print("  ✓ Patched cellpose_omni __main__ (train channels for nchan==2)")
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

    ok_io = False
    print("\nPatching cellpose_omni io (NumPy 2.0 TIFF)...")
    try:
        ok_io = patch_cellpose_omni_io()
    except Exception as e:
        print(f"  ✗ Error: {e}")

    ok_cellpose_pretrained = False
    print("\nPatching cellpose_omni Cellpose (pretrained_model path)...")
    try:
        ok_cellpose_pretrained = patch_cellpose_omni_cellpose_pretrained()
    except Exception as e:
        print(f"  ✗ Error: {e}")

    ok_train_channels = False
    print("\nPatching cellpose_omni __main__ (train channels for nchan==2)...")
    try:
        ok_train_channels = patch_cellpose_omni_main_train_channels()
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 50)
    print(f"Done. peakdetect: {'OK' if ok_peak else 'skipped/failed'}; cellpose_omni: {'OK' if ok_cellpose else 'skipped/failed'}; transforms: {'OK' if ok_transforms else 'skipped/failed'}; io: {'OK' if ok_io else 'skipped/failed'}; Cellpose pretrained_model: {'OK' if ok_cellpose_pretrained else 'skipped/failed'}; train_channels: {'OK' if ok_train_channels else 'skipped/failed'}")
    print("=" * 50)
    return ok_peak or ok_cellpose or ok_io or ok_cellpose_pretrained


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
