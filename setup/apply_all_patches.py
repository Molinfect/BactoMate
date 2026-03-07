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
7. cellpose_omni models - Cellpose.__init__ accepts pretrained_model (path) for user-trained models;
   nchan/nclasses from checkpoint top-level only; nclasses 2 or 3 (default 3), never infer 4 from state_dict
8. cellpose_omni __main__ - pass channels to model.train() when nchan==2 so grayscale is duplicated to 2 channels
9. cellpose_omni core - DataParallel unwrap when saving model (self.net.module.save_model when wrapped)
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


def _find_all_in_syspath(relative_path: str) -> list[Path]:
    """Return all existing paths under sys.path for relative_path (deduplicated, order preserved)."""
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sys.path:
        candidate = Path(p).resolve() / relative_path
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


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


def patch_cellpose_omni_dynamics() -> bool:
    """Patch cellpose_omni/dynamics.py: steps2D must set tr when calc_trace is False to avoid UnboundLocalError."""
    dynamics_file = _find_in_syspath("cellpose_omni/dynamics.py")
    if not dynamics_file:
        print("  WARNING: Could not find cellpose_omni/dynamics.py")
        return False

    content = dynamics_file.read_text()
    old = """    shape = p.shape[1:]
    if calc_trace:
        Ly = shape[0]
        Lx = shape[1]
        tr = np.zeros((niter,2,Ly,Lx))
    for t in range(niter):"""
    new = """    shape = p.shape[1:]
    if calc_trace:
        Ly = shape[0]
        Lx = shape[1]
        tr = np.zeros((niter,2,Ly,Lx))
    else:
        tr = None
    for t in range(niter):"""
    if new in content:
        print("  ✓ cellpose_omni dynamics (steps2D tr) already patched")
        return True
    if old not in content:
        print("  WARNING: Could not find steps2D block in cellpose_omni/dynamics.py")
        return False

    content = content.replace(old, new)
    dynamics_file.write_text(content)
    print("  ✓ Patched cellpose_omni dynamics (steps2D tr when calc_trace=False)")
    return True


def patch_cellpose_omni_resnet_load_checkpoint() -> bool:
    """Patch resnet_torch load_model to accept checkpoints with 'state_dict' + nchan/nclasses (BactiMate-trained)."""
    resnet_file = _find_in_syspath("cellpose_omni/resnet_torch.py")
    if not resnet_file:
        print("  WARNING: Could not find cellpose_omni/resnet_torch.py")
        return False

    content = resnet_file.read_text()
    # GPU branch: unwrap checkpoint so load_state_dict gets state_dict only
    old_gpu = """            self.load_state_dict(torch.load(filename,
                                            map_location=torch_GPU,
                                            weights_only=True))"""
    new_gpu = """            ckpt = torch.load(filename, map_location=torch_GPU, weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            if state_dict and next(iter(state_dict.keys()), '').startswith('module.'):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)"""
    # Upgrade existing patch: add module. strip + strict=False if we only have the unwrap
    # Upgrade: already has unwrap (weights_only=False) but missing module. strip and strict=False
    if "state_dict = ckpt.get('state_dict'" in content and "strict=False" not in content and "self.load_state_dict(state_dict)" in content:
        content = content.replace(
            "            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt\n            self.load_state_dict(state_dict)",
            "            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt\n            if state_dict and next(iter(state_dict.keys()), '').startswith('module.'):\n                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}\n            self.load_state_dict(state_dict, strict=False)",
        )
        if "strict=False" in content:
            resnet_file.write_text(content)
            print("  ✓ cellpose_omni resnet_torch (added module. strip + strict=False)")
            return True
    if "state_dict = ckpt.get('state_dict'" in content and "strict=False" in content:
        print("  ✓ cellpose_omni resnet_torch (load checkpoint + module strip) already patched")
        return True
    if old_gpu not in content and "state_dict = ckpt.get('state_dict'" not in content:
        print("  WARNING: Could not find load_model GPU branch in cellpose_omni/resnet_torch.py")
        return False

    content = content.replace(old_gpu, new_gpu)
    # CPU branch
    old_cpu = """            state_dict = torch.load(filename, map_location=torch_CPU, weights_only=True)"""
    new_cpu = """            ckpt = torch.load(filename, map_location=torch_CPU, weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            if state_dict and next(iter(state_dict.keys()), '').startswith('module.'):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}"""
    if old_cpu in content:
        content = content.replace(old_cpu, new_cpu)
    elif "ckpt.get('state_dict', ckpt)" in content and "state_dict.keys(), '').startswith('module.')" not in content:
        # CPU already patched; add module strip
        content = content.replace(
            "            ckpt = torch.load(filename, map_location=torch_CPU, weights_only=False)\n            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt",
            "            ckpt = torch.load(filename, map_location=torch_CPU, weights_only=False)\n            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt\n            if state_dict and next(iter(state_dict.keys()), '').startswith('module.'):\n                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}",
        )
    resnet_file.write_text(content)
    print("  ✓ Patched cellpose_omni resnet_torch (load checkpoint with nchan/nclasses)")
    return True


def patch_omnipose_gpu_empty_except() -> bool:
    """[DEBUG] omnipose/gpu.py: fix SyntaxError 'expected an indented block after except' (empty except block). Run with: --only omnipose_gpu_empty_except"""
    gpu_file = _find_in_syspath("omnipose/gpu.py")
    if not gpu_file:
        print("  WARNING: Could not find omnipose/gpu.py")
        return False
    content = gpu_file.read_text()
    lines = content.splitlines(keepends=True)
    patched = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("except") or ":" not in stripped:
            continue
        except_indent = line[: len(line) - len(stripped)]
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            break
        next_line = lines[j]
        next_indent = next_line[: len(next_line) - len(next_line.lstrip())]
        body_indent = except_indent + "    "
        # Case 1: next line has less indent -> truly empty except block
        if len(next_indent) < len(except_indent):
            body = f"{body_indent}pass  # empty except fixed by BactiMate patch\n"
            if "Exception as" not in stripped:
                lines[i] = line.replace("except:", "except Exception as e:", 1).replace("except :", "except Exception as e:", 1)
            lines.insert(i + 1, body)
            content = "".join(lines)
            patched = True
            break
        # Case 2: next line has same indent as except -> body not indented (SyntaxError); re-indent those lines
        if len(next_indent) == len(except_indent) and next_indent:
            k = j
            while k < len(lines):
                line_indent = lines[k][: len(lines[k]) - len(lines[k].lstrip())]
                if lines[k].strip() and len(line_indent) < len(except_indent):
                    break
                if lines[k].strip() and line_indent == except_indent:
                    lines[k] = body_indent + lines[k].lstrip()
                k += 1
            content = "".join(lines)
            patched = True
            break
    # Repair: if we previously inserted "pass" but the next line was gpu_logger.info at same indent as except, we broke the file. Remove our pass and indent the following block to proper body indent (except + 4 spaces).
    if not patched and "pass  # empty except fixed by BactiMate patch" in content:
        lines = content.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if "pass  # empty except fixed by BactiMate patch" not in line:
                continue
            pass_indent = line[: len(line) - len(line.lstrip())]
            body_indent = pass_indent  # our pass was at except_indent+4, so body indent = pass_indent (8 spaces)
            except_indent_len = len(pass_indent) - 4  # except is 4 spaces less (e.g. 4)
            lines.pop(i)
            j = i
            while j < len(lines):
                if not lines[j].strip():
                    j += 1
                    continue
                line_indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
                if len(line_indent) <= except_indent_len:
                    break
                if len(line_indent) < len(body_indent):
                    lines[j] = body_indent + lines[j].lstrip()
                j += 1
            content = "".join(lines)
            patched = True
            break
    if patched:
        gpu_file.write_text(content)
        print("  ✓ Patched omnipose gpu.py (empty except block fixed)")
    else:
        print("  ✓ omnipose gpu.py: no empty except block found (already OK)")
    return True


def patch_omnipose_gpu_mps() -> bool:
    """Improve MPS (Apple Silicon) GPU detection in omnipose/gpu.py: broader ARM check and log real exception when device test fails."""
    gpu_file = _find_in_syspath("omnipose/gpu.py")
    if not gpu_file:
        print("  WARNING: Could not find omnipose/gpu.py")
        return False
    content = gpu_file.read_text()
    patched = False
    # 1) Broaden ARM detection (platform.processor() can vary; platform.machine() == 'arm64' is reliable on Mac)
    old_arm = "ARM = 'arm' in platform.processor()"
    new_arm = "ARM = ('arm' in platform.processor()) or (platform.machine() == 'arm64')"
    if new_arm in content:
        pass
    elif old_arm in content:
        content = content.replace(old_arm, new_arm, 1)
        patched = True
    # 2) Log actual exception when GPU test fails (upstream uses 1-space indent in omnipose/gpu.py)
    for (old_except, new_except) in [
        (
            " except:# Exception as e:\n gpu_logger.info('TORCH GPU version not installed/working.')#, e)",
            " except Exception as e:\n gpu_logger.info('TORCH GPU version not installed/working: %s', e)",
        ),
        (
            "except:# Exception as e:\n    gpu_logger.info('TORCH GPU version not installed/working.')#, e)",
            "except Exception as e:\n    gpu_logger.info('TORCH GPU version not installed/working: %s', e)",
        ),
        (
            "except:# Exception as e:\n gpu_logger.info('TORCH GPU version not installed/working.')#, e)",
            "except Exception as e:\n gpu_logger.info('TORCH GPU version not installed/working: %s', e)",
        ),
    ]:
        if "not installed/working: %s" in content:
            break
        if old_except in content:
            content = content.replace(old_except, new_except, 1)
            patched = True
            break
    # 3) Regex fallback: match "except:" or "except Exception" followed by the info line (any indent)
    if "not installed/working: %s" not in content and "gpu_logger.info('TORCH GPU version not installed/working." in content:
        import re
        # Match except with optional Exception as e, then the info line without %s
        regex = re.compile(
            r"(\s*)except\s*:?\s*#?\s*Exception as e:\s*\n\s*gpu_logger\.info\('TORCH GPU version not installed/working\.'\)\s*#?,?\s*e?\s*\)?",
            re.MULTILINE,
        )
        def repl(m: re.Match) -> str:
            indent = m.group(1) or " "
            return f"{indent}except Exception as e:\n{indent}gpu_logger.info('TORCH GPU version not installed/working: %s', e)"
        new_content = regex.sub(repl, content, count=1)
        if new_content != content:
            content = new_content
            patched = True
    # 4) Fix device string: MPS must be "mps" not "mps:True" (PyTorch rejects "mps:True")
    import re
    # Match "mps:" + str(use_torch) or 'mps:'+str(gpu) etc.
    mps_bug_pattern = re.compile(r'[\"\']mps:[\"\']\s*\+\s*str\s*\(\s*\w+\s*\)')
    if mps_bug_pattern.search(content):
        content = mps_bug_pattern.sub('"mps"', content)
        patched = True
    # 4b) Fix f-string form that produces 'mps:True' when variable is True (e.g. f'mps:{use_torch}' in omnipose/gpu.py)
    mps_fstring_pattern = re.compile(r"f['\"]mps:\s*\{\s*\w+\s*\}['\"]")
    if mps_fstring_pattern.search(content):
        content = mps_fstring_pattern.sub("'mps'", content)
        patched = True
    # 5) Fix empty except block (SyntaxError: expected an indented block after 'except')
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("except") or ":" not in stripped:
            continue
        except_indent = line[: len(line) - len(stripped)]
        # Find next non-empty line
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            break
        next_line = lines[j]
        next_indent = next_line[: len(next_line) - len(next_line.lstrip())]
        # If next line is not indented more than except, the block is empty
        if len(next_indent) <= len(except_indent):
            body_indent = except_indent + "    "
            body = f"{body_indent}pass  # empty except fixed by BactiMate patch\n"
            # Normalize except to "except Exception as e:" so variable e exists if used below
            if "Exception as" not in stripped:
                lines[i] = line.replace("except:", "except Exception as e:", 1).replace("except :", "except Exception as e:", 1)
            lines.insert(i + 1, body)
            content = "".join(lines)
            patched = True
            break
    if patched:
        gpu_file.write_text(content)
        print("  ✓ Patched omnipose gpu.py (MPS ARM detection and/or GPU exception logging)")
    elif "not installed/working: %s" in content or new_arm in content:
        print("  ✓ omnipose gpu.py (MPS) already patched")
    return True


def patch_omnipose_core_bce_boundary_range() -> bool:
    """Fix RuntimeError: all elements of input should be between 0 and 1. Boundary head outputs logits; BCELoss expects [0,1]. Apply sigmoid to prediction and clamp target."""
    core_files = _find_all_in_syspath("omnipose/core.py")
    if not core_files:
        print("  WARNING: Could not find omnipose/core.py")
        return False

    old = """        if self.nclasses==(self.dim+2):
            bd = y[:,self.dim+1]
            bd_loss = self.BCELoss(bd,boundary) #BCElogits"""
    new = """        if self.nclasses==(self.dim+2):
            bd = y[:,self.dim+1]
            # Boundary head outputs logits; BCELoss expects [0,1]. Apply sigmoid and ensure target is float.
            bd_prob = torch.sigmoid(bd)
            boundary_float = boundary.float().clamp(0., 1.)
            bd_loss = self.BCELoss(bd_prob, boundary_float)"""
    patched = False
    for core_file in core_files:
        content = core_file.read_text()
        if "bd_prob = torch.sigmoid(bd)" in content and "boundary_float = boundary.float()" in content:
            continue
        if old not in content:
            continue
        content = content.replace(old, new)
        core_file.write_text(content)
        print("  ✓ Patched omnipose core (BCE boundary range fix): {}".format(core_file))
        patched = True
    if patched:
        return True
    print("  ✓ omnipose core (BCE boundary range) already patched ({} copy/copies)".format(len(core_files)))
    return True


def patch_omnipose_core_grid_sample_dtype() -> bool:
    """Ensure pt and flow have same dtype in steps_batch so grid_sample does not get Double vs Float (RuntimeError)."""
    core_file = _find_in_syspath("omnipose/core.py")
    if not core_file:
        print("  WARNING: Could not find omnipose/core.py")
        return False

    content = core_file.read_text()
    old = """    for k in range(d): 
        pt[...,k] = 2*pt[...,k]/shape[k] - 1 
        flow[:,k] = 2*flow[:,k]/shape[k]
    
    if calc_trace:"""
    new = """    for k in range(d): 
        pt[...,k] = 2*pt[...,k]/shape[k] - 1 
        flow[:,k] = 2*flow[:,k]/shape[k]
    # Ensure same dtype for grid_sample (shape is numpy float64 and can promote pt to double)
    pt = pt.to(flow.dtype)
    
    if calc_trace:"""
    if "pt = pt.to(flow.dtype)" in content:
        print("  ✓ omnipose core (grid_sample dtype) already patched")
        return True
    if old not in content:
        print("  WARNING: Could not find steps_batch pt/flow block in omnipose/core.py")
        return False

    content = content.replace(old, new)
    core_file.write_text(content)
    print("  ✓ Patched omnipose core (grid_sample Float/Double fix)")
    return True


def patch_omnipose_core_follow_flows_dtype() -> bool:
    """Cast final_p to final_points.dtype in follow_flows so index put does not get Float vs Double (RuntimeError)."""
    core_file = _find_in_syspath("omnipose/core.py")
    if not core_file:
        print("  WARNING: Could not find omnipose/core.py")
        return False

    content = core_file.read_text()
    old = "final_points[cell_px] = final_p.squeeze()"
    new = "final_points[cell_px] = final_p.squeeze().to(final_points.dtype)"
    if ".to(final_points.dtype)" in content:
        print("  ✓ omnipose core (follow_flows dtype) already patched")
        return True
    if old not in content:
        print("  WARNING: Could not find follow_flows final_points assignment in omnipose/core.py")
        return False

    content = content.replace(old, new)
    core_file.write_text(content)
    print("  ✓ Patched omnipose core (follow_flows Float/Double fix)")
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
    import re
    models_file = _find_in_syspath("cellpose_omni/models.py")
    if not models_file:
        print("  WARNING: Could not find cellpose_omni/models.py")
        return False
    content = models_file.read_text()
    # Repair broken debug write (unterminated string) if present. Run first so broken files get fixed
    # before any "already patched" logic. Match: try block containing _f.write("...state_dict_sample=%s)
    # then newline then " % ( (syntax error). Replace whole try/except with safe one-line write.
    new_try_block = """            try:
                import datetime as _dt
                _dp = os.environ.get("BACTIMATE_OMNI_DEBUG_LOG")
                if _dp and _ckpt is not None and isinstance(_ckpt, dict):
                    _st = _ckpt.get('state_dict') or _ckpt.get('model_state_dict') or _ckpt
                    _sk = list(_st.keys())[:12] if isinstance(_st, dict) else []
                    _ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(_dp, "a", encoding="utf-8") as _f:
                        _f.write("[%s] [cellpose_omni load] path=%r nchan=%s nclasses=%s keys=%s\\n" % (_ts, self.pretrained_model, user_nchan, user_nclasses, list(_ckpt.keys())[:25]))
            except Exception:
                pass"""
    # Regex: try block containing _f.write with newline inside the string (broken).
    broken_try_re = re.compile(
        r'\s+try:\s+import datetime as _dt\s+_dp = os\.environ\.get\("BACTIMATE_OMNI_DEBUG_LOG"\).*?'
        r'with open\(_dp, "a", encoding="utf-8"\) as _f:\s+_f\.write\("\[%s\].*?state_dict_sample=%s\s+" % \(_ts, self\.pretrained_model, user_nchan, user_nclasses, list\(_ckpt\.keys\(\)\)\[:25\], _sk\)\)\s+except Exception:\s+pass',
        re.DOTALL,
    )
    if broken_try_re.search(content):
        content = broken_try_re.sub(new_try_block, content, count=1)
        models_file.write_text(content)
        print("  ✓ Repaired cellpose_omni models.py (unterminated debug string)")
        return True
    # Exact string repair (for same newline/indent as in repo)
    broken_marker = 'state_dict_sample=%s\n" % (_ts, self.pretrained_model'
    if broken_marker in content:
        old_try_block = (
            '            try:\n'
            '                import datetime as _dt\n'
            '                _dp = os.environ.get("BACTIMATE_OMNI_DEBUG_LOG")\n'
            '                if _dp and _ckpt is not None and isinstance(_ckpt, dict):\n'
            '                    _st = _ckpt.get(\'state_dict\') or _ckpt.get(\'model_state_dict\') or _ckpt\n'
            '                    _sk = list(_st.keys())[:12] if isinstance(_st, dict) else []\n'
            '                    _ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")\n'
            '                    with open(_dp, "a", encoding="utf-8") as _f:\n'
            '                        _f.write("[%s] [cellpose_omni load] path=%r inferred nchan=%s nclasses=%s ckpt_keys=%s state_dict_sample=%s\n'
            '\n'
            '" % (_ts, self.pretrained_model, user_nchan, user_nclasses, list(_ckpt.keys())[:25], _sk))\n'
            '            except Exception:\n'
            '                pass'
        )
        if old_try_block in content:
            content = content.replace(old_try_block, new_try_block, 1)
            models_file.write_text(content)
            print("  ✓ Repaired cellpose_omni models.py (unterminated debug string)")
            return True
    # 1) Add pretrained_model=None to Cellpose.__init__ signature (support use_torch or torch param name)
    new_sig_4 = """    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None, 
                 use_torch=True, model_dir=None, dim=2, omni=None, pretrained_model=None):"""
    old_sig_4 = """    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None, 
                 use_torch=True, model_dir=None, dim=2, omni=None):"""
    # Installed cellpose_omni often has torch=True (not use_torch)
    old_sig_torch = """    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None, 
                 torch=True, model_dir=None, dim=2, omni=None):"""
    new_sig_torch = """    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None, 
                 torch=True, model_dir=None, dim=2, omni=None, pretrained_model=None):"""
    if new_sig_4 in content or new_sig_torch in content:
        pass  # already patched
    elif "pretrained_model=None" in content and "def __init__(self, gpu=False, model_type='cyto'" in content:
        pass  # already patched (regex or other variant)
    elif old_sig_4 in content:
        content = content.replace(old_sig_4, new_sig_4)
    elif old_sig_torch in content:
        content = content.replace(old_sig_torch, new_sig_torch)
    else:
        # Regex fallback: match torch=True or use_torch=True
        sig_pattern = re.compile(
            r"( def __init__\(self,\s*gpu=False,\s*model_type='cyto',\s*net_avg=True,\s*device=None,\s*\n\s*(?:use_)?torch=True,\s*model_dir=None,\s*dim=2,\s*omni=None)\):"
        )
        if sig_pattern.search(content) and "pretrained_model=None" not in content:
            content = sig_pattern.sub(r"\1, pretrained_model=None):", content, count=1)
        elif "pretrained_model=None" not in content:
            print("  WARNING: Could not find Cellpose.__init__ signature in cellpose_omni/models.py")
            return False

    # 2) After "self.dim = dim # 2D vs 3D", branch on pretrained_model path
    # Vanilla upstream block (no model_type inside CellposeModel call)
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
    # Vanilla block with torch= (installed cellpose_omni uses torch not use_torch)
    old_block_torch = """        # for now, omni models cannot do net_avg 
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
                                diam_mean=self.diam_mean, torch=self.torch, 
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
    # 4-space block WITH model_type in CellposeModel (already-patched or older BactiMate)
    old_block_with_fix = """        # for now, omni models cannot do net_avg 
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
            # BactiMate: user checkpoints must use nchan/nclasses from checkpoint top-level only (Omnipose training uses nclasses=3).
            # Do NOT infer nclasses from state_dict (e.g. 4 output channels) or loader will show nclasses=4.
            user_nchan = 1
            user_nclasses = 3
            _ckpt = None
            try:
                import torch as _torch
                _ckpt = _torch.load(self.pretrained_model, map_location='cpu')
                if isinstance(_ckpt, dict):
                    if 'nchan' in _ckpt:
                        try:
                            user_nchan = int(_ckpt['nchan'])
                        except (TypeError, ValueError):
                            pass
                    if 'nclasses' in _ckpt:
                        try:
                            n = int(_ckpt['nclasses'])
                            if n in (2, 3):
                                user_nclasses = n
                            # else: keep 3 (BactiMate/Omnipose default; avoid 4 which is wrong for our training)
                        except (TypeError, ValueError):
                            pass
                    # If nchan not at top level, infer from first 4D weight in state_dict (training saves nested)
                    if user_nchan == 1:
                        _state = _ckpt.get('state_dict') or _ckpt.get('model_state_dict') or _ckpt
                        if isinstance(_state, dict):
                            for _k, _v in _state.items():
                                if not hasattr(_v, 'shape') or len(_v.shape) != 4:
                                    continue
                                # Conv2d weight is (out_c, in_c, kH, kW); in_c = nchan
                                if 'weight' in _k and ('conv' in _k or 'encoder' in _k or '.0.weight' in _k or '.0.' in _k):
                                    user_nchan = int(_v.shape[1])
                                    break
                            if user_nchan == 1:
                                for _k, _v in _state.items():
                                    if hasattr(_v, 'shape') and len(_v.shape) == 4:
                                        user_nchan = int(_v.shape[1])
                                        break
            except Exception:
                pass
            # Debug log for user-model load (compare with built-in in Logs/debug_*.log)
            try:
                import datetime as _dt
                _dp = os.environ.get("BACTIMATE_OMNI_DEBUG_LOG")
                if _dp and _ckpt is not None and isinstance(_ckpt, dict):
                    _st = _ckpt.get('state_dict') or _ckpt.get('model_state_dict') or _ckpt
                    _sk = list(_st.keys())[:12] if isinstance(_st, dict) else []
                    _ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(_dp, "a", encoding="utf-8") as _f:
                        _f.write(
                            "[%s] [cellpose_omni load] path=%r inferred nchan=%s nclasses=%s ckpt_keys=%s state_dict_sample=%s\\n"
                            % (_ts, self.pretrained_model, user_nchan, user_nclasses, list(_ckpt.keys())[:25], _sk))
            except Exception:
                pass
            self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                    pretrained_model=self.pretrained_model,
                                    model_type=None,
                                    diam_mean=self.diam_mean, use_torch=self.torch,
                                    dim=self.dim, omni=self.omni,
                                    nchan=user_nchan,
                                    nclasses=user_nclasses)
            self.cp.model_type = 'bact_phase_omni'
            # BactiMate: use built-in size model so diameter=0 triggers estimation; on 404 create fallback
            try:
                self.pretrained_size = size_model_path('bact_phase_omni', torch)
                self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size, cp_model=self.cp)
                self.sz.model_type = 'bact_phase_omni'
            except Exception as _e:
                try:
                    from urllib.error import HTTPError
                    if isinstance(_e, HTTPError) and getattr(_e, 'code', None) == 404:
                        _fb_name = 'size_bact_phase_omnitorch_0.npy' if torch else 'size_bact_phase_omni_0.npy'
                        _fb_path = os.fspath(MODEL_DIR.joinpath(_fb_name))
                        if not os.path.exists(_fb_path):
                            MODEL_DIR.mkdir(parents=True, exist_ok=True)
                            _params = {'A': np.zeros(256), 'smean': np.zeros(256), 'diam_mean': 30., 'ymean': 0.}
                            np.save(_fb_path, _params)
                            models_logger.info('BactiMate: created fallback size model (remote 404): %s', _fb_path)
                        self.pretrained_size = _fb_path
                        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size, cp_model=self.cp)
                        self.sz.model_type = 'bact_phase_omni'
                    else:
                        try:
                            models_logger.warning("BactiMate: could not load size model for user model (%s), diameter=0 will not rescale: %s", type(_e).__name__, _e)
                        except Exception:
                            pass
                        self.pretrained_size = None
                except Exception:
                    try:
                        models_logger.warning("BactiMate: could not load size model for user model (%s), diameter=0 will not rescale: %s", type(_e).__name__, _e)
                    except Exception:
                        pass
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
    new_block_torch = new_block.replace("use_torch=self.torch", "torch=self.torch")
    # Already patched with full nchan/nclasses inference + size model
    if new_block in content or new_block_torch in content:
        print("  ✓ cellpose_omni Cellpose (pretrained_model path) already patched")
        return True
    # Upgrade: already have user-model block but still self.pretrained_size = None -> add size model for diameter=0
    old_no_size = """            self.cp.model_type = 'bact_phase_omni'
            self.pretrained_size = None
        else:"""
    new_with_size = """            self.cp.model_type = 'bact_phase_omni'
            # BactiMate: use built-in size model so diameter=0 triggers estimation; on 404 create fallback
            try:
                self.pretrained_size = size_model_path('bact_phase_omni', torch)
                self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size, cp_model=self.cp)
                self.sz.model_type = 'bact_phase_omni'
            except Exception as _e:
                try:
                    from urllib.error import HTTPError
                    if isinstance(_e, HTTPError) and getattr(_e, 'code', None) == 404:
                        _fb_name = 'size_bact_phase_omnitorch_0.npy' if torch else 'size_bact_phase_omni_0.npy'
                        _fb_path = os.fspath(MODEL_DIR.joinpath(_fb_name))
                        if not os.path.exists(_fb_path):
                            MODEL_DIR.mkdir(parents=True, exist_ok=True)
                            _params = {'A': np.zeros(256), 'smean': np.zeros(256), 'diam_mean': 30., 'ymean': 0.}
                            np.save(_fb_path, _params)
                            models_logger.info('BactiMate: created fallback size model (remote 404): %s', _fb_path)
                        self.pretrained_size = _fb_path
                        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size, cp_model=self.cp)
                        self.sz.model_type = 'bact_phase_omni'
                    else:
                        try:
                            models_logger.warning("BactiMate: could not load size model for user model (%s), diameter=0 will not rescale: %s", type(_e).__name__, _e)
                        except Exception:
                            pass
                        self.pretrained_size = None
                except Exception:
                    try:
                        models_logger.warning("BactiMate: could not load size model for user model (%s), diameter=0 will not rescale: %s", type(_e).__name__, _e)
                    except Exception:
                        pass
                    self.pretrained_size = None
        else:"""
    if old_no_size in content and "size_model_path('bact_phase_omni'" not in content:
        content = content.replace(old_no_size, new_with_size, 1)
        models_file.write_text(content)
        print("  ✓ Updated cellpose_omni Cellpose patch (added size model for user models, diameter=0)")
        return True
    # Upgrade: already have size model block but silent except -> add logging so we see why size model fails
    silent_except = """            except Exception:
                self.pretrained_size = None
        else:"""
    logging_except = """            except Exception as _e:
                try:
                    models_logger.warning("BactiMate: could not load size model for user model (%s), diameter=0 will not rescale: %s", type(_e).__name__, _e)
                except Exception:
                    pass
                self.pretrained_size = None
        else:"""
    # 404 fallback: when remote size model returns 404, create minimal size model so diameter=0 still works
    fallback_except = """            except Exception as _e:
                try:
                    from urllib.error import HTTPError
                    if isinstance(_e, HTTPError) and getattr(_e, 'code', None) == 404:
                        _fb_name = 'size_bact_phase_omnitorch_0.npy' if torch else 'size_bact_phase_omni_0.npy'
                        _fb_path = os.fspath(MODEL_DIR.joinpath(_fb_name))
                        if not os.path.exists(_fb_path):
                            MODEL_DIR.mkdir(parents=True, exist_ok=True)
                            _params = {'A': np.zeros(256), 'smean': np.zeros(256), 'diam_mean': 30., 'ymean': 0.}
                            np.save(_fb_path, _params)
                            models_logger.info('BactiMate: created fallback size model (remote 404): %s', _fb_path)
                        self.pretrained_size = _fb_path
                        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size, cp_model=self.cp)
                        self.sz.model_type = 'bact_phase_omni'
                    else:
                        try:
                            models_logger.warning("BactiMate: could not load size model for user model (%s), diameter=0 will not rescale: %s", type(_e).__name__, _e)
                        except Exception:
                            pass
                        self.pretrained_size = None
                except Exception:
                    try:
                        models_logger.warning("BactiMate: could not load size model for user model (%s), diameter=0 will not rescale: %s", type(_e).__name__, _e)
                    except Exception:
                        pass
                    self.pretrained_size = None
        else:"""
    if "size_model_path('bact_phase_omni'" in content and logging_except in content and "created fallback size model" not in content:
        content = content.replace(logging_except, fallback_except, 1)
        models_file.write_text(content)
        print("  ✓ Updated cellpose_omni Cellpose patch (404 fallback: create local size model)")
        return True
    if "size_model_path('bact_phase_omni'" in content and silent_except in content and "BactiMate: could not load size model" not in content:
        content = content.replace(silent_except, logging_except, 1)
        models_file.write_text(content)
        print("  ✓ Updated cellpose_omni Cellpose patch (size model failure now logged)")
        return True
    # Upgrade: already patched but with narrow nchan inference (only 'conv' in key); relax to encoder/.0./any 4D
    old_inner_loop = """                    # If nchan not at top level, infer from first 4D conv in state_dict (training saves nested)
                    if user_nchan == 1:
                        _state = _ckpt.get('state_dict') or _ckpt.get('model_state_dict') or _ckpt
                        if isinstance(_state, dict):
                            for _k, _v in _state.items():
                                if hasattr(_v, 'shape') and len(_v.shape) == 4 and 'weight' in _k and 'conv' in _k:
                                    user_nchan = int(_v.shape[1])
                                    break"""
    new_inner_loop = """                    # If nchan not at top level, infer from first 4D weight in state_dict (training saves nested)
                    if user_nchan == 1:
                        _state = _ckpt.get('state_dict') or _ckpt.get('model_state_dict') or _ckpt
                        if isinstance(_state, dict):
                            for _k, _v in _state.items():
                                if not hasattr(_v, 'shape') or len(_v.shape) != 4:
                                    continue
                                # Conv2d weight is (out_c, in_c, kH, kW); in_c = nchan
                                if 'weight' in _k and ('conv' in _k or 'encoder' in _k or '.0.weight' in _k or '.0.' in _k):
                                    user_nchan = int(_v.shape[1])
                                    break
                            if user_nchan == 1:
                                for _k, _v in _state.items():
                                    if hasattr(_v, 'shape') and len(_v.shape) == 4:
                                        user_nchan = int(_v.shape[1])
                                        break"""
    if old_inner_loop in content:
        content = content.replace(old_inner_loop, new_inner_loop, 1)
        models_file.write_text(content)
        print("  ✓ Updated cellpose_omni Cellpose patch (relaxed nchan inference for encoder/.0./any 4D weight)")
        return True
    # Try vanilla upstream block (torch= first, then use_torch=), then older BactiMate block (with FIX)
    if old_block_torch in content:
        content = content.replace(old_block_torch, new_block_torch, 1)
        models_file.write_text(content)
        print("  ✓ Patched cellpose_omni Cellpose (pretrained_model path, vanilla upstream torch=)")
        return True
    if old_block in content:
        content = content.replace(old_block, new_block, 1)
        models_file.write_text(content)
        print("  ✓ Patched cellpose_omni Cellpose (pretrained_model path, vanilla upstream)")
        return True
    if old_block_with_fix in content:
        content = content.replace(old_block_with_fix, new_block, 1)
        models_file.write_text(content)
        print("  ✓ Patched cellpose_omni Cellpose (pretrained_model path, with FIX block)")
        return True
    # Upgrade: had top-level-only nchan inference; add state_dict lookup and nclasses from checkpoint
    old_patch_state_dict = """            # Infer nchan from checkpoint so architecture matches (avoid size mismatch when loading state_dict)
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
                                    nclasses=3)"""
    if old_patch_state_dict in content:
        content = content.replace(
            old_patch_state_dict,
            """            # Infer nchan/nclasses from checkpoint (top-level or inside state_dict) so architecture matches
            user_nchan = 1
            user_nclasses = 3
            try:
                import torch as _torch
                _ckpt = _torch.load(self.pretrained_model, map_location='cpu')
                if isinstance(_ckpt, dict):
                    if 'nchan' in _ckpt:
                        try:
                            user_nchan = int(_ckpt['nchan'])
                        except (TypeError, ValueError):
                            pass
                    if 'nclasses' in _ckpt:
                        try:
                            user_nclasses = int(_ckpt['nclasses'])
                        except (TypeError, ValueError):
                            pass
                    # If nchan not at top level, infer from first 4D weight in state_dict (training saves nested)
                    if user_nchan == 1:
                        _state = _ckpt.get('state_dict') or _ckpt.get('model_state_dict') or _ckpt
                        if isinstance(_state, dict):
                            for _k, _v in _state.items():
                                if not hasattr(_v, 'shape') or len(_v.shape) != 4:
                                    continue
                                # Conv2d weight is (out_c, in_c, kH, kW); in_c = nchan
                                if 'weight' in _k and ('conv' in _k or 'encoder' in _k or '.0.weight' in _k or '.0.' in _k):
                                    user_nchan = int(_v.shape[1])
                                    break
                            if user_nchan == 1:
                                for _k, _v in _state.items():
                                    if hasattr(_v, 'shape') and len(_v.shape) == 4:
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
                                    nclasses=user_nclasses)""",
        )
        models_file.write_text(content)
        print("  ✓ Updated cellpose_omni Cellpose patch (nchan/nclasses from state_dict + top-level)")
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
    # 1) Fix model.train(..., channels=None, ...) so it passes the channels variable (required for nchan==2 grayscale).
    train_call_none = "channels=None,  # or channels if you prefer"
    train_call_var = "channels=channels,"
    if train_call_none in content:
        content = content.replace(train_call_none, train_call_var, 1)
        main_file.write_text(content)
        print("  ✓ Patched cellpose_omni __main__ (train channels: pass channels to model.train)")
        return True
    if train_call_var in content:
        print("  ✓ cellpose_omni __main__ (train channels) already patched")
        return True
    # 2) Legacy: ensure assignment exists so channels is set when nchan==2 (some installs have it, some need it).
    # Upstream sets "channels = None" (or "channels = None ") before model.train(..., channels=channels, ...)
    old_assign_variants = [
        "                channels = None ",
        "                channels = None",
        "            channels = None ",
        "            channels = None",
        "        channels = None",
    ]
    replaced = False
    for old in old_assign_variants:
        if old in content:
            indent = old[: len(old) - len(old.lstrip())]
            new_line = indent + "channels = [args.chan, args.chan2] if nchan == 2 else None"
            content = content.replace(old, new_line, 1)
            replaced = True
            break
    if not replaced:
        # Fallback: regex for "channels = None" assignment (any indent)
        import re
        pattern = re.compile(r"^(\s+)channels = None\s*$", re.MULTILINE)
        if pattern.search(content):
            content = pattern.sub(
                r'\1channels = [args.chan, args.chan2] if nchan == 2 else None',
                content,
                count=1,
            )
            replaced = True
    if not replaced:
        print("  WARNING: Could not find expected 'channels = None' in cellpose_omni/__main__.py")
        return False
    main_file.write_text(content)
    print("  ✓ Patched cellpose_omni __main__ (train channels for nchan==2)")
    return True


def patch_cellpose_omni_core_save_model() -> bool:
    """Patch cellpose_omni/core.py: when self.net is DataParallel, call save_model on .module (fixes AttributeError in frozen/build).
    Handles both: single line self.net.save_model(file_name) and the if self.torch and self.gpu / else variant (else branch fails on CPU when net is still DataParallel)."""
    core_file = _find_in_syspath("cellpose_omni/core.py")
    if not core_file:
        print("  WARNING: Could not find cellpose_omni/core.py")
        return False
    content = core_file.read_text()
    # Already fully patched: single-line safe call or two-line (net = ...; net.save_model)
    if "self.net.save_model(file_name)" not in content and (
        "(self.net.module if hasattr(self.net, 'module') else self.net).save_model(file_name)" in content
        or "net = self.net.module if hasattr(self.net, 'module') else self.net" in content
    ):
        print("  ✓ cellpose_omni core (DataParallel save_model) already patched")
        return True
    # Variant A: if/else block (app bundle / some installs) — CPU branch uses self.net.save_model and can be DataParallel
    block_old = """                        if self.torch and self.gpu:
                            self.net.module.save_model(file_name)
                        else:
                            self.net.save_model(file_name)"""
    block_new = """                        net = self.net.module if hasattr(self.net, 'module') else self.net
                        net.save_model(file_name)"""
    if block_old in content:
        backup = core_file.with_suffix(".py.backup")
        if not backup.exists():
            backup.write_text(content)
            print(f"  Created backup at: {backup}")
        content = content.replace(block_old, block_new, 1)
        core_file.write_text(content)
        print("  ✓ Patched cellpose_omni core (DataParallel save_model, if/else variant)")
        return True
    # Variant B: single line
    old = "self.net.save_model(file_name)"
    new = "(self.net.module if hasattr(self.net, 'module') else self.net).save_model(file_name)"
    if old in content:
        backup = core_file.with_suffix(".py.backup")
        if not backup.exists():
            backup.write_text(content)
            print(f"  Created backup at: {backup}")
        content = content.replace(old, new, 1)
        core_file.write_text(content)
        print("  ✓ Patched cellpose_omni core (DataParallel save_model)")
        return True
    print("  WARNING: Could not find save_model block in cellpose_omni/core.py")
    return False


def patch_cellpose_omni_core_mps_device() -> bool:
    """Fix MPS device string in cellpose_omni/core.py: use 'mps' not 'mps:{gpu_number}'.
    When use_gpu(True) is called, gpu_number is True and f'mps:{gpu_number}' becomes 'mps:True', which PyTorch rejects.
    Many cellpose_omni versions get device from omnipose.gpu.get_device(); in that case nothing to patch here."""
    core_file = _find_in_syspath("cellpose_omni/core.py")
    if not core_file:
        print("  WARNING: Could not find cellpose_omni/core.py")
        return False
    content = core_file.read_text()
    old_line = "device = torch.device(f'mps:{gpu_number}') if ARM else torch.device(f'cuda:{gpu_number}')"
    new_line = "device = torch.device('mps') if ARM else torch.device(f'cuda:{gpu_number}')"
    if new_line in content:
        print("  ✓ cellpose_omni core (MPS device string) already patched")
        return True
    if old_line in content:
        content = content.replace(old_line, new_line, 1)
        core_file.write_text(content)
        print("  ✓ Patched cellpose_omni core (MPS device string: use 'mps' not 'mps:True')")
        return True
    old_alt = 'device = torch.device(f"mps:{gpu_number}") if ARM else torch.device(f"cuda:{gpu_number}")'
    if old_alt in content:
        content = content.replace(old_alt, new_line, 1)
        core_file.write_text(content)
        print("  ✓ Patched cellpose_omni core (MPS device string: use 'mps' not 'mps:True')")
        return True
    # No device-assignment line in core; device may come from omnipose.gpu.get_device() (patched in omnipose_gpu_mps)
    if "get_device" in content and "omnipose" in content:
        print("  ✓ cellpose_omni core (MPS): device from omnipose.gpu (patched there); nothing to change in core.py")
        return True
    print("  WARNING: Could not find MPS device line in cellpose_omni/core.py (patch may not be needed for this version)")
    return False


def patch_cellpose_omni_core_ntrain_label() -> bool:
    """Rename training log label ntrain -> ntrain_img in cellpose_omni/core.py."""
    core_files = _find_all_in_syspath("cellpose_omni/core.py")
    if not core_files:
        print("  WARNING: Could not find cellpose_omni/core.py")
        return False
    patched_any = False
    for core_file in core_files:
        content = core_file.read_text()
        if "ntrain_img =" in content:
            print("  ✓ cellpose_omni core (ntrain_img label) already patched")
            patched_any = True
            continue
        if "ntrain =" not in content:
            continue
        content = content.replace(">>>> ntrain = ", ">>>> ntrain_img = ", 2)
        core_file.write_text(content)
        patched_any = True
        print("  ✓ Patched cellpose_omni core (log label ntrain -> ntrain_img)")
    return patched_any


# Ordered list of (name, description, patch_func) for --only / --list. Names are used as: --only resnet_load,cellpose_pretrained
PATCH_LIST = [
    ("peakdetect", "peakdetect: scipy ifft compatibility", patch_peakdetect),
    ("cellpose_omni", "cellpose_omni: model_type + model URL", patch_cellpose_omni),
    ("transforms", "cellpose_omni transforms: nchan trim", patch_cellpose_omni_transforms),
    ("io", "cellpose_omni io: NumPy 2.0 TIFF read", patch_cellpose_omni_io),
    ("dynamics", "cellpose_omni dynamics: steps2D tr", patch_cellpose_omni_dynamics),
    ("resnet_load", "cellpose_omni resnet_torch: load checkpoint unwrap", patch_cellpose_omni_resnet_load_checkpoint),
    ("omnipose_gpu_mps", "omnipose gpu: MPS (Apple Silicon) detection + log GPU exception", patch_omnipose_gpu_mps),
    ("omnipose_gpu_empty_except", "[DEBUG] omnipose gpu: fix empty except SyntaxError only", patch_omnipose_gpu_empty_except),
    ("omnipose_bce_boundary", "omnipose core: BCE boundary range (training)", patch_omnipose_core_bce_boundary_range),
    ("omnipose_dtype", "omnipose core: grid_sample dtype", patch_omnipose_core_grid_sample_dtype),
    ("omnipose_follow_flows", "omnipose core: follow_flows dtype", patch_omnipose_core_follow_flows_dtype),
    ("cellpose_pretrained", "cellpose_omni Cellpose: pretrained_model path (user models)", patch_cellpose_omni_cellpose_pretrained),
    ("train_channels", "cellpose_omni __main__: train channels for nchan==2", patch_cellpose_omni_main_train_channels),
    ("cellpose_core_save", "cellpose_omni core: DataParallel save_model", patch_cellpose_omni_core_save_model),
    ("core_mps_device", "cellpose_omni core: MPS device string (use 'mps' not 'mps:True')", patch_cellpose_omni_core_mps_device),
    ("core_ntrain_label", "cellpose_omni core: log label ntrain -> ntrain_img", patch_cellpose_omni_core_ntrain_label),
]


def main() -> bool:
    argv = list(sys.argv)
    # When building the app bundle, pass --target-site-packages so we patch the copied venv, not another path.
    if len(argv) >= 3 and argv[1] == "--target-site-packages":
        target = Path(argv[2]).resolve()
        if target.is_dir():
            sys.path.insert(0, str(target))
            print("Target site-packages: {}".format(target))
        argv = argv[3:]  # consume so --list/--only are visible
    else:
        argv = argv[1:]

    only_names: set[str] | None = None
    if argv and argv[0] == "--list":
        print("Available patches (use with --only name1,name2):")
        for name, desc, _ in PATCH_LIST:
            print("  {}  {}".format(name, desc))
        return True
    if argv and argv[0] == "--only" and len(argv) >= 2:
        only_names = {s.strip() for s in argv[1].split(",") if s.strip()}
        valid = {n for n, _, _ in PATCH_LIST}
        unknown = only_names - valid
        if unknown:
            print("Unknown patch names: {}. Use --list to see available names.".format(unknown))
            return False

    print("=" * 50)
    print("Applying BactiMate compatibility patches" + (" (selected only)" if only_names else ""))
    print("=" * 50)

    results: dict[str, bool] = {}
    for name, desc, patch_func in PATCH_LIST:
        if only_names is not None and name not in only_names:
            continue
        print("\nPatching {}...".format(desc))
        try:
            results[name] = patch_func()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[name] = False

    print("\n" + "=" * 50)
    summary = "; ".join("{}: {}".format(n, "OK" if r else "skipped/failed") for n, r in results.items())
    print(summary)
    print("=" * 50)
    return any(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
