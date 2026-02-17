#!/usr/bin/env python3
"""
Apply the OMNI_MODELS patch to omnipose.core so that imports of cellpose_omni
do not fail with "cannot import name 'OMNI_MODELS' from 'omnipose.core'".

Usage:

  # Only apply the patch (in this process) and exit:
  python -m bactimate.scripts.patch_omnipose_omni_models

  # Apply the patch then run the diagnostic in the same process (so the patch is active):
  python -m bactimate.scripts.patch_omnipose_omni_models --run-diagnose -- omnipose_user_models/model.pth --training-dir /path/to/train

  # From your own script or REPL, patch before importing cellpose_omni:
  from bactimate.scripts.patch_omnipose_omni_models import apply_omnipose_omni_models_patch
  apply_omnipose_omni_models_patch()
  from cellpose_omni import models  # now works
"""
from __future__ import annotations

import sys
from pathlib import Path


def apply_omnipose_omni_models_patch() -> bool:
    """
    Ensure omnipose.core has OMNI_MODELS (fixes 'cannot import name OMNI_MODELS' in some installs).
    Call this before any import that pulls in cellpose_omni or omnipose.core.
    Returns True if the patch was applied or OMNI_MODELS was already present.
    """
    try:
        import omnipose.core
        if hasattr(omnipose.core, "OMNI_MODELS"):
            return True
        omni = []
        for attr in ("C1_MODELS", "C2_MODELS", "C1_BD_MODELS", "C2_BD_MODELS"):
            if hasattr(omnipose.core, attr):
                omni.extend(getattr(omnipose.core, attr))
        seen: set = set()
        omni = [x for x in omni if not (x in seen or seen.add(x))]
        if not omni:
            omni = ["bact_omni", "bact_phase_omni", "bact_fluor_omni"]
        omnipose.core.OMNI_MODELS = omni
        return True
    except Exception as e:
        print(f"Patch failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    argv = list(sys.argv[1:])
    run_diagnose = False
    if argv and argv[0] == "--run-diagnose":
        run_diagnose = True
        argv.pop(0)
    if argv and argv[0] == "--":
        argv.pop(0)

    if apply_omnipose_omni_models_patch():
        print("OMNI_MODELS patch applied (omnipose.core.OMNI_MODELS is set).")
    else:
        return 1

    if run_diagnose:
        if not argv:
            print("Error: --run-diagnose requires model path after --", file=sys.stderr)
            return 1
        # Run diagnose_flat_model main with remaining argv
        sys.argv = ["diagnose_flat_model"] + argv
        from bactimate.scripts.diagnose_flat_model import main as diagnose_main
        return diagnose_main()

    return 0


if __name__ == "__main__":
    sys.exit(main())
