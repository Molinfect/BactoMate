#!/usr/bin/env python3
"""Verify bact_phase_affinity loads with nclasses=3 (run from repo root)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    ckpt = root / "omnipose_pretrained" / "bact_phase_affinitytorch_0"
    if not ckpt.is_file():
        print(f"Missing checkpoint: {ckpt}")
        return 1

    try:
        import omnipose.core as oc

        if not hasattr(oc, "OMNI_MODELS"):
            names: list = []
            for attr in ("C1_MODELS", "C2_MODELS", "C1_BD_MODELS", "C2_BD_MODELS"):
                names.extend(getattr(oc, attr, []) or [])
            oc.OMNI_MODELS = list(dict.fromkeys(names)) or [
                "bact_omni",
                "bact_phase_omni",
                "bact_fluor_omni",
                "bact_phase_affinity",
            ]
    except ImportError as e:
        print(f"omnipose import failed: {e}")
        return 1

    from bactomate.gui.omnipose_utils import (
        BACTOMATE_AFFINITY_LOADER_ID,
        load_bact_phase_affinity_model,
        materialize_bact_phase_affinity_checkpoint,
    )

    safe = materialize_bact_phase_affinity_checkpoint(ckpt)
    print(f"affinity_loader={BACTOMATE_AFFINITY_LOADER_ID}")
    print(f"source={ckpt}")
    print(f"materialized={safe}")

    model = load_bact_phase_affinity_model(gpu=False, checkpoint_path=ckpt)
    nc = getattr(model, "nclasses", None)
    omni = getattr(model, "omni", None)
    nchan = getattr(model, "nchan", None)
    print(f"OK class={type(model).__name__} nchan={nchan} nclasses={nc} omni={omni}")
    if nc != 3 or omni:
        print("FAIL: expected nclasses=3 and omni=False")
        return 1
    print("Affinity model load verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
