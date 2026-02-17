# BactiMate and Omnipose: How We Use It

BactiMate does **not** call most of `omnipose.core` / `omnipose.utils` directly. Those run **inside Omnipose** when we start training (CLI) or run segmentation (cellpose_omni API). This note maps what we control to the relevant Omnipose behavior and confirms we use the training-related parts correctly.

## Training: We Invoke the CLI

We run:

```text
python -m omnipose --train --dir <path> --nchan <n> --nclasses <n> --channel_axis 0 --all_channels ...
```

Omnipose’s `__main__.py` and cellpose_omni then:

- Load image/mask pairs from `--dir` (with `--img_filter` / `--mask_filter`).
- Use **channel_axis** and **nchan** when reading images → correct layout for **random_crop_warp** and **random_rotate_and_resize** (augmentation).
- Use **labels_to_flows** (and related) to build training targets from masks.
- Use the **loss** from `omnipose.core` (flow, distance, boundary for nclasses=3).

So for **training**, the “commands” that matter are used **by Omnipose**; we only need to pass the right CLI args and provide correctly formatted data.

### What We Ensure for Training

| Omnipose expectation | What we do |
|----------------------|------------|
| **Channel format** | CYX = (nChan, nY, nX). We write/convert training images to (C, Y, X) and pass `--channel_axis 0`. |
| **channel_axis** | We pass `--channel_axis 0` when nchan ≥ 2 so `random_crop_warp` and data loaders see (C, Y, X). |
| **nchan / nclasses** | We pass `--nchan` and `--nclasses` (default 3). User-model loader uses only top-level nchan/nclasses from checkpoint (default nclasses=3). |
| **Masks** | Instance labels (background=0, each cell = unique ID). We prepare/copy masks and optionally tile; we do not call `masks_to_flows` / `labels_to_flows` ourselves—Omnipose does. |
| **Grayscale → 2 channels** | Patched `cellpose_omni/__main__.py` so when nchan==2 we pass `channels=[chan,chan2]` into `model.train(...)`, so grayscale is duplicated and **random_crop_warp** does not IndexError. |

So for **training**, we are using the relevant “commands” (data format + CLI) correctly; the actual functions (`labels_to_flows`, `random_crop_warp`, `random_rotate_and_resize`, `loss`, etc.) are used by Omnipose internally.

---

## Inference: cellpose_omni + omnipose.core

At inference we use cellpose_omni’s `CellposeModel.eval()` (and our patch for user-model loading). That path ends up calling:

- **compute_masks** (in omnipose/core) → uses **get_masks**, flow/distance/boundary, dynamics.
- **get_masks** (omnipose mask reconstruction).

We do **not** call `compute_masks` or `get_masks` ourselves from BactiMate code; we only patch them where needed.

### Patches That Touch omnipose.core

| Patch | Function(s) affected | Purpose |
|-------|----------------------|--------|
| **get_masks** | `omnipose.core.get_masks` | Handle `bd=None` in frozen builds (supply a small default so the function does not crash). |
| **OMNI_MODELS** | `omnipose.core.OMNI_MODELS` | Ensure the list exists in frozen builds (fallback from C1_MODELS, etc.). |

So for **inference**, we are not “using” the full list of core/utils functions ourselves; we only patch those code paths where needed for our environment.

---

## omnipose.utils

We do **not** import or call `omnipose.utils` from BactiMate. Normalization, tiling at test time, etc. are used inside cellpose_omni/Omnipose when they run the model. Our job is to pass the right model path, diameter, and thresholds; we do not call e.g. `normalize99`, `make_tiles_ND`, `rescale`, etc. directly.

---

## Summary

- **Training**: We use Omnipose correctly by (1) preparing data in CYX and instance masks, (2) passing `--channel_axis 0`, `--nchan`, `--nclasses`, and (3) patching `__main__.py` so `model.train(..., channels=...)` is set for nchan==2. The functions you listed (e.g. `labels_to_flows`, `random_crop_warp`, `random_rotate_and_resize`, `loss`) are used by Omnipose; we only ensure inputs and CLI are correct.
- **Inference**: We use cellpose_omni’s eval path; it calls `compute_masks` / `get_masks` internally. We patch `get_masks` (bd=None) and OMNI_MODELS where needed for our builds.
- **omnipose.utils**: Not called from our code; used internally by Omnipose/cellpose_omni.

So yes: we are using the **training-relevant** behavior (and the inference code paths we depend on) correctly; the “commands” you listed are used by Omnipose under the hood, and we align our data and flags with that.
