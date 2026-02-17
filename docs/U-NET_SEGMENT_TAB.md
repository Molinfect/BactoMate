# U-NET Segment Tab — Plan and Open Decisions

## What Was Done

### 1. New tab: **U-NET Segment**
- **Location:** `bactimate/gui/tabs/unet_segment.py`
- **Layout:** Mirrors Omnipose Segment:
  - **Input File/Folder** — path line edit, Browse File, Browse Folder, drag-and-drop
  - **U-Net Parameters** — Model dropdown (trained `.keras` from BactiMate U-Net models dir), Refresh, **Train new U-Net model...** button
  - **Preview U-Net Segmentation** — run model on first frame/file
  - **Process File or Folder** — batch prediction, output next to input or in `unet_segmentation/`

### 2. Training dialog (same ideas as Omnipose)
- **Training directory** — folder with image/mask pairs
- **Image filter / Mask filter** — e.g. `_img`, `_mask` (or `_Simple Segmentation` for Building U-NET)
- **Check for names** — uses `get_training_pairs()` to show pair count
- **Make masks binary (U-Net)** — `prepare_unet_masks_binary()` → writes to `unet_masks_binary/`, copies images; masks become strict [0, 255]
- **Tile and prepare for training** — reuses `tile_training_data()` → output to `unet_masks_tiled/` (U-Net–specific; separate from Omnipose)
- **Building U-NET project path** — optional; when set, training and prediction delegate to your Building U-NET scripts (GPU used there)
- **Epochs, Batch size, Use GPU**
- **Model name** — saved under BactiMate’s U-Net models dir as `{name}.keras` and `{name}_patch_config.json`
- **Start training** — runs `unet_analyze_patch_config` then `unet_train_patches` (with Building U-NET path if set)

### 3. Paths and storage
- **U-Net models dir:** `get_unet_models_dir()` → `unet_user_models/` (e.g. `<repo>/unet_user_models/` or `~/Library/Application Support/BactiMate/Models/unet_user_models/`). Separate from Omnipose.
- **Building U-NET path** — stored in preferences (`get_building_unet_path()` / `set_building_unet_path()`).

### 4. Scripts (under `bactimate/scripts/`)
- **unet_analyze_patch_config.py** — CLI: `--train-path`, `--output`, `--img-filter`, `--mask-filter`, `--force-patch-size`, `--no-visualizations`, `--building-unet-path`. If `--building-unet-path` is set, delegates to Building U-NET’s `analyze_patch_config`; else writes a minimal `patch_config.json` from image dimensions.
- **unet_train_patches.py** — CLI: `--train-path`, `--config-path`, `--output-model-path`, `--epochs`, `--batch-size`, `--use-gpu`, `--img-filter`, `--mask-filter`, `--building-unet-path`. If Building U-NET path is set: copies your training data into `Building U-NET/Train_files/`, copies config to `Building U-NET/patch_config.json`, runs `U-net-patches.py` (with `yes` piped for the prompt), then copies `model_patches.keras` (and config) to the requested output path.
- **unet_predict_patches.py** — CLI: `-i`, `-o`, `-m`, `--config-path`, `--building-unet-path`. If Building U-NET path is set: copies model and config into that folder and runs `predict_patches.py`.

### 5. Mask utilities
- **prepare_unet_masks_binary()** in `mask_utils.py` — converts all masks in a folder to binary [0, 255], writes to `unet_masks_binary/` and copies images (same pairing as `get_training_pairs`).
- **convert_single_mask_to_binary_unet()** — single-file helper.

### 6. GPU
- When using **Building U-NET path**, GPU is used by the Building U-NET scripts (TensorFlow/Metal as in your project).
- “Use GPU” in the training dialog is passed as `--use-gpu 1/0`; the actual GPU usage is in the Building U-NET training script.

---

## How to Use (Workflow)

1. **Set Building U-NET path (once)**  
   Open **Train new U-Net model...** → set **Building U-NET project path** to your `Building U-NET` folder → it is saved in preferences.

2. **Training**
   - Set **Training directory** to a folder with image/mask pairs.
   - Set **Image filter** / **Mask filter** to match your names (e.g. `` and `_mask`, or `` and `_Simple Segmentation` for Building U-NET).
   - **Check for names** to confirm pair count.
   - Optionally **Make masks binary** → use the new `unet_masks_binary/` as training dir if you want strict binary masks.
   - Optionally **Tile and prepare** → use the tiled folder as training dir.
   - Set **Epochs**, **Batch size**, **Use GPU**, **Model name**.
   - **Start training** → analyze config runs, then training; model is saved in BactiMate’s U-Net models dir.

3. **Inference**
   - In the main tab, choose a **Model** from the dropdown (or Refresh after training).
   - Set **Input File/Folder**.
   - **Preview** or **Process File or Folder** (Building U-NET path must be set for prediction).

---

## Open Questions / Decisions

1. **Building U-NET mask naming**  
   Building U-NET’s `U-net-patches.py` expects masks like `{base}_Simple Segmentation.tiff`. When you use “Make masks binary” or your own `_mask` names, the **copied** files in `Train_files/` keep their names. If your masks are `img_mask.tif`, the script may not find them. **Decision:** Either (a) document that for “Train with Building U-NET” the training folder should use Building U-NET naming, or (b) add a step that renames/copies masks to `*_Simple Segmentation.tiff` when copying into `Train_files/`.

2. **Training in background**  
   Currently “Start training” runs the training in the same thread (blocking). **Decision:** Move to a `QThread` worker so the dialog stays responsive and the user can cancel (like Omnipose training).

3. **Preview UX**  
   Preview runs prediction and writes to a temp folder; it does not open an overlay window like Omnipose. **Decision:** Either keep “see output folder” or add a simple overlay dialog that loads the first output mask and shows it over the first input image.

4. **TensorFlow dependency**  
   BactiMate does not require TensorFlow by default. Training/prediction without Building U-NET would need an in-repo TensorFlow-based implementation. **Decision:** Keep “Building U-NET path required” for now, or add optional `tensorflow` and a minimal in-repo train/predict path later.

5. **Single-file process**  
   For “Process File or Folder” with a single file, the predict script is called with the parent folder; output is `{stem}_unet_segmentation/`. **Decision:** Confirm whether single-file output naming (e.g. one mask next to the file) is preferred.

6. **Patch config source**  
   When Building U-NET path is set, “Analyze patch config” can delegate to Building U-NET’s analyzer (full geometry/periodicity). When not set, a minimal config is written from image dimensions. **Decision:** Is the minimal config sufficient for your typical data, or should we always recommend setting Building U-NET path for training?

7. **Session logs**  
   The new tab is registered in `SESSION_TAB_KEYS` and `TAB_WIDGET_ATTR` so session logs can include it. **Decision:** Whether to add a `get_session_params()`-style export for the U-NET tab (e.g. model name, building path, last dir) for session log apply.

---

## File Summary

| File | Change |
|------|--------|
| `bactimate/gui/paths.py` | `get_omnipose_user_models_dir()` → `omnipose_user_models/`; `get_unet_models_dir()` → `unet_user_models/`. |
| `bactimate/gui/preferences.py` | Added `LAST_DIR_UNET_SEGMENT`, `get_building_unet_path()`, `set_building_unet_path()`. |
| `bactimate/gui/mask_utils.py` | Added `UNET_MASKS_BINARY_SUBDIR`, `convert_single_mask_to_binary_unet()`, `prepare_unet_masks_binary()`. |
| `bactimate/gui/tabs/unet_segment.py` | New tab: file group, params (model + train button), preview, process; training dialog with dir, filters, check names, make binary, tile, Building U-NET path, epochs/batch/GPU, model name, start training. |
| `bactimate/gui/tabs/__init__.py` | Export `UNetSegmentTab`. |
| `bactimate/gui/main_window.py` | Create and add **U-NET Segment** tab. |
| `bactimate/gui/constants.py` | Added `unet_segment` to `SESSION_TAB_KEYS` and `TAB_WIDGET_ATTR`. |
| `bactimate/scripts/unet_analyze_patch_config.py` | New: minimal or Building U-NET analyze, writes `patch_config.json`. |
| `bactimate/scripts/unet_train_patches.py` | New: copy data + config to Building U-NET, run `U-net-patches.py`, copy model back. |
| `bactimate/scripts/unet_predict_patches.py` | New: copy model/config to Building U-NET, run `predict_patches.py`. |
| `docs/U-NET_SEGMENT_TAB.md` | This plan and open-questions document. |
