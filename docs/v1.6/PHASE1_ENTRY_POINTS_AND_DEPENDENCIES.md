# Phase 1.1: Entry points and dependencies map

**Done when:** One clear map of which GUI tab runs which script, what each script imports from the package, and which paths are used vs legacy/dead.

---

## 1. Scripts invoked by the GUI (subprocess or module)

| GUI tab / action | How invoked | Entry point | Script / module |
|------------------|-------------|-------------|------------------|
| **TIFF Conversion** | `WorkerThread(script_path, cmd_args)` with `script_path = CONVERT_ND2_SCRIPT` | File path to script; `cmd = [python_exe, script_path] + args` | `convert_nd2_to_tif.py` (via `bactimate/scripts/` or fallback `get_script_path`) |
| **Omnipose Segment** | `WorkerThread(script_path, cmd_args)` with `script_path = BATCH_PROCESS_TIFF_SCRIPT` | Same | `batch_process_tiff_sequence.py` |
| **Analyze Masks** (single + folder) | `WorkerThread(script_path, run_args)` with `script_path = "-m"`, `run_args = ["bactimate.scripts.analyze_cell_areas", ...cli] | `cmd = [python_exe, "-m", module] + cli` | `bactimate.scripts.analyze_cell_areas` (module, no file path) |
| **U-NET Segment – Training “Analyze”** | `subprocess.run(analyze_cmd)` (blocking) | `[sys.executable, "-m", "bactimate.scripts.unet_analyze_patch_config", ...]` | `bactimate.scripts.unet_analyze_patch_config` |
| **U-NET Segment – Training “Start”** | `_UnetTrainWorker(train_cmd)` → `Popen(train_cmd)` | `[sys.executable, "-m", "bactimate.scripts.unet_train_patches", ...]` | `bactimate.scripts.unet_train_patches` |
| **U-NET Segment – Process** | `subprocess.run(cmd)` (blocking) | `[sys.executable, "-m", "bactimate.scripts.unet_predict_patches", ...]` | `bactimate.scripts.unet_predict_patches` |
| **Tracking** | In-process: `TrackingWorker(QThread)` runs tracking logic inside the app | No subprocess | (no script) |
| **Post-processing – Growth curve video** | In-process: `GrowthCurveVideoWorker(QThread)` | No subprocess | (no script) |

**Re-run analysis (session):** Applies session params then calls, in order, tab methods that themselves use the above: `convert_nd2` → `process_file_or_folder` → `analyze_masks` → `run_tracking`. So the same scripts/modules as above.

---

## 2. Where script paths come from

- **constants.py**
  - `CONVERT_ND2_SCRIPT` = `get_app_root() / "bactimate" / "scripts" / "convert_nd2_to_tif.py"` if that file exists, else `get_script_path("convert_nd2_to_tif.py")`.
  - `BATCH_PROCESS_TIFF_SCRIPT` = same pattern for `batch_process_tiff_sequence.py`.
- **paths.py**
  - `get_scripts_dir()` = `get_app_root() / "scripts"` (legacy: top-level `scripts/`).
  - `get_script_path(basename)` = `get_scripts_dir() / basename` → in dev repo this is `repo/scripts/<basename>`; that directory is **not** present in the repo (scripts live under `bactimate/scripts/`). So the fallback is only used when the package script path does not exist (e.g. some installed or frozen layout).

**Conclusion:** In normal development, only the package paths `bactimate/scripts/convert_nd2_to_tif.py` and `bactimate/scripts/batch_process_tiff_sequence.py` are used. `get_script_path` points to a **legacy** location (`app_root/scripts/`) that is **unused** when package scripts exist.

---

## 3. Scripts in `bactimate/scripts/` – used by GUI vs not

| Script | Invoked by GUI? | Notes |
|--------|------------------|--------|
| `convert_nd2_to_tif.py` | Yes | TIFF Conversion tab |
| `batch_process_tiff_sequence.py` | Yes | Omnipose Segment tab |
| `analyze_cell_areas.py` | Yes | As module `bactimate.scripts.analyze_cell_areas` from Analyze Masks |
| `unet_analyze_patch_config.py` | Yes | U-NET tab, training “Analyze” step |
| `unet_train_patches.py` | Yes | U-NET tab, training “Start” |
| `unet_predict_patches.py` | Yes | U-NET tab, “Process” |
| `batch_process_folder.py` | No (directly) | Imported by `batch_process_tiff_sequence`; not run as entry by GUI |
| `assign_persistent_ids.py` | No | Not invoked by GUI (could be CLI or future use) |
| `check_environment.py` | No | CLI / setup |
| `check_tiff_format.py` | No | CLI / utility |
| `check_omnipose_pipeline.py` | No | CLI / setup (referenced from diagnose, patch scripts) |
| `convert_checkpoint_to_plain.py` | No | CLI (referenced from gui omnipose_utils) |
| `diagnose_flat_model.py` | No | CLI (referenced from patch_omnipose_omni_models) |
| `fix_user_model_metadata.py` | No | CLI |
| `inspect_omnipose_checkpoint.py` | No | CLI |
| `patch_omnipose_omni_models.py` | No | Setup |
| `unet_predict_impl.py` | No | Library; imported by unet_seg (preview) and unet_predict_patches |
| `unet_train_impl.py` | No | Library; imported by unet_train_patches |
| `unet_train_patches.py` | Yes | See above |

---

## 4. Package imports from scripts (scripts → bactimate.*)

- **convert_nd2_to_tif.py** → `bactimate.modules.tiff_metadata`
- **analyze_cell_areas.py** → `bactimate.modules.foci_utils`, `bactimate.modules.validation`; optionally `bactimate.gui.constants.LOGS_DIR`
- **batch_process_tiff_sequence.py** → `bactimate.gui.omnipose_utils`, `bactimate.modules.tiff_metadata`
- **batch_process_folder.py** → `bactimate.scripts.batch_process_tiff_sequence`, `bactimate.scripts.analyze_cell_areas`
- **unet_analyze_patch_config.py** → `bactimate.gui.mask_utils`
- **unet_train_patches.py** → `bactimate.scripts.unet_train_impl`; unet_train_impl → `bactimate.gui.mask_utils`
- **unet_predict_patches.py** → `bactimate.scripts.unet_predict_impl`; unet_predict_impl → `bactimate.modules.tiff_metadata`
- **assign_persistent_ids.py** → `bactimate.scripts.analyze_cell_areas`
- **check_environment.py** → `bactimate.modules.bioio_reader`
- **check_omnipose_pipeline.py**, **diagnose_flat_model.py**, **fix_user_model_metadata.py**, **inspect_omnipose_checkpoint.py**, **convert_checkpoint_to_plain.py**, **patch_omnipose_omni_models.py** → various `bactimate.gui.*` (paths, omnipose_utils, utils, mask_utils)

---

## 5. GUI in-process imports of script code (no subprocess)

- **file_conversion.py** → `bactimate.scripts.convert_nd2_to_tif.bleach_correct_stack`
- **analyze_masks.py** → `bactimate.scripts.analyze_cell_areas`: `calculate_cell_properties`, `filter_rod_shaped_cells`, `filter_by_size` (preview); and for running: module `bactimate.scripts.analyze_cell_areas` via `-m`
- **unet_seg.py** → `bactimate.scripts.unet_predict_impl.load_model_and_config`, `predict_one_image` (preview)

---

## 6. Paths summary: used vs legacy

| Path / concept | Status |
|----------------|--------|
| `bactimate/scripts/*.py` | **Used** – primary location for all GUI-invoked scripts and modules |
| `get_app_root() / "bactimate" / "scripts" / "<name>.py"` | **Used** – source of CONVERT_ND2_SCRIPT and BATCH_PROCESS_TIFF_SCRIPT when file exists |
| `get_scripts_dir()` = `get_app_root() / "scripts"` | **Legacy** – no `scripts/` at repo root; only used as fallback when pkg path missing |
| `get_script_path(basename)` | **Legacy** – used only in constants fallback; points to `app_root/scripts/<basename>` |
| Top-level `scripts/` (e.g. from git “D scripts/…”) | **Unused / deleted** – not present in repo; no references in code to old top-level script names as paths |

---

## 7. One-page summary

- **Tab → script:** TIFF Conversion → `convert_nd2_to_tif.py` (path); Omnipose → `batch_process_tiff_sequence.py` (path); Analyze Masks → `bactimate.scripts.analyze_cell_areas` (-m); U-NET → `unet_analyze_patch_config`, `unet_train_patches`, `unet_predict_patches` (-m); Tracking / Growth curve → in-process, no script.
- **Scripts** live in `bactimate/scripts/`; constants prefer that; fallback is `get_script_path` → `app_root/scripts/` (legacy, often missing in dev).
- **No remaining references** to the old top-level `scripts/` filenames as invocation paths; only `bactimate/scripts/` and the legacy fallback are used.

Phase 1.1 **done**. Proceed to Phase 1.2 (identify dead and redundant code).
