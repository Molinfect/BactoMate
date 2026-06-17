# BactoMate scripts

Python backends invoked from the GUI (subprocess workers) or run manually from the CLI.

Run from the project root with the project venv active:

```bash
python -m bactomate.scripts.<module> [args]
```

---

## GUI pipeline (automatic)

These are launched by the GUI; you normally do not run them by hand.

| Script | Module | When it runs |
|--------|--------|--------------|
| `convert_nd2_to_tif.py` | File conversion | ND2/LIF/CZI → TIFF |
| `batch_process_tiff_sequence.py` | Segmentation | Omnipose / Cellpose on a mask TIFF stack |
| `analyze_cell_areas.py` | Analyze masks | Morphology, FI, foci, Excel/CSV output |
| `unet_predict_patches.py` / `unet_predict_impl.py` | U-Net segmentation | Patch inference |
| `unet_train_patches.py` / `unet_train_impl.py` | U-Net training | Training loop |

---

## CLI utilities (manual)

| Script | Purpose | Example |
|--------|---------|---------|
| `batch_process_folder.py` | Segment + analyze every TIFF in a folder | `python -m bactomate.scripts.batch_process_folder /path/to/folder --model bact_phase_omni` |
| `assign_persistent_ids.py` | Assign persistent CellIDs across frames (raw mask stack) | `python -m bactomate.scripts.assign_persistent_ids masks.tif -o masks_cellids.tif` |
| `check_tiff_format.py` | Inspect or reshape TIFF stack layout | `python -m bactomate.scripts.check_tiff_format stack.tif` |
| `generate_bactomate_pdf.py` | Build user-manual PDF from markdown | `python -m bactomate.scripts.generate_bactomate_pdf` |
| `check_environment.py` | Dependency / GPU smoke report | `python -m bactomate.scripts.check_environment` |

---

## Omnipose / model diagnostics

Used from the Segmentation tab (“Check pipeline”) or for manual troubleshooting.

| Script | Purpose |
|--------|---------|
| `check_omnipose_pipeline.py` | End-to-end Omnipose readiness check |
| `diagnose_flat_model.py` | Debug flat/wrong segmentation output |
| `inspect_omnipose_checkpoint.py` | Inspect checkpoint metadata |
| `fix_user_model_metadata.py` | Repair user model sidecar files |
| `convert_checkpoint_to_plain.py` | Convert checkpoint to plain state_dict |
| `verify_affinity_model.py` | Verify affinity model load |
| `patch_omnipose_omni_models.py` | Patch Omnipose model registry (setup) |

---

## Setup / patches (not under `bactomate/scripts/`)

| Path | Purpose |
|------|---------|
| `setup/apply_all_patches.py` | Apply Omnipose/Cellpose site-package patches (run by launchers) |
| `setup/install_mac.sh` / `setup/install_windows.bat` | Create venv and install dependencies |
| `launch_gui.sh` / `launch_windows.bat` | Start the GUI |

See [docs/README.md](../../docs/README.md) for install, launch, and bundle build instructions.
