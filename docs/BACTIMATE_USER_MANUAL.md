# BactiMate — Complete User Manual

**Version:** 1.0  
**Document type:** Detailed user manual (GUI reference)  
**Approximate length:** 30–40 A4 pages when printed (body text ~12,000+ words; with headings and lists, fills 30–40 pages at normal font size).  
**Audience:** Researchers and users who want a full, structured reference to every tab, setting, and workflow in BactiMate.  
**Printing:** Export this Markdown to PDF (e.g. via Pandoc, VS Code, or a Markdown-to-PDF tool) for a printable manual. Use a normal font size (e.g. 11 pt body) and standard margins to achieve roughly 30–40 A4 pages.

---

## Table of Contents

1. [Introduction and scope](#1-introduction-and-scope)
2. [Application overview](#2-application-overview)
3. [TIFF Conversion tab](#3-tiff-conversion-tab)
4. [Omnipose Segment tab](#4-omnipose-segment-tab)
5. [Analyze Masks tab](#5-analyze-masks-tab)
6. [Tracking tab](#6-tracking-tab)
7. [Video Graph tab](#7-video-graph-tab)
8. [Session Logs tab](#8-session-logs-tab)
9. [File types, naming, and workflows](#9-file-types-naming-and-workflows)
10. [Common scenarios and questions](#10-common-scenarios-and-questions)
11. [References and further documentation](#11-references-and-further-documentation)

---

## 1. Introduction and scope

### 1.1 What this manual covers

This manual describes **every way a user can interact with BactiMate** through its graphical interface:

- **Tabs:** What each tab does, in what order they appear, and how they relate to each other.
- **Inputs:** File and folder selection (including drag-and-drop), placeholders, and validation.
- **Parameters:** Every spin box, combo box, check box, and text field: name, range, default, and effect.
- **Info boxes:** The exact text of each **(?)** info icon (clickable tooltips) next to settings.
- **Buttons and actions:** What each button does, when it is enabled, and what dialogs or outputs result.
- **Outputs:** Where files are written, naming conventions, and how to open or reuse them in other tabs.

It is intended as a **deep reference**, not a quick start. For a shorter overview and installation, see the main project `README.md`.

### 1.2 How to use this manual

- **Look up a tab:** Use the table of contents to jump to the tab (e.g. “Omnipose Segment” or “Analyze Masks”).
- **Look up a setting:** Within each tab section, settings are grouped by panel (e.g. “Model Parameters”, “Run Parameters”). Use the subsection headings or search for the label (e.g. “Flow Threshold”, “Pixel size (µm)”).
- **Understand a workflow:** Read the tab section from top to bottom; “Workflow” subsections describe typical steps. Cross-tab workflows are in Section 9.
- **Answer “what does this (?) mean?”:** Each parameter that has an info icon has its tooltip text reproduced in a **Info (?) — …** block.

### 1.3 Conventions

- **Default** means the value or state when the tab is first loaded or after “Reset to Default” (where available).
- **Range** is given as [min, max] or as “allowed values” for combo boxes.
- **Placeholder** is the gray hint text in empty line edits (e.g. “Select a TIFF file or folder...”).

---

## 2. Application overview

### 2.1 Main window

- **Title:** “BactiMate”
- **Subtitle (in title bar):** “BactiMate - your mate for bacteria time-lapse data”
- **Minimum size:** 1200×700 pixels; default resize 1200×900.
- **Layout:** Top title bar with logo; central **tab widget**; bottom **Status & Log** group.

On startup, the application restores the last window geometry and tab state if available (saved in QSettings). On close, it saves geometry, state, and the current session to a session log file (see Section 8).

### 2.2 Title bar

- **Logo:** Image from `img/logo2.png` (scaled to height 60 px if larger). If the file is missing, the text “BactiMate” is shown instead.
- **Title text:** “BactiMate - your mate for bacteria time-lapse data” (large, bold, light color).

### 2.3 Tab bar (order of tabs)

Tabs appear in this order:

1. **TIFF Conversion** — Convert ND2/LIF/CZI (and optionally TIFF) microscopy files to TIFF.
2. **Omnipose Segment** — Segment cells with Omnipose (single file or folder), preview, batch process, and train custom models.
3. **Analyze Masks** — Analyze mask TIFFs (single file or folder): morphological and size filters, Excel export, optional fluorescence intensity (FI) analysis.
4. **Tracking** — Run Trackastra on mask time series; optional link to original images and Excel analysis.
5. **Video Graph** — Create a growth-curve video: TIFF sequence + Excel analysis → side-by-side graph and video.
6. **Session Logs** — List recent session logs, view saved parameters, and apply them to the current session.

**Note:** A “SAM2 Segment” tab exists in the codebase but is not shown in the tab bar (commented out). This manual does not document it.

### 2.4 Status & Log (bottom panel)

- **Progress bar:** Hidden by default; shown during long-running jobs (e.g. batch segmentation, conversion). Range 0–100; format “%p%”.
- **Log text area:** Read-only, monospace font (Monaco or Courier New, 11 pt). Minimum height 100 px, maximum 150 px. Used for:
  - Startup message: “Omnipose Batch Processing GUI started.” and “Select a tab and configure parameters to begin.”
  - Per-tab messages: selected paths, job start/end, errors, and script output.

Tabs and scripts call `parent_window.log_output(...)` to append lines to this area.

### 2.5 Status bar

- **Message:** “Ready” at startup. Can be updated by tabs to show short status (e.g. “Processing...”, “Done”).

### 2.6 Session saving

When the window is closed, the application:

- Saves window geometry and state (e.g. which tab was active).
- Writes a **session log** JSON file under the logs directory (e.g. `Logs/`) with a timestamped filename. The session log contains parameter key–value pairs per tab (e.g. segmentation, analyze_masks, tracking, growth_curve_video, nd2) so that “Session Logs” can later apply them. File paths are typically not changed when applying a session (only parameters).

---

## 3. TIFF Conversion tab

**Purpose:** Convert microscopy files (ND2, LIF, CZI, and optionally TIFF) to TIFF with configurable frame count, channel handling, and optional image corrections (drift, photobleaching). Metadata can be displayed after loading a file or folder.

### 3.1 Input: Microscopy File Selection

- **Control:** A drag-and-drop group box titled **“Microscopy File Selection”**.
- **Path field:** Line edit; placeholder: “Select a microscopy file or folder... (or drag and drop)”. Minimum width 500 px; font 12 pt.
- **Browse File...:** Opens file dialog. Filter: ND2, LIF, CZI, TIFF (all common extensions). On selection, the path is set and, for a single file, `load_file_info` is called to fill metadata and enable Preview/Convert.
- **Browse Folder...:** Opens folder dialog. On selection, the path is set to the folder and the app looks for supported files in that folder to display metadata (or a message if none found).
- **Drag-and-drop:** Dropping a file or folder sets the path and triggers the same load logic as above (single-file info or folder list).

Supported extensions: `.nd2`, `.ND2`, `.lif`, `.LIF`, `.czi`, `.CZI`, `.tif`, `.tiff`, `.TIF`, `.TIFF`.

### 3.2 Conversion Parameters

**Group title:** “Conversion Parameters”

**Number of Frames**

- **Label:** “Number of Frames:”
- **Control:** Spin box; range [1, 10000]; default 100.
- **Info (?) —** “Number of time frames to extract from the microscopy file. Source: Hardcoded default value. The value will automatically update to the maximum number of frames when a file is loaded. Note: Extracting more frames takes longer and uses more disk space.”

**Channel Handling**

- **Label:** “Channel Handling:”
- **“Save channels separately (recommended)”:** Check box; default **checked**. When checked, each channel is written to its own TIFF file (or as specified by the conversion script).
- **“Max intensity projection for multiple Z stack”:** Check box; default **unchecked**.
- **Info (?) (Z projection) —** “Apply maximum intensity projection across Z-slices when Z > 1. Default: Disabled. When enabled: For each pixel, takes the maximum intensity value across all Z-slices. This creates a single 2D image from a 3D Z-stack. When disabled: Uses only the first Z-slice (Z=1). Use when: You have Z-stack data and want to combine all slices into one image.”
- **Channel select:** Combo box; first item “All channels”; populated and enabled when a multi-channel file is loaded. Allows restricting conversion to specific channels.

**Image Corrections**

- **“Apply drift correction (X,Y registration)”:** Check box; default **unchecked**.
  - **Info (?) —** “Apply X,Y drift correction to compensate for stage movement during time-lapse imaging. Default: Disabled. Method: Phase cross-correlation (frame-by-frame alignment). Each frame is aligned to the PREVIOUS frame (incremental correction). Use when: You notice the sample drifting over time. Note: Increases processing time. The first frame remains unchanged; subsequent frames are incrementally aligned.”
- **“Apply photobleaching correction”:** Check box; default **unchecked**.
- **Method combo:** “Exponential (intensity preservation)”, “Linear decay”, “Histogram matching”, “Per-frame min-max (visualization)”. Default: first (Exponential).
- **Info (?) (bleach) —** “Photobleaching correction. Which channels: Tick only fluorescence channels; phase/transmission leave unticked. Methods: • Exponential: I(t)=I₀×exp(-t/τ), normalizes to first frame. Use for intensity quantification. • Linear: Linear decay fit, scale to first frame. • Histogram matching: Match each frame's histogram to frame 0. Good for segmentation; not for quantification. • Per-frame min-max: Stretch each frame min→0, max→full range; visualization only.”
- After loading a multi-channel file, checkboxes for each channel appear to select which channels get bleach correction.

### 3.3 Output Directory

- **Group title:** “Output Directory”
- **Path field:** Placeholder: “Auto: creates 'Tif_1-N' folder in input directory”. Minimum width 500 px; font 12 pt. If left empty, the conversion script typically creates a `Tif_1-N` (or similar) folder next to the input.
- **Browse...:** Opens folder dialog to set a custom output directory.

### 3.4 Extracted Metadata

- **Group title:** “Extracted Metadata”
- **Control:** Read-only text edit; min height 120 px, max 200 px; font 12 pt; placeholder: “Metadata will be displayed here after loading microscopy file...”. Displays metadata (e.g. dimensions, channels, pixel size) after loading a file or folder.

### 3.5 Buttons

- **Preview File:** Enabled when a valid file (or folder with files) is loaded. Shows a preview of the first frame (or first file) in a dialog.
- **Convert to TIFF:** Enabled when a valid file or folder is set. Runs the ND2 conversion script with the current parameters and writes TIFFs to the chosen output directory.

### 3.6 Workflow (typical)

1. Select a microscopy file or folder (Browse or drag-and-drop).
2. Check “Extracted Metadata” and adjust “Number of Frames” and channel options if needed.
3. Optionally set “Output Directory”; otherwise use auto output folder.
4. Optionally enable drift or photobleaching correction and set method/channels.
5. Click “Preview File” to verify, then “Convert to TIFF” to run conversion.

---

## 4. Omnipose Segment tab

**Purpose:** Run Omnipose cell segmentation on a single TIFF file or on all TIFF files in a folder; preview results with optional cache; train custom Omnipose models from image/mask pairs. Supports user-trained models and automatic flow-threshold presets (0 for user models, 0.4 for built-in).

### 4.1 Input: File/Folder

- **Group:** “Input File/Folder” (drag-and-drop).
- **Path field:** Placeholder: “Select a TIFF file or folder... (or drag and drop)”. Minimum width 500 px; font 12 pt.
- **Browse File...:** Opens “Select TIFF File”; filter `*.tif *.tiff`. Sets path and updates overlay checkbox state and button states.
- **Browse Folder...:** Opens “Select Folder”. Sets path to the folder.
- **Drag-and-drop:** Sets the path to the dropped file or folder.

**Button states:** “Preview Omnipose Segmentation” and “Process File or Folder” are **disabled** until the path is non-empty and exists as a file or directory.

### 4.2 Segmentation Parameters

**Group:** “Segmentation Parameters”, with two sub-groups side by side: **Model Parameters** (left) and **Run Parameters** (right), plus **Reset to Default** at the bottom.

#### 4.2.1 Model Parameters

**Model**

- **Label:** “Model:”
- **Info (?) —** “Omnipose model type for cell segmentation. Default: 'bact_phase_omni'. Models are downloaded from: https://github.com/kevinjohncutler/omnipose-models. Documentation: https://omnipose.readthedocs.io/models.html. Available: bact_phase_omni (bacterial phase contrast, recommended), bact_fluor_omni (bacterial fluorescence), bact_phase_affinity (bacterial phase, affinity-based). Tip: Choose based on cell type and imaging modality.”
- **Status label:** Shows “✓ User model”, “✓ Cached”, “Will download on first use”, or empty (on error). Green for ready, orange for “will download”.
- **Combo box:** Lists built-in models (bact_phase_omni, bact_fluor_omni, bact_phase_affinity) plus user-trained model names (stem of `.pth` files in the user models directory). Default selection: bact_phase_omni if available.
- **Tooltip (combo)** — “Omnipose model type for cell segmentation. Default: 'bact_phase_omni'. User-trained models appear here after training via 'Train model…'.”
- **Train model** button — Opens the “Train Omnipose model” dialog. Tooltip: “Train a new Omnipose model from your images and ground-truth masks.”

When the user selects a **user model**, flow threshold is automatically set to **0.0**; when selecting a **built-in** model, flow threshold is set to **0.4**.

**Diameter (0=auto)**

- **Control:** Double spin box; range [0, 1000]; default 0; decimals 1; width 100 px.
- **Info (?) —** “Expected cell diameter in pixels (only diameter field; Pixel size (µm) below is for scale bar). Default: 0 (auto-detect). Specify a value (e.g., 20-50 pixels for bacteria) for more consistent results. If you know size in µm: diameter_px = diameter_µm ÷ Pixel size (µm). Typical: Bacteria 20-50, Mammalian 30-100, Nuclei 15-30. Custom/user model: try Flow threshold 0, Mask threshold -2, Diameter 0 or cell size (px). See docs/CUSTOM_MODEL_TROUBLESHOOTING.md.”

**Pixel size (µm)**

- **Control:** Double spin box; range [0, 10]; default 0; decimals 3; step 0.01; special value text “0” (when 0 = no µm scale). Used for scale bar and calibration only; diameter is always in pixels.
- **Info (?) —** “Pixel size in µm (for scale bar and calibration). Diameter above is always in pixels. To get diameter in px from µm: diameter_px = diameter_µm ÷ pixel_size_µm. Common: 100× ~0.108, 60× ~0.18, 40× ~0.27. Can be filled from file metadata when available.”

**Flow Threshold**

- **Control:** Double spin box; range [0.0, 1.0]; default 0.4 (or 0.0 when a user model is selected); step 0.1; decimals 2.
- **Info (?) —** “Flow error threshold for cell detection. Default: 0.4. Lower values (0.2-0.3): more sensitive; higher (0.5-0.7): stricter. Range: 0.0 to 1.0. Tip: If missing cells, try 0.3; if too many false positives, try 0.5.”

**Mask threshold**

- **Control:** Double spin box; range [0.0, 1.0]; default 0.0; step 0.1; decimals 2.
- **Info (?) —** “Mask probability threshold. Default: 0.0. Lower: more permissive; higher: stricter. Tip: Usually 0.0 is fine; increase if masks look too loose.”

#### 4.2.2 Run Parameters

- **Use GPU (if available):** Check box; default **checked**. Info (?) — Use GPU for faster processing; disable for CPU-only or if OOM. Falls back to CPU if no GPU.
- **Use DBSCAN clustering:** Check box; default **unchecked**. Info (?) — Helps thin/elongated cells; slightly increases time.
- **Create overlay visualization:** Check box; default **checked**. When checked, creates overlay TIFF (masks on image). Tooltip in file mode: “Saved as: filename_overlay.tif”. In folder mode: “Creates overlay for each file in the folder. Saved as: filename_overlay.tif (in each file's output directory).”

**Reset to Default** — Restores all segmentation parameters to defaults. Tooltip: “Reset all segmentation parameters to their default values”.

### 4.3 Buttons (main tab)

- **Preview Omnipose Segmentation:** Opens a preview dialog: loads the first frame (or first file in folder), runs segmentation with current parameters, shows overlay and frame navigation. Disabled until a valid file/folder is set.
- **Process File or Folder:** Starts batch processing (single file or folder). Disabled until path is set. Progress and log go to the main window Status & Log. Cancel is available during run.
- **Cancel:** Visible only when a job is running. Tooltip: “Stop the current segmentation job.” Sets a user-cancelled flag and terminates the worker process.

### 4.4 Train Omnipose model dialog

Opened by **Train model** in the Model Parameters area.

**Training directory (images + masks)**

- **Path:** Placeholder: “Absolute path to folder with image/mask pairs (or drag and drop)”. Drag-and-drop: directory sets path; file sets path to its parent.
- **Info (?) —** “Folder containing your training data: one image file and one mask file per pair. Omnipose expects: for each image 'NAME.tif', a mask file 'NAME<Mask filter>.tif' (e.g. NAME_masks.tif). Mask format: instance label matrices (background=0, each cell=1,2,3,…). Not binary. See docs/TRAINING_MASKS.md.”
- **Browse…:** Select training directory.

**Mask format help (instance vs binary)** — Button opens `docs/TRAINING_MASKS.md` if present (system open).

**File name filters**

- **Image filter:** Line edit; placeholder “empty or _img”. Empty if images are e.g. A.tif; use _img for A_img.tif.
- **Mask filter:** Line edit; default “_mask”. Suffix on mask files (e.g. _masks → cell1_masks.tif).
- **Info (?) —** “Suffix that identifies image vs mask files. Image filter: leave empty if A.tif; use _img if A_img.tif. Mask filter: must match your filenames exactly.”

**Convert masks for Omnipose** — Converts binary masks to instance labels and writes (with image copies when naming matches) to subfolder `omnipose_masks`. Does **not** change the training directory path; when you click “Start training”, the app uses `omnipose_masks` automatically and copies images there if needed.
- **Info (?) —** “Convert binary masks to instance label matrices. Converted masks (and image copies when naming matches) are written to subfolder 'omnipose_masks'. Keep the training directory as the folder that contains your images and masks. When you click Start training, we use the omnipose_masks subfolder automatically and copy images there if needed.”

**Model name:** Line edit; placeholder “e.g. my_bacteria_v1”. Required; no path separators. This is the **output** name: the trained checkpoint is saved as `<model_name>.pth` in the **user models folder** and appears in the model dropdown. User models folder: when running from the **repo** it is `<repo_root>/user_models/`; when **installed** it is `~/Library/Application Support/BactiMate/Models` (macOS) or `%APPDATA%/BactiMate/Models` (Windows). The success dialog after training shows the full save path and, if you used transfer learning, a note “(Transfer learning from …)”.

**Training parameters**

- **Epochs:** Spin box [100, 50000]; default 4000. Info (?) — Full passes over training set; Omnipose often benefits from more (e.g. 4000 for bacteria).
- **Learning rate:** Double spin [0.001, 1.0]; default 0.1. Info (?) — Step size; larger faster but unstable, smaller slower but stable.
- **Batch size:** Spin [1, 128]; default 16. Info (?) — Larger batches faster with enough GPU memory; smaller use less memory.
- **Start from (transfer learning):** Dropdown. Options: **From scratch** (no pretrained model; nchan=2, nclasses=3); or a built-in model by name (**bact_phase_omni**, **bact_fluor_omni**, **bact_phase_affinity**) to initialize from that checkpoint—you get a new model file (your name) trained on your data. **nchan** and **nclasses** are locked to match the chosen model (required by Omnipose). See the (?) tooltip for details.
- **Check compatibility with selected model:** Button — Validates that the training folder has image/mask pairs and that image dimensions match the selected model (e.g. 2D images for bact_phase_omni). Use this before starting training when using transfer learning.
- **Channels (nchan):** Spin [1, 8]; default **2** (grayscale duplicated; Omnipose/patch uses channels=[0,0]). Info (?) — 1 = grayscale only; 2 = two-channel like bact_phase_omni. Set nclasses to match when evaluating. **Locked** when a pretrained model is selected.
- **Classes (nclasses):** Spin [2, 4]; default 3. Info (?) — 2 = flow+distance; 3 adds boundary. Use 3 for typical Omnipose bacteria models. **Locked** when a pretrained model is selected.
- **Use GPU:** Check box; default checked. Info (?) — Use GPU (CUDA or Apple MPS) if available.

Before starting, the dialog checks for image/mask pairs. When you use a **pretrained model**, it also checks that images are 2D if the model is 2D (e.g. bact_phase_omni); 3D stacks trigger a warning and you can cancel or continue. If **Image filter** is empty and no pairs are found, it **infers** the image filter from filenames (e.g. 634_phase.tif + 634_masks.tif → _phase) and fills the field so training can proceed. It also shows: “Found N image/mask pair(s). All will be used every epoch for M epochs.” (When using transfer learning it adds: “Transfer learning: starting from … (nchan/nclasses locked to match).”) Then it runs Omnipose training; log is shown in the dialog. **On success**, a dialog shows the full path where the model was saved (user models folder), and if you used transfer learning, a line “(Transfer learning from …)”. The model list is refreshed and the new model appears in the main tab’s model dropdown.

### 4.5 Preview dialog (Omnipose)

- **Title:** “Preview Omnipose Segmentation”
- **Content:** File or folder path; “Select a file to preview segmentation:” and a list (for folder) or single file; image canvas with overlay, scale bar (if pixel size > 0), frame navigation (“Frame:” spin + “of N”), status label (e.g. “Model loaded”, “Ready”), **Re-segment** and **Clear Cache** buttons. Tooltip on canvas: “Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).”
- **Re-segment:** Re-runs segmentation for the current frame with current parameters. Tooltip explains it updates with current diameter, flow, mask threshold, etc.
- **Clear Cache:** Clears the in-memory (and if applicable disk) preview cache for the current file/folder.

### 4.6 Batch processing (Process File or Folder)

- **Single file:** Runs Omnipose on the multipage TIFF; writes masks (and optionally overlay) to an output folder (e.g. `filename_omnipose/`).
- **Folder:** For each TIFF in the folder, runs segmentation and analysis (if applicable); can produce a combined Excel and per-file outputs. Progress and cancel are shown in the main window. **Cancel** button appears during the run and terminates the worker process when clicked.
- **Outputs:** Masks TIFF(s) (e.g. `*_masks.tif`), optional overlay TIFF(s) (e.g. `*_overlay.tif`); when processing a folder, a combined batch analysis Excel (e.g. `folder_batch_analysis.xlsx`) and per-file outputs (e.g. each file in its own `*_omnipose/` subfolder with masks, overlay, and per-file analysis Excel) may be produced depending on the batch script. Output directory is typically a dedicated results folder (e.g. `omnipose_batch_results/` under the selected folder) or next to each input file.
- **Progress:** The main window progress bar and log show file-by-file or frame-by-frame progress; completion messages and any errors are appended to the log.

### 4.7 Session parameters (Omnipose Segment)

Session log stores (among others): file path, model, diameter, pixel_size_um, flow_threshold, mask_threshold, use_gpu, cluster, create_overlay. Each time you complete a training run, a **training record** is appended and the session is saved (see Section 8). The record includes: **train_dir**, **model_name**, **n_epochs**, **learning_rate**, **batch_size**, **nchan**, **nclasses**, **use_gpu**, **mask_filter**, **img_filter**, and **pretrained_model** (e.g. `"bact_phase_omni"` when you used transfer learning, or absent/null when from scratch). Session files are written to the **Logs** folder with a timestamp (e.g. `Logs/2026-02-09_14-30-00_session.json`).

### 4.8 User-trained vs built-in models (no cells / wrong detections)

**Why custom models can fail on the same images they were trained on**

Inference must use the **same architecture** as training ([Omnipose docs](https://omnipose.readthedocs.io)):

- **"omni" in the filename (main cause)**: cellpose_omni sets **omni=True** (Omnipose dynamics) only when the checkpoint **filename stem** contains the word **"omni"**. Built-in models are named e.g. `bact_phase_omni`, so they work. BactiMate saves user models as e.g. `mf_test.pth` → stem has no "omni" → the library loaded them as standard Cellpose (omni=False), producing wrong or random detections. **BactiMate now fixes this automatically**: when you run segmentation or preview with a user model, it passes a path whose name contains "omni" (via a temporary link/copy) so the loader uses Omnipose dynamics. You do not need to rename files.
- **nchan** (number of input channels): BactiMate trains with **nchan=2**. If the loaded model had nchan=1, input shape would be wrong.
- **nclasses** (output classes): Training default is **nclasses=3**. Built-in bacteria models use nclasses=4. Wrong nclasses → wrong outputs.
- **Diameter**: User models have no size model; **diameter=0** gives no cells. BactiMate uses diameter=30 when you leave it at 0, and **flow_threshold=0** for custom models.
- **Flow threshold**: For custom models use **0**; 0.4 often removes all detections.

**Comparing user vs built-in**

When you run segmentation with a **user model**, the batch script logs:

- Checkpoint keys and (if present) **nchan** / **nclasses** from the .pth file.
- **Loaded model architecture: nchan=…, nclasses=…** after the model is loaded.

If you see `nchan=1` or `nclasses=2` for a model you trained in BactiMate (which uses 2 and 3), that mismatch can explain wrong or no detections.

**Transfer learning (start from a pretrained model)**

In the **Train model** dialog, **Start from** offers: **From scratch** (no pretrained model); or a built-in model by name (**bact_phase_omni**, **bact_fluor_omni**, **bact_phase_affinity**) to initialize from that checkpoint and train on your data. Choosing a pretrained option adds your data on top of it; **nchan** and **nclasses** are locked to match (Omnipose requirement). Use **Check compatibility with selected model** to validate image/mask pairs and that image dimensions match (e.g. 2D for bact_phase_omni). See [Omnipose Training](https://omnipose.readthedocs.io).

**Inspect a checkpoint manually**

From the project root:

```bash
# Show where BactiMate stores user models and list .pth files (use if you don't know the path)
python -m bactimate.scripts.inspect_omnipose_checkpoint --list-user-models

# Inspect a specific model (use the path shown above or your actual path)
python -m bactimate.scripts.inspect_omnipose_checkpoint path/to/your_model.pth
```

User models folder: when running from the **repo** it is `<repo_root>/user_models/`; when **installed** it is `~/Library/Application Support/BactiMate/Models` (macOS) or `%APPDATA%/BactiMate/Models` (Windows).

This prints the checkpoint’s top-level keys, any stored nchan/nclasses, and the first conv layer shape (from which input channels can be inferred). Compare with BactiMate’s inference defaults: **nchan=2, nclasses=3**. Models trained with other tools may use different nchan/nclasses; those same values must be used at inference (future BactiMate versions may allow setting them in the UI).

**Why bact_phase_affinity works without "omni" in the name**

Built-in models are loaded by **model type** (e.g. `bact_phase_affinity`), not by file path. The library decides how to run them from the model type string and internal lists. **bact_phase_affinity** is an affinity-based model and uses a different code path than flow-based Omnipose models; it does not rely on the filename. The **"omni in filename"** rule only applies when the library loads a **custom checkpoint from a file path**: in that case it infers `omni` from the file’s stem. So built-in names (bact_phase_omni, bact_phase_affinity, etc.) are handled by model type; only user `.pth` paths are affected by the filename.

**User model (e.g. after transfer learning) detects no cells while bact_phase_omni does**

Omnipose/cellpose_omni often saves trained checkpoints with the network state under a **nested** key (e.g. `state_dict` or `model_state_dict`), not at the top level. BactiMate’s patch must infer **nchan** and **nclasses** from the checkpoint to build the model with the same architecture. If the patch only looked at top-level keys, it never saw the conv weights and assumed nchan=1, so the model was built for 1 channel while the weights expected 2 → wrong input shape → no or random detections. **Fix:** Re-run `python setup/apply_all_patches.py` so the patch is updated to read nchan/nclasses from top-level keys when present and to infer nchan from the first conv weight **inside** `state_dict` / `model_state_dict` when needed. Then restart BactiMate and run segmentation again with your user model (e.g. mf_bact_phase_omni).

**Debug log (user vs built-in model)**

When loading models and running segmentation, BactiMate writes lines to **Logs/omnipose_debug.log** (project `Logs` folder, or under Application Support on installed builds). Each line is timestamped. You’ll see:
- **USER_MODEL pre-load**: checkpoint inspection (nchan, nclasses, first_layer_shape, keys) before the patched loader runs.
- **cellpose_omni load**: what the patched loader inferred (nchan, nclasses, ckpt_keys, state_dict sample).
- **USER_MODEL / BUILTIN post-load**: `model.cp.nchan` and `model.cp.nclasses` after loading.
- **eval**: kind (USER_MODEL vs BUILTIN), model id, diameter, flow/mask threshold, frame shape, and result **n_cells**, mask_min, mask_max.

Compare a run with a built-in model (e.g. bact_phase_omni) and one with your user model: same image and similar diameter/flow/mask. If the built-in run shows `n_cells > 0` and the user run shows `n_cells=0` and/or different nchan/nclasses in post-load, the issue is likely architecture (nchan/nclasses) or parameters; use the log to confirm what was inferred and passed to eval.

**Why renaming to mf_lasse_omni.pth might not fix it**

1. **BactiMate patch**: If you ran `setup/apply_all_patches.py`, BactiMate patches cellpose_omni so that when a **path** to a user model is passed, it sets `omni=True` and infers **nchan** and **nclasses** from the checkpoint (top-level or inside state_dict). In that case the **filename is ignored** for the omni flag. So renaming alone doesn’t change behavior when the patch is applied; the patch already forces the right mode. If user models still fail, ensure the patch was applied (run the patch script from the repo) and that you fully restarted BactiMate after patching.
2. **Dropdown selection**: If you rename (or copy) to `mf_lasse_omni.pth`, the model list shows **stems** of `.pth` files. You must choose **"mf_lasse_omni"** in the dropdown so BactiMate actually uses that file. If you still have `mf_lasse` selected, it will use `mf_lasse.pth`. Refresh the model list (e.g. switch tab and back, or restart) if the new file doesn’t appear.
3. **Without the patch**: If the patch is not applied, the vanilla cellpose_omni `Cellpose` class may not accept a user path at all, so passing a path can fail or be ignored. The "omni in filename" workaround (and BactiMate’s symlink/copy with an `_omni` name) helps when the library does accept a path and only then infers `omni` from the filename.

**One model works, another does not**

If **mf_test** gives random detections but **mf_lasse** gives none, possible causes:

- Different training settings (nchan/nclasses) between the two; check with the inspector.
- Different diameter or flow threshold; try diameter ≈ your cell size in pixels and flow_threshold=0 for both.
- Different image type or normalization; keep training and inference image format and bit depth consistent.

---

## 5. Analyze Masks tab

**Purpose:** Analyze mask TIFF files (single file or folder) with morphological and size filters; export to Excel; optionally analyze fluorescence intensity (FI) from a separate channel TIFF; create filter visualizations and combined summary.

### 5.1 Mask File/Folder input

- **Group:** “Mask File/Folder” (drag-and-drop). Placeholder: “Select a mask TIFF file or folder... (or drag and drop)”. Browse File / Browse Folder set the path. Accepted: single TIFF or folder of mask TIFFs.

### 5.2 Combine group

- **Create combined summary:** Check box. When analyzing a folder, create one extra Excel that combines all Summary sheets. Output: `{folder_name}_combined_summary.xlsx` with sheet “Combined Summary (wide)” (one row per frame, one set of columns per file). Info (?) explains use for downstream plotting and comparing samples.

### 5.3 Analysis Parameters

**Morphological Filters**

- **Pixel Size (µm):** Double spin [0.001, 10.0]; default 0.108. Used for area calculations (pixel area → µm²). Info (?) — Common values 100× ~0.108, 60× ~0.18, 40× ~0.27; can be auto-filled from metadata. Indicator label “⚠ Check value!” when default is used (user should verify).
- **Min Aspect Ratio:** [1.0, 20.0]; default 1.5. Info (?) — length/width; filters out circular objects. Rod-shaped bacteria often 2.0–5.0.
- **Max Aspect Ratio:** [1.0, 50.0]; default 10.0. Info (?) — Filters too-elongated artifacts.
- **Min Eccentricity:** [0.0, 1.0]; default 0.5. Info (?) — 0=circle, 1=line; filters circular objects.
- **Min Solidity:** [0.0, 1.0]; default 0.7. Info (?) — area/convex_hull_area; filters irregular shapes and debris.

**Size Filters**

- **Min Area (µm², 0=no limit):** [0, 10000]; default 0. Info (?) — Filters cells smaller than value; 0 disables.
- **Max Area (µm², 0=no limit):** [0, 10000]; default 0. Info (?) — Filters oversized objects; 0 disables.
- **Min Width (µm, 0=no limit):** [0, 100]; default 0. Info (?) — Minor axis; filters too-narrow objects.
- **Max Width (µm, 0=no limit):** [0, 100]; default 0. Info (?) — Filters too-wide objects.
- **Min Length (µm, 0=no limit):** [0, 100]; default 0. Info (?) — Major axis; filters too-short objects.

**Reset to Default** — Restores all analysis parameters to defaults.

### 5.4 Options

- **Include detailed per-cell data:** Check box; default **checked**. Info (?) — When enabled, Excel includes a sheet with per-cell measurements (ID, area, width, length, aspect ratio, eccentricity, solidity, etc.); when disabled, only summary (total area per frame).
- **Create filter visualization:** Check box; default **checked**. Info (?) — Creates TIFF with green=accepted, red=rejected. Saved as `*_filter_visualization.tif`. Useful for QC.
- **Create combined summary:** Described in Combine group above.
- **Exclude masks at borders:** Check box; default **unchecked**. Info (?) — When enabled, any mask that touches the image border is excluded from counts and area (useful to avoid partial cells at the image edge). In **Preview Filter Effects**, border-touching masks are shown in red when this option is on; use **Re-apply** to refresh the preview after toggling.

### 5.5 Combine existing summaries from folder

- **Group:** “Combine existing summaries from folder”. Optional workflow: select a folder that contains `*_analysis.xlsx` (or `*_masks_analysis.xlsx`) files, possibly in subfolders. A **Combine** or similar action creates a single combined summary Excel (e.g. one row per frame, columns per file). Placeholder: “Folder with *_analysis.xlsx (subfolders)...”. This is separate from the “Create combined summary” checkbox used when analyzing a folder in the main Analyze path.

### 5.6 Analyze FI (fluorescence intensity)

- **Check box “Analyze FI”:** Default **unchecked**. When enabled, the FI file path and Contour FI options become active.
- **Info (?) —** “Analyze fluorescence intensity (FI) from a separate FI channel image. When enabled: Requires selecting an FI channel TIFF file. The analysis will always add: Mean FI intensity (average pixel intensity within each cell). Optional (if 'Contour FI' is enabled): Contour FI intensity (average intensity at cell boundary), FI ratio (contour FI / mean FI). Use when: You have a separate fluorescence channel image and want to quantify fluorescence intensity per cell.”
- **FI file path:** Line edit; placeholder “Select FI channel TIFF file... (or drag and drop)”. Browse FI File. Disabled until “Analyze FI” is checked.
- **Contour FI:** Check box; default unchecked; disabled until “Analyze FI” is checked. When checked, analysis adds contour-based FI (boundary intensity and FI ratio).
- **Info (?) (Contour FI) —** “Compute contour-based FI metrics (boundary intensity + FI ratio). When disabled: Only Mean FI Intensity will be calculated. Contour thickness is ignored.”
- **Contour thickness (pixels):** Spin box [1, 10]; default 1. Info (?) — “Thickness of the contour region used for FI intensity calculation. Default: 1 pixel (single-pixel boundary). The contour is the region at the cell boundary. Increasing thickness includes more pixels near the boundary. Note: This affects both the preview visualization and the final analysis.” Disabled until Contour FI is relevant (e.g. when Analyze FI is on).

### 5.7 Buttons and actions

- **Preview Filter Effects:** Tooltip: “Preview how current filter parameters would affect the selected mask file”. Opens a preview (e.g. first frame or first file) with accepted/rejected cells shown. Disabled until a mask file or folder is selected.
- **Analyze Masks:** Runs full analysis with current morphological and size filters; writes Excel, optional filter visualization TIFF, optional combined summary. Disabled until path is set. Progress in main window log.
- **Export last result to CSV:** Tooltip: “Export the most recent analysis Excel (Summary or Detailed sheet) to a CSV file.” Exports the last generated analysis Excel to CSV (user chooses path). Useful for downstream scripts or plotting.

### 5.8 Session parameters (Analyze Masks)

Stored for session log: pixel_size, min/max aspect ratio, min eccentricity, min solidity, min/max area, min/max width, min length, detailed, create_vis, create_combined_summary, exclude_border, analyze_fi, fi_file_path, analyze_fi_contour, fi_contour_thickness, file_path, combine_folder_path, etc.

---

## 6. Tracking tab

**Purpose:** Run Trackastra cell tracking on a mask TIFF time series; optionally link original TIFF and Excel analysis file for visualization and enriched output.

### 6.1 Inputs

**Mask File** (required)

- **Placeholder:** “Select mask TIFF file (time series)... (or drag and drop)”. Browse File; drag-and-drop accepts TIFF only. Run Tracking is enabled only when mask path is non-empty.

**Original Data File (optional)**

- **Placeholder:** “Select original TIFF file for visualization (optional)... (or drag and drop)”. For overlay/visualization.

**Excel Analysis File (optional)**

- **Placeholder:** “Select Excel file with cell analysis data (e.g., *_masks_analysis.xlsx)... (or drag and drop)”. Info (?) — Link to Excel from Analyze Masks; tracking links cell IDs across frames and can preserve cell properties from the analysis file.

### 6.2 Tracking Parameters

**Model**

- **Combo:** “general_2d”, “general_2d_w_SAM2_features”. Default: general_2d.
- **Info (?) —** general_2d: fast, good for most cell types. general_2d_w_SAM2_features: uses SAM2 features, better for some data (e.g. bacteria), slower, benefits from GPU.

**Mode**

- **Combo:** “greedy”, “greedy_nodiv”, “ilp”. Default: greedy.
- **Info (?) —** greedy: division support, fast. greedy_nodiv: no divisions, faster. ilp: Integer Linear Programming, optimal but slower, may need extra deps (e.g. Gurobi/SCIP).

**Device**

- **Combo:** “CPU” and, if available, “CUDA” and/or “GPU” (Apple MPS). Default: CPU.
- **Info (?) —** CPU works everywhere; CUDA/GPU faster when available.

### 6.3 Run Tracking

- **Button:** “Run Tracking”. Runs Trackastra with selected mask (and optional original + Excel). Outputs and progress depend on the worker implementation (e.g. tracked masks, lineage, Excel with linked IDs).

---

## 7. Video Graph tab

**Purpose:** Create a growth-curve video: combine a TIFF sequence (video input) with an Excel analysis file from the Analyze Masks tab, and render a side-by-side video with graph (relative cell area over time) and image sequence.

### 7.1 Video Input (TIFF Sequence)

- **Group:** Drag-and-drop. Placeholder: “Select TIFF file or folder containing TIFF sequence... (or drag and drop)”. Browse File / Browse Folder.

### 7.2 Excel Analysis File

- **Placeholder:** “Select Excel file from Analyze Masks tab (e.g., *_masks_analysis.xlsx)... (or drag and drop)”. Browse.
- **Info (?) —** “Excel file from the Analyze Masks tab containing growth data. Required columns: Frame, Total Cell Area (Relative) (normalized, first frame = 1.0).”

### 7.3 Time Settings

- **Time per frame (min.):** Double spin; min 0, max 10000; default 10.0; suffix “ min”. Info (?) — Time between consecutive frames; used to convert frame index to time (hours) on the x-axis of the growth curve.

### 7.4 Graph Settings

- **Y-axis min:** Double spin; minimum 0, maximum 100; default 0; special value “Auto”. When 0, axis is auto-scaled from data.
- **Y-axis max:** Double spin; same range; default 0; special value “Auto”.
- **Info (?) —** “Configure the Y-axis range for the growth curve graph. Y-axis represents: Relative cell area (normalized to first frame = 1.0). Y-axis min/max: Set to 'Auto' (0.0) to automatically scale based on data; set custom values to fix the Y-axis range; useful for comparing multiple videos with consistent scaling. Example: Set min=0.0, max=3.0 to show growth from 1x to 3x initial area.”

### 7.5 Video Settings

- **Frames per second (FPS):** Double spin; range [0.1, 60.0]; default 5.0; suffix “ fps”. Info (?) — “Playback speed for the output video. Higher values = faster playback; lower = slower. Default: 5.0 fps. Typical range: 1–10 fps for time-lapse videos.”

### 7.6 Output Formats

- **Group:** “Output Formats”. Checkboxes: **Select All**, **TIFF**, **AVI**, **GIF**, **MP4**. Default: TIFF and AVI checked. At least one format must be selected for the button to be enabled.
- **Select All:** Toggles all four format checkboxes at once. When any format is unchecked, “Select All” is unchecked; when all four are checked, “Select All” is checked.

### 7.7 Create Growth Curve Video button

- **Button:** “Create Growth Curve Video”. Enabled when: (1) Video input path is set, (2) Excel path is set, (3) At least one output format (TIFF, AVI, GIF, MP4) is checked.
- **Behavior:** Reads the Excel Summary (Frame, Total Cell Area (Relative)), loads the TIFF sequence from the video path, builds the growth curve (time in hours from “Time per frame (min.)”), and renders a side-by-side layout (graph + video). Writes output files in the chosen formats (e.g. same base name with .tif, .avi, .gif, .mp4) to an output directory (often next to the Excel or the video input, with logic to go up one level if the path is inside a Tif_* or _segmentation folder).

---

## 8. Session Logs tab

**Purpose:** List recent session log files, show an overview of saved parameters, and apply those parameters to the current session (without changing file paths).

### 8.1 Recent session logs (last 10)

- **Group:** “Recent session logs (last 10)”. List widget shows up to 10 most recent `.json` session files from the logs directory (e.g. `Logs/`), sorted by modification time (newest first). Each item shows timestamp and filename.
- **Load other...:** Opens file dialog (JSON). Tooltip: “Open a session log file that is not in the list above”.

### 8.2 Overview of selected session

- **Group:** “Overview of selected session”. Read-only text area; placeholder: “Select a log file above or click 'Load other...' to see parameters.” When a log is selected, it shows schema version, timestamp, and per-tab parameter summary (keys/values).

### 8.3 Apply now to current session

- **Button:** “Apply now to current session”. Enabled when a session file is loaded. Tooltip: “Apply the parameters from the selected log to all tabs (parameters only; file paths are not changed).”
- **Behavior:** Reads the selected JSON and applies stored parameter key–value pairs to the corresponding tab widgets (via `apply_to_widgets`). Paths (file/folder fields) are typically not overwritten so the user keeps their current input paths.

---

## 9. File types, naming, and workflows

### 9.1 Supported file types (by tab)

- **TIFF Conversion:** Input: ND2, LIF, CZI, TIFF. Output: TIFF (multi-page or per channel/frame as per script).
- **Omnipose Segment:** Input: TIFF (single or folder). Output: masks TIFF (`*_masks.tif`), overlay TIFF (`*_overlay.tif`), and batch analysis outputs (e.g. Excel) when processing a folder.
- **Analyze Masks:** Input: mask TIFF(s) (single or folder). Optional: FI channel TIFF. Output: Excel (`*_analysis.xlsx`, `*_masks_analysis.xlsx`), optional filter visualization TIFF, optional combined summary Excel.
- **Tracking:** Input: mask TIFF (time series); optional original TIFF, optional Excel from Analyze Masks. Output: as implemented by Trackastra worker (e.g. tracked masks, Excel with lineage).
- **Video Graph:** Input: TIFF file or folder (sequence), Excel from Analyze Masks. Output: growth curve video (side-by-side graph + video).

### 9.2 Naming conventions

- **Omnipose training:** Image and mask pairs: `NAME<Image filter>.tif` and `NAME<Mask filter>.tif` (e.g. `634_phase.tif` + `634_masks.tif` with Image filter `_phase`, Mask filter `_masks`). Base name is derived by stripping the Image filter from the image filename; the mask must then be `base + Mask filter + extension`. If Image filter is empty and filenames follow e.g. `634_phase.tif` / `634_masks.tif`, the training dialog can auto-infer `_phase` and fill Image filter.
- **omnipose_masks folder:** After “Convert masks for Omnipose”, converted masks (and copied images when naming matches) live in `training_folder/omnipose_masks/`. The training directory field stays on the parent folder; at “Start training” the app uses `omnipose_masks` and syncs missing images from the parent.

### 9.3 Cross-tab workflows

1. **ND2 → Segment → Analyze:** Convert ND2 in TIFF Conversion → open Omnipose Segment, set path to converted TIFF or folder → Process → open Analyze Masks, set path to mask file or folder → set pixel size and filters → Analyze Masks.
2. **Segment → Analyze → Tracking:** After segmentation, analyze masks (Analyze Masks) → in Tracking, set mask TIFF, optional original TIFF, optional Excel from Analyze Masks → Run Tracking.
3. **Analyze → Video Graph:** After Analyze Masks, use the Excel output and the same TIFF sequence in Video Graph; set time per frame → Create Growth Curve Video.
4. **Training:** In Omnipose Segment, Train model → set training directory (folder with image/mask pairs), set Image/Mask filters, optionally “Convert masks for Omnipose” → set model name and hyperparameters → Start training. New model appears in Model dropdown.

---

## 10. Common scenarios and questions

**Q: I have 634_phase.tif and 634_masks.tif; training says “No matching mask files found”.**  
A: Set **Image filter** to `_phase` so the base name is “634” and the app looks for `634_masks.tif`. You can leave Image filter empty and click Start training; the dialog will try to infer `_phase` from the filenames and fill it in.

**Q: After “Convert masks for Omnipose”, the path switched to omnipose_masks and training fails (no images).**  
A: The app no longer switches the path. Keep the training directory on the **parent** folder (the one that contains both images and masks). When you click Start training, the app uses the `omnipose_masks` subfolder and copies missing images from the parent. If it still fails, the error message explains that images must match mask names (e.g. NAME.tif for NAME_masks.tif) and suggests checking the parent folder and Image filter.

**Q: Which flow threshold for my custom model?**  
A: When you select a **user model** in the dropdown, flow threshold is set to **0** automatically (recommended for custom/well-trained models). For built-in models it is set to **0.4**.

**Q: Where is the number of image/mask pairs for training shown?**  
A: In the Train dialog, after validation and before “Starting Omnipose training…”, the log shows: “Found N image/mask pair(s). All will be used every epoch for M epochs.” N and M are computed from the training folder (and the omnipose_masks subfolder if present).

**Q: Pixel size in Omnipose Segment vs Analyze Masks?**  
A: In Omnipose Segment, “Pixel size (µm)” is for **scale bar and calibration** only; diameter is always in **pixels**. In Analyze Masks, “Pixel Size (µm)” is used to convert pixel areas to **µm²** for filtering and Excel output.

**Q: Session Logs — what gets applied?**  
A: Parameter values (model, diameter, thresholds, filters, options, etc.) are applied to each tab’s widgets. **File/folder paths are not changed** so you keep your current inputs.

**Q: I only have masks (no images) in omnipose_masks; I didn’t click “Convert masks for Omnipose”. Will training still work?**  
A: Yes. If there is **no** `omnipose_masks` subfolder, the training directory is used as-is (your folder with both images and masks). Training runs on that single folder. The `omnipose_masks` logic only applies when that subfolder exists (e.g. after you ran “Convert masks for Omnipose” at least once).

**Q: Analyze Masks — what Excel columns are required for Video Graph?**  
A: Video Graph expects an Excel from Analyze Masks with at least **Frame** (frame number) and **Total Cell Area (Relative)** (normalized so first frame = 1.0). These are produced by the Analyze Masks tab’s Summary sheet.

**Q: Can I use a single multipage TIFF for Tracking?**  
A: Yes. The Tracking tab expects a mask TIFF that is a time series (multiple frames). That can be a single multipage TIFF where each page is one time point. The “Original Data File” can be the corresponding multipage TIFF of raw images for visualization.

**Q: ND2 Conversion — when do channel checkboxes for bleach correction appear?**  
A: After you load a **multi-channel** microscopy file, the tab builds a set of checkboxes (one per channel) so you can select which channels are fluorescence and should receive photobleaching correction. Phase/transmission channels are typically left unchecked.

---

## 11. References and further documentation

- **Omnipose:** https://omnipose.readthedocs.io/ (models, training, settings).
- **Omnipose models (GitHub):** https://github.com/kevinjohncutler/omnipose-models
- **Trackastra:** https://github.com/weigertlab/trackastra (tracking models and modes).
- **Project README:** `README.md` in the project root (installation, quick start, command-line, troubleshooting).
- **Training mask format:** See `docs/TRAINING_MASKS.md` if present (instance vs binary masks).
- **Custom model troubleshooting:** See `docs/CUSTOM_MODEL_TROUBLESHOOTING.md` if present (flow threshold, diameter, mask threshold for user models).

---

## Appendix A: Verbatim info texts (quick reference)

This appendix reproduces the **(?)** info icon texts as they appear in the application, for quick lookup. Section references point to where the setting is described in the main body.

**Omnipose Segment — Model (Section 4.2.1)**  
“Omnipose model type for cell segmentation. Default: 'bact_phase_omni' Models are downloaded from: https://github.com/kevinjohncutler/omnipose-models Documentation: https://omnipose.readthedocs.io/models.html Available Omnipose models: bact_phase_omni: Bacterial phase contrast (recommended for bacteria), bact_fluor_omni: Bacterial fluorescence, bact_phase_affinity: Bacterial phase contrast (affinity-based). Tip: Choose based on your cell type and imaging modality.”

**Omnipose Segment — Diameter (Section 4.2.1)**  
“Expected cell diameter in pixels (only diameter field; Pixel size (µm) below is for scale bar). Default: 0 (auto-detect). Set to 0 for automatic detection (recommended for first try). Specify a value (e.g., 20-50 pixels for bacteria) for more consistent results. If you know size in µm: diameter_px = diameter_µm ÷ Pixel size (µm). Typical values: Bacteria: 20-50 pixels, Mammalian cells: 30-100 pixels, Nuclei: 15-30 pixels. Tip: Auto-detection works well, but specifying diameter often gives better results. Custom/user model: try Flow threshold 0, Mask threshold -2, Diameter 0 or cell size (px). See docs/CUSTOM_MODEL_TROUBLESHOOTING.md.”

**Omnipose Segment — Pixel size (µm) (Section 4.2.1)**  
“Pixel size in µm (for scale bar and calibration). Diameter above is always in pixels. To get diameter in px from µm: diameter_px = diameter_µm ÷ pixel_size_µm. Common: 100× ~0.108, 60× ~0.18, 40× ~0.27. Can be filled from file metadata when available.”

**Omnipose Segment — Flow Threshold (Section 4.2.1)**  
“Flow error threshold for cell detection. Default: 0.4. Source: Omnipose default value. Controls sensitivity: Lower values (0.2-0.3): More sensitive; Higher values (0.5-0.7): More strict. Range: 0.0 to 1.0. Tip: If missing cells, try 0.3; if too many false positives, raise to 0.5.”

**Omnipose Segment — Mask threshold (Section 4.2.1)**  
“Mask probability threshold for cell detection. Default: 0.0. Lower values: more permissive; higher: stricter. Range: 0.0 to 1.0. Tip: Usually 0.0 is fine; increase if masks look too loose.”

**Omnipose Segment — Use GPU (Section 4.2.2)**  
“Use GPU acceleration for faster processing. Default: Enabled. When enabled: Uses GPU if available (10-100x faster). When disabled: Forces CPU. Note: Automatically falls back to CPU if no GPU. Tip: Disable if you encounter out-of-memory errors.”

**Omnipose Segment — Use DBSCAN clustering (Section 4.2.2)**  
“Use DBSCAN clustering to help segment thin/elongated cells. Default: Disabled. When enabled: Improves segmentation of thin/elongated cells (e.g. bacteria, worms), cells with complex shapes, overlapping cells. Use when: Standard segmentation struggles with elongated structures. Note: Slightly increases processing time.”

**Omnipose Segment — Create overlay visualization (Section 4.2.2)**  
“Create overlay visualization showing masks overlaid on original images. Default: Enabled. When enabled: Creates a colored overlay TIFF. When disabled: Skips overlay (faster). Saved as: filename_overlay.tif. Tip: Disable to speed up when you only need masks.”

**Train dialog — Training directory (Section 4.4)**  
“Folder containing your training data: one image file and one mask file per pair. Omnipose expects: for each image 'NAME.tif', a mask file 'NAME<Mask filter>.tif' (e.g. NAME_masks.tif). Mask format: instance label matrices (background=0, each cell=1,2,3,…). Not binary. See docs/TRAINING_MASKS.md.”

**Train dialog — File name filters (Section 4.4)**  
“Suffix that identifies image vs mask files. Image filter: leave empty if filenames are e.g. A.tif; use _img if you have A_img.tif. Mask filter: suffix on mask files. With _masks, mask for 'cell1.tif' must be 'cell1_masks.tif'. Must match your filenames exactly.”

**Train dialog — Convert masks for Omnipose (Section 4.4)**  
“Convert binary masks (e.g. 0 and 255 from Ilastik) to instance label matrices. Converted masks (and image copies when naming matches) are written to a subfolder 'omnipose_masks'; your originals are never overwritten. Keep the training directory as the folder that contains your images and masks. When you click Start training, we use the omnipose_masks subfolder automatically and copy images there if needed.”

**Train dialog — Model name (Section 4.4)**  
“A short name for your model (letters, numbers, underscores). It will appear in the model dropdown and is used to save the .pth file. Example: my_bacteria_v1. Do not use path separators. Models are saved in BactiMate's user models folder. User models folder: [path shown].”

**Train dialog — Epochs (Section 4.4)**  
“Number of full passes over the training set. Omnipose often benefits from more epochs (e.g. 4000 for bacteria). Fewer epochs (500–1000) can be used for quick tests.”

**Train dialog — Learning rate (Section 4.4)**  
“Step size for the optimizer (default 0.1). Larger values train faster but can be unstable; smaller values are slower but more stable.”

**Train dialog — Batch size (Section 4.4)**  
“Number of image crops per training step. Larger batches (e.g. 16) train faster if you have enough GPU memory; smaller batches (e.g. 4–8) use less memory.”

**Train dialog — Channels (nchan) (Section 4.4)**  
“Number of image channels the model expects. 1 = grayscale only. 2 = two-channel (like bact_phase_omni; grayscale duplicated at training). Both work in Preview/Batch. Always set nchan and nclasses to match when evaluating.”

**Train dialog — Classes (nclasses) (Section 4.4)**  
“Number of prediction classes (2 = flow + distance; 3 adds boundary). Use 3 for typical Omnipose bacteria models (recommended).”

**Analyze Masks — Pixel Size (µm) (Section 5.3)**  
“Pixel size in micrometers (µm) for area calculations. Default: 0.108 µm. This value is used to convert pixel areas to µm². Common: 100x ~0.108, 60x ~0.18, 40x ~0.27. Tip: Check your microscope settings or ND2 metadata.”

**Analyze Masks — Min/Max Aspect Ratio, Min Eccentricity, Min Solidity (Section 5.3)**  
See Section 5.3 for full text; each describes the morphological filter (length/width, eccentricity 0–1, solidity area/convex_hull) and typical values for rod-shaped cells.

**Analyze Masks — Size filters (Section 5.3)**  
Min/Max Area (µm²), Min/Max Width (µm), Min Length (µm): 0 means no limit; otherwise objects outside the range are excluded. Info texts describe filtering of undersized/oversized and too-narrow/too-wide/too-short objects.

**Tracking — Model (Section 6.2)**  
“Pretrained Trackastra model. general_2d: General-purpose 2D, fast. general_2d_w_SAM2_features: Enhanced with SAM2, better for some data (e.g. bacteria), slower. Note: Trackastra won the 7th Cell Tracking Challenge (ISBI 2024).”

**Tracking — Mode (Section 6.2)**  
“Linking mode. greedy: Greedy with division support (recommended). greedy_nodiv: No divisions. ilp: Integer Linear Programming, optimal but slower, may need Gurobi/SCIP.”

**Video Graph — Excel (Section 7.2)**  
“Excel file from the Analyze Masks tab containing growth data. Required columns: Frame, Total Cell Area (Relative) (first frame = 1.0).”

**Video Graph — Time per frame (Section 7.3)**  
“Time difference between consecutive frames. Used to convert frame numbers to time (hours) for the x-axis. Example: If images were taken every 10 minutes, enter 10.0.”

**Video Graph — Y-axis (Section 7.4)**  
“Y-axis = relative cell area (first frame = 1.0). Set to Auto (0.0) to auto-scale; set custom values to fix range. Example: min=0, max=3 to show 1x to 3x initial area.”

**Video Graph — FPS (Section 7.5)**  
“Playback speed for the output video. Higher = faster; lower = slower. Default: 5.0 fps. Typical range: 1–10 fps for time-lapse.”

**ND2 — Number of Frames (Section 3.2)**  
“Number of time frames to extract. Value will automatically update to the maximum when a file is loaded. Note: More frames takes longer and uses more disk space.”

**ND2 — Max Z projection (Section 3.2)**  
“Apply maximum intensity projection across Z-slices when Z > 1. When enabled: single 2D image from 3D Z-stack. When disabled: first Z-slice only.”

**ND2 — Drift correction (Section 3.2)**  
“Apply X,Y drift correction. Method: Phase cross-correlation, frame-by-frame alignment to previous frame. First frame unchanged; subsequent frames incrementally aligned.”

**ND2 — Photobleaching correction (Section 3.2)**  
“Tick only fluorescence channels. Methods: Exponential (intensity preservation), Linear decay, Histogram matching, Per-frame min-max (visualization only).”

**Session Logs — Apply (Section 8.3)**  
“Apply the parameters from the selected log to all tabs (parameters only; file paths are not changed).”

---

*End of BactiMate User Manual.*
