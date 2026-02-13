# BactiMate

Bacterial cell segmentation and analysis tool using Omnipose with a user-friendly GUI and batch processing capabilities.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Graphical User Interface](#graphical-user-interface)
- [Command Line Usage](#command-line-usage)
- [Batch Processing](#batch-processing)
- [Cell Area Analysis](#cell-area-analysis)
- [Building App Bundles](#building-app-bundles)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Available Scripts](#available-scripts)

## Quick Start

### Launch the GUI

```bash
./launch_gui.sh
```

This opens the BactiMate GUI with three main tabs:
- **ND2 to TIFF Conversion**: Convert ND2 microscopy files to TIFF
- **Single File Processing**: Process individual TIFF files
- **Batch Folder Processing**: Process all TIFF files in a folder with analysis
- **Analyze Existing Masks**: Re-analyze masks with different parameters

### Process a Single File (Command Line)

```bash
source venv/bin/activate
python -m scripts.batch_process_tiff_sequence your_sequence.tif --diameter 30
```

### Process All Files in a Folder

```bash
python -m scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni \
    --diameter 30 \
    --pixel-size 0.108 \
    --min-aspect-ratio 2.0 \
    --max-area 50 \
    --max-width 2.0
```

## Installation

### Prerequisites

- Python 3.11+
- macOS 10.13+ or Windows 10/11

### Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **For NVIDIA GPU (CUDA) support (optional):**
   - **macOS/Linux:** Run `./setup/install_cuda_pytorch.sh` after installing requirements
   - Or manually: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
   - Replace `cu118` with your CUDA version (check with `nvidia-smi` or `nvcc --version`)

3. **Optional extras (pyproject.toml):**
   - Install with SAM2 extra (e.g. newer torch): `pip install -e .[sam2]`
   - Key dependencies in `requirements.txt` are pinned for reproducible builds (e.g. PyQt6, numpy).

4. **For fresh setup (optional):**
   ```bash
   ./setup/setup_fresh_env.sh
   ```

### Installation Details

- **Python Version**: 3.11+
- **Omnipose Version**: 1.0.6
- **GUI Framework**: PyQt6
- **Virtual Environment**: `venv/`

## Graphical User Interface

### Launching the GUI

```bash
./launch_gui.sh
```

Or manually:
```bash
source venv/bin/activate
python main.py
```

### GUI Tabs

#### Tab 1: ND2 to TIFF Conversion

Convert ND2 microscopy files to TIFF format with metadata preservation.

**Steps:**
1. Click "Browse File..." or "Browse Folder..." to select ND2 file(s)
2. Click "Preview ND2 File" to view first frame (optional)
3. Adjust conversion parameters:
   - Number of frames to extract
   - Channel handling (separate or combined)
   - Image corrections (optional)
4. Click "Convert ND2 to TIFF"

**Features:**
- Automatic pixel size extraction from ND2 metadata
- Metadata preservation
- Flexible frame selection
- Preview option
- Multi-channel support

**Note:** Requires `nd2` library: `pip install nd2`

#### Tab 2: Single File Processing

Process a single multipage TIFF file with Omnipose segmentation.

**Steps:**
1. Click "Browse..." to select a TIFF file
2. Choose model (default: `bact_phase_omni`)
3. Set diameter (0 = auto-detect)
4. Adjust flow threshold if needed
5. Check/uncheck GPU and clustering options
6. Click "Process File"

**Output:**
- Masks saved to `filename_omnipose/filename_masks.tif`
- Overlay visualization saved to `filename_omnipose/filename_overlay.tif`

#### Tab 3: Batch Folder Processing

Process all TIFF files in a folder with segmentation and analysis.

**Steps:**
1. Click "Browse..." to select folder containing TIFF files
2. Configure segmentation parameters (model, diameter, flow threshold)
3. Configure analysis parameters (pixel size, morphological filters, size filters)
4. Choose options:
   - **Skip segmentation**: Check if masks already exist (re-analyze only)
   - Include detailed per-cell data
   - Create filter visualizations
5. Click "Process All Files in Folder"

**Output:**
- For each file: masks, visualizations, individual Excel files
- Combined Excel file: `folder_batch_analysis.xlsx` with all files in one table

#### Tab 4: Analyze Existing Masks

Analyze existing mask files with morphological filtering.

**Steps:**
1. Click "Browse..." to select a mask TIFF file
2. Set pixel size
3. Configure morphological filters
4. Set size constraints
5. Choose options (detailed data, visualization)
6. Click "Analyze Masks"

**Output:**
- Excel file with area analysis
- Filter visualization (if enabled): green=accepted, red=rejected

### Parameter Guide

#### Segmentation Parameters

- **Model**: Choose the Omnipose model type
  - `bact_phase_omni`: Best for bacterial phase contrast images (default)
  - `cyto2_omni`: General purpose
  - Other models available in dropdown

- **Diameter**: Cell diameter in pixels
  - Set to 0 for automatic detection
  - Typical values: 20-50 pixels for bacteria

- **Flow Threshold**: Flow error threshold (0.0-1.0)
  - Lower = more sensitive, may include false positives
  - Default: 0.4

- **Use GPU**: Enable GPU acceleration (faster if available)

- **Use DBSCAN clustering**: Helps with thin/elongated cells

#### Analysis Parameters

- **Pixel Size**: Physical pixel size in micrometers (default: 0.108 µm)
- **Min Aspect Ratio**: Minimum length/width ratio for rod-shaped cells (default: 1.5)
- **Max Aspect Ratio**: Maximum length/width ratio (default: 10.0)
- **Min Eccentricity**: Minimum eccentricity (0=circle, 1=line, default: 0.5)
- **Min Solidity**: Minimum solidity (default: 0.7)
- **Max Area**: Maximum cell area in µm² (0 = no limit)
- **Max Width**: Maximum cell width in µm (0 = no limit)

## Command Line Usage

### Single File Processing

```bash
# Basic usage
python -m scripts.batch_process_tiff_sequence input.tif --diameter 30

# With specific model
python -m scripts.batch_process_tiff_sequence input.tif --model bact_phase_omni --diameter 30

# With channels
python -m scripts.batch_process_tiff_sequence input.tif --channels 0 1 --diameter 25

# Adjust sensitivity
python -m scripts.batch_process_tiff_sequence input.tif --flow-threshold 0.3 --mask-threshold -2

# Use CPU instead of GPU
python -m scripts.batch_process_tiff_sequence input.tif --no-gpu

# Skip overlay creation (faster)
python -m scripts.batch_process_tiff_sequence input.tif --no-overlay
```

### Command Line Options

**Model Options:**
- `--model`: Model type
  - `cyto2_omni` (default) - General cytoplasm segmentation
  - `bact_phase_omni` - Bacterial phase contrast
  - `bact_omni` - Bacterial cells
  - `nuclei` - Nucleus segmentation

**Segmentation Parameters:**
- `--diameter`: Cell diameter in pixels (0 = auto-detect)
- `--channels`: `[main_channel nuclear_channel]` (0=GRAY, 1=RED, 2=GREEN, 3=BLUE)
- `--flow-threshold`: Flow error threshold (default: 0.4)
- `--mask-threshold`: Mask threshold (default: 0.0)

**Advanced Options:**
- `--cluster`: Use DBSCAN clustering (helps with thin/elongated cells)
- `--no-gpu`: Force CPU processing
- `--batch-size`: Batch size for processing (default: 8)
- `--verbose`: Show detailed progress

**Output Options:**
- `--output` / `-o`: Custom output directory
- `--save-frames`: Save individual frames to directory
- `--no-overlay`: Skip creating overlay visualization

## Batch Processing

### Process Multiple Files in a Folder

```bash
python -m scripts.batch_process_folder /path/to/folder \
    --model bact_phase_omni \
    --diameter 30 \
    --pixel-size 0.108 \
    --min-aspect-ratio 2.0 \
    --max-area 50 \
    --max-width 2.0
```

**Options:**
- `--recursive, -r`: Search subdirectories recursively
- `--skip-processing`: Only analyze, don't segment (if masks exist)
- `--detailed`: Include per-cell data sheets
- `--no-visualization`: Skip filter visualization TIFFs
- `--output, -o`: Custom output directory
- `--excel-name`: Custom Excel filename

**Output Structure:**
```
folder/
├── omnipose_batch_results/
│   ├── folder_batch_analysis.xlsx      # Combined summary
│   ├── file1_omnipose/
│   │   ├── file1_masks.tif
│   │   ├── file1_filter_visualization.tif
│   │   └── file1_analysis.xlsx
│   └── file2_omnipose/
│       └── ...
```

### Temporal Analysis

For multi-time-point TIFF sequences:

1. **Format Requirements**: TIFF sequences should be in **ZCYX format** where:
   - **Z**: Number of time points
   - **C**: Number of channels
   - **Y**: Image height
   - **X**: Image width

2. **Automatic Detection**: The GUI automatically detects multiple time points and processes them as a 3D volume.

3. **Verifying Format**:
   ```python
   import tifffile
   data = tifffile.imread('your_sequence.tif')
   print(f"Image shape: {data.shape}")
   ```

## Cell Area Analysis

### Basic Usage

```bash
# Analyze masks with default parameters
python -m scripts.analyze_cell_areas masks.tif

# Specify pixel size
python -m scripts.analyze_cell_areas masks.tif --pixel-size 0.108

# Adjust morphological filters
python -m scripts.analyze_cell_areas masks.tif \
    --min-aspect-ratio 2.0 \
    --min-eccentricity 0.7 \
    --min-solidity 0.8 \
    --max-area 30 \
    --max-width 1.5
```

### Morphological Filtering

The script filters for rod-shaped cells using:

1. **Aspect Ratio** (length/width):
   - `--min-aspect-ratio`: Minimum ratio (default: 1.5)
   - `--max-aspect-ratio`: Maximum ratio (default: 10.0)

2. **Eccentricity** (0 = circle, 1 = line):
   - `--min-eccentricity`: Minimum eccentricity (default: 0.5)

3. **Solidity** (area / convex_hull_area):
   - `--min-solidity`: Minimum solidity (default: 0.7)

4. **Size Filters**:
   - `--max-area`: Maximum cell area in µm²
   - `--max-width`: Maximum cell width in µm
   - `--min-size`: Minimum object size in pixels (default: 10)

### Output Options

- `--detailed`: Include per-cell data sheet
- `--visualization`: Create visualization TIFF (green=accepted, red=rejected)
- `--output, -o`: Custom output Excel file path

### Example Workflows

**Bacterial Cells (Typical Settings):**
```bash
python -m scripts.analyze_cell_areas masks.tif \
    --pixel-size 0.108 \
    --min-aspect-ratio 2.0 \
    --max-aspect-ratio 8.0 \
    --min-eccentricity 0.6 \
    --min-solidity 0.75 \
    --max-area 30 \
    --max-width 1.5
```

**Complete Workflow:**
```bash
# Step 1: Process images
python -m scripts.batch_process_tiff_sequence input.tif --model bact_phase_omni --diameter 30

# Step 2: Analyze results
python -m scripts.analyze_cell_areas input_masks.tif --pixel-size 0.108 --max-area 50 --max-width 2.0
```

## Building App Bundles

BactiMate can be packaged into a macOS `.app` bundle that includes the Python environment, similar to how [ilastik](https://github.com/ilastik/ilastik) distributes their application.

### Overview

Instead of compiling with Nuitka (which has PyQt6 compatibility issues on macOS), we bundle the entire project folder including the Python virtual environment into a macOS `.app` bundle. This approach:

- ✅ Avoids Nuitka compilation issues
- ✅ Works with PyQt6 without modifications
- ✅ Keeps the environment small and self-contained
- ✅ Easy to maintain and update
- ✅ Similar to how ilastik distributes their application

### Prerequisites

1. **Set up the environment:**
   ```bash
   ./setup/setup_fresh_env.sh
   ```

2. **For macOS DMG creation (optional):**
   ```bash
   brew install create-dmg
   ```

### Build Steps

1. **Create icons (first time only):**
   ```bash
   ./build/create_icons.sh
   ```

2. **Build the app bundle:**
   ```bash
   ./build/build_app_bundle.sh
   ```
   
   This will create `BactiMate.app` in the project root directory.

3. **Test the app bundle:**
   ```bash
   open BactiMate.app
   ```
   
   Or double-click `BactiMate.app` in Finder.

4. **Create a DMG for distribution (optional):**
   ```bash
   ./build/create_dmg.sh
   ```
   
   This creates `dist/BactiMate.dmg` that users can mount and drag to Applications.

**Output:** `BactiMate.app` and optionally `dist/BactiMate.dmg`

The app bundle includes the entire project folder with the Python virtual environment, making it self-contained and easy to distribute.

### App Bundle Structure

```
BactiMate.app/
├── Contents/
│   ├── MacOS/
│   │   └── BactiMate          # Launcher executable (runs launch_gui.sh)
│   ├── Resources/             # All project files
│   │   ├── venv/              # Python virtual environment
│   │   ├── main.py
│   │   ├── launch_gui.sh      # Main launch script
│   │   └── ... (all other project files)
│   └── Info.plist             # macOS app metadata
```

The launcher (`Contents/MacOS/BactiMate`) is a shell script that:
1. Finds the project directory inside the app bundle
2. Changes to that directory
3. Executes `launch_gui.sh` which activates the venv and runs the Python script

### Distribution

**For Users:**
1. Download the `.dmg` file
2. Mount it by double-clicking
3. Copy `BactiMate.app` to their Applications folder
4. Control-click and choose "Open" the first time (to bypass Gatekeeper)
5. Double-click to launch thereafter

**Creating a Distribution Archive:**
```bash
# Create a tar.bz2 archive (similar to ilastik)
tar -cjf BactiMate-macOS.tar.bz2 BactiMate.app
```

Or use the DMG method:
```bash
./build/create_dmg.sh
```

### Cross-Machine Compatibility

The app bundle is designed to work across different Macs. Key features:

1. **Relocatable Virtual Environment**: Uses `PYTHONHOME` environment variable to make the venv relocatable
2. **Direct Python Execution**: Uses venv Python directly instead of activation (more reliable)
3. **Error Logging**: Logs to the BactiMate Logs folder (`Logs/launch.log` in the project, or `~/Library/Application Support/BactiMate/Logs` when run from app bundle) for debugging
4. **User-Friendly Error Dialogs**: Shows error messages if launch fails

**Key Environment Variables:**
```bash
export PYTHONHOME="$SCRIPT_DIR/venv"  # Makes venv relocatable
export PATH="$SCRIPT_DIR/venv/bin:$PATH"  # Prioritize venv binaries
```

### Advantages Over Nuitka

1. **No compilation required** - Just bundle existing files
2. **Faster builds** - No long compilation process
3. **Easier debugging** - Can inspect/modify files inside the bundle
4. **PyQt6 compatible** - No need to switch to PySide6
5. **Smaller learning curve** - Standard macOS app bundle structure

### File Size

The app bundle includes:
- Python virtual environment (~500MB-1GB depending on dependencies)
- All project files
- Models and data files

Total size is typically 500MB-1.5GB, which is reasonable for a scientific application with ML dependencies.

### Building on Different Platforms

**macOS Bundle:**
- Run `./build/build_app_bundle.sh` on macOS
- Creates `BactiMate.app` that can be distributed to other Macs
- No compilation needed - just bundles the venv and files

**macOS-only:** This repo focuses on the macOS app bundle. Windows build scripts have been removed.

## Troubleshooting

### User-trained models: why different parameters?

Models trained with BactiMate use the same Omnipose code (`python -m omnipose --train ...`), so in principle they should behave like the built-in `bact_phase_omni` and the same parameters (e.g. mask threshold 0) would apply. In practice, user checkpoints often produce a distance map in a different range (e.g. constant or shifted negative values), so with the default mask threshold 0 you get "No cell pixels".

**Likely causes:**

- **Input preprocessing**: At inference, BactiMate (and cellpose_omni) normalize and resize images. If that differs from what Omnipose used during training (e.g. per-tile vs per-image normalization, or different rescaling when diameter=0), the network sees a different input distribution and can output a collapsed or shifted distance channel.
- **Size model**: Built-in models can use a size model so that diameter=0 triggers rescaling; user models typically have no size model, so rescaling and effective scale at the network input can differ.
- **Training run**: Different data or training length can change the scale of the distance head output even with the same architecture.

So the difference is not that "training broke the model"—it's that inference conditions (preprocessing, rescaling) don’t fully match training, so the output range of the distance channel differs. Using a **negative mask threshold** (e.g. -2.5) for user models is a pragmatic workaround so that pixels still count as "cell" and segmentation runs. The proper long-term fix is to align inference with training (same normalization and rescaling) so the distance channel stays in the same range as the pretrained model.

### Making user-trained models match pretrained behavior

To get user-trained models to behave like the built-in Omnipose models (same parameters, no need for a negative mask threshold), follow the [Omnipose Training](https://omnipose.readthedocs.io/training.html) and [Models](https://omnipose.readthedocs.io/models.html) docs:

1. **Same architecture as pretrained**  
   When training from a pretrained model (e.g. **bact_phase_omni**), BactiMate locks **nchan** and **nclasses** to the model's values (e.g. nchan=2, nclasses=3 for bact_phase_omni). Use "Check compatibility" and keep these locked so the saved checkpoint matches the pretrained architecture.

2. **Checkpoint metadata**  
   After training, BactiMate rewrites the saved `.pth` to include **nchan** and **nclasses** in the checkpoint. That way, at inference the same architecture is used and the distance channel is read correctly. Models trained in BactiMate (with the current version) get this automatically; older or externally trained `.pth` files may not have it, so the loader infers nchan from weights and defaults nclasses=3.

3. **Training settings**  
   Train with **diameter=0** (no rescaling), as in the Omnipose docs. Use the same channel setup as at inference (e.g. grayscale → nchan=2 with channels=[0,0]).

4. **Inference settings**  
   For well-trained models, [Settings](https://omnipose.readthedocs.io/settings.html) recommends **flow_threshold=0.0** and **mask_threshold=0.0**. If your user model still gives 0 cells with mask_threshold=0, try the negative mask threshold workaround (e.g. -2.5) or re-check that training data and preprocessing match inference.

5. **Training data and architecture**  
   When you click **Start training**, BactiMate always prepares the **omnipose_masks** subfolder (masks normalized to instance labels) and always uses the **same architecture as the built-in model**. If you choose **"New model (same architecture as bact_phase_omni)"**, BactiMate initializes from bact_phase_omni and trains on your data, so the result is identical to the built-in model (same architecture, same init) and works at inference. You can also choose to fine-tune from bact_phase_omni or another built-in model explicitly.

### App Bundle Issues

#### App won't launch

1. Check that `launch_gui.sh` is executable:
   ```bash
   chmod +x BactiMate.app/Contents/Resources/launch_gui.sh
   chmod +x BactiMate.app/Contents/Resources/setup/launch_gui.sh
   ```

2. Check console logs:
   ```bash
   open Console.app
   # Then try launching BactiMate.app and look for errors
   ```

3. Check the log file:
   ```bash
   cat Logs/launch.log
   ```
   
   The log will show:
   - Launch attempts with timestamps
   - Python version detection
   - Qt plugin paths
   - Any errors that occur

4. Test the launcher manually:
   ```bash
   cd BactiMate.app/Contents/Resources
   ./launch_gui.sh
   ```

#### Code signing issues

The build script applies ad-hoc code signatures. For distribution outside the App Store, you may need:
- A Developer ID certificate from Apple
- Proper code signing with `codesign --sign "Developer ID Application: Your Name"`

#### Gatekeeper warnings

Users may see a warning the first time they open the app. They need to:
1. Control-click the app
2. Choose "Open"
3. Click "Open" in the security dialog

After the first launch, macOS will remember the app is safe.

#### Cross-machine compatibility issues

If the app fails on another machine, check the log file:
```bash
cat Logs/launch.log
```

Common issues:
- **Hardcoded Python paths**: The venv's `pyvenv.cfg` may contain absolute paths that don't exist on other machines. The launch script uses `PYTHONHOME` to work around this.
- **Missing dependencies**: Ensure all dependencies are installed in the venv before building the bundle.
- **Qt plugin issues**: The launch script automatically detects and sets Qt plugin paths.

### PyQt6 Issues (macOS)

If you see "Could not find the Qt platform plugin 'cocoa'":

```bash
./setup/repair_pyqt6.sh
```

**Why this happens:**
- macOS version updates with stricter security policies
- PyQt6 version changes during reinstallation
- Virtual environment recreation

**Prevention:**
- PyQt6 is pinned to a specific version in `requirements.txt`
- Launch script automatically tests and repairs if needed

### GUI Won't Launch

1. Make sure venv is activated: `source venv/bin/activate`
2. Check PyQt6 is installed: `pip show PyQt6`
3. Run repair script: `./setup/repair_pyqt6.sh`
4. Try fresh setup: `./setup/setup_fresh_env.sh`

### Processing Fails

1. Check the Status & Log panel in GUI for error messages
2. Verify input files exist and are valid TIFF files
3. Ensure sufficient disk space for outputs
4. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory Errors

```bash
# Reduce batch size
python -m scripts.batch_process_tiff_sequence input.tif --batch-size 4

# Or use CPU (slower but uses less memory)
python -m scripts.batch_process_tiff_sequence input.tif --no-gpu
```

### Poor Segmentation

1. Try specifying diameter explicitly
2. Adjust flow and mask thresholds
3. Try different model: `--model cyto_omni`
4. Enable clustering: `--cluster`

### Model Download Issues

Models are downloaded at runtime to `~/.cellpose/models`. If corrupted:

```bash
./setup/clear_models.sh
```

Models will be re-downloaded on next use.

### Format Issues

Check TIFF format:
```bash
python -m scripts.check_tiff_format your_file.tif
```

Reshape if needed:
```bash
python -m scripts.check_tiff_format input.tif output.tif
```

## Testing

### Automated tests

Optional unit tests (no Qt required for session_params; safe_imread needs numpy/tifffile):

```bash
pip install pytest   # optional, or use plain python
python tests/test_session_params.py
python tests/test_safe_imread.py
# Or with pytest:
pytest tests/ -v
```

- **tests/test_session_params.py** – session param get/apply round-trip with mock widgets
- **tests/test_safe_imread.py** – TIFF reading via `gui.utils.safe_imread`

### Pre-Build Setup

- [ ] Ensure virtual environment is activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] All patches applied (`python3 setup/apply_all_patches.py` or run via launch_gui.sh)

### Basic Functionality Tests

#### Application Launch
- [ ] Application launches successfully
- [ ] No import errors in console/logs
- [ ] GUI displays correctly
- [ ] All tabs/buttons visible and accessible
- [ ] No Qt plugin errors

#### Model Loading
- [ ] Can select different model types (bact_phase_omni, cyto2_omni, etc.)
- [ ] Models download correctly if not cached
- [ ] No model architecture mismatch errors
- [ ] Model files found in correct location

#### Segmentation Tests

**Single Image Segmentation:**
- [ ] Load a single TIFF image
- [ ] Run segmentation with default settings
- [ ] Segmentation completes successfully
- [ ] Results display correctly
- [ ] Can export results

**Batch Processing:**
- [ ] Load a folder of images
- [ ] Run batch segmentation
- [ ] All images process without errors
- [ ] Progress bar updates correctly
- [ ] Results saved correctly

**Different Model Types:**
- [ ] Test with `bact_phase_omni`
- [ ] Test with `cyto2_omni`
- [ ] Test with `bact_fluor_omni` (if available)
- [ ] Each model type works correctly

**Different Diameter Settings:**
- [ ] Test with auto-detect (diameter = 0)
- [ ] Test with specific diameter values (20, 30, 40 pixels)
- [ ] No errors with any diameter setting

#### File Format Support
- [ ] TIFF files work correctly
- [ ] ND2 files can be loaded (if applicable)
- [ ] Multi-page TIFF sequences work
- [ ] Export to Excel/CSV works

### Error Handling Tests

**Known Issues to Verify Fixed:**
- [ ] No `TypeError: '>' not supported between instances of 'NoneType' and 'int'`
- [ ] No model architecture mismatch errors
- [ ] No missing dependency errors
- [ ] No Qt plugin errors

**Edge Cases:**
- [ ] Handle empty image folders gracefully
- [ ] Handle corrupted/invalid image files gracefully
- [ ] Handle missing model files gracefully
- [ ] Handle network errors during model download

### Performance Tests

- [ ] Startup time is reasonable (< 10 seconds)
- [ ] Segmentation speed is acceptable
- [ ] Memory usage is reasonable
- [ ] No memory leaks during batch processing

### Platform-Specific Tests (macOS)

- [ ] Application bundle structure is correct
- [ ] Can create DMG file (if applicable)
- [ ] Application runs on clean macOS system (no venv)
- [ ] Code signing works (if applicable)
- [ ] Gatekeeper doesn't block the app

### Cross-Machine Testing

- [ ] Build app bundle on build machine
- [ ] Test on build machine (should still work)
- [ ] Copy to different Mac (different user, different location)
- [ ] Test launch on different Mac
- [ ] Check log file if it fails: `Logs/launch.log` (or `~/Library/Application Support/BactiMate/Logs/launch.log` when run from app)
- [ ] Verify error dialogs appear if there are critical errors

## Available Scripts

### Project layout
- `main.py` - Application entrypoint (run from project root)
- `gui/` - GUI code (tabs, workers, widgets)
- `scripts/` - Backend Python scripts (run as `python -m scripts.analyze_cell_areas ...`, etc.)
- `setup/` - Setup and maintenance scripts
- `tests/` - Automated tests
- `docs/` - Documentation
- `Logs/` - Session log files

### Main entrypoints
- `main.py` - Batch processing GUI
- `launch_gui.sh` - Launch GUI (creates venv, applies patches, runs main.py)

### Scripts (run from project root)
- `python -m scripts.analyze_cell_areas` - Analyze masks and extract cell areas
- `python -m scripts.batch_process_tiff_sequence` - Process single multipage TIFF with Omnipose
- `python -m scripts.batch_process_folder` - Process folder: segment + analyze
- `python -m scripts.convert_nd2_to_tif` - Convert ND2/LIF/CZI to TIFF
- `python -m scripts.check_tiff_format` - Verify/reshape TIFF format

### Setup and utilities
- `setup/setup_fresh_env.sh` - Setup fresh virtual environment
- `setup/apply_all_patches.py` - Compatibility patches (run automatically by launch_gui)
- `setup/repair_pyqt6.sh` - Repair PyQt6 installation
- `setup/install_cuda_pytorch.sh` - Install PyTorch with CUDA
- `setup/clear_models.sh` - Clear corrupted model cache
- `build/build_app_bundle.sh` - Build macOS application bundle
- `build/create_dmg.sh` - Create macOS DMG file
- `build/create_icons.sh` - Create application icons from `img/logo2.png`

## Technical Notes

### PyQt6 Migration

BactiMate uses PyQt6 (migrated from PyQt5) for better:
- macOS support (especially macOS 13+)
- Active development and long-term support
- Performance improvements
- Nuitka compatibility

### Virtual Environment Management

The project uses Python 3.11+ with a virtual environment. The launch script (`launch_gui.sh`) uses the venv Python directly and sets `PYTHONHOME` to make the environment relocatable, which is essential for app bundle distribution.

## License

[Add your license information here]

## References

- [Omnipose Documentation](https://omnipose.readthedocs.io/)
- [Cellpose Documentation](https://cellpose.readthedocs.io/)
