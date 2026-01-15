# BactiMate

Bacterial cell segmentation and analysis tool using Omnipose with a user-friendly GUI and batch processing capabilities.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Graphical User Interface](#graphical-user-interface)
- [Command Line Usage](#command-line-usage)
- [Batch Processing](#batch-processing)
- [Cell Area Analysis](#cell-area-analysis)
- [Building Executables](#building-executables)
- [Troubleshooting](#troubleshooting)

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
python batch_process_tiff_sequence.py your_sequence.tif --diameter 30
```

### Process All Files in a Folder

```bash
python batch_process_folder.py /path/to/folder \
    --model bact_phase_omni \
    --diameter 30 \
    --pixel-size 0.108 \
    --min-aspect-ratio 2.0 \
    --max-area 50 \
    --max-width 2.0
```

## Installation

### Prerequisites

- Python 3.9+
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

3. **For fresh setup (optional):**
   ```bash
   ./setup_fresh_env.sh
   ```

### Installation Details

- **Python Version**: 3.9.6
- **Omnipose Version**: 1.0.6
- **GUI Framework**: PyQt5
- **Virtual Environment**: `venv/`

## Graphical User Interface

### Launching the GUI

```bash
./launch_gui.sh
```

Or manually:
```bash
source venv/bin/activate
python omnipose_batch_gui.py
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
python batch_process_tiff_sequence.py input.tif --diameter 30

# With specific model
python batch_process_tiff_sequence.py input.tif --model bact_phase_omni --diameter 30

# With channels
python batch_process_tiff_sequence.py input.tif --channels 0 1 --diameter 25

# Adjust sensitivity
python batch_process_tiff_sequence.py input.tif --flow-threshold 0.3 --mask-threshold -2

# Use CPU instead of GPU
python batch_process_tiff_sequence.py input.tif --no-gpu

# Skip overlay creation (faster)
python batch_process_tiff_sequence.py input.tif --no-overlay
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
python batch_process_folder.py /path/to/folder \
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
python analyze_cell_areas.py masks.tif

# Specify pixel size
python analyze_cell_areas.py masks.tif --pixel-size 0.108

# Adjust morphological filters
python analyze_cell_areas.py masks.tif \
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
python analyze_cell_areas.py masks.tif \
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
python batch_process_tiff_sequence.py input.tif --model bact_phase_omni --diameter 30

# Step 2: Analyze results
python analyze_cell_areas.py input_masks.tif --pixel-size 0.108 --max-area 50 --max-width 2.0
```

## Building Executables

BactiMate can be packaged into standalone executables for Windows (.exe) and macOS (.dmg) using PyInstaller.

### Prerequisites

1. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

2. **For macOS DMG creation (optional):**
   ```bash
   brew install create-dmg
   ```

### macOS Build

```bash
# 1. Create icons (first time only)
./create_icons.sh

# 2. Build the app
./build_macos.sh

# 3. Create DMG
./create_dmg.sh
```

**Output:** `dist/BactiMate.app` and `dist/BactiMate.dmg`

### Windows Build

```cmd
build_windows.bat
```

Or manually:
```cmd
pyinstaller BactiMate.spec
```

**Output:** `dist\BactiMate.exe`

### Building on Different Platforms

**Cross-platform builds are not supported.** You must build on each platform:
- macOS builds only work on macOS
- Windows builds only work on Windows

**Recommended:** Use GitHub Actions for automated builds (see `.github/workflows/build.yml`)

### Distribution

- **macOS**: Share the `.dmg` file (users mount and drag app to Applications)
- **Windows**: Share the `.exe` file directly

**Note:** Executables are large (500MB-1GB) because they include Python, PyTorch, PyQt5, and all dependencies.

## Troubleshooting

### PyQt5 Issues (macOS)

If you see "Could not find the Qt platform plugin 'cocoa'":

```bash
./repair_pyqt5.sh
```

**Why this happens:**
- macOS version updates with stricter security policies
- PyQt5 version changes during reinstallation
- Virtual environment recreation

**Prevention:**
- PyQt5 is pinned to version 5.15.10 in `requirements.txt`
- Launch script automatically tests and repairs if needed

### GUI Won't Launch

1. Make sure venv is activated: `source venv/bin/activate`
2. Check PyQt5 is installed: `pip show PyQt5`
3. Run repair script: `./repair_pyqt5.sh`
4. Try fresh setup: `./setup_fresh_env.sh`

### Processing Fails

1. Check the Status & Log panel in GUI for error messages
2. Verify input files exist and are valid TIFF files
3. Ensure sufficient disk space for outputs
4. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory Errors

```bash
# Reduce batch size
python batch_process_tiff_sequence.py input.tif --batch-size 4

# Or use CPU (slower but uses less memory)
python batch_process_tiff_sequence.py input.tif --no-gpu
```

### Poor Segmentation

1. Try specifying diameter explicitly
2. Adjust flow and mask thresholds
3. Try different model: `--model cyto_omni`
4. Enable clustering: `--cluster`

### Model Download Issues

Models are downloaded at runtime to `~/.cellpose/models`. If corrupted:

```bash
./clear_models.sh
```

Models will be re-downloaded on next use.

### Format Issues

Check TIFF format:
```bash
python check_tiff_format.py your_file.tif
```

Reshape if needed:
```bash
python check_tiff_format.py input.tif output.tif
```

## Available Scripts

### Main Scripts
- `omnipose_batch_gui.py` - Batch processing GUI
- `batch_process_tiff_sequence.py` - Process single multipage TIFF file
- `batch_process_folder.py` - Process all TIFF files in a folder
- `analyze_cell_areas.py` - Analyze masks and extract cell areas with filtering
- `check_tiff_format.py` - Verify/reshape TIFF format
- `convert_nd2_to_tif.py` - Convert ND2 files to TIFF

### Utility Scripts
- `launch_gui.sh` - Launch Batch Processing GUI
- `launch_omnipose.sh` - Launch Omnipose GUI
- `setup_fresh_env.sh` - Setup fresh virtual environment
- `repair_pyqt5.sh` - Repair PyQt5 installation
- `clear_models.sh` - Clear corrupted model cache
- `build_macos.sh` - Build macOS application bundle
- `create_dmg.sh` - Create macOS DMG file
- `create_icons.sh` - Create application icons

## License

[Add your license information here]

## References

- [Omnipose Documentation](https://omnipose.readthedocs.io/)
- [Cellpose Documentation](https://cellpose.readthedocs.io/)