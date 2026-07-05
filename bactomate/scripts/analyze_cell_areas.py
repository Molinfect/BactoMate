#!/usr/bin/env python3
"""
Analyze cell areas from Omnipose masks with morphological filtering.

This script:
1. Loads masks and optionally original images
2. Applies morphological filtering (rod-shaped cells, max area, max width)
3. Calculates total cell area per frame in µm²
4. Normalizes to first frame (first frame = 1.0)
5. Exports results to Excel

Usage:
    python analyze_cell_areas.py masks.tif [original.tif] [options]
"""

import argparse
import json
import logging
import sys
import numpy as np
import tifffile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, binary_dilation, binary_erosion
from skimage.segmentation import find_boundaries
import warnings
warnings.filterwarnings('ignore')

from bactomate.modules.label_render import draw_hq_labels
from bactomate.modules.image_stack_utils import coerce_plane_to_2d, coerce_stack_to_tyx
from bactomate.modules.spatial_metrics import (
    apply_foci_spatial_to_frame,
    apply_temporal_metrics,
    compute_fi_distribution_metrics,
    enrich_cell_geometry,
    frame_summary_spatial_metrics,
)

SCRIPT_LOG = logging.getLogger(__name__)

try:
    from bactomate.modules.foci_utils import (
        detect_peaks_skimage,
        detect_peaks_skimage_auto,
        detect_peaks_cell_first_region_grow,
        detect_peaks_spotiflow,
        refine_spotiflow_peaks,
        assign_peaks_to_cells,
        focus_size_and_mean_at_peak,
        get_focus_blob_mask,
    )
except ImportError:
    detect_peaks_skimage = None
    detect_peaks_skimage_auto = None
    detect_peaks_cell_first_region_grow = None
    detect_peaks_spotiflow = None
    refine_spotiflow_peaks = None
    assign_peaks_to_cells = None
    focus_size_and_mean_at_peak = None
    get_focus_blob_mask = None

try:
    from bactomate.modules.validation import paths_exist, mask_fi_spatial_shape_match
except ImportError:
    paths_exist = None
    mask_fi_spatial_shape_match = None


def _setup_script_logging(log_to_file: bool = True) -> None:
    """Configure logging for this script: stdout (for GUI capture). When run from the GUI, the script logger
    propagates to the BactoMate logger, so output also goes to bactomate_gui_*.log. When run from CLI without
    the GUI, we do not add a separate analyze_masks.log; use stdout or the caller's logging."""
    log = SCRIPT_LOG
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(message)s")
    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(fmt)
    log.addHandler(out)
    # Do not add a separate analyze_masks.log: when run from GUI, output goes to bactomate_gui_*.log via
    # propagation to the "bactomate" logger; when run from CLI, stdout is sufficient.


def calculate_cell_properties(mask, pixel_size=0.108, exclude_border=False, progress_callback=None):
    """
    Calculate properties for each cell in a mask.

    Parameters:
    -----------
    mask : array
        Labeled mask image (0 = background, 1,2,3... = cell IDs)
    pixel_size : float
        Pixel size in µm (default: 0.108)
    exclude_border : bool
        If True, skip regions that touch the image border (default: False).

    Returns:
    --------
    properties : list of dict
        List of cell properties (area, width, length, aspect_ratio, etc.)
    """
    props = []
    h, w = mask.shape[:2]

    regions = regionprops(mask)
    n_regions = len(regions)
    report_every = max(1, n_regions // 24) if n_regions else 1

    for idx, region in enumerate(regions):
        if progress_callback is not None and (
            idx == 0 or idx == n_regions - 1 or idx % report_every == 0
        ):
            try:
                progress_callback(int(idx), int(n_regions))
            except Exception:
                pass
        # Skip background (label 0)
        if region.label == 0:
            continue
        if exclude_border:
            min_row, min_col, max_row, max_col = region.bbox
            if min_row <= 0 or min_col <= 0 or max_row >= h or max_col >= w:
                continue

        # Basic properties
        area_pixels = region.area
        area_um2 = area_pixels * (pixel_size ** 2)
        
        # Bounding box dimensions
        min_row, min_col, max_row, max_col = region.bbox
        bbox_width = (max_col - min_col) * pixel_size
        bbox_height = (max_row - min_row) * pixel_size
        
        # Use major and minor axis lengths for rod-shaped analysis
        # major_axis_length is the longest axis
        # minor_axis_length is perpendicular to major axis
        major_axis = region.major_axis_length * pixel_size
        minor_axis = region.minor_axis_length * pixel_size if region.minor_axis_length > 0 else pixel_size
        
        # Aspect ratio (length/width for rods)
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
        
        # Equivalent diameter (diameter of circle with same area)
        equiv_diameter = region.equivalent_diameter * pixel_size
        
        # Eccentricity (0 = circle, 1 = line)
        eccentricity = region.eccentricity
        
        # Solidity (area / convex_hull_area)
        solidity = region.solidity
        
        props.append({
            'label': region.label,
            'area_um2': area_um2,
            'area_pixels': area_pixels,
            'major_axis_um': major_axis,
            'minor_axis_um': minor_axis,
            'aspect_ratio': aspect_ratio,
            'width_um': minor_axis,  # Width is the minor axis for rods
            'length_um': major_axis,  # Length is the major axis for rods
            'equiv_diameter_um': equiv_diameter,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'bbox_width_um': bbox_width,
            'bbox_height_um': bbox_height,
        })
        enrich_cell_geometry(props[-1], region, pixel_size)
    
    return props


def calculate_fi_intensities(mask, fi_image, cell_label, boundaries=None, contour_thickness=1, do_contour=True):
    """
    Calculate fluorescence intensity metrics for a single cell.
    
    Parameters:
    -----------
    mask : array
        Labeled mask image (0 = background, 1,2,3... = cell IDs)
    fi_image : array
        FI channel image (2D, same shape as mask)
    cell_label : int
        Label ID of the cell to analyze
    boundaries : array, optional
        Pre-computed boundary map from find_boundaries(). If None, will compute it.
        Pass this to avoid redundant computation when processing multiple cells.
    contour_thickness : int, default=1
        Thickness of the contour region in pixels. 1 = single-pixel boundary.
        Values > 1 will dilate the boundary to include more pixels.
        
    Returns:
    --------
    mean_fi : float
        Mean pixel intensity within the cell
    contour_fi : float
        Mean pixel intensity at the cell contour/boundary
    fi_ratio : float
        Ratio of contour FI to mean FI (contour_fi / mean_fi)
    """
    # Create binary mask for this cell
    cell_mask = (mask == cell_label)
    
    if not np.any(cell_mask):
        return np.nan, np.nan, np.nan
    
    # Ensure FI image is 2D and matches mask shape
    if fi_image.ndim > 2:
        # Take first channel if multi-channel
        fi_image = fi_image[:, :, 0] if fi_image.shape[2] < 10 else fi_image[0, :, :]
    
    if fi_image.shape != mask.shape:
        # Try to match shapes
        if fi_image.shape[0] == mask.shape[0] and fi_image.shape[1] == mask.shape[1]:
            pass  # Shapes match
        else:
            SCRIPT_LOG.warning("FI image shape %s doesn't match mask shape %s", fi_image.shape, mask.shape)
            return np.nan, np.nan, np.nan
    
    # Mean FI intensity of entire cell: FI channel pixel values at cell mask (raw fluorescence)
    cell_pixels = fi_image[cell_mask]
    mean_fi = np.mean(cell_pixels) if len(cell_pixels) > 0 else np.nan

    if not do_contour:
        return mean_fi, np.nan, np.nan

    # Find contour/boundary pixels using morphological edge detection
    # This method gets the actual edge pixels by eroding the cell mask
    # The difference between original and eroded mask gives us the edge
    if boundaries is None:
        # Use erosion-based edge detection for more accurate membrane detection
        # This gets the actual outermost pixels of the cell
        cell_mask_eroded = binary_erosion(cell_mask)
        # Edge pixels = pixels in cell_mask but not in eroded version
        cell_contour = cell_mask & (~cell_mask_eroded)
        
        # Apply thickness using dilation if thickness > 1
        if contour_thickness > 1:
            # Dilate the contour to include more pixels near the edge
            for _ in range(contour_thickness - 1):
                cell_contour = binary_dilation(cell_contour) & cell_mask  # Keep only within cell
    else:
        # Use pre-computed boundaries if provided (legacy support)
        # Get boundary pixels that belong to this cell
        cell_contour = boundaries & cell_mask
        
        # Apply thickness if needed (though boundaries should already be at right thickness)
        if contour_thickness > 1:
            for _ in range(contour_thickness - 1):
                cell_contour = binary_dilation(cell_contour) & cell_mask
    
    # Calculate mean FI intensity at contour (yellow pixels)
    if np.any(cell_contour):
        contour_pixels = fi_image[cell_contour]
        contour_fi = np.mean(contour_pixels) if len(contour_pixels) > 0 else np.nan
    else:
        # If no contour found (very small cell), set to NaN
        contour_fi = np.nan
    
    # Calculate interior pixels (green mask MINUS yellow contour)
    cell_interior = cell_mask & (~cell_contour)
    
    # Calculate mean FI intensity of interior pixels only (excluding contour)
    if np.any(cell_interior):
        interior_pixels = fi_image[cell_interior]
        interior_fi = np.mean(interior_pixels) if len(interior_pixels) > 0 else np.nan
    else:
        # If no interior (cell is all contour), use contour FI
        interior_fi = contour_fi
    
    # Calculate ratio: contour FI / interior FI (this is the meaningful comparison)
    # This compares membrane (yellow) to cytoplasm (green interior)
    if not np.isnan(interior_fi) and interior_fi > 0:
        fi_ratio = contour_fi / interior_fi
    else:
        fi_ratio = np.nan
    
    return mean_fi, contour_fi, fi_ratio


def filter_rod_shaped_cells(properties, min_aspect_ratio=1.5, max_aspect_ratio=10.0,
                            min_eccentricity=0.5, min_solidity=0.7):
    """
    Filter cells to keep only rod-shaped ones.
    
    Parameters:
    -----------
    properties : list of dict
        Cell properties
    min_aspect_ratio : float
        Minimum aspect ratio (length/width) for rod-shaped cells
    max_aspect_ratio : float
        Maximum aspect ratio to avoid artifacts
    min_eccentricity : float
        Minimum eccentricity (0=circle, 1=line)
    min_solidity : float
        Minimum solidity (area/convex_hull_area) to filter out irregular shapes
        
    Returns:
    --------
    filtered : list of dict
        Filtered cell properties
    """
    filtered = []
    
    for prop in properties:
        aspect = prop['aspect_ratio']
        ecc = prop['eccentricity']
        sol = prop['solidity']
        
        # Check if cell is rod-shaped
        if (min_aspect_ratio <= aspect <= max_aspect_ratio and
            ecc >= min_eccentricity and
            sol >= min_solidity):
            filtered.append(prop)
    
    return filtered


def filter_by_size(
    properties,
    min_area_um2=None,
    max_area_um2=None,
    min_width_um=None,
    max_width_um=None,
    min_length_um=None,
    max_length_um=None,
):
    """
    Filter cells by minimum/maximum area, width, and length.
    
    Parameters:
    -----------
    properties : list of dict
        Cell properties
    min_area_um2 : float, optional
        Minimum cell area in µm²
    max_area_um2 : float, optional
        Maximum cell area in µm²
    min_width_um : float, optional
        Minimum cell width in µm
    max_width_um : float, optional
        Maximum cell width in µm
    min_length_um : float, optional
        Minimum cell length in µm (major axis)
    max_length_um : float, optional
        Maximum cell length in µm (major axis)

    Returns:
    --------
    filtered : list of dict
        Filtered cell properties
    """
    filtered = properties
    
    if min_area_um2 is not None:
        filtered = [p for p in filtered if p['area_um2'] >= min_area_um2]
    
    if max_area_um2 is not None:
        filtered = [p for p in filtered if p['area_um2'] <= max_area_um2]
    
    if min_width_um is not None:
        filtered = [p for p in filtered if p['width_um'] >= min_width_um]
    
    if max_width_um is not None:
        filtered = [p for p in filtered if p['width_um'] <= max_width_um]
    
    if min_length_um is not None:
        filtered = [p for p in filtered if p['length_um'] >= min_length_um]

    if max_length_um is not None:
        filtered = [p for p in filtered if p['length_um'] <= max_length_um]

    return filtered


def analyze_masks(masks_path, pixel_size=0.108,
                  min_aspect_ratio=1.5, max_aspect_ratio=10.0,
                  min_eccentricity=0.5, min_solidity=0.7,
                  min_area_um2=None, max_area_um2=None, min_width_um=None, max_width_um=None, min_length_um=None,
                  remove_small=True, min_size_pixels=10, fi_image_path=None, fi_contour_thickness=1, fi_do_contour=True,
                  exclude_border=False,
                  run_foci=False, foci_detection_mode=0, foci_min_rel=2.0, foci_min_rel_mode="per_cell", foci_min_rel_cell=2.0, foci_min_separation=2, foci_min_size=1, foci_max_size=None,
                  foci_min_peak_raw=0, foci_min_focus_mean=0, foci_blob_threshold=0.8,
                  foci_image_path=None,
                  foci_exclude_outside_cells=False,
                  spotiflow_options=None,
                  fi_channel_name=None):
    """
    Analyze masks and extract cell areas with morphological filtering.
    
    Parameters:
    -----------
    masks_path : str
        Path to mask TIFF file
    pixel_size : float
        Pixel size in µm
    min_aspect_ratio : float
        Minimum aspect ratio for rod-shaped cells
    max_aspect_ratio : float
        Maximum aspect ratio
    min_eccentricity : float
        Minimum eccentricity
    min_solidity : float
        Minimum solidity
    min_area_um2 : float, optional
        Minimum cell area in µm²
    max_area_um2 : float, optional
        Maximum cell area in µm²
    min_width_um : float, optional
        Minimum cell width in µm
    max_width_um : float, optional
        Maximum cell width in µm
    min_length_um : float, optional
        Minimum cell length in µm (major axis)
    remove_small : bool
        Remove small objects before analysis
    min_size_pixels : int
        Minimum size in pixels for objects to keep
    fi_image_path : str, optional
        Path to FI channel TIFF file for fluorescence intensity analysis
    foci_image_path : str, optional
        Optional path to foci-detection TIFF file. When omitted, ``fi_image_path`` is used.
    fi_contour_thickness : int, default=1
        Thickness of the contour region in pixels for FI intensity calculation.
        1 = single-pixel boundary, >1 dilates the boundary to include more pixels.
        
    Returns:
    --------
    results : list of dict
        Results per frame with total area and relative area
    all_frame_data : list of dict
        Detailed data including accepted/rejected cell labels
    masks : array
        Original mask array
    """
    # Pre-run validation: paths exist and (if FI) spatial shape match
    if paths_exist is not None:
        ok, missing = paths_exist([masks_path])
        if not ok:
            raise FileNotFoundError(f"Mask file not found: {missing[0]}")
    if fi_image_path and paths_exist is not None:
        ok, missing = paths_exist([fi_image_path])
        if not ok:
            raise FileNotFoundError(f"FI image file not found: {missing[0]}")
    if fi_image_path and mask_fi_spatial_shape_match is not None:
        ok, err = mask_fi_spatial_shape_match(masks_path, fi_image_path)
        if not ok and err:
            raise ValueError(err)
    if foci_image_path and paths_exist is not None:
        ok, missing = paths_exist([foci_image_path])
        if not ok:
            raise FileNotFoundError(f"Foci image file not found: {missing[0]}")
    if foci_image_path and mask_fi_spatial_shape_match is not None:
        ok, err = mask_fi_spatial_shape_match(masks_path, foci_image_path)
        if not ok and err:
            raise ValueError(err)
    try:
        foci_blob_threshold = float(foci_blob_threshold)
    except (TypeError, ValueError):
        foci_blob_threshold = 0.8
    if not np.isfinite(foci_blob_threshold):
        foci_blob_threshold = 0.8
    foci_blob_threshold = float(np.clip(foci_blob_threshold, 0.05, 1.0))

    SCRIPT_LOG.info("Loading masks from: %s", masks_path)
    masks = tifffile.imread(masks_path)
    
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    
    num_frames = masks.shape[0]
    SCRIPT_LOG.info("Found %s frames", num_frames)
    
    fi_images = None

    def _load_stack_tyx(path_str: str, label: str) -> np.ndarray:
        arr = tifffile.imread(path_str)
        stack = coerce_stack_to_tyx(arr)
        if stack is None:
            raise ValueError(f"{label} image shape {np.asarray(arr).shape} is not supported")
        return np.asarray(stack)

    def _align_stack_to_frames(stack: np.ndarray, label: str) -> np.ndarray:
        nonlocal masks, num_frames, fi_images
        if stack.shape[0] == num_frames:
            return stack
        SCRIPT_LOG.warning("%s image has %s frames but masks have %s frames", label, stack.shape[0], num_frames)
        if stack.shape[0] == 1:
            return np.repeat(stack, num_frames, axis=0)
        min_frames = min(stack.shape[0], num_frames)
        stack = stack[:min_frames]
        masks = masks[:min_frames]
        if fi_images is not None and fi_images.shape[0] != min_frames:
            fi_images = fi_images[:min_frames]
        num_frames = min_frames
        SCRIPT_LOG.warning("Using first %s frames from masks and %s image", min_frames, label)
        return stack

    # Load FI image for FI intensity analysis
    if fi_image_path:
        SCRIPT_LOG.info("Loading FI image from: %s", fi_image_path)
        fi_images = _load_stack_tyx(fi_image_path, "FI")
        fi_images = _align_stack_to_frames(fi_images, "FI")
        SCRIPT_LOG.info("FI image loaded: %s", fi_images.shape)

    # Load foci-detection image, defaulting to FI image when not set
    foci_images = None
    if foci_image_path:
        SCRIPT_LOG.info("Loading foci image from: %s", foci_image_path)
        foci_images = _load_stack_tyx(foci_image_path, "Foci")
        foci_images = _align_stack_to_frames(foci_images, "Foci")
        SCRIPT_LOG.info("Foci image loaded: %s", foci_images.shape)
    elif fi_images is not None:
        foci_images = fi_images
        if run_foci:
            SCRIPT_LOG.info("Using FI image for foci detection (no separate --foci-image provided).")
    
    results = []
    all_frame_data = []  # Store detailed data for each frame
    foci_table = []  # Foci rows (cell_id updated to persistent ID after assignment)
    next_foci_id = 1
    
    SCRIPT_LOG.info("Analyzing frames...")
    SCRIPT_LOG.info("[ANALYZE_PROGRESS] phase=frames current=0 total=%s", num_frames)
    frame_progress_every = max(1, num_frames // 100)
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        mask = masks[frame_idx]
        
        # Remove small objects (debris)
        if remove_small:
            mask_cleaned = remove_small_objects(mask, min_size=min_size_pixels)
        else:
            mask_cleaned = mask
        
        # Calculate cell properties.
        #
        # Important: analysis trusts the provided masks and does not apply additional
        # morphology/size filtering (those are handled upstream in the Filtering stage).
        # We also intentionally ignore exclude_border here so analysis does not depend on
        # filtering UI state.
        properties = calculate_cell_properties(mask_cleaned, pixel_size=pixel_size, exclude_border=False)

        # Use all detected cells as-is.
        filtered = list(properties)
        all_labels = {prop['label'] for prop in properties}
        accepted_labels = {cell['label'] for cell in filtered}
        
        # Step 3: Calculate FI intensities if FI image is provided
        if fi_images is not None:
            fi_image = fi_images[frame_idx]
            # Don't pre-compute boundaries - use per-cell erosion method for better accuracy
            # This gives us the actual edge pixels, not just pixels near the edge
            boundaries = None  # Will be computed per-cell using erosion method
            
            for cell in filtered:
                cell_label = cell['label']
                if fi_do_contour:
                    mean_fi, contour_fi, fi_ratio = calculate_fi_intensities(
                        mask_cleaned,
                        fi_image,
                        cell_label,
                        boundaries=None,
                        contour_thickness=fi_contour_thickness,
                        do_contour=True,
                    )
                    cell['mean_fi'] = mean_fi
                    cell['contour_fi'] = contour_fi
                    cell['fi_ratio'] = fi_ratio
                else:
                    mean_fi, _, _ = calculate_fi_intensities(
                        mask_cleaned,
                        fi_image,
                        cell_label,
                        boundaries=None,
                        contour_thickness=fi_contour_thickness,
                        do_contour=False,
                    )
                    cell['mean_fi'] = mean_fi
                fi_extra = compute_fi_distribution_metrics(
                    mask_cleaned,
                    fi_image,
                    cell_label,
                    contour_thickness=fi_contour_thickness,
                    do_contour=fi_do_contour,
                )
                cell['interior_fi'] = fi_extra.get('interior_fi', np.nan)
                cell['fi_std'] = fi_extra.get('fi_std', np.nan)
                cell['fi_cv'] = fi_extra.get('fi_cv', np.nan)
                cell['fi_top10_mean'] = fi_extra.get('fi_top10_mean', np.nan)
        
        # Foci analysis (per-frame detection and assignment to cells)
        if run_foci and foci_images is not None:
            for cell in filtered:
                cell.setdefault('foci_count', 0)
                cell.setdefault('foci_ids', [])
        if (run_foci and foci_images is not None and assign_peaks_to_cells is not None):
            foci_image = np.asarray(foci_images[frame_idx])
            foci_plane = np.asarray(coerce_plane_to_2d(foci_image), dtype=np.float64)
            if foci_plane.shape[:2] != mask_cleaned.shape[:2]:
                SCRIPT_LOG.warning(
                    "Foci image shape %s != mask shape %s, skipping foci for frame %s",
                    foci_plane.shape[:2],
                    mask_cleaned.shape[:2],
                    frame_idx + 1,
                )
            else:
                # foci_detection_mode: 0=manual, 1=auto, 2=spotiflow
                peaks = None
                use_per_cell = False
                cell_factor = None
                if foci_detection_mode == 2:
                    if detect_peaks_spotiflow is None:
                        raise RuntimeError("Spotiflow detection requested, but spotiflow is unavailable.")
                    sf_opts = dict(spotiflow_options) if spotiflow_options else {}
                    sf_post_on = bool(sf_opts.get("postfilters_enabled", False))
                    sf_kw = {
                        k: v
                        for k, v in sf_opts.items()
                        if k
                        not in {
                            "postfilters_enabled",
                            "pf_area_enabled",
                            "pf_min_peak_raw_enabled",
                            "pf_min_focus_mean_enabled",
                            "pf_per_cell_ratio_enabled",
                            "pf_cell_ratio_factor",
                            "pf_min_peak_raw",
                            "pf_min_focus_mean",
                            "pf_min_size_px",
                            "pf_max_size_px",
                            "preview_render_cap",
                            "preview_show_detected_vs_rendered",
                        }
                    }
                    sf_kw["raise_on_error"] = True
                    try:
                        peaks = detect_peaks_spotiflow(foci_plane, **sf_kw)
                    except TypeError:
                        peaks = detect_peaks_spotiflow(
                            foci_plane,
                            model_name=str(sf_kw.get("model_name") or "general"),
                            raise_on_error=True,
                        )
                    if peaks is None:
                        raise RuntimeError("Spotiflow detection returned no result.")
                    if refine_spotiflow_peaks is not None:
                        min_sz_sf = None
                        max_sz_sf = None
                        if sf_post_on and bool(sf_opts.get("pf_area_enabled", False)):
                            try:
                                min_sz_sf = int(sf_opts.get("pf_min_size_px", foci_min_size))
                            except (TypeError, ValueError):
                                min_sz_sf = int(foci_min_size)
                            try:
                                max_sz_sf = int(sf_opts.get("pf_max_size_px", foci_max_size or 0))
                            except (TypeError, ValueError):
                                max_sz_sf = int(foci_max_size or 0)
                            if max_sz_sf is not None and max_sz_sf <= 0:
                                max_sz_sf = None
                        min_peak_raw_sf = None
                        if sf_post_on and bool(sf_opts.get("pf_min_peak_raw_enabled", False)):
                            try:
                                min_peak_raw_sf = float(sf_opts.get("pf_min_peak_raw", foci_min_peak_raw))
                            except (TypeError, ValueError):
                                min_peak_raw_sf = float(foci_min_peak_raw or 0)
                        min_focus_mean_sf = None
                        if sf_post_on and bool(sf_opts.get("pf_min_focus_mean_enabled", False)):
                            try:
                                min_focus_mean_sf = float(sf_opts.get("pf_min_focus_mean", foci_min_focus_mean))
                            except (TypeError, ValueError):
                                min_focus_mean_sf = float(foci_min_focus_mean or 0)
                        per_cell_ratio_sf = None
                        if sf_post_on and bool(sf_opts.get("pf_per_cell_ratio_enabled", False)):
                            try:
                                per_cell_ratio_sf = float(sf_opts.get("pf_cell_ratio_factor", foci_min_rel))
                            except (TypeError, ValueError):
                                per_cell_ratio_sf = float(foci_min_rel or 0)
                        peaks = refine_spotiflow_peaks(
                            foci_plane,
                            peaks,
                            labeled_mask=mask_cleaned,
                            threshold_frac=foci_blob_threshold,
                            max_radius=5,
                            min_size=min_sz_sf,
                            max_size=max_sz_sf,
                            min_peak_raw=min_peak_raw_sf,
                            min_focus_mean=min_focus_mean_sf,
                            per_cell_ratio=per_cell_ratio_sf,
                        )
                elif foci_detection_mode == 1 and detect_peaks_cell_first_region_grow is not None:
                    peaks = detect_peaks_cell_first_region_grow(
                        foci_plane,
                        labeled_mask=mask_cleaned,
                    )
                elif detect_peaks_skimage is not None:
                    mode = (foci_min_rel_mode or "per_cell").strip().lower()
                    use_per_frame = mode in ("per_frame", "both")
                    use_per_cell = mode in ("per_cell", "both")
                    cell_factor = float(foci_min_rel_cell) if use_per_cell and mode == "both" else (float(foci_min_rel) if use_per_cell else None)
                    peaks = detect_peaks_skimage(
                        foci_plane,
                        min_rel=foci_min_rel if use_per_frame else None,
                        min_distance=foci_min_separation,
                        min_size=foci_min_size,
                        max_size=foci_max_size,
                        size_threshold_frac=foci_blob_threshold,
                        min_peak_raw=foci_min_peak_raw if foci_min_peak_raw and float(foci_min_peak_raw) > 0 else None,
                        min_focus_mean=foci_min_focus_mean if foci_min_focus_mean and float(foci_min_focus_mean) > 0 else None,
                        labeled_mask=mask_cleaned,
                    )
                if peaks is not None:
                    # Spotiflow yields point detections; derive blob area/mean using the configured blob threshold.
                    if (
                        foci_detection_mode == 2
                        and peaks
                        and focus_size_and_mean_at_peak is not None
                        and refine_spotiflow_peaks is None
                    ):
                        fi_flat = np.asarray(foci_plane, dtype=np.float64)
                        peaks_refined = []
                        for p in peaks:
                            yi, xi = int(p.get("y", 0)), int(p.get("x", 0))
                            size_px, mean_int = focus_size_and_mean_at_peak(
                                fi_flat,
                                yi,
                                xi,
                                threshold_frac=foci_blob_threshold,
                                max_radius=5,
                                labeled_mask=mask_cleaned,
                            )
                            q = dict(p)
                            q["size_px"] = int(size_px)
                            q["mean_intensity"] = float(mean_int)
                            peaks_refined.append(q)
                        peaks = peaks_refined
                    foci_rows, cell_foci_ids, next_foci_id = assign_peaks_to_cells(
                        peaks, mask_cleaned, dilation_px=1, frame_index=frame_idx, start_foci_id=next_foci_id
                    )
                    if foci_detection_mode == 0 and use_per_cell and cell_factor is not None and cell_factor > 0 and filtered and get_focus_blob_mask is not None:
                        # Per-cell filter: keep only foci where (focus blob mean) / (mean of cell excluding focus) >= cell_factor.
                        fi_flat = np.asarray(foci_plane, dtype=np.float64)
                        new_foci_rows = []
                        for row in foci_rows:
                            if not row.get('in_cell') or row.get('cell_id') is None:
                                new_foci_rows.append(row)
                                continue
                            cell_id = row['cell_id']
                            yi, xi = int(row.get('y', 0)), int(row.get('x', 0))
                            blob_mask = get_focus_blob_mask(
                                fi_flat, yi, xi,
                                threshold_frac=foci_blob_threshold, max_radius=5,
                                labeled_mask=mask_cleaned,
                            )
                            rest_of_cell = (mask_cleaned == cell_id) & ~blob_mask
                            n_rest = int(np.sum(rest_of_cell))
                            if n_rest == 0:
                                continue
                            rest_mean = float(np.mean(fi_flat[rest_of_cell]))
                            if not np.isfinite(rest_mean) or rest_mean <= 0:
                                continue
                            focus_mean = row.get('mean_intensity')
                            if focus_mean is None or not np.isfinite(focus_mean):
                                continue
                            ratio = float(focus_mean) / rest_mean
                            if ratio >= cell_factor:
                                new_foci_rows.append(row)
                        foci_rows = new_foci_rows
                        cell_foci_ids = {}
                        for row in foci_rows:
                            if row.get('in_cell') and row.get('cell_id') is not None:
                                key = (row['cell_id'], frame_idx)
                                cell_foci_ids.setdefault(key, []).append(row['foci_id'])
                    # Attach foci count and foci_ids to each cell (key by label, frame_idx)
                    for cell in filtered:
                        key = (cell['label'], frame_idx)
                        ids = cell_foci_ids.get(key, [])
                        cell['foci_count'] = len(ids)
                        cell['foci_ids'] = ids
                    apply_foci_spatial_to_frame(
                        filtered,
                        foci_rows,
                        mask_cleaned,
                        pixel_size,
                        contour_thickness=fi_contour_thickness if fi_do_contour else 1,
                    )
                    for row in foci_rows:
                        if foci_exclude_outside_cells:
                            if not row.get('in_cell', False) or row.get('cell_id') not in accepted_labels:
                                continue
                        r = row.copy()
                        r['channel_name'] = fi_channel_name or 'FI'
                        if foci_detection_mode == 2:
                            r['detection_method'] = 'spotiflow'
                        foci_table.append(r)
        
        # Determine rejected labels
        rejected_labels = all_labels - accepted_labels
        
        # Calculate total area
        total_area_um2 = sum(cell['area_um2'] for cell in filtered)
        num_cells = len(filtered)
        
        # Store frame data with label information
        all_frame_data.append({
            'frame': frame_idx + 1,
            'num_cells': num_cells,
            'total_area_um2': total_area_um2,
            'cells': filtered,
            'accepted_labels': accepted_labels,
            'rejected_labels': rejected_labels,
            'all_labels': all_labels
        })
        
        results.append({
            'frame': frame_idx + 1,
            'total_area_um2': total_area_um2,
            'num_cells': num_cells
        })
        done_frames = frame_idx + 1
        if done_frames == 1 or done_frames == num_frames or done_frames % frame_progress_every == 0:
            SCRIPT_LOG.info(
                "[ANALYZE_PROGRESS] phase=frames current=%s total=%s",
                done_frames,
                num_frames,
            )
    
    # Calculate relative areas (normalized to first frame)
    if len(results) > 0 and results[0]['total_area_um2'] > 0:
        first_frame_area = results[0]['total_area_um2']
        for result in results:
            result['total_area_relative'] = result['total_area_um2'] / first_frame_area
    else:
        SCRIPT_LOG.warning("First frame has zero area, cannot calculate relative values")
        for result in results:
            result['total_area_relative'] = 0.0
    
    # Check if input masks already have persistent CellIDs (e.g. from "Assign persistent IDs" after segmentation)
    total_unique = len(np.unique(masks))
    per_frame_uniques = [len(np.unique(masks[i])) for i in range(num_frames)]
    avg_per_frame = sum(per_frame_uniques) / len(per_frame_uniques) if per_frame_uniques else 0
    input_has_persistent_ids = avg_per_frame > 0 and total_unique <= 2.0 * avg_per_frame

    if input_has_persistent_ids:
        # Keep existing CellIDs; only filter (accepted = keep ID, rejected = remove from output mask)
        SCRIPT_LOG.info("Input masks already have persistent CellIDs; keeping them (filtering only).")
        for frame_data in all_frame_data:
            for cell in frame_data['cells']:
                cell['cell_id'] = cell['label']  # existing label is the CellID
        masks_with_cellids = np.zeros_like(masks, dtype=masks.dtype)
        for frame_idx, frame_data in enumerate(all_frame_data):
            for label in frame_data['accepted_labels']:
                masks_with_cellids[frame_idx][masks[frame_idx] == label] = label
        if masks_with_cellids.max() < 65535 and masks_with_cellids.dtype != np.uint16:
            masks_with_cellids = masks_with_cellids.astype(np.uint16)
    else:
        # Assign new persistent CellIDs to accepted cells (original behaviour)
        SCRIPT_LOG.info("Assigning persistent CellIDs to accepted cells...")
        SCRIPT_LOG.info("  Processing %s frames...", len(all_frame_data))
        masks_with_cellids, cellid_mapping = assign_persistent_cell_ids(masks, all_frame_data, pixel_size)
        for frame_idx, frame_data in enumerate(all_frame_data):
            for cell in frame_data['cells']:
                mapping_key = (frame_idx, cell['label'])
                cell['cell_id'] = cellid_mapping.get(mapping_key)
    
    # Update foci_table: replace mask label with persistent cell_id
    if foci_table:
        label_to_cellid = {}
        for frame_idx, frame_data in enumerate(all_frame_data):
            for cell in frame_data['cells']:
                cid = cell.get('cell_id')
                if cid is not None:
                    label_to_cellid[(frame_idx, cell['label'])] = cid
        for row in foci_table:
            if row.get('in_cell') and row.get('cell_id') is not None:
                frame_idx = row['frame']
                if isinstance(frame_idx, int) and frame_idx >= 1 and frame_idx <= len(all_frame_data):
                    # Legacy rows may store 1-based frame numbers.
                    frame_idx = frame_idx - 1
                key = (int(frame_idx), row['cell_id'])
                row['cell_id'] = label_to_cellid.get(key, row['cell_id'])
    
    apply_temporal_metrics(all_frame_data)
    
    # Return both masks_with_cellids (for tracking/analysis) and original masks (for visualization)
    return results, all_frame_data, masks_with_cellids, masks, foci_table


def link_persistent_cell_ids_raw(masks, pixel_size=0.108, max_distance_um=5.0):
    """
    Assign persistent CellIDs to all cells in a mask stack (no filtering).
    Use this right after segmentation so the same cell keeps the same ID through the pipeline.
    Frame-to-frame linking by Hungarian assignment on centroid distance.

    Parameters:
    -----------
    masks : array
        Mask array (T, Y, X) with per-frame instance labels (0, 1, 2, ... per frame).
    pixel_size : float
        Pixel size in µm for distance calculations.
    max_distance_um : float
        Maximum distance in µm to link the same cell between frames.

    Returns:
    --------
    masks_with_cellids : array
        Mask array with persistent CellIDs (1, 2, 3, ... across frames).
    """
    from bactomate.modules.cell_id_linking import centroids_from_mask, hungarian_link_centroids

    num_frames = masks.shape[0]
    masks_with_cellids = np.zeros_like(masks, dtype=np.uint32)
    next_cell_id = 1
    frame_centroids_cache = {}
    frame_cellids_cache = {}
    max_distance_pixels = max_distance_um / pixel_size

    mask_0 = masks[0]
    frame_centroids_cache[0] = centroids_from_mask(mask_0, pixel_size=pixel_size)
    frame_cellids_cache[0] = {}
    for label in sorted(frame_centroids_cache[0]):
        frame_cellids_cache[0][label] = next_cell_id
        masks_with_cellids[0][mask_0 == label] = next_cell_id
        next_cell_id += 1

    for frame_idx in range(1, num_frames):
        mask = masks[frame_idx]
        centroids_current = centroids_from_mask(mask, pixel_size=pixel_size)
        frame_centroids_cache[frame_idx] = centroids_current
        frame_cellids_cache[frame_idx] = {}

        prev_frame_idx = frame_idx - 1
        centroids_prev = {}
        for prev_label, prev_cellid in frame_cellids_cache[prev_frame_idx].items():
            if prev_label in frame_centroids_cache[prev_frame_idx]:
                centroids_prev[prev_cellid] = frame_centroids_cache[prev_frame_idx][prev_label]

        current_labels = sorted(centroids_current.keys())
        if centroids_prev and current_labels:
            prev_cellids = np.array(list(centroids_prev.keys()), dtype=int)
            prev_points = np.array(list(centroids_prev.values()), dtype=np.float64)
            curr_labels_arr = np.array(current_labels, dtype=int)
            curr_points = np.array([centroids_current[l] for l in current_labels], dtype=np.float64)

            linked, unlinked = hungarian_link_centroids(
                prev_points,
                prev_cellids,
                curr_points,
                curr_labels_arr,
                max_distance_pixels,
            )
            for prev_cellid, current_label in linked:
                frame_cellids_cache[frame_idx][current_label] = prev_cellid
                masks_with_cellids[frame_idx][mask == current_label] = prev_cellid
            for current_label in unlinked:
                frame_cellids_cache[frame_idx][current_label] = next_cell_id
                masks_with_cellids[frame_idx][mask == current_label] = next_cell_id
                next_cell_id += 1
        else:
            for current_label in current_labels:
                frame_cellids_cache[frame_idx][current_label] = next_cell_id
                masks_with_cellids[frame_idx][mask == current_label] = next_cell_id
                next_cell_id += 1

    if masks_with_cellids.max() < 65535:
        masks_with_cellids = masks_with_cellids.astype(np.uint16)
    SCRIPT_LOG.info(f"  Assigned {next_cell_id - 1} unique CellIDs across {num_frames} frames (all cells linked)")
    return masks_with_cellids


def assign_persistent_cell_ids(masks, all_frame_data, pixel_size=0.108, max_distance_um=5.0):
    """
    Assign persistent CellIDs to accepted cells across frames using Hungarian linking.
    Optimized version with centroid caching for better performance.
    
    Parameters:
    -----------
    masks : array
        Original mask array (T, Y, X) with frame-specific labels
    all_frame_data : list of dict
        Frame data with accepted/rejected labels
    pixel_size : float
        Pixel size in µm for distance calculations
    max_distance_um : float
        Maximum distance in µm to link cells between frames
        
    Returns:
    --------
    masks_with_cellids : array
        Mask array with persistent CellIDs
    cellid_mapping : dict
        Mapping from (frame_idx, original_label) to cell_id
    """
    from bactomate.modules.cell_id_linking import centroids_from_mask, hungarian_link_centroids
    
    num_frames = masks.shape[0]
    masks_with_cellids = np.zeros_like(masks, dtype=np.uint32)  # Use uint32 for more cell IDs
    cellid_mapping = {}  # {(frame_idx, original_label): cell_id}
    next_cell_id = 1
    
    # Cache for centroids and cellids per frame (to avoid recalculating regionprops)
    frame_centroids_cache = {}  # {frame_idx: {label: centroid}}
    frame_cellids_cache = {}  # {frame_idx: {label: cell_id}}
    
    # Process first frame - assign initial CellIDs
    frame_0_data = all_frame_data[0]
    mask_0 = masks[0]
    accepted_labels_0 = set(frame_0_data['accepted_labels'])
    
    frame_centroids_cache[0] = centroids_from_mask(
        mask_0, labels=accepted_labels_0, pixel_size=pixel_size
    )
    frame_cellids_cache[0] = {}
    for label in sorted(frame_centroids_cache[0]):
        cellid_mapping[(0, label)] = next_cell_id
        frame_cellids_cache[0][label] = next_cell_id
        masks_with_cellids[0][mask_0 == label] = next_cell_id
        next_cell_id += 1
    
    # Link cells across subsequent frames
    max_distance_pixels = max_distance_um / pixel_size
    
    # Progress indicator for large datasets
    print_every = max(1, num_frames // 20)  # Print progress every 5%
    
    for frame_idx in range(1, num_frames):
        if frame_idx % print_every == 0:
            SCRIPT_LOG.info(f"  Processing frame {frame_idx + 1}/{num_frames}...")
        frame_data = all_frame_data[frame_idx]
        mask = masks[frame_idx]
        accepted_labels = set(frame_data['accepted_labels'])
        
        centroids_current = centroids_from_mask(
            mask, labels=accepted_labels, pixel_size=pixel_size
        )
        frame_centroids_cache[frame_idx] = centroids_current
        
        # Get centroids from previous frame with their CellIDs (use cache!)
        centroids_prev = {}
        prev_frame_idx = frame_idx - 1
        
        if prev_frame_idx in frame_cellids_cache:
            for prev_label, prev_cellid in frame_cellids_cache[prev_frame_idx].items():
                if prev_label in frame_centroids_cache[prev_frame_idx]:
                    centroids_prev[prev_cellid] = frame_centroids_cache[prev_frame_idx][prev_label]
        
        current_labels = sorted(centroids_current.keys())
        frame_cellids_cache[frame_idx] = {}

        if centroids_prev and current_labels:
            prev_cellids = np.array(list(centroids_prev.keys()), dtype=int)
            prev_points = np.array(list(centroids_prev.values()), dtype=np.float64)
            curr_labels_arr = np.array(current_labels, dtype=int)
            curr_points = np.array([centroids_current[l] for l in current_labels], dtype=np.float64)

            linked, unlinked = hungarian_link_centroids(
                prev_points,
                prev_cellids,
                curr_points,
                curr_labels_arr,
                max_distance_pixels,
            )
            for prev_cellid, current_label in linked:
                cellid_mapping[(frame_idx, current_label)] = prev_cellid
                frame_cellids_cache[frame_idx][current_label] = prev_cellid
                masks_with_cellids[frame_idx][mask == current_label] = prev_cellid
            for current_label in unlinked:
                cellid_mapping[(frame_idx, current_label)] = next_cell_id
                frame_cellids_cache[frame_idx][current_label] = next_cell_id
                masks_with_cellids[frame_idx][mask == current_label] = next_cell_id
                next_cell_id += 1
        else:
            for current_label in current_labels:
                cellid_mapping[(frame_idx, current_label)] = next_cell_id
                frame_cellids_cache[frame_idx][current_label] = next_cell_id
                masks_with_cellids[frame_idx][mask == current_label] = next_cell_id
                next_cell_id += 1
    
    # Convert back to uint16 if needed (for compatibility)
    if masks_with_cellids.max() < 65535:
        masks_with_cellids = masks_with_cellids.astype(np.uint16)
    
    SCRIPT_LOG.info(f"  Assigned {next_cell_id - 1} unique CellIDs across {num_frames} frames")
    return masks_with_cellids, cellid_mapping


def _draw_foci_on_frame(vis_frame, foci_rows, color_rgb, radius=2):
    """Draw foci as small circles on an RGB frame (H, W, 3). foci_rows: list of dict with 'y', 'x' (int)."""
    h, w = vis_frame.shape[0], vis_frame.shape[1]
    r = max(1, min(radius, 8))
    for row in foci_rows:
        y, x = int(row.get('y', 0)), int(row.get('x', 0))
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2:
                    vis_frame[yy, xx, :] = color_rgb


def _normalize_plane_for_visualization_u8(plane):
    """Normalize a 2D image plane to uint8 [0..255] for display."""
    try:
        img = np.asarray(coerce_plane_to_2d(np.asarray(plane)), dtype=np.float32)
    except Exception:
        return None
    if img is None or img.ndim != 2 or img.size == 0:
        return None
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros(img.shape, dtype=np.uint8)
    vals = img[finite]
    # Robust normalization: suppress hot-pixel outliers.
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    out = (np.clip((img - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


def create_filter_visualization(
    masks,
    all_frame_data,
    output_path,
    masks_original=None,
    foci_table=None,
    background_images=None,
):
    """
    Create visualization showing accepted (green) and rejected (red) cells with persistent CellIDs.
    When foci_table is provided (Run foci), detected foci are drawn as yellow dots.
    
    Parameters:
    -----------
    masks : array
        Mask array (T, Y, X) with persistent CellIDs (for accepted cells only)
    all_frame_data : list of dict
        Frame data with cells containing cell_id and accepted/rejected labels
    output_path : str
        Path to save visualization TIFF
    masks_original : array, optional
        Original mask array (T, Y, X) with all labels (accepted + rejected) before CellID assignment
        If provided, used to show rejected cells. If None, only accepted cells will be shown.
    foci_table : list of dict, optional
        Foci rows (frame 0-based, y, x, ...). When provided, foci are drawn as yellow circles.
    background_images : array, optional
        Optional grayscale/intensity stack (T, Y, X) used as normalized background.
        If provided, accepted/rejected masks are alpha-tinted over the normalized image.
    """
    SCRIPT_LOG.info("Creating filter visualization...")
    
    num_frames = masks.shape[0]
    height, width = masks.shape[1], masks.shape[2]
    
    # Create RGB visualization with CellID labels
    visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    
    # Try to import PIL for text rendering
    try:
        from PIL import Image
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
        SCRIPT_LOG.warning("Pillow missing (required dependency); labels disabled. Run: pip install -r requirements.txt")
    
    for frame_idx in tqdm(range(num_frames), desc="Creating visualization"):
        mask = masks[frame_idx]  # Mask with persistent CellIDs (accepted cells only)
        frame_data = all_frame_data[frame_idx]
        
        # Get accepted CellIDs and rejected labels from frame data
        accepted_cellids = {cell.get('cell_id') for cell in frame_data['cells'] if cell.get('cell_id') is not None}
        rejected_labels = frame_data.get('rejected_labels', set())
        
        # Use original mask if provided to show rejected cells
        if masks_original is not None:
            mask_orig = masks_original[frame_idx]
        else:
            mask_orig = mask
        
        accepted_arr = np.array([cid for cid in accepted_cellids if cid is not None], dtype=np.int64)
        accepted_labels_img = (
            np.where(np.isin(mask, accepted_arr), mask, 0).astype(np.int32)
            if accepted_arr.size > 0
            else np.zeros((height, width), dtype=np.int32)
        )
        rejected_arr = np.array([lab for lab in rejected_labels if lab > 0], dtype=np.int64)
        rejected_labels_img = (
            np.where(np.isin(mask_orig, rejected_arr), mask_orig, 0).astype(np.int32)
            if rejected_arr.size > 0
            else np.zeros((height, width), dtype=np.int32)
        )
        accepted_contour = find_boundaries(accepted_labels_img, mode="inner") if accepted_arr.size > 0 else np.zeros((height, width), dtype=bool)
        rejected_contour = find_boundaries(rejected_labels_img, mode="inner") if rejected_arr.size > 0 else np.zeros((height, width), dtype=bool)

        bg_u8 = None
        if background_images is not None and frame_idx < background_images.shape[0]:
            bg_u8 = _normalize_plane_for_visualization_u8(background_images[frame_idx])

        if bg_u8 is None or bg_u8.shape != (height, width):
            frame_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame_rgb = np.stack([bg_u8, bg_u8, bg_u8], axis=-1).astype(np.uint8)

        # Draw cell contours only (no full-area fill)
        if np.any(rejected_contour):
            frame_rgb[rejected_contour] = np.array([255, 0, 0], dtype=np.uint8)
        if np.any(accepted_contour):
            frame_rgb[accepted_contour] = np.array([0, 255, 0], dtype=np.uint8)

        visualization[frame_idx] = frame_rgb
        
        foci_in_frame = [r for r in foci_table if r.get('frame') == frame_idx] if foci_table else []
        # Overlay detected foci as yellow dots when Run foci was used
        if foci_in_frame:
            _draw_foci_on_frame(visualization[frame_idx], foci_in_frame, (255, 255, 0), radius=2)
        
        # Add CellID and foci_id labels if PIL is available
        if PIL_AVAILABLE and (mask.max() > 0 or foci_in_frame):
            try:
                # Convert to PIL Image
                pil_img = Image.fromarray(visualization[frame_idx])
                
                labels_to_draw = []
                if mask.max() > 0:
                    # Get cell centroids and CellIDs (masks now have persistent CellIDs)
                    from skimage.measure import regionprops
                    regions = regionprops(mask)
                    for region in regions:
                        if region.label == 0:
                            continue
                        
                        # Get centroid
                        centroid_y, centroid_x = region.centroid
                        x, y = int(centroid_x), int(centroid_y)
                        
                        cell_id_str = str(region.label)
                        text_x, text_y = x + 2, y - 5
                        labels_to_draw.append((cell_id_str, text_x, text_y))

                if foci_in_frame:
                    for row in foci_in_frame:
                        fid = row.get('foci_id')
                        if fid is None:
                            continue
                        x, y = int(row.get('x', 0)), int(row.get('y', 0))
                        labels_to_draw.append((f"F{fid}", x + 3, y - 3))

                if labels_to_draw:
                    pil_img = draw_hq_labels(
                        pil_img,
                        labels_to_draw,
                        font_size=12,
                        scale=2,
                        stroke_width=2,
                    )
                    visualization[frame_idx] = np.array(pil_img)
            except Exception:
                # If text rendering fails, continue without labels
                pass
    
    # Save visualization
    SCRIPT_LOG.info("Saving visualization to: %s", output_path)
    tifffile.imwrite(output_path, visualization)
    try:
        from bactomate.modules.pipeline_outputs import write_png_preview_from_tiff

        write_png_preview_from_tiff(output_path)
    except Exception:
        pass
    SCRIPT_LOG.info("Visualization saved")
    total_accepted = sum(len(fd['accepted_labels']) for fd in all_frame_data)
    total_rejected = sum(len(fd['rejected_labels']) for fd in all_frame_data)
    rate = 100 * total_accepted / (total_accepted + total_rejected) if (total_accepted + total_rejected) > 0 else 0
    SCRIPT_LOG.info("Visualization statistics: accepted %s, rejected %s, rate %.1f%%", total_accepted, total_rejected, rate)


def create_fi_contour_visualization(masks, all_frame_data, fi_images, output_path, masks_original=None, foci_table=None):
    """
    Create visualization showing cell masks with FI contours highlighted.
    When foci_table is provided (Run foci), detected foci are drawn as cyan dots.
    
    Parameters:
    -----------
    masks : array
        Mask array (T, Y, X) with persistent CellIDs (for accepted cells only)
    all_frame_data : list of dict
        Frame data with cells containing cell_id and accepted/rejected labels
    fi_images : array
        FI channel images (T, Y, X)
    output_path : str
        Path to save visualization TIFF
    masks_original : array, optional
        Original mask array before CellID assignment (for showing all cells)
    foci_table : list of dict, optional
        Foci rows (frame 0-based, y, x, ...). When provided, foci are drawn as cyan circles.
    """
    SCRIPT_LOG.info("Creating FI contour visualization...")
    
    num_frames = masks.shape[0]
    height, width = masks.shape[1], masks.shape[2]
    
    # Create RGB visualization
    visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    
    # Normalize FI images for background display (percentile 1-99)
    fi_images_norm = []
    for fi_img in fi_images:
        fi_norm = _normalize_plane_for_visualization_u8(fi_img)
        if fi_norm is None:
            fi_norm = np.zeros((height, width), dtype=np.uint8)
        fi_images_norm.append(fi_norm)
    fi_images_norm = np.stack(fi_images_norm)
    
    for frame_idx in tqdm(range(num_frames), desc="Creating contour visualization"):
        mask = masks[frame_idx]  # Mask with persistent CellIDs (accepted cells only)
        frame_data = all_frame_data[frame_idx]
        fi_image = fi_images_norm[frame_idx]
        
        # Get accepted CellIDs
        accepted_cellids = {cell.get('cell_id') for cell in frame_data['cells'] if cell.get('cell_id') is not None}
        
        # Use original mask if provided to show all cells
        if masks_original is not None:
            mask_orig = masks_original[frame_idx]
        else:
            mask_orig = mask
        
        # Start with FI image as background (grayscale in all channels)
        visualization[frame_idx, :, :, 0] = fi_image  # Red channel
        visualization[frame_idx, :, :, 1] = fi_image  # Green channel
        visualization[frame_idx, :, :, 2] = fi_image  # Blue channel
        
        # Draw cell masks in green (semi-transparent overlay)
        for cell_id in accepted_cellids:
            if cell_id is not None:
                cell_mask = (mask == cell_id)
                # Blend green with background (50% opacity)
                visualization[frame_idx, :, :, 1][cell_mask] = np.clip(
                    visualization[frame_idx, :, :, 1][cell_mask] * 0.5 + 255 * 0.5, 0, 255
                ).astype(np.uint8)
        
        # Find and draw contours in bright yellow/cyan
        # Find boundaries for all accepted cells
        boundaries = find_boundaries(mask, mode='inner')
        
        # Draw contours in bright yellow (red + green channels)
        contour_mask = boundaries & (mask > 0)
        visualization[frame_idx, :, :, 0][contour_mask] = 255  # Red channel
        visualization[frame_idx, :, :, 1][contour_mask] = 255  # Green channel
        visualization[frame_idx, :, :, 2][contour_mask] = 0     # Blue channel (yellow = red+green)
        
        # Overlay detected foci as cyan dots when Run foci was used
        if foci_table:
            foci_in_frame = [r for r in foci_table if r.get('frame') == frame_idx]
            if foci_in_frame:
                _draw_foci_on_frame(visualization[frame_idx], foci_in_frame, (0, 255, 255), radius=2)
    
    # Save visualization
    SCRIPT_LOG.info("Saving FI contour visualization to: %s", output_path)
    tifffile.imwrite(output_path, visualization)
    try:
        from bactomate.modules.pipeline_outputs import write_png_preview_from_tiff

        write_png_preview_from_tiff(output_path)
    except Exception:
        pass
    SCRIPT_LOG.info("FI contour visualization saved")


def save_to_excel(results, output_path, detailed_data=None, foci_table=None, fi_channel_name=None, export_csv=False, pixel_size=0.108):
    """
    Save results to Excel file and optionally to CSV files.
    
    Parameters:
    -----------
    results : list of dict
        Results with frame, total_area_um2, total_area_relative
    output_path : str
        Path to save Excel file
    detailed_data : list of dict, optional
        Detailed per-cell data for additional sheets
    foci_table : list of dict, optional
        Foci rows (foci_id, cell_id, frame, channel_name, y, x, intensity, size_px, mean_intensity, rel_intensity, in_cell)
    fi_channel_name : str, optional
        When set (from FI filename, e.g. GFP), Per-Cell FI/foci columns include this tag in the header
        (e.g. ``Mean FI Intensity (whole cell) [GFP]``) so multi-channel batch runs are distinguishable.
        Also used for foci count / foci_ids column prefixes (e.g. GFP_foci_count).
    export_csv : bool, optional
        If True, also write Summary, Per-Cell Data, and Foci (if present) as CSV files
        next to the Excel file (same stem with _summary.csv, _per_cell.csv, _foci.csv).
    pixel_size : float, optional
        Pixel size in µm; used to add focus area (µm²) = area_px * pixel_size² to the Foci sheet (default 0.108).
    """
    output_path = Path(output_path)
    # Create main results DataFrame
    df_main = pd.DataFrame(results)
    
    # Reorder columns
    df_main = df_main[['frame', 'total_area_um2', 'total_area_relative', 'num_cells']]
    
    # Rename columns for clarity
    df_main.columns = ['Frame', 'Total Cell Area (µm²)', 'Total Cell Area (Relative)', 'Number of Cells']
    
    # Per-cell foci aggregates and per-focus lists from foci_table (frame in foci_table is 0-based; we use 1-based to match Summary/Per-Cell)
    foci_per_cell_agg = {}
    foci_per_cell_list = {}  # (frame_1based, cell_id) -> list of {foci_id, mean_intensity, size_px} in foci_id order
    if foci_table and len(foci_table) > 0:
        for r in foci_table:
            if not r.get('in_cell') or r.get('cell_id') is None:
                continue
            frame_1based = r.get('frame', 0) + 1 if isinstance(r.get('frame'), int) else r.get('frame')
            key = (frame_1based, r['cell_id'])
            intensity = r.get('intensity')
            size_px = r.get('size_px', 1)
            mean_int = r.get('mean_intensity')
            if key not in foci_per_cell_agg:
                foci_per_cell_agg[key] = {'intensities': [], 'sizes': []}
                foci_per_cell_list[key] = []
            if intensity is not None and (isinstance(intensity, (int, float)) and np.isfinite(intensity)):
                foci_per_cell_agg[key]['intensities'].append(float(intensity))
            foci_per_cell_agg[key]['sizes'].append(size_px if size_px is not None else 1)
            rel_int = r.get('rel_intensity')
            foci_per_cell_list[key].append({
                'foci_id': r.get('foci_id'),
                'intensity': intensity if intensity is not None and np.isfinite(intensity) else np.nan,
                'mean_intensity': mean_int if mean_int is not None and np.isfinite(mean_int) else np.nan,
                'rel_intensity': rel_int if rel_int is not None and np.isfinite(rel_int) else np.nan,
                'size_px': size_px if size_px is not None else 1,
            })
        for k, v in foci_per_cell_agg.items():
            intensities = v['intensities']
            sizes = v['sizes']
            v['mean_intensity'] = np.mean(intensities) if intensities else np.nan
            v['sum_intensity'] = np.sum(intensities) if intensities else np.nan
            v['max_intensity'] = np.max(intensities) if intensities else np.nan
            v['mean_size_px'] = np.mean(sizes) if sizes else np.nan
            v['total_area_px'] = np.sum(sizes) if sizes else np.nan
        # Sort per-cell focus lists by foci_id so order matches foci_ids column
        for k in foci_per_cell_list:
            foci_per_cell_list[k].sort(key=lambda x: (x['foci_id'] is None, x['foci_id'] or 0))
    
    # Add Summary sheet foci columns when run_foci was on
    if foci_table and len(foci_table) > 0:
        frame_to_foci = {}
        for r in foci_table:
            if not r.get('in_cell'):
                continue
            frame_1based = r.get('frame', 0) + 1 if isinstance(r.get('frame'), int) else r.get('frame')
            frame_to_foci.setdefault(frame_1based, []).append(r)
        total_foci = []
        mean_foci_per_cell = []
        for _, row in df_main.iterrows():
            f = int(row['Frame'])
            n_cells = int(row['Number of Cells'])
            n_foci = len(frame_to_foci.get(f, []))
            total_foci.append(n_foci)
            mean_foci_per_cell.append(n_foci / n_cells if n_cells > 0 else np.nan)
        df_main['Total Foci'] = total_foci
        df_main['Mean Foci per Cell'] = mean_foci_per_cell

        def _foci_row_summary_frame(r: dict) -> object:
            fr = r.get("frame")
            if isinstance(fr, int):
                return int(fr) + 1
            try:
                if fr is not None and isinstance(fr, (float, np.floating)) and np.isfinite(fr):
                    return int(fr) + 1
            except Exception:
                pass
            return fr

        chans = sorted(
            {
                str(r.get("channel_name") or fi_channel_name or "FI").strip() or "FI"
                for r in foci_table
            },
            key=lambda s: s.lower(),
        )
        if len(chans) > 1:
            from bactomate.gui.analysis_workbook import apply_foci_summary_columns

            for ch in chans:
                totals_by_frame: dict[int, int] = {}
                within_by_frame: dict[int, int] = {}
                for _, row in df_main.iterrows():
                    f = int(row["Frame"])
                    n_tot = 0
                    n_in = 0
                    for r in foci_table:
                        rch = str(r.get("channel_name") or fi_channel_name or "FI").strip() or "FI"
                        if rch != ch:
                            continue
                        try:
                            sf = _foci_row_summary_frame(r)
                            if sf is None or (isinstance(sf, float) and not np.isfinite(sf)):
                                continue
                            sf_i = int(sf)
                        except Exception:
                            continue
                        if sf_i != f:
                            continue
                        n_tot += 1
                        if bool(r.get("in_cell")) and r.get("cell_id") not in (None, "", "None"):
                            try:
                                if int(r["cell_id"]) > 0:
                                    n_in += 1
                            except Exception:
                                pass
                    totals_by_frame[f] = int(n_tot)
                    within_by_frame[f] = int(n_in)
                df_main = apply_foci_summary_columns(
                    df_main,
                    label=ch,
                    totals_by_frame=totals_by_frame,
                    within_by_frame=within_by_frame,
                )

    if detailed_data is not None:
        by_frame = {int(fd['frame']): fd for fd in detailed_data}
        pop_rows = []
        for _, row in df_main.iterrows():
            fd = by_frame.get(int(row['Frame']))
            cells = fd.get('cells') if fd else []
            pop_rows.append(frame_summary_spatial_metrics(cells or []))
        if pop_rows:
            pop_df = pd.DataFrame(pop_rows)
            for col in pop_df.columns:
                df_main[col] = pop_df[col].values
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main summary sheet
        df_main.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed per-cell data (optional)
        df_detailed = None
        # IMPORTANT: when the caller requests detailed output, always create the
        # "Per-Cell Data" sheet even if it's empty. Downstream post-analysis steps
        # (e.g. colocalization) depend on the sheet existing.
        if detailed_data is not None:
            detailed_rows = []
            ch_name = fi_channel_name or 'FI'
            foci_count_col = f'{ch_name}_foci_count'
            foci_ids_col = f'foci_ids_{ch_name}'
            ch_tag = (fi_channel_name or "").strip() or None

            def _s(base: str) -> str:
                """Suffix Per-Cell column names when a channel tag is known (multi-FI exports)."""
                return f"{base} [{ch_tag}]" if ch_tag else base

            for frame_data in detailed_data:
                frame_num = frame_data['frame']
                for cell in frame_data['cells']:
                    row = {
                        'Frame': frame_num,
                        'Cell_ID': cell.get('cell_id', cell.get('label', None)),  # Use persistent CellID if available
                        'Cell Label': cell['label'],  # Keep original label for reference
                        'Area (µm²)': cell['area_um2'],
                        'Length (µm)': cell['length_um'],
                        'Width (µm)': cell['width_um'],
                        'Aspect Ratio': cell['aspect_ratio'],
                        'Eccentricity': cell['eccentricity'],
                        'Solidity': cell['solidity'],
                        'Centroid Y': cell.get('centroid_y'),
                        'Centroid X': cell.get('centroid_x'),
                        'Orientation (deg)': cell.get('orientation_deg'),
                        'Perimeter (µm)': cell.get('perimeter_um'),
                        'Circularity': cell.get('circularity'),
                    }
                    # Add FI columns if available (contour metrics may be disabled)
                    if 'mean_fi' in cell:
                        if ch_tag:
                            row[_s('Mean FI Intensity (whole cell)')] = cell['mean_fi']
                        else:
                            row['Mean FI Intensity'] = cell['mean_fi']
                    if 'interior_fi' in cell:
                        row[_s('Interior FI Mean')] = cell.get('interior_fi')
                    if 'fi_std' in cell:
                        row[_s('FI Std')] = cell.get('fi_std')
                    if 'fi_cv' in cell:
                        row[_s('FI CV')] = cell.get('fi_cv')
                    if 'fi_top10_mean' in cell:
                        row[_s('FI Top 10% Mean')] = cell.get('fi_top10_mean')
                    if 'contour_fi' in cell:
                        row[_s('Contour FI Intensity')] = cell['contour_fi']
                    if 'fi_ratio' in cell:
                        row[_s('FI Ratio (Contour/Mean)')] = cell['fi_ratio']
                    # Foci columns: count and IDs (existing)
                    if 'foci_count' in cell:
                        row[_s('Number of Foci')] = cell['foci_count']
                        row[foci_count_col] = cell['foci_count']
                    if 'foci_ids' in cell:
                        row[foci_ids_col] = ','.join(map(str, cell['foci_ids'])) if cell['foci_ids'] else ''
                    # Foci intensity/size aggregates and per-focus value lists (when run_foci was on)
                    cell_id = cell.get('cell_id', cell.get('label'))
                    agg_key = (frame_num, cell_id) if cell_id is not None else None
                    ps = float(pixel_size) if pixel_size and pixel_size > 0 else 0.108
                    if agg_key and agg_key in foci_per_cell_agg:
                        agg = foci_per_cell_agg[agg_key]
                        row[_s('Foci Mean Intensity')] = agg['mean_intensity']
                        row[_s('Foci Total Intensity')] = agg['sum_intensity']
                        row[_s('Foci Max Intensity')] = agg['max_intensity']
                        row[_s('Foci Mean Area (px)')] = agg['mean_size_px']
                        row[_s('Foci Total Area (px)')] = agg['total_area_px']
                        # Foci mean intensity ratio to whole-cell mean FI (when both available)
                        mean_fi = cell.get('mean_fi')
                        if mean_fi is not None and isinstance(mean_fi, (int, float)) and np.isfinite(mean_fi) and mean_fi > 0:
                            fi_val = float(mean_fi)
                            foci_mean = agg.get('mean_intensity')
                            if foci_mean is not None and np.isfinite(foci_mean):
                                row[_s('Foci Mean Intensity Ratio')] = float(foci_mean) / fi_val
                        # Per-focus values (value1, value2, ...) in same order as foci_ids — every foci stat as list
                        if agg_key in foci_per_cell_list and cell.get('foci_ids'):
                            foci_list = foci_per_cell_list[agg_key]
                            by_id = {f['foci_id']: f for f in foci_list if f.get('foci_id') is not None}
                            mean_fi = cell.get('mean_fi')
                            fi_val = float(mean_fi) if mean_fi is not None and isinstance(mean_fi, (int, float)) and np.isfinite(mean_fi) and mean_fi > 0 else None
                            peak_vals = []
                            mean_vals = []
                            area_px_vals = []
                            area_um2_vals = []
                            rel_vals = []
                            ratio_vals = []
                            for fid in cell['foci_ids']:
                                f = by_id.get(fid)
                                if f is not None:
                                    peak = f.get('intensity')
                                    peak_vals.append(f"{peak:.4g}" if peak is not None and np.isfinite(peak) else "")
                                    mi = f.get('mean_intensity')
                                    mean_vals.append(f"{mi:.4g}" if mi is not None and np.isfinite(mi) else "")
                                    sx = f.get('size_px') or 1
                                    area_px_vals.append(f"{sx:.4g}")
                                    area_um2 = sx * (ps ** 2)
                                    area_um2_vals.append(f"{area_um2:.4g}")
                                    ri = f.get('rel_intensity')
                                    rel_vals.append(f"{ri:.4g}" if ri is not None and np.isfinite(ri) else "")
                                    if fi_val and mi is not None and np.isfinite(mi):
                                        ratio_vals.append(f"{mi / fi_val:.4g}")
                                    else:
                                        ratio_vals.append("")
                                else:
                                    peak_vals.append("")
                                    mean_vals.append("")
                                    area_px_vals.append("")
                                    area_um2_vals.append("")
                                    rel_vals.append("")
                                    ratio_vals.append("")
                            row[_s('Foci intensities (peak)')] = ", ".join(peak_vals) if peak_vals else ""
                            row[_s('Foci mean intensities')] = ", ".join(mean_vals) if mean_vals else ""
                            row[_s('Foci areas (px)')] = ", ".join(area_px_vals) if area_px_vals else ""
                            row[_s('Foci areas (µm²)')] = ", ".join(area_um2_vals) if area_um2_vals else ""
                            row[_s('Foci rel intensities')] = ", ".join(rel_vals) if rel_vals else ""
                            row[_s('Foci mean intensity ratios')] = ", ".join(ratio_vals) if ratio_vals else ""
                    for src_key, col_name in (
                        ('foci_min_dist_contour_um', 'Min Foci-Contour Distance (µm)'),
                        ('foci_mean_dist_contour_um', 'Mean Foci-Contour Distance (µm)'),
                        ('foci_mean_longitudinal_position', 'Mean Foci Longitudinal Position'),
                        ('foci_longitudinal_spread', 'Foci Longitudinal Spread'),
                        ('foci_mean_radial_offset_um', 'Mean Foci Radial Offset (µm)'),
                        ('foci_min_dist_nearest_pole_um', 'Min Foci-Pole Distance (µm)'),
                        ('foci_nearest_neighbor_um', 'Foci Nearest-Neighbor Distance (µm)'),
                        ('foci_membrane_proximal_fraction', 'Membrane-Proximal Foci Fraction'),
                    ):
                        if src_key in cell:
                            row[_s(col_name)] = cell.get(src_key)
                    for src_key, col_name in (
                        ('delta_area_um2', 'Delta Area (µm²)'),
                        ('delta_area_um2_per_frame', 'Delta Area per Frame (µm²)'),
                        ('delta_mean_fi', 'Delta Mean FI'),
                        ('delta_foci_count', 'Delta Foci Count'),
                        ('delta_foci_min_dist_contour_um', 'Delta Min Foci-Contour Distance (µm)'),
                        ('delta_foci_mean_dist_contour_um', 'Delta Mean Foci-Contour Distance (µm)'),
                        ('delta_contour_fi', 'Delta Contour FI'),
                        ('delta_interior_fi', 'Delta Interior FI'),
                        ('time_of_first_focus', 'Time of First Focus (frame)'),
                    ):
                        if src_key in cell:
                            row[col_name] = cell.get(src_key)
                    detailed_rows.append(row)
            
            if detailed_rows:
                df_detailed = pd.DataFrame(detailed_rows)
            else:
                # Preserve a stable schema even when no cells exist (or all frames empty).
                df_detailed = pd.DataFrame(
                    columns=[
                        'Frame',
                        'Cell_ID',
                        'Cell Label',
                        'Area (µm²)',
                        'Length (µm)',
                        'Width (µm)',
                        'Aspect Ratio',
                        'Eccentricity',
                        'Solidity',
                    ]
                )
            df_detailed.to_excel(writer, sheet_name='Per-Cell Data', index=False)
        
        # Foci sheet (when run_foci was on)
        df_foci = None
        if foci_table is not None:
            df_foci = pd.DataFrame(foci_table)
            if len(df_foci) > 0:
                # Add area (µm²) = area_px * pixel_size²
                ps = float(pixel_size) if pixel_size and pixel_size > 0 else 0.108
                df_foci['area_um2'] = df_foci['size_px'].astype(float).mul(ps ** 2)
                # Reorder and rename columns for clarity
                col_map = {'size_px': 'Area (px)', 'area_um2': 'Area (µm²)', 'mean_intensity': 'Mean intensity (focus blob)', 'rel_intensity': 'Rel intensity (peak/frame median)'}
                spatial_rename = {
                    'distance_to_contour_um': 'Distance to contour (µm)',
                    'longitudinal_position': 'Longitudinal position',
                    'radial_offset_um': 'Radial offset (µm)',
                    'distance_to_nearest_pole_um': 'Distance to nearest pole (µm)',
                    'membrane_proximal': 'Membrane proximal',
                    'cell_orientation_deg': 'Cell orientation (deg)',
                }
                col_map.update(spatial_rename)
                df_foci = df_foci.rename(columns={k: v for k, v in col_map.items() if k in df_foci.columns})
                cols = [
                    'foci_id', 'cell_id', 'frame', 'channel_name', 'y', 'x', 'intensity',
                    'Area (px)', 'Area (µm²)', 'Mean intensity (focus blob)', 'Rel intensity (peak/frame median)',
                    'Distance to contour (µm)', 'Longitudinal position', 'Radial offset (µm)',
                    'Distance to nearest pole (µm)', 'Membrane proximal', 'Cell orientation (deg)',
                    'in_cell',
                ]
                df_foci = df_foci[[c for c in cols if c in df_foci.columns]]
            else:
                df_foci = pd.DataFrame(columns=['foci_id', 'cell_id', 'frame', 'channel_name', 'y', 'x', 'intensity', 'Area (px)', 'Area (µm²)', 'Mean intensity (focus blob)', 'Rel intensity (peak/frame median)', 'in_cell'])
            df_foci.to_excel(writer, sheet_name='Foci', index=False)

        try:
            from bactomate.modules.excel_autofit import autofit_openpyxl_workbook

            autofit_openpyxl_workbook(writer.book)
        except Exception:
            pass

    # Optionally write CSV files (same stem as Excel)
    if export_csv:
        stem = output_path.stem  # e.g. "foo_analysis"
        parent = output_path.parent
        df_main.to_csv(parent / f"{stem}_summary.csv", index=False)
        if df_detailed is not None:
            df_detailed.to_csv(parent / f"{stem}_per_cell.csv", index=False)
        if df_foci is not None and len(df_foci) > 0:
            df_foci.to_csv(parent / f"{stem}_foci.csv", index=False)
        SCRIPT_LOG.info("Results saved to: %s (and CSV)", output_path)
    else:
        SCRIPT_LOG.info("Results saved to: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze cell areas from Omnipose masks with morphological filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python analyze_cell_areas.py masks.tif
  
  # Specify pixel size and filtering parameters
  python analyze_cell_areas.py masks.tif --pixel-size 0.108 --max-area 50 --max-width 2.0
  
  # Stricter rod-shaped filtering
  python analyze_cell_areas.py masks.tif --min-aspect-ratio 2.0 --min-eccentricity 0.7
        """
    )
    
    parser.add_argument('masks', type=str, help='Path to mask TIFF file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output Excel file path (default: masks_filename_analysis.xlsx)')
    parser.add_argument('--pixel-size', type=float, default=0.108,
                       help='Pixel size in µm (default: 0.108)')
    
    # Rod-shaped filtering
    parser.add_argument('--min-aspect-ratio', type=float, default=1.5,
                       help='Minimum aspect ratio (length/width) for rod-shaped cells (default: 1.5)')
    parser.add_argument('--max-aspect-ratio', type=float, default=10.0,
                       help='Maximum aspect ratio to avoid artifacts (default: 10.0)')
    parser.add_argument('--min-eccentricity', type=float, default=0.5,
                       help='Minimum eccentricity (0=circle, 1=line) (default: 0.5)')
    parser.add_argument('--min-solidity', type=float, default=0.7,
                       help='Minimum solidity (area/convex_hull_area) (default: 0.7)')
    
    # Size filtering
    parser.add_argument('--min-area', type=float, default=None,
                       help='Minimum cell area in µm² (default: no limit)')
    parser.add_argument('--max-area', type=float, default=None,
                       help='Maximum cell area in µm² (default: no limit)')
    parser.add_argument('--min-width', type=float, default=None,
                       help='Minimum cell width in µm (default: no limit)')
    parser.add_argument('--max-width', type=float, default=None,
                       help='Maximum cell width in µm (default: no limit)')
    parser.add_argument('--min-length', type=float, default=None,
                       help='Minimum cell length in µm (default: no limit)')
    
    # Debris removal
    parser.add_argument('--min-size', type=int, default=10,
                       help='Minimum object size in pixels to keep (default: 10)')
    parser.add_argument('--no-remove-small', action='store_true',
                       help='Disable removal of small objects')
    
    # Detailed output
    parser.add_argument('--detailed', action='store_true',
                       help='Include detailed per-cell data in Excel output')
    parser.add_argument('--visualization', action='store_true', default=True,
                       help='Create visualization TIFF showing accepted (green) and rejected (red) cells (default: on)')
    parser.add_argument('--no-visualization', dest='visualization', action='store_false',
                       help='Skip filter/FI contour visualization TIFFs')
    
    # FI analysis
    parser.add_argument('--fi-image', type=str, default=None,
                       help='Path to FI channel TIFF file for fluorescence intensity analysis')
    parser.add_argument('--foci-image', type=str, default=None,
                       help='Optional path to TIFF file used only for foci detection (defaults to --fi-image)')
    parser.add_argument('--fi-contour-thickness', type=int, default=1,
                       help='Thickness of contour region in pixels for FI intensity calculation (default: 1)')
    parser.add_argument('--no-fi-contour', action='store_true',
                       help='Disable contour-based FI metrics (only compute Mean FI Intensity)')
    parser.add_argument('--exclude-border', action='store_true',
                       help='Exclude cells that touch the image border from counts and area')
    # Foci analysis
    parser.add_argument('--run-foci', action='store_true',
                       help='Run foci detection on the FI channel and add foci count and Foci sheet')
    parser.add_argument('--foci-detection-mode', type=int, default=0, choices=[0, 1, 2],
                       help='0=manual (use foci-min-rel etc.), 1=auto (data-driven), 2=Spotiflow deep learning (default: 0)')
    parser.add_argument('--foci-min-rel', type=float, default=2.0,
                       help='Min relative intensity factor for foci (default: 2.0)')
    parser.add_argument('--foci-min-rel-mode', type=str, default='per_cell', choices=['per_frame', 'per_cell', 'both'],
                       help='Apply min relative intensity per frame, per cell, or both (default: per_cell)')
    parser.add_argument('--foci-min-rel-cell', type=float, default=2.0,
                       help='Cell factor when --foci-min-rel-mode is both (focus / cell mean FI ≥ this value) (default: 2.0)')
    parser.add_argument('--foci-min-separation', type=int, default=2,
                       help='Min distance between foci in pixels (default: 2)')
    parser.add_argument('--foci-min-size', type=int, default=1,
                       help='Min focus size in pixels (default: 1)')
    parser.add_argument('--foci-max-size', type=int, default=None,
                       help='Max focus size in pixels (0 or omit = no limit)')
    parser.add_argument('--foci-min-peak-raw', type=float, default=0,
                       help='Min raw intensity at focus peak; 0 = off (default: 0)')
    parser.add_argument('--foci-min-focus-mean', type=float, default=0,
                       help='Min mean intensity of focus blob; 0 = off (default: 0)')
    parser.add_argument('--foci-blob-threshold', type=float, default=0.8,
                       help='Blob threshold fraction for focus area/mean around each peak (default: 0.8)')
    parser.add_argument('--foci-exclude-outside-cells', action='store_true',
                       help='Exclude foci that fall outside any cell from counts and Foci sheet')
    parser.add_argument('--spotiflow-config', type=str, default=None,
                       help='JSON file with Spotiflow options (same keys as GUI foci_spotiflow snapshot)')
    parser.add_argument('--foci-channel-name', type=str, default=None,
                       help='Channel name for foci columns/sheet (default: from FI filename, e.g. GFP)')
    parser.add_argument('--export-csv', action='store_true',
                       help='Also save Summary, Per-Cell Data, and Foci as CSV files next to the Excel file')
    
    args = parser.parse_args()
    _setup_script_logging(log_to_file=True)

    # Check input file
    if not Path(args.masks).exists():
        SCRIPT_LOG.error("Mask file not found: %s", args.masks)
        return 1
    
    # Set output path
    if args.output is None:
        input_path = Path(args.masks)
        output_path = input_path.parent / f"{input_path.stem}_analysis.xlsx"
    else:
        output_path = Path(args.output)
        # Ensure it has .xlsx extension
        if not output_path.suffix:
            output_path = output_path.with_suffix('.xlsx')
    
    # Check FI image if provided
    fi_image_path = None
    if args.fi_image:
        if not Path(args.fi_image).exists():
            SCRIPT_LOG.error("FI image file not found: %s", args.fi_image)
            return 1
        fi_image_path = args.fi_image
    foci_image_path = None
    if args.foci_image:
        if not Path(args.foci_image).exists():
            SCRIPT_LOG.error("Foci image file not found: %s", args.foci_image)
            return 1
        foci_image_path = args.foci_image
    
    spotiflow_options = None
    sc = getattr(args, "spotiflow_config", None)
    if sc:
        sc_path = Path(sc)
        if sc_path.is_file():
            try:
                with open(sc_path, encoding="utf-8") as sf:
                    spotiflow_options = json.load(sf)
            except Exception as e:
                SCRIPT_LOG.warning("Could not read --spotiflow-config %s: %s", sc_path, e)

    def _is_unlikely_channel_token(token: str) -> bool:
        t = (token or "").strip().lower()
        if not t:
            return True
        if t in {"fi", "foci", "mask", "masks", "phase", "bf", "dic", "seg", "segmentation"}:
            return True
        if t.startswith("t=") or t.startswith("z="):
            return True
        if t.endswith("x") and t[:-1].isdigit():
            return True
        return False

    def _infer_foci_channel_name(path_str: str) -> str:
        stem = Path(path_str).stem
        if stem.endswith("_fi"):
            return "FI"
        if "_fi_" in stem:
            rem = stem.split("_fi_", 1)[1].strip("_")
            if rem.startswith("foci_"):
                rem = rem[len("foci_") :].strip("_")
            if rem.endswith("_foci"):
                rem = rem[: -len("_foci")].strip("_")
            return rem or "FI"
        if stem.startswith("foci_"):
            rem = stem[len("foci_") :].strip("_")
            return rem or "FI"
        if "_foci_" in stem:
            rem = stem.split("_foci_", 1)[1].strip("_")
            return rem or "FI"
        if stem.endswith("_foci"):
            prefix = stem[: -len("_foci")].strip("_")
            if "_" in prefix:
                token = prefix.rsplit("_", 1)[-1].strip("_")
                if token and not _is_unlikely_channel_token(token):
                    return token
            return "FI"
        return "FI"

    # Foci channel name from foci image path (or FI path) if not set.
    fi_channel_name = args.foci_channel_name
    if fi_channel_name is None and (foci_image_path or fi_image_path):
        fi_channel_name = _infer_foci_channel_name(foci_image_path or fi_image_path)
    
    # Analyze masks
    results, detailed_data, masks_with_cellids, masks_original, foci_table = analyze_masks(
        args.masks,
        pixel_size=args.pixel_size,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_eccentricity=args.min_eccentricity,
        min_solidity=args.min_solidity,
        min_area_um2=args.min_area,
        max_area_um2=args.max_area,
        min_width_um=args.min_width,
        max_width_um=args.max_width,
        min_length_um=args.min_length,
        remove_small=not args.no_remove_small,
        min_size_pixels=args.min_size,
        fi_image_path=fi_image_path,
        foci_image_path=foci_image_path,
        fi_contour_thickness=args.fi_contour_thickness,
        fi_do_contour=not args.no_fi_contour,
        exclude_border=args.exclude_border,
        run_foci=args.run_foci,
        foci_detection_mode=getattr(args, 'foci_detection_mode', 0),
        foci_min_rel=args.foci_min_rel,
        foci_min_rel_mode=getattr(args, 'foci_min_rel_mode', 'per_cell'),
        foci_min_rel_cell=getattr(args, 'foci_min_rel_cell', 2.0),
        foci_min_separation=args.foci_min_separation,
        foci_min_size=args.foci_min_size,
        foci_max_size=args.foci_max_size,
        foci_min_peak_raw=getattr(args, 'foci_min_peak_raw', 0),
        foci_min_focus_mean=getattr(args, 'foci_min_focus_mean', 0),
        foci_blob_threshold=getattr(args, 'foci_blob_threshold', 0.8),
        foci_exclude_outside_cells=getattr(args, 'foci_exclude_outside_cells', False),
        spotiflow_options=spotiflow_options,
        fi_channel_name=fi_channel_name,
    )
    
    # Get input mask filename stem for output naming
    input_mask_stem = Path(args.masks).stem
    
    # Use a single _analysis folder when --output is already inside _analysis; else one subfolder per mask
    if output_path.parent.name == "_analysis":
        output_dir = output_path.parent
    else:
        output_dir = output_path.parent / f"{input_mask_stem}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Excel path: same directory (no extra subfolder when using _analysis)
    excel_in_folder = output_dir / output_path.name
    # We'll save directly to the folder, so update output_path
    output_path = excel_in_folder
    
    # Derive per-task artifact stem so multi-channel runs do not overwrite visualizations.
    output_stem = Path(output_path).stem
    default_analysis_stem = f"{input_mask_stem}_analysis"
    vis_stem = input_mask_stem if output_stem == default_analysis_stem else output_stem

    # Create visualization if requested
    vis_output = None
    if args.visualization:
        vis_output = output_dir / f"{vis_stem}_filter_visualization.tif"
        vis_bg_images = None
        vis_bg_source = args.foci_image or args.fi_image
        if vis_bg_source:
            try:
                vis_bg_images = coerce_stack_to_tyx(tifffile.imread(vis_bg_source))
                if vis_bg_images is None:
                    SCRIPT_LOG.warning(
                        "Skipping visualization background normalization: unsupported image shape for %s",
                        vis_bg_source,
                    )
                else:
                    num_frames = masks_with_cellids.shape[0]
                    if vis_bg_images.shape[0] != num_frames:
                        if vis_bg_images.shape[0] == 1:
                            vis_bg_images = np.repeat(vis_bg_images, num_frames, axis=0)
                        else:
                            min_frames = min(vis_bg_images.shape[0], num_frames)
                            vis_bg_images = vis_bg_images[:min_frames]
                            if min_frames < num_frames:
                                SCRIPT_LOG.warning(
                                    "Visualization background has fewer frames (%s) than masks (%s); trailing frames use no background.",
                                    min_frames,
                                    num_frames,
                                )
            except Exception as e:
                SCRIPT_LOG.warning("Could not load visualization background image %s: %s", vis_bg_source, e)
                vis_bg_images = None
        create_filter_visualization(
            masks_with_cellids, detailed_data, str(vis_output),
            masks_original=masks_original,
            foci_table=foci_table if args.run_foci else None,
            background_images=vis_bg_images,
        )
    
    # Create FI contour visualization only if contour FI was computed
    fi_contour_output = None
    if args.fi_image and (not args.no_fi_contour):
        # Reload FI images for visualization (they were already loaded in analyze_masks, but we need them here)
        fi_images = coerce_stack_to_tyx(tifffile.imread(args.fi_image))
        if fi_images is None:
            SCRIPT_LOG.warning("Skipping FI contour visualization: unsupported FI image shape for %s", args.fi_image)
            fi_images = None

        if fi_images is not None:
            # Ensure FI images match mask frames
            num_frames = masks_with_cellids.shape[0]
            if fi_images.shape[0] != num_frames:
                if fi_images.shape[0] == 1:
                    fi_images = np.repeat(fi_images, num_frames, axis=0)
                else:
                    min_frames = min(fi_images.shape[0], num_frames)
                    fi_images = fi_images[:min_frames]

            fi_contour_output = output_dir / f"{vis_stem}_fi_contours_visualization.tif"
            create_fi_contour_visualization(
                masks_with_cellids, detailed_data, fi_images, str(fi_contour_output),
                masks_original=masks_original,
                foci_table=foci_table if args.run_foci else None,
            )
    
    # Print summary statistics
    SCRIPT_LOG.info("=" * 60)
    SCRIPT_LOG.info("Summary Statistics")
    SCRIPT_LOG.info("=" * 60)
    total_cells = sum(r['num_cells'] for r in results)
    avg_cells_per_frame = total_cells / len(results) if results else 0
    avg_area = np.mean([r['total_area_um2'] for r in results])
    SCRIPT_LOG.info("Total frames analyzed: %s", len(results))
    SCRIPT_LOG.info("Total cells detected: %s", total_cells)
    SCRIPT_LOG.info("Average cells per frame: %.1f", avg_cells_per_frame)
    SCRIPT_LOG.info("Average total area per frame: %.2f µm²", avg_area)
    SCRIPT_LOG.info("First frame area: %.2f µm²", results[0]['total_area_um2'])
    SCRIPT_LOG.info("Last frame area: %.2f µm²", results[-1]['total_area_um2'])
    if results[0]['total_area_um2'] > 0:
        growth_factor = results[-1]['total_area_relative']
        SCRIPT_LOG.info("Relative growth (last/first): %.2fx", growth_factor)
    SCRIPT_LOG.info("=" * 60)

    # Save to Excel (and optionally CSV)
    save_to_excel(
        results, output_path,
        detailed_data if args.detailed else None,
        foci_table=foci_table if args.run_foci else None,
        fi_channel_name=fi_channel_name if (args.fi_image or args.foci_image) else None,
        export_csv=getattr(args, 'export_csv', False),
        pixel_size=args.pixel_size,
    )
    
    SCRIPT_LOG.info("Analysis complete!")
    SCRIPT_LOG.info("  Excel file saved to: %s", output_path)
    SCRIPT_LOG.info("  All files saved in folder: %s", output_dir)
    if args.visualization and vis_output is not None:
        SCRIPT_LOG.info("  Visualization saved to: %s", vis_output)
    if fi_contour_output is not None:
        SCRIPT_LOG.info("  FI contour visualization saved to: %s", fi_contour_output)
    return 0


if __name__ == "__main__":
    exit(main())
