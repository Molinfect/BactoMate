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

try:
    from bactimate.foci_utils import detect_peaks_skimage, assign_peaks_to_cells
except ImportError:
    detect_peaks_skimage = None
    assign_peaks_to_cells = None


def calculate_cell_properties(mask, pixel_size=0.108, exclude_border=False):
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

    # Get region properties
    regions = regionprops(mask)

    for region in regions:
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
            print(f"Warning: FI image shape {fi_image.shape} doesn't match mask shape {mask.shape}")
            return np.nan, np.nan, np.nan
    
    # Calculate mean FI intensity of entire cell (always available)
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


def filter_by_size(properties, min_area_um2=None, max_area_um2=None, min_width_um=None, max_width_um=None, min_length_um=None):
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
    
    return filtered


def analyze_masks(masks_path, pixel_size=0.108,
                  min_aspect_ratio=1.5, max_aspect_ratio=10.0,
                  min_eccentricity=0.5, min_solidity=0.7,
                  min_area_um2=None, max_area_um2=None, min_width_um=None, max_width_um=None, min_length_um=None,
                  remove_small=True, min_size_pixels=10, fi_image_path=None, fi_contour_thickness=1, fi_do_contour=True,
                  exclude_border=False,
                  run_foci=False, foci_min_rel=2.0, foci_min_separation=2, foci_min_size=1, foci_max_size=None,
                  foci_exclude_outside_cells=False,
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
    print(f"Loading masks from: {masks_path}")
    masks = tifffile.imread(masks_path)
    
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    
    num_frames = masks.shape[0]
    print(f"Found {num_frames} frames")
    
    # Load FI image if provided
    fi_images = None
    if fi_image_path:
        print(f"Loading FI image from: {fi_image_path}")
        fi_images = tifffile.imread(fi_image_path)
        # Handle different dimensions
        if fi_images.ndim == 2:
            fi_images = fi_images[np.newaxis, ...]  # Add time dimension
        elif fi_images.ndim == 4:
            # (T, C, Y, X) - take first channel
            fi_images = fi_images[:, 0, :, :]
        elif fi_images.ndim == 3:
            # Could be (T, Y, X) or (Y, X, C)
            if fi_images.shape[-1] < 10:  # Likely channels last
                fi_images = fi_images.transpose(2, 0, 1)[np.newaxis, ...]  # (1, C, Y, X)
                fi_images = fi_images[:, 0, :, :]  # (1, Y, X)
            # Otherwise assume (T, Y, X)
        
        # Ensure FI images match mask frames
        if fi_images.shape[0] != num_frames:
            print(f"Warning: FI image has {fi_images.shape[0]} frames but masks have {num_frames} frames")
            if fi_images.shape[0] == 1:
                # Single FI frame - use for all mask frames
                fi_images = np.repeat(fi_images, num_frames, axis=0)
            else:
                # Use minimum number of frames
                min_frames = min(fi_images.shape[0], num_frames)
                fi_images = fi_images[:min_frames]
                masks = masks[:min_frames]
                num_frames = min_frames
                print(f"Using first {min_frames} frames from both images")
        
        print(f"✓ FI image loaded: {fi_images.shape}")
    
    results = []
    all_frame_data = []  # Store detailed data for each frame
    foci_table = []  # Foci rows (cell_id updated to persistent ID after assignment)
    
    print("\nAnalyzing frames...")
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        mask = masks[frame_idx]
        
        # Remove small objects (debris)
        if remove_small:
            mask_cleaned = remove_small_objects(mask, min_size=min_size_pixels)
        else:
            mask_cleaned = mask
        
        # Calculate cell properties
        properties = calculate_cell_properties(mask_cleaned, pixel_size=pixel_size, exclude_border=exclude_border)
        
        # Get all cell labels
        all_labels = {prop['label'] for prop in properties}
        
        # Apply morphological filtering
        # Step 1: Filter for rod-shaped cells
        rod_shaped = filter_rod_shaped_cells(
            properties,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            min_eccentricity=min_eccentricity,
            min_solidity=min_solidity
        )
        rod_shaped_labels = {cell['label'] for cell in rod_shaped}
        
        # Step 2: Filter by size constraints
        filtered = filter_by_size(
            rod_shaped,
            min_area_um2=min_area_um2,
            max_area_um2=max_area_um2,
            min_width_um=min_width_um,
            max_width_um=max_width_um,
            min_length_um=min_length_um
        )
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
        
        # Foci analysis (per-frame detection and assignment to cells)
        if run_foci and fi_images is not None:
            for cell in filtered:
                cell.setdefault('foci_count', 0)
                cell.setdefault('foci_ids', [])
        if (run_foci and fi_images is not None and detect_peaks_skimage is not None and assign_peaks_to_cells is not None):
            if fi_image.shape[:2] != mask_cleaned.shape[:2]:
                print(f"Warning: FI image shape {fi_image.shape[:2]} != mask shape {mask_cleaned.shape[:2]}, skipping foci for frame {frame_idx + 1}")
            else:
                peaks = detect_peaks_skimage(
                    fi_image,
                    min_rel=foci_min_rel,
                    min_distance=foci_min_separation,
                    min_size=foci_min_size,
                    max_size=foci_max_size,
                )
                foci_rows, cell_foci_ids = assign_peaks_to_cells(
                    peaks, mask_cleaned, dilation_px=1, frame_index=frame_idx
                )
                # Attach foci count and foci_ids to each cell (key by label, frame_idx)
                for cell in filtered:
                    key = (cell['label'], frame_idx)
                    ids = cell_foci_ids.get(key, [])
                    cell['foci_count'] = len(ids)
                    cell['foci_ids'] = ids
                for row in foci_rows:
                    if foci_exclude_outside_cells:
                        if not row.get('in_cell', False) or row.get('cell_id') not in accepted_labels:
                            continue
                    r = row.copy()
                    r['channel_name'] = fi_channel_name or 'FI'
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
    
    # Calculate relative areas (normalized to first frame)
    if len(results) > 0 and results[0]['total_area_um2'] > 0:
        first_frame_area = results[0]['total_area_um2']
        for result in results:
            result['total_area_relative'] = result['total_area_um2'] / first_frame_area
    else:
        print("WARNING: First frame has zero area, cannot calculate relative values")
        for result in results:
            result['total_area_relative'] = 0.0
    
    # Check if input masks already have persistent CellIDs (e.g. from "Assign persistent IDs" after segmentation)
    total_unique = len(np.unique(masks))
    per_frame_uniques = [len(np.unique(masks[i])) for i in range(num_frames)]
    avg_per_frame = sum(per_frame_uniques) / len(per_frame_uniques) if per_frame_uniques else 0
    input_has_persistent_ids = avg_per_frame > 0 and total_unique <= 2.0 * avg_per_frame

    if input_has_persistent_ids:
        # Keep existing CellIDs; only filter (accepted = keep ID, rejected = remove from output mask)
        print("\nInput masks already have persistent CellIDs; keeping them (filtering only).")
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
        print("\nAssigning persistent CellIDs to accepted cells...")
        print(f"  Processing {len(all_frame_data)} frames...")
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
                key = (row['frame'] - 1, row['cell_id'])
                row['cell_id'] = label_to_cellid.get(key, row['cell_id'])
    
    # Return both masks_with_cellids (for tracking/analysis) and original masks (for visualization)
    return results, all_frame_data, masks_with_cellids, masks, foci_table


def link_persistent_cell_ids_raw(masks, pixel_size=0.108, max_distance_um=5.0):
    """
    Assign persistent CellIDs to all cells in a mask stack (no filtering).
    Use this right after segmentation so the same cell keeps the same ID through the pipeline.
    Frame-to-frame linking by nearest-neighbor centroid distance.

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
    from scipy.spatial.distance import cdist
    from skimage.measure import regionprops

    num_frames = masks.shape[0]
    masks_with_cellids = np.zeros_like(masks, dtype=np.uint32)
    next_cell_id = 1
    frame_centroids_cache = {}
    frame_cellids_cache = {}
    max_distance_pixels = max_distance_um / pixel_size

    # Frame 0: assign initial IDs to all labels
    mask_0 = masks[0]
    frame_centroids_cache[0] = {}
    frame_cellids_cache[0] = {}
    for region in regionprops(mask_0):
        if region.label == 0:
            continue
        label = region.label
        centroid = np.array(region.centroid) * pixel_size
        frame_centroids_cache[0][label] = centroid
        frame_cellids_cache[0][label] = next_cell_id
        masks_with_cellids[0][mask_0 == label] = next_cell_id
        next_cell_id += 1

    for frame_idx in range(1, num_frames):
        mask = masks[frame_idx]
        regions = regionprops(mask)
        centroids_current = {r.label: np.array(r.centroid) * pixel_size for r in regions if r.label > 0}
        frame_centroids_cache[frame_idx] = centroids_current
        frame_cellids_cache[frame_idx] = {}

        prev_frame_idx = frame_idx - 1
        centroids_prev = {}
        for prev_label, prev_cellid in frame_cellids_cache[prev_frame_idx].items():
            if prev_label in frame_centroids_cache[prev_frame_idx]:
                centroids_prev[prev_cellid] = frame_centroids_cache[prev_frame_idx][prev_label]

        current_labels = list(centroids_current.keys())
        if len(centroids_prev) > 0 and len(current_labels) > 0:
            prev_points = np.array(list(centroids_prev.values()))
            prev_cellids = np.array(list(centroids_prev.keys()))
            current_points = np.array([centroids_current[l] for l in current_labels])
            distances = cdist(prev_points, current_points)
            used_prev = set()
            used_current = set()
            assignments = []
            for i in range(len(prev_cellids)):
                for j in range(len(current_labels)):
                    if distances[i, j] <= max_distance_pixels:
                        assignments.append((distances[i, j], prev_cellids[i], current_labels[j]))
            assignments.sort()
            for dist, prev_cellid, current_label in assignments:
                if prev_cellid not in used_prev and current_label not in used_current:
                    frame_cellids_cache[frame_idx][current_label] = prev_cellid
                    masks_with_cellids[frame_idx][mask == current_label] = prev_cellid
                    used_prev.add(prev_cellid)
                    used_current.add(current_label)
            for current_label in current_labels:
                if current_label not in used_current:
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
    print(f"  Assigned {next_cell_id - 1} unique CellIDs across {num_frames} frames (all cells linked)")
    return masks_with_cellids


def assign_persistent_cell_ids(masks, all_frame_data, pixel_size=0.108, max_distance_um=5.0):
    """
    Assign persistent CellIDs to accepted cells across frames using simple nearest-neighbor linking.
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
    from scipy.spatial.distance import cdist
    from skimage.measure import regionprops
    import numpy as np
    
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
    
    # Get centroids for accepted cells in frame 0 (cache them)
    regions_0 = regionprops(mask_0)
    centroids_0 = {}
    frame_centroids_cache[0] = {}
    frame_cellids_cache[0] = {}
    
    for region in regions_0:
        label = region.label
        if label in accepted_labels_0:
            centroid = np.array(region.centroid) * pixel_size
            centroids_0[label] = centroid
            frame_centroids_cache[0][label] = centroid
            cellid_mapping[(0, label)] = next_cell_id
            frame_cellids_cache[0][label] = next_cell_id
            # Update mask efficiently using direct indexing
            mask_indices = mask_0 == label
            masks_with_cellids[0][mask_indices] = next_cell_id
            next_cell_id += 1
    
    # Link cells across subsequent frames
    max_distance_pixels = max_distance_um / pixel_size
    
    # Progress indicator for large datasets
    print_every = max(1, num_frames // 20)  # Print progress every 5%
    
    for frame_idx in range(1, num_frames):
        if frame_idx % print_every == 0:
            print(f"  Processing frame {frame_idx + 1}/{num_frames}...")
        frame_data = all_frame_data[frame_idx]
        mask = masks[frame_idx]
        accepted_labels = set(frame_data['accepted_labels'])
        
        # Get centroids for accepted cells in current frame (cache them)
        regions = regionprops(mask)
        centroids_current = {}
        frame_centroids_cache[frame_idx] = {}
        
        for region in regions:
            label = region.label
            if label in accepted_labels:
                centroid = np.array(region.centroid) * pixel_size
                centroids_current[label] = centroid
                frame_centroids_cache[frame_idx][label] = centroid
        
        # Get centroids from previous frame with their CellIDs (use cache!)
        centroids_prev = {}
        prev_frame_idx = frame_idx - 1
        
        # Use cached centroids and cellids from previous frame
        if prev_frame_idx in frame_cellids_cache:
            for prev_label, prev_cellid in frame_cellids_cache[prev_frame_idx].items():
                if prev_label in frame_centroids_cache[prev_frame_idx]:
                    centroids_prev[prev_cellid] = frame_centroids_cache[prev_frame_idx][prev_label]
        
        # Link cells using nearest neighbor
        if len(centroids_prev) > 0 and len(centroids_current) > 0:
            # Build distance matrix
            prev_points = np.array(list(centroids_prev.values()))
            prev_cellids = np.array(list(centroids_prev.keys()))
            current_points = np.array(list(centroids_current.values()))
            current_labels = np.array(list(centroids_current.keys()))
            
            if len(prev_points) > 0 and len(current_points) > 0:
                distances = cdist(prev_points, current_points)
                
                # Greedy assignment: assign each current cell to nearest previous cell
                used_prev = set()
                used_current = set()
                
                # Create assignments list more efficiently (only within max_distance)
                assignments = []
                for i in range(len(prev_cellids)):
                    for j in range(len(current_labels)):
                        dist = distances[i, j]
                        if dist <= max_distance_pixels:
                            assignments.append((dist, prev_cellids[i], current_labels[j]))
                
                # Sort by distance and assign
                assignments.sort()
                
                frame_cellids_cache[frame_idx] = {}
                for dist, prev_cellid, current_label in assignments:
                    if prev_cellid not in used_prev and current_label not in used_current:
                        # Link this cell
                        cellid_mapping[(frame_idx, current_label)] = prev_cellid
                        frame_cellids_cache[frame_idx][current_label] = prev_cellid
                        # Update mask efficiently
                        mask_indices = mask == current_label
                        masks_with_cellids[frame_idx][mask_indices] = prev_cellid
                        used_prev.add(prev_cellid)
                        used_current.add(current_label)
                
                # Assign new CellIDs to unlinked cells
                for current_label in current_labels:
                    if current_label not in used_current:
                        cellid_mapping[(frame_idx, current_label)] = next_cell_id
                        frame_cellids_cache[frame_idx][current_label] = next_cell_id
                        # Update mask efficiently
                        mask_indices = mask == current_label
                        masks_with_cellids[frame_idx][mask_indices] = next_cell_id
                        next_cell_id += 1
        else:
            # No previous cells or no current cells - assign new IDs
            frame_cellids_cache[frame_idx] = {}
            for current_label in accepted_labels:
                cellid_mapping[(frame_idx, current_label)] = next_cell_id
                frame_cellids_cache[frame_idx][current_label] = next_cell_id
                # Update mask efficiently
                mask_indices = mask == current_label
                masks_with_cellids[frame_idx][mask_indices] = next_cell_id
                next_cell_id += 1
    
    # Convert back to uint16 if needed (for compatibility)
    if masks_with_cellids.max() < 65535:
        masks_with_cellids = masks_with_cellids.astype(np.uint16)
    
    print(f"  Assigned {next_cell_id - 1} unique CellIDs across {num_frames} frames")
    return masks_with_cellids, cellid_mapping


def create_filter_visualization(masks, all_frame_data, output_path, masks_original=None):
    """
    Create visualization showing accepted (green) and rejected (red) cells with persistent CellIDs.
    
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
    """
    print(f"\nCreating filter visualization...")
    
    num_frames = masks.shape[0]
    height, width = masks.shape[1], masks.shape[2]
    
    # Create RGB visualization with CellID labels
    visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    
    # Try to import PIL for text rendering
    try:
        from PIL import Image, ImageDraw, ImageFont
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
        print("WARNING: Pillow missing (required dependency); labels disabled. Run: pip install -r requirements.txt")
    
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
        
        # Create RGB channels
        # Green channel (accepted cells with CellIDs)
        green = np.zeros((height, width), dtype=np.uint8)
        for cell_id in accepted_cellids:
            if cell_id is not None:
                green[mask == cell_id] = 255
        
        # Red channel (rejected cells - use original labels from original mask)
        red = np.zeros((height, width), dtype=np.uint8)
        for rejected_label in rejected_labels:
            if rejected_label > 0:  # Skip background
                red[mask_orig == rejected_label] = 255
        
        # Blue channel (background, can be used for original image overlay if needed)
        blue = np.zeros((height, width), dtype=np.uint8)
        
        # Combine into RGB
        visualization[frame_idx, :, :, 0] = red    # Red channel
        visualization[frame_idx, :, :, 1] = green  # Green channel
        visualization[frame_idx, :, :, 2] = blue   # Blue channel
        
        # Add CellID labels if PIL is available
        if PIL_AVAILABLE and mask.max() > 0:
            try:
                # Convert to PIL Image
                pil_img = Image.fromarray(visualization[frame_idx])
                draw = ImageDraw.Draw(pil_img)
                
                # Try to use a nice font, fallback to default if not available
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 10)
                    except:
                        font = ImageFont.load_default()
                
                # Get cell centroids and CellIDs (masks now have persistent CellIDs)
                from skimage.measure import regionprops
                regions = regionprops(mask)
                for region in regions:
                    if region.label == 0:
                        continue
                    
                    # Get centroid
                    centroid_y, centroid_x = region.centroid
                    x, y = int(centroid_x), int(centroid_y)
                    
                    # Draw CellID text with shadow for readability (label is now persistent CellID)
                    cell_id_str = str(region.label)
                    text_x, text_y = x + 2, y - 5
                    
                    # Draw text shadow (black)
                    draw.text((text_x + 1, text_y + 1), cell_id_str, fill=(0, 0, 0), font=font)
                    # Draw text (white)
                    draw.text((text_x, text_y), cell_id_str, fill=(255, 255, 255), font=font)
                
                # Convert back to numpy array
                visualization[frame_idx] = np.array(pil_img)
            except Exception as e:
                # If text rendering fails, continue without labels
                pass
    
    # Save visualization
    print(f"Saving visualization to: {output_path}")
    tifffile.imwrite(output_path, visualization)
    print("✓ Visualization saved")
    
    # Print statistics
    total_accepted = sum(len(fd['accepted_labels']) for fd in all_frame_data)
    total_rejected = sum(len(fd['rejected_labels']) for fd in all_frame_data)
    print(f"\nVisualization statistics:")
    print(f"  Total accepted cells (green): {total_accepted}")
    print(f"  Total rejected cells (red): {total_rejected}")
    print(f"  Acceptance rate: {100*total_accepted/(total_accepted+total_rejected):.1f}%" if (total_accepted+total_rejected) > 0 else "  Acceptance rate: 0%")


def create_fi_contour_visualization(masks, all_frame_data, fi_images, output_path, masks_original=None):
    """
    Create visualization showing cell masks with FI contours highlighted.
    
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
    """
    print(f"\nCreating FI contour visualization...")
    
    num_frames = masks.shape[0]
    height, width = masks.shape[1], masks.shape[2]
    
    # Create RGB visualization
    visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    
    # Normalize FI images for background display
    fi_images_norm = []
    for fi_img in fi_images:
        fi_min, fi_max = fi_img.min(), fi_img.max()
        if fi_max > fi_min:
            fi_norm = ((fi_img - fi_min) / (fi_max - fi_min) * 255).astype(np.uint8)
        else:
            fi_norm = fi_img.astype(np.uint8)
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
    
    # Save visualization
    print(f"Saving FI contour visualization to: {output_path}")
    tifffile.imwrite(output_path, visualization)
    print("✓ FI contour visualization saved")


def save_to_excel(results, output_path, detailed_data=None, foci_table=None, fi_channel_name=None):
    """
    Save results to Excel file.
    
    Parameters:
    -----------
    results : list of dict
        Results with frame, total_area_um2, total_area_relative
    output_path : str
        Path to save Excel file
    detailed_data : list of dict, optional
        Detailed per-cell data for additional sheets
    foci_table : list of dict, optional
        Foci rows (foci_id, cell_id, frame, channel_name, y, x, intensity, size_px, in_cell)
    fi_channel_name : str, optional
        Channel name for foci count/foci_ids columns (e.g. GFP)
    """
    # Create main results DataFrame
    df_main = pd.DataFrame(results)
    
    # Reorder columns
    df_main = df_main[['frame', 'total_area_um2', 'total_area_relative', 'num_cells']]
    
    # Rename columns for clarity
    df_main.columns = ['Frame', 'Total Cell Area (µm²)', 'Total Cell Area (Relative)', 'Number of Cells']
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main summary sheet
        df_main.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed per-cell data (optional)
        if detailed_data:
            detailed_rows = []
            ch_name = fi_channel_name or 'FI'
            foci_count_col = f'{ch_name}_foci_count'
            foci_ids_col = f'foci_ids_{ch_name}'
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
                        'Solidity': cell['solidity']
                    }
                    # Add FI columns if available (contour metrics may be disabled)
                    if 'mean_fi' in cell:
                        row['Mean FI Intensity'] = cell['mean_fi']
                    if 'contour_fi' in cell:
                        row['Contour FI Intensity'] = cell['contour_fi']
                    if 'fi_ratio' in cell:
                        row['FI Ratio (Contour/Mean)'] = cell['fi_ratio']
                    # Foci columns
                    if 'foci_count' in cell:
                        row[foci_count_col] = cell['foci_count']
                    if 'foci_ids' in cell:
                        row[foci_ids_col] = ','.join(map(str, cell['foci_ids'])) if cell['foci_ids'] else ''
                    detailed_rows.append(row)
            
            if detailed_rows:
                df_detailed = pd.DataFrame(detailed_rows)
                df_detailed.to_excel(writer, sheet_name='Per-Cell Data', index=False)
        
        # Foci sheet (when run_foci was on)
        if foci_table is not None:
            df_foci = pd.DataFrame(foci_table)
            if len(df_foci) > 0:
                # Reorder columns for clarity
                cols = ['foci_id', 'cell_id', 'frame', 'channel_name', 'y', 'x', 'intensity', 'size_px', 'in_cell']
                df_foci = df_foci[[c for c in cols if c in df_foci.columns]]
            else:
                df_foci = pd.DataFrame(columns=['foci_id', 'cell_id', 'frame', 'channel_name', 'y', 'x', 'intensity', 'size_px', 'in_cell'])
            df_foci.to_excel(writer, sheet_name='Foci', index=False)
    
    print(f"\n✓ Results saved to: {output_path}")


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
    parser.add_argument('--visualization', action='store_true',
                       help='Create visualization TIFF showing accepted (green) and rejected (red) cells')
    
    # FI analysis
    parser.add_argument('--fi-image', type=str, default=None,
                       help='Path to FI channel TIFF file for fluorescence intensity analysis')
    parser.add_argument('--fi-contour-thickness', type=int, default=1,
                       help='Thickness of contour region in pixels for FI intensity calculation (default: 1)')
    parser.add_argument('--no-fi-contour', action='store_true',
                       help='Disable contour-based FI metrics (only compute Mean FI Intensity)')
    parser.add_argument('--exclude-border', action='store_true',
                       help='Exclude cells that touch the image border from counts and area')
    # Foci analysis
    parser.add_argument('--run-foci', action='store_true',
                       help='Run foci detection on the FI channel and add foci count and Foci sheet')
    parser.add_argument('--foci-min-rel', type=float, default=2.0,
                       help='Min relative intensity (peak/median) for foci (default: 2.0)')
    parser.add_argument('--foci-min-separation', type=int, default=2,
                       help='Min distance between foci in pixels (default: 2)')
    parser.add_argument('--foci-min-size', type=int, default=1,
                       help='Min focus size in pixels (default: 1)')
    parser.add_argument('--foci-max-size', type=int, default=None,
                       help='Max focus size in pixels (0 or omit = no limit)')
    parser.add_argument('--foci-exclude-outside-cells', action='store_true',
                       help='Exclude foci that fall outside any cell from counts and Foci sheet')
    parser.add_argument('--foci-channel-name', type=str, default=None,
                       help='Channel name for foci columns/sheet (default: from FI filename, e.g. GFP)')
    
    args = parser.parse_args()
    
    # Check input file
    if not Path(args.masks).exists():
        print(f"ERROR: Mask file not found: {args.masks}")
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
            print(f"ERROR: FI image file not found: {args.fi_image}")
            return 1
        fi_image_path = args.fi_image
    
    # Foci channel name from FI path if not set (e.g. stem_fi_GFP.tif -> GFP)
    fi_channel_name = args.foci_channel_name
    if fi_channel_name is None and fi_image_path:
        stem = Path(fi_image_path).stem
        if stem.endswith('_fi'):
            fi_channel_name = 'FI'
        elif '_fi_' in stem:
            fi_channel_name = stem.split('_fi_')[-1].strip('_')
        else:
            fi_channel_name = 'FI'
    
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
        fi_contour_thickness=args.fi_contour_thickness,
        fi_do_contour=not args.no_fi_contour,
        exclude_border=args.exclude_border,
        run_foci=args.run_foci,
        foci_min_rel=args.foci_min_rel,
        foci_min_separation=args.foci_min_separation,
        foci_min_size=args.foci_min_size,
        foci_max_size=args.foci_max_size,
        foci_exclude_outside_cells=getattr(args, 'foci_exclude_outside_cells', False),
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
    
    # Create visualization if requested
    if args.visualization:
        vis_output = output_dir / f"{input_mask_stem}_filter_visualization.tif"
        create_filter_visualization(masks_with_cellids, detailed_data, str(vis_output), masks_original=masks_original)
    
    # Create FI contour visualization only if contour FI was computed
    if args.fi_image and (not args.no_fi_contour):
        # Reload FI images for visualization (they were already loaded in analyze_masks, but we need them here)
        fi_images = tifffile.imread(args.fi_image)
        # Handle different dimensions (same logic as in analyze_masks)
        if fi_images.ndim == 2:
            fi_images = fi_images[np.newaxis, ...]
        elif fi_images.ndim == 4:
            fi_images = fi_images[:, 0, :, :]
        elif fi_images.ndim == 3:
            if fi_images.shape[-1] < 10:
                fi_images = fi_images.transpose(2, 0, 1)[np.newaxis, ...]
                fi_images = fi_images[:, 0, :, :]
        
        # Ensure FI images match mask frames
        num_frames = masks_with_cellids.shape[0]
        if fi_images.shape[0] != num_frames:
            if fi_images.shape[0] == 1:
                fi_images = np.repeat(fi_images, num_frames, axis=0)
            else:
                min_frames = min(fi_images.shape[0], num_frames)
                fi_images = fi_images[:min_frames]
        
        fi_contour_output = output_dir / f"{input_mask_stem}_fi_contours_visualization.tif"
        create_fi_contour_visualization(masks_with_cellids, detailed_data, fi_images, str(fi_contour_output), masks_original=masks_original)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    total_cells = sum(r['num_cells'] for r in results)
    avg_cells_per_frame = total_cells / len(results) if results else 0
    avg_area = np.mean([r['total_area_um2'] for r in results])
    print(f"Total frames analyzed: {len(results)}")
    print(f"Total cells detected: {total_cells}")
    print(f"Average cells per frame: {avg_cells_per_frame:.1f}")
    print(f"Average total area per frame: {avg_area:.2f} µm²")
    print(f"First frame area: {results[0]['total_area_um2']:.2f} µm²")
    print(f"Last frame area: {results[-1]['total_area_um2']:.2f} µm²")
    if results[0]['total_area_um2'] > 0:
        growth_factor = results[-1]['total_area_relative']
        print(f"Relative growth (last/first): {growth_factor:.2f}x")
    print("="*60)
    
    # Save to Excel
    save_to_excel(
        results, output_path,
        detailed_data if args.detailed else None,
        foci_table=foci_table if args.run_foci else None,
        fi_channel_name=fi_channel_name if args.run_foci else None,
    )
    
    print("\n✓ Analysis complete!")
    print(f"  Excel file saved to: {output_path}")
    print(f"  All files saved in folder: {output_dir}")
    if args.visualization:
        print(f"  Visualization saved to: {vis_output}")
    if args.fi_image and (not args.no_fi_contour):
        fi_contour_output = output_dir / f"{input_mask_stem}_fi_contours_visualization.tif"
        print(f"  FI contour visualization saved to: {fi_contour_output}")
    return 0


if __name__ == "__main__":
    exit(main())
