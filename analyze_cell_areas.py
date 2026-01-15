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
from skimage.morphology import remove_small_objects
import warnings
warnings.filterwarnings('ignore')


def calculate_cell_properties(mask, pixel_size=0.108):
    """
    Calculate properties for each cell in a mask.
    
    Parameters:
    -----------
    mask : array
        Labeled mask image (0 = background, 1,2,3... = cell IDs)
    pixel_size : float
        Pixel size in µm (default: 0.108)
        
    Returns:
    --------
    properties : list of dict
        List of cell properties (area, width, length, aspect_ratio, etc.)
    """
    props = []
    
    # Get region properties
    regions = regionprops(mask)
    
    for region in regions:
        # Skip background (label 0)
        if region.label == 0:
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


def filter_by_size(properties, max_area_um2=None, max_width_um=None):
    """
    Filter cells by maximum area and width.
    
    Parameters:
    -----------
    properties : list of dict
        Cell properties
    max_area_um2 : float, optional
        Maximum cell area in µm²
    max_width_um : float, optional
        Maximum cell width in µm
        
    Returns:
    --------
    filtered : list of dict
        Filtered cell properties
    """
    filtered = properties
    
    if max_area_um2 is not None:
        filtered = [p for p in filtered if p['area_um2'] <= max_area_um2]
    
    if max_width_um is not None:
        filtered = [p for p in filtered if p['width_um'] <= max_width_um]
    
    return filtered


def analyze_masks(masks_path, pixel_size=0.108, 
                  min_aspect_ratio=1.5, max_aspect_ratio=10.0,
                  min_eccentricity=0.5, min_solidity=0.7,
                  max_area_um2=None, max_width_um=None,
                  remove_small=True, min_size_pixels=10):
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
    max_area_um2 : float, optional
        Maximum cell area in µm²
    max_width_um : float, optional
        Maximum cell width in µm
    remove_small : bool
        Remove small objects before analysis
    min_size_pixels : int
        Minimum size in pixels for objects to keep
        
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
    
    results = []
    all_frame_data = []  # Store detailed data for each frame
    
    print("\nAnalyzing frames...")
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        mask = masks[frame_idx]
        
        # Remove small objects (debris)
        if remove_small:
            mask_cleaned = remove_small_objects(mask, min_size=min_size_pixels)
        else:
            mask_cleaned = mask
        
        # Calculate cell properties
        properties = calculate_cell_properties(mask_cleaned, pixel_size=pixel_size)
        
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
            max_area_um2=max_area_um2,
            max_width_um=max_width_um
        )
        accepted_labels = {cell['label'] for cell in filtered}
        
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
    
    return results, all_frame_data, masks


def create_filter_visualization(masks, all_frame_data, output_path):
    """
    Create visualization showing accepted (green) and rejected (red) cells.
    
    Parameters:
    -----------
    masks : array
        Mask array (T, Y, X)
    all_frame_data : list of dict
        Frame data with accepted/rejected labels
    output_path : str
        Path to save visualization TIFF
    """
    print(f"\nCreating filter visualization...")
    
    num_frames = masks.shape[0]
    height, width = masks.shape[1], masks.shape[2]
    
    # Create RGB visualization
    visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    
    for frame_idx in tqdm(range(num_frames), desc="Creating visualization"):
        mask = masks[frame_idx]
        frame_data = all_frame_data[frame_idx]
        
        accepted_labels = frame_data['accepted_labels']
        rejected_labels = frame_data['rejected_labels']
        
        # Create RGB channels
        # Green channel (accepted cells)
        green = np.zeros((height, width), dtype=np.uint8)
        for label in accepted_labels:
            green[mask == label] = 255
        
        # Red channel (rejected cells)
        red = np.zeros((height, width), dtype=np.uint8)
        for label in rejected_labels:
            red[mask == label] = 255
        
        # Blue channel (background, can be used for original image overlay if needed)
        blue = np.zeros((height, width), dtype=np.uint8)
        
        # Combine into RGB
        visualization[frame_idx, :, :, 0] = red    # Red channel
        visualization[frame_idx, :, :, 1] = green  # Green channel
        visualization[frame_idx, :, :, 2] = blue   # Blue channel
    
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


def save_to_excel(results, output_path, detailed_data=None):
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
            for frame_data in detailed_data:
                frame_num = frame_data['frame']
                for cell in frame_data['cells']:
                    row = {
                        'Frame': frame_num,
                        'Cell Label': cell['label'],
                        'Area (µm²)': cell['area_um2'],
                        'Length (µm)': cell['length_um'],
                        'Width (µm)': cell['width_um'],
                        'Aspect Ratio': cell['aspect_ratio'],
                        'Eccentricity': cell['eccentricity'],
                        'Solidity': cell['solidity']
                    }
                    detailed_rows.append(row)
            
            if detailed_rows:
                df_detailed = pd.DataFrame(detailed_rows)
                df_detailed.to_excel(writer, sheet_name='Per-Cell Data', index=False)
    
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
    parser.add_argument('--max-area', type=float, default=None,
                       help='Maximum cell area in µm² (default: no limit)')
    parser.add_argument('--max-width', type=float, default=None,
                       help='Maximum cell width in µm (default: no limit)')
    
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
    
    # Analyze masks
    results, detailed_data, masks = analyze_masks(
        args.masks,
        pixel_size=args.pixel_size,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_eccentricity=args.min_eccentricity,
        min_solidity=args.min_solidity,
        max_area_um2=args.max_area,
        max_width_um=args.max_width,
        remove_small=not args.no_remove_small,
        min_size_pixels=args.min_size
    )
    
    # Create visualization if requested
    if args.visualization:
        vis_output = output_path.parent / f"{output_path.stem}_filter_visualization.tif"
        create_filter_visualization(masks, detailed_data, vis_output)
    
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
    save_to_excel(results, output_path, detailed_data if args.detailed else None)
    
    print("\n✓ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
