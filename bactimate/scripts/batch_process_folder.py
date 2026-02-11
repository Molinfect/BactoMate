#!/usr/bin/env python3
"""
Batch process all TIFF files in a folder with Omnipose and analyze results.

This script:
1. Finds all TIFF files in a folder
2. Processes each with Omnipose (using batch_process_tiff_sequence.py logic)
3. Analyzes each with morphological filtering (using analyze_cell_areas.py logic)
4. Combines all results into a single Excel file with filename-prefixed columns

Usage:
    python batch_process_folder.py /path/to/folder [options]
"""

import argparse
import os
import sys
import warnings
import numpy as np
import tifffile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import tempfile
import shutil
import subprocess

# Suppress FutureWarning from omnipose about deprecated area_threshold parameter
warnings.filterwarnings('ignore', category=FutureWarning, message='.*area_threshold.*')

# Import functions from other scripts in this package
from bactimate.scripts.batch_process_tiff_sequence import (
    split_multipage_tiff,
    process_frames_with_omnipose,
    concatenate_masks,
    overlay_masks_on_images,
)
from bactimate.scripts.analyze_cell_areas import (
    analyze_masks,
    save_to_excel,
    create_filter_visualization,
)


def find_tiff_files(folder_path, recursive=False):
    """
    Find all TIFF files in a folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder
    recursive : bool
        Search subdirectories recursively
        
    Returns:
    --------
    tiff_files : list
        List of TIFF file paths
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if recursive:
        tiff_files = list(folder.rglob("*.tif")) + list(folder.rglob("*.tiff"))
    else:
        tiff_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    
    # Filter out output files (masks, overlays, visualizations)
    tiff_files = [f for f in tiff_files 
                  if not any(x in f.name.lower() for x in ['_masks', '_overlay', '_visualization', '_analysis'])]
    
    return sorted(tiff_files)


def process_single_file(input_file, output_base_dir, model_type='bact_phase_omni',
                        diameter=0, channels=[0, 0], use_gpu=True,
                        flow_threshold=0.4, mask_threshold=0.0,
                        omni=True, cluster=False,
                        pixel_size=0.108,
                        min_aspect_ratio=1.5, max_aspect_ratio=10.0,
                        min_eccentricity=0.5, min_solidity=0.7,
                        max_area_um2=None, max_width_um=None,
                        min_size_pixels=10, remove_small=True,
                        save_intermediate=False, create_visualization=True, create_overlay=True, verbose=False):
    """
    Process a single TIFF file: segment and analyze.
    
    Returns:
    --------
    results : dict
        Analysis results with filename
    masks_path : str
        Path to saved masks
    """
    input_file = Path(input_file)
    file_stem = input_file.stem
    
    # Create output directory for this file
    file_output_dir = output_base_dir / f"{file_stem}_omnipose"
    os.makedirs(file_output_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {input_file.name}")
        print(f"{'='*60}")
    
    # Step 1: Split and process with Omnipose
    frame_dir = file_output_dir / "frames" if save_intermediate else None
    frame_paths, temp_dir, image_shape = split_multipage_tiff(
        str(input_file),
        output_dir=str(frame_dir) if frame_dir else None,
        prefix="frame"
    )
    
    try:
        # Process frames
        masks = process_frames_with_omnipose(
            frame_paths,
            model_type=model_type,
            diameter=diameter,
            channels=channels,
            use_gpu=use_gpu,
            flow_threshold=flow_threshold,
            mask_threshold=mask_threshold,
            omni=omni,
            cluster=cluster,
            verbose=verbose
        )
        
        # Save masks
        masks_path = file_output_dir / f"{file_stem}_masks.tif"
        concatenate_masks(masks, str(masks_path))
        
        # Step 1.5: Create overlay (if requested) - before analysis to avoid reloading images
        overlay_path = None
        if create_overlay:
            try:
                # Reload original images for overlay
                images = tifffile.imread(str(input_file))
                # Handle different formats (same logic as batch_process_tiff_sequence.py)
                if images.ndim == 3:
                    if images.shape[-1] < 10:
                        images = images.transpose(2, 0, 1)[np.newaxis, ...]
                    else:
                        images = images[:, np.newaxis, :, :]
                elif images.ndim == 4:
                    if images.shape[-1] < 10:
                        images = images.transpose(0, 3, 1, 2)
                    elif images.shape[0] < 10:
                        images = images.transpose(1, 0, 2, 3)
                
                overlay_path = file_output_dir / f"{file_stem}_overlay.tif"
                overlay_masks_on_images(images, masks, str(overlay_path))
            except Exception as e:
                if verbose:
                    print(f"WARNING: Could not create overlay: {e}")
        
        # Step 2: Analyze masks
        results, detailed_data, masks_with_cellids, masks_original = analyze_masks(
            str(masks_path),
            pixel_size=pixel_size,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            min_eccentricity=min_eccentricity,
            min_solidity=min_solidity,
            max_area_um2=max_area_um2,
            max_width_um=max_width_um,
            remove_small=remove_small,
            min_size_pixels=min_size_pixels
        )
        
        # Step 3: Create filter visualization (if requested)
        vis_path = None
        if create_visualization:
            vis_path = file_output_dir / f"{file_stem}_filter_visualization.tif"
            create_filter_visualization(masks_with_cellids, detailed_data, str(vis_path), masks_original=masks_original)
        
        # Add filename and paths to results
        for result in results:
            result['filename'] = input_file.name
            result['file_stem'] = file_stem
        
        return {
            'results': results,
            'detailed_data': detailed_data,
            'masks_path': str(masks_path),
            'vis_path': str(vis_path) if vis_path else None,
            'filename': input_file.name,
            'file_stem': file_stem,
            'output_dir': file_output_dir  # Store output directory
        }
        
    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def combine_results(all_results):
    """
    Combine results from multiple files into a single DataFrame.
    
    Parameters:
    -----------
    all_results : list of dict
        Results from each file
        
    Returns:
    --------
    combined_df : DataFrame
        Combined results with filename-prefixed columns
    """
    # Check if all files have same number of frames
    num_frames_list = [len(r['results']) for r in all_results]
    if len(set(num_frames_list)) == 1:
        # All same length - use single Frame column
        num_frames = num_frames_list[0]
        use_single_frame_col = True
    else:
        # Different lengths - need separate frame columns
        use_single_frame_col = False
    
    # Build combined DataFrame
    if use_single_frame_col:
        # All files have same number of frames
        combined_data = {'Frame': list(range(1, num_frames + 1))}
        
        for file_result in all_results:
            file_stem = file_result['file_stem']
            results = file_result['results']
            
            # Add columns with filename prefix
            combined_data[f'{file_stem}_Total Area (µm²)'] = [r['total_area_um2'] for r in results]
            combined_data[f'{file_stem}_Total Area (Relative)'] = [r['total_area_relative'] for r in results]
            combined_data[f'{file_stem}_Number of Cells'] = [r['num_cells'] for r in results]
        
        df = pd.DataFrame(combined_data)
        
    else:
        # Different frame counts - need to handle separately
        # Use longest frame count and pad with NaN
        max_frames = max(num_frames_list)
        combined_data = {'Frame': list(range(1, max_frames + 1))}
        
        for file_result in all_results:
            file_stem = file_result['file_stem']
            results = file_result['results']
            num_frames = len(results)
            
            # Pad with NaN if shorter
            total_areas = [r['total_area_um2'] for r in results] + [np.nan] * (max_frames - num_frames)
            relative_areas = [r['total_area_relative'] for r in results] + [np.nan] * (max_frames - num_frames)
            num_cells = [r['num_cells'] for r in results] + [np.nan] * (max_frames - num_frames)
            
            combined_data[f'{file_stem}_Total Area (µm²)'] = total_areas
            combined_data[f'{file_stem}_Total Area (Relative)'] = relative_areas
            combined_data[f'{file_stem}_Number of Cells'] = num_cells
        
        df = pd.DataFrame(combined_data)
    
    return df


def save_combined_excel(combined_df, output_path):
    """
    Save combined summary results to Excel (only Combined Summary sheet).
    
    Parameters:
    -----------
    combined_df : DataFrame
        Combined summary data
    output_path : str
        Path to save Excel file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main combined summary sheet only
        combined_df.to_excel(writer, sheet_name='Combined Summary', index=False)
    
    print(f"\n✓ Combined results saved to: {output_path}")


def save_individual_excel(file_result, output_dir, include_detailed=False):
    """
    Save individual file results to Excel in the file's output directory.
    
    Parameters:
    -----------
    file_result : dict
        File result with results and detailed_data
    output_dir : Path
        Directory where the file's masks are saved
    include_detailed : bool
        Include detailed per-cell data sheet
    """
    file_stem = file_result['file_stem']
    results = file_result['results']
    detailed_data = file_result['detailed_data']
    
    # Create Excel file path in the same directory as masks
    excel_path = output_dir / f"{file_stem}_analysis.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Summary sheet
        df_file = pd.DataFrame(results)
        df_file = df_file[['frame', 'total_area_um2', 'total_area_relative', 'num_cells']]
        df_file.columns = ['Frame', 'Total Cell Area (µm²)', 'Total Cell Area (Relative)', 'Number of Cells']
        df_file.to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed per-cell data (optional)
        if include_detailed:
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
    
    return excel_path


def main():
    parser = argparse.ArgumentParser(
        description='Batch process all TIFF files in a folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all TIFF files in a folder
  python batch_process_folder.py /path/to/folder --model bact_phase_omni --diameter 30
  
  # With custom analysis parameters
  python batch_process_folder.py /path/to/folder \\
      --model bact_phase_omni \\
      --diameter 30 \\
      --pixel-size 0.108 \\
      --min-aspect-ratio 2.0 \\
      --max-area 50 \\
      --max-width 2.0
        """
    )
    
    parser.add_argument('folder', type=str, help='Folder containing TIFF files')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory (default: folder/omnipose_batch_results)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search subdirectories recursively')
    
    # Omnipose processing parameters
    parser.add_argument('--model', type=str, default='bact_phase_omni',
                       choices=['cyto2_omni', 'cyto_omni', 'bact_omni', 'bact_phase_omni', 
                               'bact_fluor_omni', 'worm_omni', 'nuclei', 'cyto2', 'cyto'],
                       help='Omnipose model type (default: bact_phase_omni)')
    parser.add_argument('--diameter', type=float, default=0,
                       help='Cell diameter in pixels (0 for auto-detection)')
    parser.add_argument('--channels', type=int, nargs=2, default=[0, 0],
                       metavar=('CHAN1', 'CHAN2'),
                       help='Channels for segmentation (default: 0 0)')
    parser.add_argument('--flow-threshold', type=float, default=0.4,
                       help='Flow error threshold (default: 0.4)')
    parser.add_argument('--mask-threshold', type=float, default=0.0,
                       help='Mask threshold (default: 0.0)')
    parser.add_argument('--no-omni', action='store_true',
                       help='Disable Omnipose mask reconstruction')
    parser.add_argument('--cluster', action='store_true',
                       help='Use DBSCAN clustering')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU (use CPU)')
    
    # Analysis parameters
    parser.add_argument('--pixel-size', type=float, default=0.108,
                       help='Pixel size in µm (default: 0.108)')
    parser.add_argument('--min-aspect-ratio', type=float, default=1.5,
                       help='Minimum aspect ratio for rod-shaped cells (default: 1.5)')
    parser.add_argument('--max-aspect-ratio', type=float, default=10.0,
                       help='Maximum aspect ratio (default: 10.0)')
    parser.add_argument('--min-eccentricity', type=float, default=0.5,
                       help='Minimum eccentricity (default: 0.5)')
    parser.add_argument('--min-solidity', type=float, default=0.7,
                       help='Minimum solidity (default: 0.7)')
    parser.add_argument('--max-area', type=float, default=None,
                       help='Maximum cell area in µm²')
    parser.add_argument('--max-width', type=float, default=None,
                       help='Maximum cell width in µm')
    parser.add_argument('--min-size', type=int, default=10,
                       help='Minimum object size in pixels (default: 10)')
    parser.add_argument('--no-remove-small', action='store_true',
                       help='Disable removal of small objects')
    
    # Output options
    parser.add_argument('--excel-name', type=str, default=None,
                       help='Output Excel filename (default: folder_name_batch_analysis.xlsx)')
    parser.add_argument('--detailed', action='store_true',
                       help='Include detailed per-cell data in Excel')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate files (masks, frames)')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip creating filter visualization TIFFs (green=accepted, red=rejected)')
    parser.add_argument('--no-overlay', action='store_true',
                       help='Skip creating overlay visualization TIFFs')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip Omnipose processing, only analyze existing masks')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check folder
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"ERROR: Folder not found: {args.folder}")
        return 1
    
    # Find TIFF files
    print(f"Searching for TIFF files in: {folder_path}")
    tiff_files = find_tiff_files(folder_path, recursive=args.recursive)
    
    if len(tiff_files) == 0:
        print(f"ERROR: No TIFF files found in {folder_path}")
        return 1
    
    print(f"Found {len(tiff_files)} TIFF file(s):")
    for f in tiff_files:
        print(f"  - {f.name}")
    
    # Setup output directory
    if args.output is None:
        output_base_dir = folder_path / "omnipose_batch_results"
    else:
        output_base_dir = Path(args.output)
    
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"\nOutput directory: {output_base_dir}")
    
    # Process each file
    all_results = []
    
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(tiff_files)}: {tiff_file.name}")
        print(f"{'='*60}")
        
        try:
            if args.skip_processing:
                # Only analyze existing masks
                masks_path = output_base_dir / f"{tiff_file.stem}_omnipose" / f"{tiff_file.stem}_masks.tif"
                if not masks_path.exists():
                    print(f"WARNING: Masks not found for {tiff_file.name}, skipping...")
                    continue
                
                # Analyze existing masks
                results, detailed_data, masks_with_cellids, masks_original = analyze_masks(
                    str(masks_path),
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
                
                # Create filter visualization (if requested)
                vis_path = None
                if not args.no_visualization:
                    vis_path = masks_path.parent / f"{tiff_file.stem}_filter_visualization.tif"
                    create_filter_visualization(masks_with_cellids, detailed_data, str(vis_path), masks_original=masks_original)
                
                for result in results:
                    result['filename'] = tiff_file.name
                    result['file_stem'] = tiff_file.stem
                
                all_results.append({
                    'results': results,
                    'detailed_data': detailed_data,
                    'masks_path': str(masks_path),
                    'vis_path': str(vis_path) if vis_path else None,
                    'filename': tiff_file.name,
                    'file_stem': tiff_file.stem,
                    'output_dir': masks_path.parent  # Store output directory
                })
            else:
                # Full processing: segment + analyze
                file_result = process_single_file(
                    tiff_file,
                    output_base_dir,
                    model_type=args.model,
                    diameter=args.diameter,
                    channels=args.channels,
                    use_gpu=not args.no_gpu,
                    flow_threshold=args.flow_threshold,
                    mask_threshold=args.mask_threshold,
                    omni=not args.no_omni,
                    cluster=args.cluster,
                    pixel_size=args.pixel_size,
                    min_aspect_ratio=args.min_aspect_ratio,
                    max_aspect_ratio=args.max_aspect_ratio,
                    min_eccentricity=args.min_eccentricity,
                    min_solidity=args.min_solidity,
                    max_area_um2=args.max_area,
                    max_width_um=args.max_width,
                    min_size_pixels=args.min_size,
                    remove_small=not args.no_remove_small,
                    save_intermediate=args.save_intermediate,
                    create_visualization=not args.no_visualization,
                    create_overlay=not args.no_overlay,
                    verbose=args.verbose
                )
                all_results.append(file_result)
        
        except Exception as e:
            print(f"\nERROR processing {tiff_file.name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping {tiff_file.name} and continuing...")
            continue
    
    if len(all_results) == 0:
        print("\nERROR: No files were successfully processed")
        return 1
    
    # Combine results
    print(f"\n{'='*60}")
    print("Combining results from all files...")
    print(f"{'='*60}")
    
    combined_df = combine_results(all_results)
    
    # Save combined Excel (only Combined Summary sheet)
    if args.excel_name is None:
        excel_name = f"{folder_path.name}_batch_analysis.xlsx"
    else:
        excel_name = args.excel_name
        if not excel_name.endswith('.xlsx'):
            excel_name += '.xlsx'
    
    excel_path = output_base_dir / excel_name
    save_combined_excel(combined_df, excel_path)
    
    # Save individual Excel files for each file
    print(f"\nSaving individual file Excel sheets...")
    individual_excel_paths = []
    for file_result in tqdm(all_results, desc="Saving individual files"):
        # Determine output directory for this file
        if 'output_dir' in file_result:
            # Skip-processing mode: use stored output_dir
            file_output_dir = Path(file_result['output_dir'])
        else:
            # Full processing mode: construct path
            file_stem = file_result['file_stem']
            file_output_dir = output_base_dir / f"{file_stem}_omnipose"
        
        excel_path = save_individual_excel(
            file_result, 
            file_output_dir, 
            include_detailed=args.detailed
        )
        individual_excel_paths.append(excel_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Batch Processing Summary")
    print(f"{'='*60}")
    print(f"Files processed: {len(all_results)}")
    print(f"Combined Excel saved to: {excel_path}")
    print(f"\nMain Excel file ({excel_path.name}) contains:")
    print(f"  - 'Combined Summary' sheet: All files in one table")
    print(f"\nIndividual Excel files saved in each file's directory:")
    for excel_path in individual_excel_paths:
        print(f"  - {excel_path.name} (in {excel_path.parent.name}/)")
    if not args.no_visualization:
        print(f"\nFilter visualizations saved:")
        for file_result in all_results:
            if file_result.get('vis_path'):
                print(f"  - {Path(file_result['vis_path']).name}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
