#!/usr/bin/env python3
"""
Script to convert ND2 microscopy files to multi-page TIFF files.
Extracts frames 1-150 from each ND2 file and saves them as TIFF files.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import nd2
import tifffile
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from skimage.exposure import match_histograms


def drift_correct_stack(frames: np.ndarray, reference_frame: int = 0, method: str = 'frame_by_frame') -> np.ndarray:
    """
    Correct for X,Y drift in a time series using phase cross-correlation.
    
    Args:
        frames: Array of shape (T, Y, X) or (T, C, Y, X)
        reference_frame: Index of reference frame (for 'all_to_first' method, default: 0)
        method: 'frame_by_frame' (recommended, more robust) or 'all_to_first' (no error accumulation)
        
    Returns:
        Drift-corrected frames
    """
    if method == 'all_to_first':
        # Original method: align all frames to first frame
        if len(frames.shape) == 3:
            # Single channel: TYX
            corrected = np.zeros_like(frames)
            reference = frames[reference_frame]
            
            for t in range(frames.shape[0]):
                if t == reference_frame:
                    corrected[t] = frames[t]
                else:
                    # Calculate shift using phase cross-correlation
                    shift_result = phase_cross_correlation(
                        reference,
                        frames[t],
                        upsample_factor=10
                    )
                    # shift_result is a tuple: (shift, error, phasediff)
                    shift_y, shift_x = shift_result[0]
                    
                    # Apply shift
                    corrected[t] = shift(frames[t], shift=(shift_y, shift_x), mode='constant', cval=0)
            
            return corrected
        
        elif len(frames.shape) == 4:
            # Multi-channel: TCYX
            corrected = np.zeros_like(frames)
            reference = frames[reference_frame]
            
            for t in range(frames.shape[0]):
                if t == reference_frame:
                    corrected[t] = frames[t]
                else:
                    # Correct each channel independently
                    for c in range(frames.shape[1]):
                        # Use first channel as reference for shift calculation
                        if c == 0:
                            shift_result = phase_cross_correlation(
                                reference[0],
                                frames[t, 0],
                                upsample_factor=10
                            )
                            # shift_result is a tuple: (shift, error, phasediff)
                            shift_y, shift_x = shift_result[0]
                        
                        # Apply same shift to all channels
                        corrected[t, c] = shift(
                            frames[t, c],
                            shift=(shift_y, shift_x),
                            mode='constant',
                            cval=0
                        )
            
            return corrected
    
    elif method == 'frame_by_frame':
        # Frame-by-frame: align each frame to previous frame (more robust for large drifts)
        if len(frames.shape) == 3:
            # Single channel: TYX
            corrected = np.zeros_like(frames)
            corrected[0] = frames[0]  # First frame unchanged
            
            # Track cumulative shifts for potential smoothing (not used yet, but could be added)
            cumulative_shift_y = 0.0
            cumulative_shift_x = 0.0
            
            for t in range(1, frames.shape[0]):
                # Align to previous corrected frame
                shift_result = phase_cross_correlation(
                    corrected[t - 1],
                    frames[t],
                    upsample_factor=10
                )
                # shift_result is a tuple: (shift, error, phasediff)
                shift_y, shift_x = shift_result[0]
                
                # Update cumulative shifts
                cumulative_shift_y += shift_y
                cumulative_shift_x += shift_x
                
                # Apply shift
                corrected[t] = shift(frames[t], shift=(shift_y, shift_x), mode='constant', cval=0)
            
            return corrected
        
        elif len(frames.shape) == 4:
            # Multi-channel: TCYX
            corrected = np.zeros_like(frames)
            corrected[0] = frames[0]  # First frame unchanged
            
            for t in range(1, frames.shape[0]):
                # Use first channel for shift calculation
                shift_result = phase_cross_correlation(
                    corrected[t - 1, 0],
                    frames[t, 0],
                    upsample_factor=10
                )
                # shift_result is a tuple: (shift, error, phasediff)
                shift_y, shift_x = shift_result[0]
                
                # Apply same shift to all channels
                for c in range(frames.shape[1]):
                    corrected[t, c] = shift(
                        frames[t, c],
                        shift=(shift_y, shift_x),
                        mode='constant',
                        cval=0
                    )
            
            return corrected
    
    else:
        raise ValueError(f"Unknown drift correction method: {method}. Use 'frame_by_frame' or 'all_to_first'")
    
    raise ValueError(f"Unexpected frame shape: {frames.shape}")


def bleach_correct_stack(
    frames: np.ndarray,
    method: str = 'exponential',
    channel_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Correct for photobleaching in fluorescence time series.
    Inspired by Fiji CorrectBleach plugin (https://github.com/fiji/CorrectBleach).

    Which channels: For multi-channel (T, C, Y, X), if channel_indices is given then
    only those channel indices are corrected; others are left unchanged. If None,
    all channels are corrected.

    Args:
        frames: Array of shape (T, Y, X) or (T, C, Y, X)
        method: One of:
            - 'exponential': Intensity preservation. I(t) = I0*exp(-t/τ), scale so
              mean intensity matches first frame (like Fiji "Exponential with offset").
            - 'linear': Linear decay fit, then scale to first frame.
            - 'histogram_matching': Match each frame's histogram to frame 0 (like
              Fiji "Histogram Matching"). Good for segmentation; not for quantification.
            - 'per_frame_minmax': Per-frame min-max stretch. Each frame is scaled so
              its minimum maps to 0 and maximum to the max of the input dtype (e.g.
              255 for uint8, 65535 for uint16). Bit depth and dtype are unchanged;
              output is still grayscale (no RGB). Good for visualization only.

    Returns:
        Bleach-corrected frames (same shape and dtype as input where possible).
    """
    if len(frames.shape) == 3:
        # Single channel: TYX
        if method == 'per_frame_minmax':
            corrected = np.zeros_like(frames, dtype=np.float64)
            out_max = (np.iinfo(frames.dtype).max if np.issubdtype(frames.dtype, np.integer) else 1.0)
            for t in range(frames.shape[0]):
                fm = frames[t].astype(np.float64)
                lo, hi = fm.min(), fm.max()
                if hi > lo:
                    corrected[t] = (fm - lo) / (hi - lo) * out_max
                else:
                    corrected[t] = fm
            return np.clip(corrected, 0, out_max).astype(frames.dtype)
        if method == 'histogram_matching':
            corrected = frames.copy()
            ref = frames[0]
            for t in range(1, frames.shape[0]):
                corrected[t] = match_histograms(frames[t], ref, channel_axis=None).astype(frames.dtype)
            return corrected
        # exponential or linear
        mean_intensities = np.mean(frames, axis=(1, 2))
        if method == 'exponential':
            I0 = mean_intensities[0]
            I_final = mean_intensities[-1]
            if I_final > 0 and I0 > I_final:
                tau = -len(mean_intensities) / np.log(I_final / I0)
            else:
                tau = len(mean_intensities) * 10
            t = np.arange(len(mean_intensities))
            expected = I0 * np.exp(-t / tau)
            correction_factors = I0 / np.maximum(expected, 1e-10)
        else:
            I0 = mean_intensities[0]
            I_final = mean_intensities[-1]
            if I_final < I0:
                slope = (I0 - I_final) / len(mean_intensities)
                expected = I0 - slope * np.arange(len(mean_intensities))
                correction_factors = I0 / np.maximum(expected, 1e-10)
            else:
                correction_factors = np.ones(len(mean_intensities))
        corrected = frames.copy()
        for t in range(frames.shape[0]):
            corrected[t] = (frames[t].astype(np.float64) * correction_factors[t]).astype(frames.dtype)
        return corrected

    if len(frames.shape) == 4:
        # Multi-channel: TCYX - correct only channels in channel_indices (or all if None)
        corrected = frames.copy()
        channels_to_correct = range(frames.shape[1]) if channel_indices is None else channel_indices
        for c in channels_to_correct:
            if c < 0 or c >= frames.shape[1]:
                continue
            channel_frames = frames[:, c, :, :]
            corrected[:, c, :, :] = bleach_correct_stack(channel_frames, method)
        return corrected

    raise ValueError(f"Unexpected frame shape: {frames.shape}")


def convert_nd2_to_tif(
    input_path: Path,
    output_dir: Optional[Path] = None,
    max_frames: int = 150,
    separate_channels: bool = True,  # Default to True for multi-channel compatibility
    channel_index: Optional[int] = None,
    drift_correct: bool = False,
    bleach_correct: bool = False,
    bleach_method: str = 'exponential',  # exponential | linear | histogram_matching | per_frame_minmax
    bleach_channel_indices: Optional[List[int]] = None,  # If set, only these channel indices get correction; else all
    pixel_size: Optional[float] = None,  # Pixel size in µm to save in metadata
    max_z_projection: bool = False,  # Max intensity projection across Z-slices
) -> None:
    """
    Convert ND2 files to multi-page TIFF files.
    
    Args:
        input_path: Path to directory containing ND2 files or a single ND2 file
        output_dir: Optional output directory. If None, creates 'Tif_1-150' in input directory
        max_frames: Maximum number of frames to extract (default: 150)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")
    
    # Determine ND2 files to process
    if input_path.is_file() and input_path.suffix.lower() == '.nd2':
        nd2_files = [input_path]
        base_dir = input_path.parent
    elif input_path.is_dir():
        nd2_files = list(input_path.glob('*.nd2')) + list(input_path.glob('*.ND2'))
        base_dir = input_path
    else:
        raise ValueError(f"Path must be a directory or an ND2 file: {input_path}")
    
    if not nd2_files:
        print(f"No ND2 files found in {input_path}")
        return
    
    # Create output directory
    if output_dir is None:
        # Dynamic folder name based on number of frames
        output_dir = base_dir / f"Tif_1-{max_frames}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(nd2_files)} ND2 file(s) to process")
    print(f"Output directory: {output_dir}")
    
    # Process each ND2 file
    for nd2_file in nd2_files:
        print(f"\nProcessing: {nd2_file.name}")
        
        try:
            # Open ND2 file
            with nd2.ND2File(nd2_file) as nd2_reader:
                # Get dimension information
                sizes = nd2_reader.sizes
                dims = nd2_reader.sizes.keys() if hasattr(nd2_reader.sizes, 'keys') else []
                
                # Get dimensions
                num_time = sizes.get('T', 1)
                num_channels = sizes.get('C', 1)
                num_z = sizes.get('Z', 1)
                
                # Get channel names if available
                channel_names = []
                if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                    # Try to get channel names from metadata
                    try:
                        meta = nd2_reader.metadata
                        # Metadata is an object, not a dict - access channels attribute
                        if hasattr(meta, 'channels') and meta.channels:
                            channel_names = [ch.channel.name if hasattr(ch, 'channel') and hasattr(ch.channel, 'name') 
                                           else f'Channel_{i+1}' 
                                           for i, ch in enumerate(meta.channels)]
                    except Exception:
                        pass
                
                # If channel names not found, create default names
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                print(f"  Dimensions: T={num_time}, C={num_channels}, Z={num_z}")
                if num_channels > 1:
                    print(f"  Channels: {', '.join(channel_names)}")
                
                # Determine number of frames to extract
                frames_to_extract = min(max_frames, num_time)
                print(f"  Extracting frames 1-{frames_to_extract}")
                
                # Extract pixel size from ND2 metadata if not provided
                extracted_pixel_size = pixel_size
                if extracted_pixel_size is None:
                    try:
                        if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                            meta = nd2_reader.metadata
                            # Pixel size is in volume.axesCalibration (first element is X, second is Y)
                            # Both should be the same for square pixels
                            if hasattr(meta, 'channels') and meta.channels:
                                # Get pixel size from first channel's volume
                                first_channel = meta.channels[0]
                                if hasattr(first_channel, 'volume') and first_channel.volume:
                                    vol = first_channel.volume
                                    if hasattr(vol, 'axesCalibration') and vol.axesCalibration:
                                        # axesCalibration is a list: [X, Y, Z] pixel sizes
                                        if len(vol.axesCalibration) >= 2:
                                            # Use X pixel size (should be same as Y for square pixels)
                                            extracted_pixel_size = vol.axesCalibration[0]
                            
                            if extracted_pixel_size:
                                print(f"  Extracted pixel size: {extracted_pixel_size:.4f} µm")
                    except Exception as e:
                        print(f"  Warning: Could not extract pixel size: {e}")
                
                # Read all frames
                print(f"  Reading frames...")
                all_frames = nd2_reader.asarray()
                
                # Handle different dimension orders
                # ND2 files can have: T, C, Z, Y, X in various orders
                # We need to extract: T (first frames_to_extract), C (all or selected), Z (first slice), Y, X
                
                # Get the shape and determine dimension order
                shape = all_frames.shape
                ndim = len(shape)
                
                print(f"  Debug: Loaded array shape: {shape}, ndim: {ndim}, size: {all_frames.size}")
                print(f"  Debug: Reported sizes - T={num_time}, C={num_channels}, Z={num_z}")
                
                # Determine which dimensions we have
                # Common orders: TCYX, CTYX, TZYX, TCZYX, TYX, CYX (when T=1), etc.
                if ndim < 2:
                    print(f"  ✗ Error: Unexpected array shape: {shape}, skipping")
                    continue
                
                if all_frames.size == 0:
                    print(f"  ✗ Error: Loaded array is empty! Shape: {shape}")
                    continue
                
                # Detect dimension order by comparing shape with reported sizes
                # Handle special case: when T=1, ND2 might return (C, Y, X) instead of (T, C, Y, X)
                # IMPORTANT: Check 3D case FIRST and handle (C, Y, X) before any slicing to avoid losing channels
                if ndim == 3:
                    # 3D array - could be (T, Y, X), (C, Y, X), or (Z, Y, X)
                    # Priority: Check if it's (C, Y, X) when we have multiple channels
                    if num_channels > 1 and shape[0] == num_channels:
                        # It's (C, Y, X) - channels first (common when T=1)
                        print(f"  Debug: Detected (C, Y, X) format - C={num_channels}, reshaping to (T, C, Y, X)")
                        # Reshape to (T, C, Y, X): add time dimension
                        all_frames = all_frames[np.newaxis, :, :, :]  # (1, C, Y, X) or (T, C, Y, X)
                        shape = all_frames.shape
                        ndim = len(shape)
                        print(f"  Debug: Reshaped to (T, C, Y, X), new shape: {shape}")
                    elif num_time > 1 and num_channels == 1 and shape[0] == num_time:
                        # (T, Y, X) format - single channel, multiple timepoints
                        print(f"  Debug: Detected (T, Y, X) format")
                        # Add channel dimension: (T, 1, Y, X)
                        all_frames = all_frames[:, np.newaxis, :, :]
                        shape = all_frames.shape
                        ndim = len(shape)
                    elif num_time == 1 and num_channels == 1:
                        # Single frame, single channel: (Y, X) - add both dimensions
                        print(f"  Debug: Detected single frame/channel, adding dimensions")
                        all_frames = all_frames[np.newaxis, np.newaxis, :, :]  # (1, 1, Y, X)
                        shape = all_frames.shape
                        ndim = len(shape)
                    else:
                        # Try to infer from shape[0]
                        print(f"  Debug: 3D array - shape[0]={shape[0]}, num_time={num_time}, num_channels={num_channels}")
                        if shape[0] == num_channels and num_channels > 1:
                            # Treat as (C, Y, X)
                            print(f"  Debug: Inferring (C, Y, X) from shape[0]==num_channels")
                            all_frames = all_frames[np.newaxis, :, :, :]
                            shape = all_frames.shape
                            ndim = len(shape)
                        elif shape[0] == num_time and num_time > 1:
                            # Treat as (T, Y, X)
                            print(f"  Debug: Inferring (T, Y, X) from shape[0]==num_time")
                            all_frames = all_frames[:, np.newaxis, :, :]
                            shape = all_frames.shape
                            ndim = len(shape)
                        else:
                            print(f"  Warning: Cannot determine 3D format, shape[0]={shape[0]}, defaulting to (C, Y, X) if num_channels>1")
                            # Last resort: if num_channels > 1, assume (C, Y, X)
                            if num_channels > 1:
                                all_frames = all_frames[np.newaxis, :, :, :]
                                shape = all_frames.shape
                                ndim = len(shape)
                                print(f"  Debug: Defaulted to (C, Y, X) -> (T, C, Y, X), new shape: {shape}")
                elif ndim == 4:
                    # 4D array - could be (T, C, Y, X), (C, T, Y, X), (T, Z, Y, X), etc.
                    if shape[0] == num_channels and shape[1] == num_time and num_channels > 1:
                        # Likely CTYX order (channels first)
                        print(f"  Debug: Detected CTYX dimension order (channels first)")
                        # Transpose to TCYX: CTYX -> TCYX
                        all_frames = all_frames.transpose(1, 0, 2, 3)
                        shape = all_frames.shape
                        print(f"  Debug: Transposed to TCYX, new shape: {shape}")
                    elif shape[0] == num_time:
                        # Likely TCYX order (time first) - this is what we want
                        print(f"  Debug: Detected TCYX dimension order (time first)")
                    else:
                        print(f"  Debug: Dimension order unclear, shape[0]={shape[0]}, num_time={num_time}, num_channels={num_channels}")
                
                # Extract time frames (first dimension should now be T after potential reshape/transpose)
                # Always slice to ensure we have exactly frames_to_extract frames
                if shape[0] > frames_to_extract:
                    all_frames = all_frames[:frames_to_extract]
                    print(f"  Debug: Sliced to {frames_to_extract} frames, new shape: {all_frames.shape}, size: {all_frames.size}")
                elif shape[0] < frames_to_extract:
                    # If we have fewer frames than requested, use what we have
                    frames_to_extract = shape[0]
                    print(f"  Warning: Only {frames_to_extract} frames available, using all available frames")
                else:
                    print(f"  Debug: Using all {frames_to_extract} frames")
                
                shape = all_frames.shape
                ndim = len(shape)
                
                # Verify we have data
                if all_frames.size == 0:
                    print(f"  ✗ Error: Extracted frames array is empty! Shape: {shape}")
                    print(f"  ✗ This should not happen - check frame extraction logic")
                    continue
                
                # Handle Z dimension first - take first Z slice or max projection if multiple exist
                # This simplifies subsequent channel handling
                if num_z > 1:
                    if max_z_projection:
                        # Apply max intensity projection across Z-slices
                        if ndim == 5:  # TCZYX
                            print(f"  Applying max intensity projection across Z={num_z} slices...")
                            all_frames = np.max(all_frames, axis=2)  # Max projection: TCYX
                            shape = all_frames.shape
                            ndim = len(shape)
                            print(f"  ✓ Max Z-projection complete: {shape}")
                        elif ndim == 4 and num_z > 1 and num_channels == 1:  # TZYX (no channels)
                            print(f"  Applying max intensity projection across Z={num_z} slices...")
                            all_frames = np.max(all_frames, axis=1)  # Max projection: TYX
                            shape = all_frames.shape
                            ndim = len(shape)
                            print(f"  ✓ Max Z-projection complete: {shape}")
                    else:
                        # Use first Z slice only
                        if ndim == 5:  # TCZYX
                            all_frames = all_frames[:, :, 0, :, :]  # Take first Z slice: TCYX
                            shape = all_frames.shape
                            ndim = len(shape)
                            print(f"  Using first Z-slice (Z=1)")
                        elif ndim == 4 and num_z > 1 and num_channels == 1:  # TZYX (no channels)
                            all_frames = all_frames[:, 0, :, :]  # Take first Z slice: TYX
                            shape = all_frames.shape
                            ndim = len(shape)
                            print(f"  Using first Z-slice (Z=1)")
                
                # Handle channel selection
                if channel_index is not None:
                    # Extract specific channel
                    if channel_index >= num_channels:
                        print(f"  Warning: Channel index {channel_index} >= {num_channels}, using channel 0")
                        channel_index = 0
                    
                    if num_channels > 1:
                        if ndim == 4:  # TCYX (after Z reduction if needed)
                            all_frames = all_frames[:, channel_index, :, :]  # Extract channel: TYX
                            print(f"  Extracted channel {channel_index + 1}: {channel_names[channel_index]}")
                            num_channels = 1  # Now it's single channel
                        else:
                            print(f"  Warning: Could not extract channel, unexpected shape: {shape}")
                
                # Now all_frames should be TYX or TCYX
                # Apply corrections if requested
                if drift_correct:
                    print(f"  Applying drift correction (frame-by-frame method)...")
                    all_frames = drift_correct_stack(all_frames, method='frame_by_frame')
                    print(f"  ✓ Drift correction complete")
                
                if bleach_correct:
                    ch_list = bleach_channel_indices
                    if ch_list is not None:
                        print(f"  Applying bleach correction ({bleach_method}) to channels: {ch_list}...")
                    else:
                        print(f"  Applying bleach correction ({bleach_method}) to all channels...")
                    all_frames = bleach_correct_stack(
                        all_frames, method=bleach_method, channel_indices=bleach_channel_indices
                    )
                    print(f"  ✓ Bleach correction complete")
                
                # Process channels
                # Note: Standard TIFF cannot store multiple grayscale channels per page
                # Options:
                # 1. Save channels separately (default): Each file has T pages (timepoints)
                # 2. Save as OME-TIFF: One file with T*C pages, but OME metadata groups them correctly
                
                # Debug: Print final shape before channel processing
                print(f"  Debug: Final shape before channel processing: {all_frames.shape}, num_channels={num_channels}, separate_channels={separate_channels}")
                
                if num_channels > 1 and channel_index is None:
                    if separate_channels:
                        # Multi-channel: save each channel separately (default and recommended)
                        # Check if we have TCYX format (4D) or need to reshape
                        if len(all_frames.shape) == 4:  # TCYX
                            # Verify the channel dimension matches
                            if all_frames.shape[1] == num_channels:
                                print(f"  Saving {num_channels} channels as separate files (each with {frames_to_extract} timepoints)")
                                for ch_idx in range(num_channels):
                                    channel_data = all_frames[:, ch_idx, :, :]  # Extract channel: TYX
                                    channel_name = channel_names[ch_idx]
                                    
                                    # Generate output filename with channel
                                    output_filename = f"{nd2_file.stem}_{channel_name}.tif"
                                    output_path = output_dir / output_filename
                                    
                                    print(f"    Saving channel {ch_idx + 1}/{num_channels}: {channel_name} (shape: {channel_data.shape})")
                                    _save_tiff_with_metadata(output_path, channel_data, frames_to_extract, pixel_size=extracted_pixel_size)
                            else:
                                print(f"  Warning: Shape mismatch - all_frames.shape[1]={all_frames.shape[1]} but num_channels={num_channels}")
                                print(f"  Falling back to single file output")
                                output_filename = nd2_file.stem + '.tif'
                                output_path = output_dir / output_filename
                                _save_tiff_with_metadata(output_path, all_frames, frames_to_extract, pixel_size=extracted_pixel_size)
                        elif len(all_frames.shape) == 3 and num_channels == 1:
                            # Single channel already extracted
                            output_filename = nd2_file.stem + '.tif'
                            output_path = output_dir / output_filename
                            _save_tiff_with_metadata(output_path, all_frames, frames_to_extract, pixel_size=extracted_pixel_size)
                        else:
                            # Unexpected shape - try to handle it
                            print(f"  Warning: Unexpected shape {all_frames.shape} for {num_channels} channels")
                            print(f"  Attempting to save channels separately anyway...")
                            # Try to reshape if possible
                            if all_frames.shape[0] == num_channels and len(all_frames.shape) == 4:
                                # Might be CTYX - transpose
                                all_frames = all_frames.transpose(1, 0, 2, 3)
                                print(f"  Transposed to TCYX, new shape: {all_frames.shape}")
                            if len(all_frames.shape) == 4 and all_frames.shape[1] == num_channels:
                                for ch_idx in range(num_channels):
                                    channel_data = all_frames[:, ch_idx, :, :]
                                    channel_name = channel_names[ch_idx]
                                    output_filename = f"{nd2_file.stem}_{channel_name}.tif"
                                    output_path = output_dir / output_filename
                                    print(f"    Saving channel {ch_idx + 1}/{num_channels}: {channel_name}")
                                    _save_tiff_with_metadata(output_path, channel_data, frames_to_extract, pixel_size=extracted_pixel_size)
                            else:
                                # Fallback: save as single file
                                output_filename = nd2_file.stem + '.tif'
                                output_path = output_dir / output_filename
                                _save_tiff_with_metadata(output_path, all_frames, frames_to_extract, pixel_size=extracted_pixel_size)
                    else:
                        # Save as single OME-TIFF file with all channels
                        output_filename = nd2_file.stem + '.tif'
                        output_path = output_dir / output_filename
                        print(f"  Saving as single OME-TIFF file ({num_channels} channels, {frames_to_extract} timepoints)")
                        _save_multichannel_tiff_with_metadata(
                            output_path,
                            all_frames,
                            frames_to_extract,
                            num_channels,
                            channel_names,
                            pixel_size=extracted_pixel_size
                        )
                else:
                    # Single channel (either only one channel, or specific channel selected)
                    if channel_index is not None:
                        channel_suffix = f"_{channel_names[channel_index]}"
                    else:
                        channel_suffix = ""
                    
                    output_filename = nd2_file.stem + channel_suffix + '.tif'
                    output_path = output_dir / output_filename
                    
                    # Extract single channel if needed
                    if len(all_frames.shape) == 4:  # TCYX but channel_index specified
                        all_frames = all_frames[:, channel_index, :, :]  # Extract to TYX
                    
                    _save_tiff_with_metadata(output_path, all_frames, frames_to_extract, pixel_size=extracted_pixel_size)
            
        except Exception as e:
            print(f"  ✗ Error processing {nd2_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Conversion complete!")


def _save_multichannel_tiff_with_metadata(
    output_path: Path,
    frames: np.ndarray,
    num_frames: int,
    num_channels: int,
    channel_names: list,
    pixel_size: Optional[float] = None,
) -> None:
    """
    Save multi-channel time series as TIFF with proper metadata.
    Uses OME-TIFF format to properly handle multi-channel time series.
    
    Args:
        output_path: Path to output file
        frames: Array of shape (T, C, Y, X)
        num_frames: Number of time frames
        num_channels: Number of channels
        channel_names: List of channel names
    """
    print(f"    Saving {num_frames} timepoints with {num_channels} channels each")
    
    # Use OME-TIFF format for proper multi-channel time series support
    # OME-TIFF properly handles TCYX dimensions where each timepoint has all channels
    # The format stores it in a way that software can interpret as multi-channel time series
    
    # For multi-channel time series, standard TIFF format limitation:
    # TIFF doesn't support multiple grayscale channels per page (only RGB with 3 channels)
    # So we need to save channels separately but in a way that software can group them
    
    # Option: Save as OME-TIFF which properly handles multi-channel time series
    # However, it still stores as separate pages but with proper metadata
    
    # Better approach: Since TIFF can't have multiple grayscale channels per page,
    # we'll save each channel as a separate file with a naming convention that indicates
    # they belong together. This is the most compatible approach.
    
    # Actually, let's try saving as ImageJ hyperstack format which can handle this
    # by using proper metadata to indicate channels and timepoints
    
    # Save with ImageJ hyperstack metadata
    # Format: We'll save it as a hyperstack where pages are organized as timepoints
    # But since TIFF doesn't support multi-channel grayscale, we need to interleave
    # and use metadata to indicate the structure
    
    # Actually, the best solution for true multi-channel is to save channels separately
    # But since user wants one file, let's use OME-TIFF with proper structure
    # Software like ImageJ/Fiji and ilastik can read OME-TIFF and interpret it correctly
    
    # Save as OME-TIFF with TCYX structure
    # Note: This will create T*C pages, but OME metadata tells software how to interpret it
    metadata = {'axes': 'TCYX'}  # Time, Channel, Y, X
    if pixel_size is not None:
        metadata['pixelSize'] = pixel_size
        metadata['unit'] = 'um'  # Use 'um' instead of 'µm' for ASCII compatibility
        print(f"    Saving pixel size: {pixel_size:.4f} µm")
    
    tifffile.imwrite(
        str(output_path),
        frames,
        photometric='minisblack',
        metadata=metadata,
        ome=True,  # Use OME-TIFF format
    )
    
    print(f"    ✓ Saved as OME-TIFF: {num_frames} timepoints × {num_channels} channels")
    print(f"      Note: OME-TIFF stores {num_frames * num_channels} pages but metadata groups them correctly")
    print(f"      Software like ImageJ/Fiji and ilastik will interpret this as {num_frames} timepoints with {num_channels} channels each")


def _save_tiff_with_metadata(output_path: Path, frames: np.ndarray, num_frames: int, pixel_size: Optional[float] = None) -> None:
    """
    Save frames as TIFF with proper time dimension metadata and pixel size.
    
    Args:
        output_path: Path to output file
        frames: Array of frames (should be TYX or YX)
        num_frames: Number of time frames
        pixel_size: Pixel size in µm (optional, will be saved in metadata)
    """
    # Check if frames array is empty
    if frames.size == 0:
        raise ValueError(f"Cannot save empty frames array! Shape: {frames.shape}")
    
    # Ensure frames are in correct shape (T, Y, X)
    if len(frames.shape) == 2:
        # Single frame - add time dimension
        frames = frames[np.newaxis, :, :]
    elif len(frames.shape) == 3:
        # Already TYX
        pass
    else:
        raise ValueError(f"Unexpected frame shape: {frames.shape}, expected (T, Y, X) or (Y, X)")
    
    # Get frame dimensions
    actual_frames, height, width = frames.shape
    
    # Verify we have valid dimensions
    if actual_frames == 0 or height == 0 or width == 0:
        raise ValueError(f"Invalid frame dimensions: {frames.shape}")
    
    print(f"    Debug: Saving frames with shape {frames.shape}, dtype {frames.dtype}, size {frames.size}")
    
    # Save with ImageJ metadata to properly indicate time dimension
    metadata = {
        'axes': 'TYX',  # Time, Y, X (not ZYX)
        'ImageJ': '1.53c',
        'images': actual_frames,
        'slices': 1,  # Only 1 Z-slice
        'frames': actual_frames,  # Number of time frames
        'loop': False,
        'mode': 'grayscale',
    }
    
    # Add pixel size to metadata if available
    # Note: Use 'um' instead of 'µm' because ImageJ metadata requires ASCII encoding
    if pixel_size is not None:
        metadata['pixelSize'] = pixel_size
        metadata['unit'] = 'um'  # Use 'um' instead of 'µm' for ASCII compatibility
        print(f"    Saving pixel size: {pixel_size:.4f} µm")
    
    try:
        tifffile.imwrite(
            str(output_path),
            frames,
            photometric='minisblack',
            metadata=metadata,
            imagej=True,
        )
        
        # Verify the file was written correctly
        file_size = output_path.stat().st_size
        if file_size < 100:  # Very small file, likely empty or corrupted
            print(f"    ⚠ Warning: Saved file is very small ({file_size} bytes), may be empty!")
            print(f"    ⚠ Check if frames array was empty or write failed")
        else:
            print(f"    ✓ Saved {actual_frames} frames to {output_path.name} ({file_size:,} bytes)")
    except Exception as e:
        print(f"    ✗ Error writing TIFF file: {e}")
        import traceback
        traceback.print_exc()
        raise


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert ND2 microscopy files to multi-page TIFF files (frames 1-150)"
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to directory containing ND2 files or a single ND2 file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory (default: creates "Tif_1-150" folder in input directory)'
    )
    parser.add_argument(
        '-f', '--frames',
        type=int,
        default=150,
        help='Maximum number of frames to extract (default: 150)'
    )
    parser.add_argument(
        '--separate-channels',
        action='store_true',
        default=True,
        help='Save each channel as a separate file (default: True). Each file will have T pages (timepoints). Note: Standard TIFF cannot store multiple grayscale channels per page, so this is the recommended approach for multi-channel data.'
    )
    parser.add_argument(
        '--combine-channels',
        action='store_true',
        help='Save all channels in one OME-TIFF file (overrides --separate-channels). Note: This will create T*C pages but OME metadata allows software to interpret it as T timepoints with C channels each.'
    )
    parser.add_argument(
        '--channel',
        type=int,
        default=None,
        help='Extract only a specific channel (0-indexed). If not specified, all channels are processed'
    )
    parser.add_argument(
        '--drift-correct',
        action='store_true',
        help='Apply X,Y drift correction using phase cross-correlation (like Fiji Fast4DReg)'
    )
    parser.add_argument(
        '--bleach-correct',
        action='store_true',
        help='Apply photobleaching correction (Fiji CorrectBleach-style). Applied to all channels unless --channel is set.'
    )
    parser.add_argument(
        '--bleach-method',
        type=str,
        choices=['exponential', 'linear', 'histogram_matching', 'per_frame_minmax'],
        default='exponential',
        help='Bleach correction method: exponential (intensity preservation), linear, histogram_matching (Fiji-like), per_frame_minmax (per-frame min-max stretch, preserves bit depth, visualization only)'
    )
    parser.add_argument(
        '--bleach-channels',
        type=int,
        nargs='*',
        default=None,
        metavar='N',
        help='Channel indices (0-based) to apply bleach correction to. If omitted and --bleach-correct is set, all channels are corrected. Example: --bleach-channels 0 2'
    )
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=None,
        help='Pixel size in µm to save in TIFF metadata (if not provided, will try to extract from ND2)'
    )
    parser.add_argument(
        '--max-z-projection',
        action='store_true',
        help='Apply maximum intensity projection across Z-slices when Z > 1. If not set, only the first Z-slice (Z=1) is used.'
    )
    
    args = parser.parse_args()
    
    # Handle channel options
    separate_channels = args.separate_channels and not args.combine_channels
    
    try:
        convert_nd2_to_tif(
            input_path=Path(args.path),
            output_dir=Path(args.output) if args.output else None,
            max_frames=args.frames,
            separate_channels=separate_channels,
            channel_index=args.channel,
            drift_correct=args.drift_correct,
            bleach_correct=args.bleach_correct,
            bleach_method=args.bleach_method,
            bleach_channel_indices=args.bleach_channels,
            pixel_size=args.pixel_size,
            max_z_projection=args.max_z_projection,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

