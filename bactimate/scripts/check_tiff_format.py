#!/usr/bin/env python3
"""
Helper script to check and optionally reshape TIFF files for Omnipose GUI.

Omnipose expects TIFF files in ZCYX format:
- Z: Number of time points (or z-slices)
- C: Number of channels
- Y: Image height
- X: Image width
"""

import sys
import numpy as np
import tifffile

def check_tiff_format(filename):
    """Check the format of a TIFF file and suggest reshaping if needed."""
    print(f"\nAnalyzing: {filename}")
    print("-" * 60)
    
    try:
        data = tifffile.imread(filename)
        original_shape = data.shape
        print(f"Current shape: {original_shape}")
        print(f"Number of dimensions: {len(original_shape)}")
        
        # Determine current format
        if len(original_shape) == 2:
            print("\nFormat: 2D image (Y, X)")
            print("⚠️  This is a single 2D image, not a sequence.")
            print("   Omnipose will treat this as a single time point.")
            
        elif len(original_shape) == 3:
            # Could be (T, Y, X), (Z, Y, X), (Y, X, C), or (C, Y, X)
            first_dim, second_dim, third_dim = original_shape
            
            if first_dim < 10 and third_dim < 10:
                # Likely (Y, X, C) - channels last
                print("\nFormat: (Y, X, C) - channels last")
                print("⚠️  Needs reshaping to ZCYX format")
                print(f"   Suggested: Reshape to (1, {third_dim}, {first_dim}, {second_dim})")
                print("   Command: data.transpose(2, 0, 1)[np.newaxis, ...]")
                
            elif first_dim < 10:
                # Likely (C, Y, X) - channels first
                print("\nFormat: (C, Y, X) - channels first, single time point")
                print("⚠️  Needs time dimension added")
                print(f"   Suggested: Reshape to (1, {first_dim}, {second_dim}, {third_dim})")
                print("   Command: data[np.newaxis, ...]")
                
            else:
                # Likely (T, Y, X) or (Z, Y, X) - single channel time sequence
                print("\nFormat: (T, Y, X) or (Z, Y, X) - single channel sequence")
                print("⚠️  Needs channel dimension added")
                print(f"   Suggested: Reshape to ({first_dim}, 1, {second_dim}, {third_dim})")
                print("   Command: data[:, np.newaxis, :, :]")
                
        elif len(original_shape) == 4:
            first_dim, second_dim, third_dim, fourth_dim = original_shape
            
            if first_dim < 10 and fourth_dim < 10:
                # Likely (Y, X, Z, C) or (T, Y, X, C)
                print("\nFormat: 4D with channels last")
                print("⚠️  Needs reordering to ZCYX")
                print("   Suggested: Reshape to (Z, C, Y, X)")
                
            elif first_dim < 10:
                # Likely (C, Z, Y, X) or (C, T, Y, X)
                print("\nFormat: (C, Z, Y, X) or (C, T, Y, X)")
                print("⚠️  Needs reordering to ZCYX")
                print(f"   Suggested: Reshape to ({second_dim}, {first_dim}, {third_dim}, {fourth_dim})")
                print("   Command: data.transpose(1, 0, 2, 3)")
                
            else:
                # Could be (T, C, Y, X) or (Z, C, Y, X) - might be correct!
                if second_dim < 10:
                    print("\nFormat: (T, C, Y, X) or (Z, C, Y, X)")
                    print("✅ This format should work with Omnipose!")
                    print(f"   Interpreted as: {first_dim} time points, {second_dim} channels")
                    print(f"   Image size: {third_dim} x {fourth_dim}")
                else:
                    print("\nFormat: 4D but unclear structure")
                    print("   Please verify the dimensions manually")
                    
        elif len(original_shape) == 5:
            print("\nFormat: 5D image")
            print("⚠️  Omnipose GUI typically handles up to 4D (ZCYX)")
            print("   You may need to reshape or use command-line interface")
            
        else:
            print("\n⚠️  Unusual number of dimensions")
            print("   Please check the file format")
            
        print("\n" + "=" * 60)
        print("Omnipose expects: (Z, C, Y, X)")
        print("  Z = time points or z-slices")
        print("  C = channels (1-3)")
        print("  Y = height")
        print("  X = width")
        print("=" * 60)
        
        return data, original_shape
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None, None


def reshape_to_zcyx(data, target_shape=None):
    """Reshape data to ZCYX format."""
    if data is None:
        return None
        
    original_shape = data.shape
    
    if len(original_shape) == 2:
        # (Y, X) -> (1, 1, Y, X)
        return data[np.newaxis, np.newaxis, :, :]
        
    elif len(original_shape) == 3:
        first, second, third = original_shape
        
        if first < 10 and third < 10:
            # (Y, X, C) -> (1, C, Y, X)
            return data.transpose(2, 0, 1)[np.newaxis, ...]
        elif first < 10:
            # (C, Y, X) -> (1, C, Y, X)
            return data[np.newaxis, ...]
        else:
            # (T, Y, X) -> (T, 1, Y, X)
            return data[:, np.newaxis, :, :]
            
    elif len(original_shape) == 4:
        first, second, third, fourth = original_shape
        
        if first < 10 and fourth < 10:
            # (Y, X, Z, C) -> (Z, C, Y, X)
            return data.transpose(2, 3, 0, 1)
        elif first < 10:
            # (C, Z, Y, X) -> (Z, C, Y, X)
            return data.transpose(1, 0, 2, 3)
        elif second < 10:
            # (T, C, Y, X) - already correct!
            return data
        else:
            print("⚠️  Unclear 4D format, returning as-is")
            return data
            
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_tiff_format.py <tiff_file> [output_file]")
        print("\nIf output_file is provided, the reshaped image will be saved.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    data, original_shape = check_tiff_format(input_file)
    
    if data is not None and output_file:
        print(f"\nReshaping to ZCYX format...")
        reshaped = reshape_to_zcyx(data)
        print(f"New shape: {reshaped.shape}")
        
        print(f"\nSaving to: {output_file}")
        tifffile.imwrite(output_file, reshaped)
        print("✅ Done!")
    elif data is not None:
        print("\n💡 Tip: To automatically reshape and save, provide an output filename:")
        print(f"   python check_tiff_format.py {input_file} output_reshaped.tif")


