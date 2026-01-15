#!/usr/bin/env python3
"""
Patch peakdetect package to fix scipy ifft import issue.

In scipy >= 1.10, ifft was moved from scipy.ifft to scipy.fft.ifft.
This script patches peakdetect to handle both old and new scipy versions.
"""

import os
import sys
from pathlib import Path

def patch_peakdetect():
    """Patch peakdetect/peakdetect.py to fix scipy import"""
    
    # Find peakdetect package
    peakdetect_file = None
    for path in sys.path:
        candidate = Path(path) / "peakdetect" / "peakdetect.py"
        if candidate.exists():
            peakdetect_file = candidate
            break
    
    if not peakdetect_file:
        print("WARNING: Could not find peakdetect/peakdetect.py")
        print("This is not critical - the package may not be installed yet.")
        return False
    
    print(f"Found peakdetect at: {peakdetect_file}")
    
    # Read the file
    with open(peakdetect_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "from scipy.fft import" in content or "try:" in content and "from scipy import fft, ifft" in content:
        # Check if it's our patch
        if "except ImportError:" in content and "scipy.fft" in content:
            print("✓ peakdetect is already patched")
            return True
    
    # Find the import line
    old_import = "from scipy import fft, ifft"
    new_import = """try:
    from scipy import fft, ifft
except ImportError:
    # For scipy >= 1.10, ifft moved to scipy.fft
    try:
        from scipy.fft import fft, ifft
    except ImportError:
        # Fallback: use numpy's fft
        from numpy.fft import fft, ifft"""
    
    if old_import in content:
        # Backup original file
        backup_file = peakdetect_file.with_suffix('.py.backup')
        if not backup_file.exists():
            with open(backup_file, 'w') as f:
                f.write(content)
            print(f"Created backup at: {backup_file}")
        
        # Apply patch
        content = content.replace(old_import, new_import)
        
        # Write patched file
        with open(peakdetect_file, 'w') as f:
            f.write(content)
        
        print("✓ Successfully patched peakdetect")
        return True
    else:
        print("WARNING: Could not find expected import line in peakdetect.py")
        print("The file may have a different structure.")
        return False

if __name__ == "__main__":
    success = patch_peakdetect()
    sys.exit(0 if success else 1)
