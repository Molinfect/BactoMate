#!/usr/bin/env python3
"""
Apply all necessary patches to fix compatibility issues.
This script checks and applies patches for:
1. peakdetect - scipy ifft import fix
2. cellpose_omni - model_type parameter fix
3. cellpose_omni - model URL fix (use GitHub instead of Cellpose server)
"""

import os
import sys
import subprocess

def run_patch_script(script_name, description):
    """Run a patch script and return True if successful"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        print(f"⚠ Warning: {script_name} not found")
        return False
    
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            # Print output but filter out "already patched" messages
            output = result.stdout.strip()
            if output:
                for line in output.split('\n'):
                    if line.strip():
                        print(f"  {line}")
            print(f"  ✓ {description} - OK")
            return True
        else:
            print(f"  ✗ {description} failed:")
            print(f"    {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  ✗ Error running {script_name}: {e}")
        return False

def main():
    """Apply all patches"""
    print("=" * 50)
    print("Applying BactiMate compatibility patches")
    print("=" * 50)
    
    patches_applied = 0
    patches_skipped = 0
    
    # Apply peakdetect patch
    if run_patch_script("patch_peakdetect.py", "Patching peakdetect (scipy compatibility)"):
        patches_applied += 1
    else:
        patches_skipped += 1
    
    # Apply cellpose_omni model_type patch
    if run_patch_script("patch_cellpose_models.py", "Patching cellpose_omni (model_type fix)"):
        patches_applied += 1
    else:
        patches_skipped += 1
    
    # Apply cellpose_omni URL patch
    if run_patch_script("patch_omnipose_model_url.py", "Patching cellpose_omni (model URL fix)"):
        patches_applied += 1
    else:
        patches_skipped += 1
    
    print("\n" + "=" * 50)
    print(f"Patches applied: {patches_applied}")
    if patches_skipped > 0:
        print(f"Patches skipped (already applied or not needed): {patches_skipped}")
    print("=" * 50)
    
    return patches_applied > 0 or patches_skipped > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
