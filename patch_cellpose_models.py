#!/usr/bin/env python3
"""
Patch cellpose_omni.models to fix the bug where Cellpose doesn't pass model_type to CellposeModel.
This fixes the architecture mismatch issue for bact_phase_omni and other C2_BD models.
"""

import os
import sys

# Find the models.py file
models_file = None
for path in sys.path:
    candidate = os.path.join(path, 'cellpose_omni', 'models.py')
    if os.path.exists(candidate):
        models_file = candidate
        break

if not models_file:
    print("ERROR: Could not find cellpose_omni/models.py")
    sys.exit(1)

print(f"Found models.py at: {models_file}")

# Read the file
with open(models_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'model_type=model_type' in content and 'self.cp = CellposeModel' in content:
    # Check if the patch is already applied by looking for the comment
    if 'model_type=model_type,  # FIX: Pass model_type for architecture detection' in content:
        print("✓ Patch already applied!")
        sys.exit(0)

# Find the line where CellposeModel is created
old_line = """        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                pretrained_model=self.pretrained_model,
                                diam_mean=self.diam_mean, use_torch=self.torch, 
                                dim=self.dim, omni=self.omni)"""

new_line = """        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                pretrained_model=self.pretrained_model,
                                model_type=model_type,  # FIX: Pass model_type for architecture detection
                                diam_mean=self.diam_mean, use_torch=self.torch, 
                                dim=self.dim, omni=self.omni)"""

if old_line in content:
    content = content.replace(old_line, new_line)
    
    # Backup original file
    backup_file = models_file + '.backup'
    with open(backup_file, 'w') as f:
        f.write(content.replace(new_line, old_line))
    print(f"Created backup at: {backup_file}")
    
    # Write patched file
    with open(models_file, 'w') as f:
        f.write(content)
    print("✓ Patch applied successfully!")
    print("\nThe bug is now fixed. You can run your script again.")
else:
    print("ERROR: Could not find the line to patch.")
    print("The file may have a different structure.")
    sys.exit(1)
