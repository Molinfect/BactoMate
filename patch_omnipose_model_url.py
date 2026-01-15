#!/usr/bin/env python3
"""
Patch cellpose_omni.models to use GitHub repository for model downloads instead of Cellpose server.
This fixes the HTTP 500 error when the Cellpose server is down.
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
if 'raw.githubusercontent.com/kevinjohncutler/omnipose-models/main' in content:
    print("✓ Model URL already patched to use GitHub repository!")
    sys.exit(0)

# Find and replace the MODEL_URL
old_line = "_MODEL_URL = 'https://www.cellpose.org/models'"
new_line = "_MODEL_URL = 'https://raw.githubusercontent.com/kevinjohncutler/omnipose-models/main'"

if old_line in content:
    content = content.replace(old_line, new_line)
    
    # Write the patched file
    with open(models_file, 'w') as f:
        f.write(content)
    
    print("✓ Successfully patched model URL!")
    print(f"  Changed from: https://www.cellpose.org/models")
    print(f"  Changed to: https://raw.githubusercontent.com/kevinjohncutler/omnipose-models/main")
    print("\nModels will now download from the Omnipose GitHub repository.")
else:
    print(f"WARNING: Could not find the line to patch in {models_file}")
    print("The file may have been modified. Please check manually.")
    sys.exit(1)
