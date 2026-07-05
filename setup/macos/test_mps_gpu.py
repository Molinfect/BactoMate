#!/usr/bin/env python3
"""
Test whether segmentation will use M-chip (Apple Silicon) GPU.
Uses the same GPU check as Omnipose/cellpose_omni.
Run with the same Python you use for BactoMate: python3 setup/test_mps_gpu.py
"""
import sys

print("Python:", sys.executable)
print()

# 1) PyTorch MPS availability (what Omnipose checks for ARM)
import torch
print("torch.version:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
if hasattr(torch.backends, "mps"):
    print("torch.backends.mps.is_available():", torch.backends.mps.is_available())
else:
    print("torch.backends.mps: not present (PyTorch built without MPS)")
print()

# 2) Omnipose gpu module: ARM + torch_GPU (cellpose_omni uses these when gpu=True)
try:
    import omnipose.gpu as gpu
    print("omnipose.gpu.ARM:", getattr(gpu, "ARM", "?"))
    print("omnipose.gpu.torch_GPU:", getattr(gpu, "torch_GPU", "?"))
    print("omnipose.gpu.torch_CPU:", getattr(gpu, "torch_CPU", "?"))
    # Run the same test Omnipose uses: can we put a tensor on torch_GPU?
    torch_GPU = getattr(gpu, "torch_GPU", None)
    if torch_GPU is not None:
        try:
            _ = torch.zeros([1, 2, 3]).to(torch_GPU)
            print("Device test (torch.zeros(...).to(torch_GPU)): OK")
            print("-> Segmentation with 'Use GPU' on should use:", torch_GPU)
        except Exception as e:
            print("Device test failed:", e)
            print("-> Segmentation will fall back to CPU (this is why you see 'TORCH GPU version not installed/working')")
    # Legacy: try get_device/use_gpu if present
    if hasattr(gpu, "get_device"):
        device, use_gpu = gpu.get_device(0, use_torch=True)
        print("get_device(0) ->", device, "use_gpu:", use_gpu)
except Exception as e:
    print("Omnipose gpu check failed:", e)
    import traceback
    traceback.print_exc()

# 3) Summary
print()
mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
if mps_ok:
    print("PyTorch sees MPS (Apple GPU). If torch_GPU above is mps and device test was OK, segmentation will use it.")
else:
    print("PyTorch does not see MPS. Install PyTorch with MPS (default pip install torch on Mac).")
