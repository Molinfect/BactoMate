#!/bin/bash
# Install PyTorch with CUDA support for NVIDIA GPUs
# This script detects your CUDA version and installs the appropriate PyTorch build

set -e

echo "=========================================="
echo "Installing PyTorch with CUDA Support"
echo "=========================================="
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. CUDA drivers may not be installed."
    echo ""
    echo "Please ensure:"
    echo "1. NVIDIA GPU is installed"
    echo "2. NVIDIA drivers are installed"
    echo "3. CUDA toolkit is installed"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    CUDA_VERSION="cu118"  # Default to CUDA 11.8
else
    # Get CUDA version from nvidia-smi
    CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "Detected NVIDIA driver version: $CUDA_DRIVER_VERSION"
    
    # Map driver version to CUDA version (simplified)
    # For more accurate detection, check: nvcc --version
    if command -v nvcc &> /dev/null; then
        CUDA_TOOLKIT_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        echo "Detected CUDA toolkit version: $CUDA_TOOLKIT_VERSION"
        
        # Convert to PyTorch format (e.g., 11.8 -> cu118)
        CUDA_MAJOR=$(echo $CUDA_TOOLKIT_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_TOOLKIT_VERSION | cut -d. -f2)
        CUDA_VERSION="cu${CUDA_MAJOR}${CUDA_MINOR}"
    else
        echo "nvcc not found. Using default CUDA 11.8 (cu118)"
        CUDA_VERSION="cu118"
    fi
fi

echo ""
echo "Installing PyTorch with CUDA ${CUDA_VERSION}..."
echo ""

# Uninstall CPU-only PyTorch if installed
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

echo ""
echo "Verifying CUDA installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✓ CUDA installation successful!')
else:
    print('✗ CUDA is not available')
    print('  This might mean:')
    print('  - PyTorch was installed without CUDA support')
    print('  - CUDA drivers are not properly installed')
    print('  - GPU is not detected')
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
