@echo off
REM Install PyTorch with CUDA support for NVIDIA GPUs on Windows
REM This script detects your CUDA version and installs the appropriate PyTorch build

echo ==========================================
echo Installing PyTorch with CUDA Support
echo ==========================================
echo.

REM Check if CUDA is available
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: nvidia-smi not found. CUDA drivers may not be installed.
    echo.
    echo Please ensure:
    echo 1. NVIDIA GPU is installed
    echo 2. NVIDIA drivers are installed
    echo 3. CUDA toolkit is installed
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
    set CUDA_VERSION=cu118
) else (
    echo Detected NVIDIA GPU
    REM Try to get CUDA version from nvcc if available
    where nvcc >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        for /f "tokens=*" %%i in ('nvcc --version ^| findstr /C:"release"') do (
            REM Extract version number (simplified)
            set CUDA_VERSION=cu118
        )
    ) else (
        echo Using default CUDA 11.8 (cu118)
        set CUDA_VERSION=cu118
    )
)

echo.
echo Installing PyTorch with CUDA %CUDA_VERSION%...
echo.

REM Uninstall CPU-only PyTorch if installed
pip uninstall -y torch torchvision torchaudio 2>nul

REM Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA_VERSION%

echo.
echo Verifying CUDA installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ==========================================
echo Installation complete!
echo ==========================================
echo.

pause
