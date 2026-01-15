# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for BactiMate
This file defines how to package the application into a standalone executable.
"""

import sys
import os
from pathlib import Path

# Get the directory containing this spec file
# When run via PyInstaller, __file__ will be the spec file path
try:
    spec_dir = Path(__file__).parent
except NameError:
    # Fallback if __file__ is not available
    spec_dir = Path.cwd()
project_dir = spec_dir

# Application metadata
app_name = 'BactiMate'
main_script = 'omnipose_batch_gui.py'

# CRITICAL: Collect all dependencies for omnipose/cellpose_omni
# This prevents "None > int" errors by ensuring all runtime dependencies are included
from PyInstaller.utils.hooks import collect_all

# Initialize collections
datas = []
binaries = []
hiddenimports = []

# Collect all dependencies for packages that omnipose/cellpose_omni rely on
# Omnipose's PyInstaller notes explicitly recommend collecting skimage and pyqtgraph
for pkg in [
    "omnipose",
    "cellpose_omni",
    "skimage",      # Explicitly recommended in Omnipose PyInstaller notes
    "pyqtgraph",    # Explicitly recommended in Omnipose PyInstaller notes
    "torch",        # Often needed due to dynamic loading
    "numba",        # Common frozen-build pain point, needs submodules
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        # If collect_all fails for a package, continue with others
        print(f"Warning: Failed to collect_all for {pkg}: {e}")

# Collect all data files that need to be included (keep existing logic)

# Include local model files if they exist (for PyInstaller builds)
# Models are typically in ~/.cellpose/models, but we can bundle local ones
models_dir = project_dir / 'models'
if models_dir.exists():
    # Include all model files from local models directory
    for model_file in models_dir.glob('*'):
        if model_file.is_file():
            datas.append((str(model_file), 'models'))

# Include logo files if they exist
for logo_file in ['logo.png', 'logo1.png', 'logo2.png']:
    logo_path = project_dir / logo_file
    if logo_path.exists():
        datas.append((str(logo_path), '.'))

# Additional hidden imports - packages that PyInstaller might miss
# (These are in addition to what collect_all found)
hiddenimports += [
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'numpy',
    'scipy',
    'scipy.sparse.csgraph._validation',
    'scipy.special.cython_special',
    'skimage',
    'skimage.filters',
    'skimage.morphology',
    'skimage.measure',
    'skimage.segmentation',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.figure',
    'pandas',
    'tifffile',
    'nd2',
    'aicsimageio',
    'readlif',
    'czifile',
    'omnipose',
    'omnipose.core',
    'cellpose_omni',
    'cellpose_omni.models',
    'cellpose_omni.io',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.distributed',
    'torch.distributed.rpc',
    'torch.distributed.rpc.backend_registry',
    'torch.testing',
    'torch.testing._internal',
    'torchvision',
    'tqdm',
    'natsort',
    'openpyxl',
    'PIL',
    'PIL.Image',
    'PIL.ImageQt',
    'pbr',
    'pbr.version',
    'pbr.git',
    'mgen',
]

# Binary exclusions (reduce size)
excludes = [
    'tkinter',
    'matplotlib.tests',
    'numpy.tests',
    'scipy.tests',
    'pandas.tests',
    'PIL.tests',
    # Note: torch.testing is now included in hiddenimports (not excluded)
    # to fix "No module named 'torch.testing'" errors in PyInstaller builds
]

# Platform-specific settings
# Use logo2.png converted to appropriate icon format
if sys.platform == 'darwin':
    icon_file = 'BactiMate.icns' if Path('BactiMate.icns').exists() else None
elif sys.platform == 'win32':
    icon_file = 'BactiMate.ico' if Path('BactiMate.ico').exists() else None
else:
    icon_file = None

# Collect PyQt5 plugins (works for both platforms)
try:
    import PyQt5
    pyqt5_path = Path(PyQt5.__file__).parent
    qt_plugins = pyqt5_path / 'Qt5' / 'plugins'
    if qt_plugins.exists():
        # Include all Qt plugins
        datas.append((str(qt_plugins), 'PyQt5/Qt5/plugins'))
except ImportError:
    pass

block_cipher = None

# Analysis - what PyInstaller should analyze
a = Analysis(
    [str(project_dir / main_script)],
    pathex=[str(project_dir)],
    binaries=binaries,  # Use collected binaries from collect_all
    datas=datas,       # Use collected datas from collect_all
    hiddenimports=hiddenimports,  # Use collected hiddenimports from collect_all
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[hook for hook in [
        'pyi_rth_torch.py' if Path('pyi_rth_torch.py').exists() else None,
        'pyi_rth_pbr.py' if Path('pyi_rth_pbr.py').exists() else None,
        'pyi_rth_omnipose.py' if Path('pyi_rth_omnipose.py').exists() else None,
    ] if hook is not None],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate entries
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
if sys.platform == 'darwin':
    # macOS: Create .app bundle using onedir mode (recommended)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=app_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,  # No console window for GUI app
        argv_emulation=True,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    
    # Collect binaries and datas for the app bundle
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=app_name,
    )
    
    # Create .app bundle
    app = BUNDLE(
        coll,
        name=f'{app_name}.app',
        icon=icon_file,
        bundle_identifier='com.bactimate.app',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'NSHumanReadableCopyright': 'Copyright © 2024',
            'LSMinimumSystemVersion': '10.13',  # macOS High Sierra or later
        },
    )
elif sys.platform == 'win32':
    # Windows: Create .exe
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=app_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,  # No console window for GUI app
        disable_windowed_traceback=False,
        icon=icon_file,
    )
