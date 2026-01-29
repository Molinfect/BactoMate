# -*- mode: python ; coding: utf-8 -*-
# BactiMate PyInstaller spec (optional alternative to Nuitka).
# PRIMARY WINDOWS BUILD: Use build_windows.bat or build_windows_bundle.bat (Nuitka).
# This .spec is for PyInstaller if you prefer it; run: pyinstaller BactiMate.spec

block_cipher = None

a = Analysis(
    ['omnipose_batch_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('logo2.png', '.')],
    hiddenimports=['numpy', 'scipy', 'PyQt6', 'tifffile', 'pandas', 'matplotlib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BactiMate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
)
