# Migrating from PyInstaller to Nuitka

## Why Switch?

We're switching from PyInstaller to Nuitka to resolve persistent issues with:
- Omnipose `bd=None` TypeError in PyInstaller builds
- Model architecture mismatches
- Dynamic import issues
- Runtime patching requirements

## Benefits of Nuitka

1. **Compiled to native code**: Avoids many dynamic import/runtime issues
2. **Better dependency handling**: More reliable inclusion of packages like omnipose/cellpose_omni
3. **Fewer runtime surprises**: Less likely to need post-install patches
4. **Better performance**: Native compilation can improve startup time

## Migration Steps

### 1. Install Nuitka

```bash
pip install nuitka
```

### 2. Build with Nuitka

**macOS:**
```bash
./build_nuitka_macos.sh
```

**Windows:**
```cmd
build_nuitka_windows.bat
```

### 3. Test the Build

1. Run the built application
2. Test segmentation with different models
3. Verify no `bd=None` errors occur
4. Check that all dependencies are included

### 4. Adjust Build Flags (if needed)

If you encounter missing dependencies, add them with:
```bash
--include-package=package_name
```

For data files:
```bash
--include-data-dir=source_dir=target_dir
```

## Removing PyInstaller Dependencies

Once Nuitka builds are stable:

1. **Optional**: Remove PyInstaller-specific files:
   - `BactiMate.spec`
   - `pyi_rth_*.py` files
   - `build_macos.sh` (or keep as backup)
   - `build_windows.bat` (or keep as backup)

2. **Update documentation**: Update README to reference Nuitka builds

3. **Simplify code**: The omnipose patch may no longer be needed with Nuitka, but keep it as a safety measure initially

## Troubleshooting

### Missing Dependencies

If you get import errors, add the missing package:
```bash
--include-package=missing_package
```

### Large Binary Size

Use exclusion flags:
```bash
--nofollow-import-to=unused_package
```

### Model Files Not Found

Include model directory:
```bash
--include-data-dir=~/.cellpose/models=models
```

## Comparison

| Feature | PyInstaller | Nuitka |
|---------|-------------|--------|
| Build time | Fast | Slower (compilation) |
| Binary size | Medium | Can be larger |
| Runtime issues | Common with complex packages | Less common |
| Dependency handling | Requires hooks | More automatic |
| Performance | Good | Better (native code) |
| Compatibility | Good | Excellent |

## Next Steps

1. Test Nuitka build thoroughly
2. Compare performance and reliability
3. If successful, make Nuitka the default build method
4. Consider removing PyInstaller support after validation period
