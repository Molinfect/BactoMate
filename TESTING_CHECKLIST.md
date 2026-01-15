# Nuitka Build Testing Checklist

## Pre-Build Setup

- [ ] Ensure virtual environment is activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] All patches applied (`./apply_all_patches.py`)
- [ ] Nuitka installed (`pip install nuitka`)

## Build Process

- [ ] Run `./build_nuitka_macos.sh` (or `build_nuitka_windows.bat` on Windows)
- [ ] Build completes without errors
- [ ] Application bundle/executable created in `dist/` directory
- [ ] Check build time (note for comparison with PyInstaller)

## Basic Functionality Tests

### Application Launch
- [ ] Application launches successfully
- [ ] No import errors in console/logs
- [ ] GUI displays correctly
- [ ] All tabs/buttons visible and accessible
- [ ] No Qt plugin errors

### Model Loading
- [ ] Can select different model types (bact_phase_omni, cyto2_omni, etc.)
- [ ] Models download correctly if not cached
- [ ] No model architecture mismatch errors
- [ ] Model files found in correct location

### Segmentation Tests

#### Single Image Segmentation
- [ ] Load a single TIFF image
- [ ] Run segmentation with default settings
- [ ] **CRITICAL**: No `bd=None` TypeError errors
- [ ] Segmentation completes successfully
- [ ] Results display correctly
- [ ] Can export results

#### Batch Processing
- [ ] Load a folder of images
- [ ] Run batch segmentation
- [ ] All images process without errors
- [ ] Progress bar updates correctly
- [ ] Results saved correctly

#### Different Model Types
- [ ] Test with `bact_phase_omni`
- [ ] Test with `cyto2_omni`
- [ ] Test with `bact_fluor_omni` (if available)
- [ ] Each model type works correctly

#### Different Diameter Settings
- [ ] Test with auto-detect (diameter = 0)
- [ ] Test with specific diameter values (20, 30, 40 pixels)
- [ ] No errors with any diameter setting

### File Format Support
- [ ] TIFF files work correctly
- [ ] ND2 files can be loaded (if applicable)
- [ ] Multi-page TIFF sequences work
- [ ] Export to Excel/CSV works

## Error Handling Tests

### Known Issues to Verify Fixed
- [ ] **PRIMARY TEST**: No `TypeError: '>' not supported between instances of 'NoneType' and 'int'`
- [ ] No model architecture mismatch errors
- [ ] No missing dependency errors
- [ ] No Qt plugin errors

### Edge Cases
- [ ] Handle empty image folders gracefully
- [ ] Handle corrupted/invalid image files gracefully
- [ ] Handle missing model files gracefully
- [ ] Handle network errors during model download

## Performance Tests

- [ ] Startup time is reasonable (< 10 seconds)
- [ ] Segmentation speed comparable to PyInstaller build
- [ ] Memory usage is reasonable
- [ ] No memory leaks during batch processing

## Comparison with PyInstaller Build

- [ ] Build time: Nuitka vs PyInstaller
- [ ] Binary size: Nuitka vs PyInstaller
- [ ] Startup time: Nuitka vs PyInstaller
- [ ] Runtime errors: Nuitka vs PyInstaller (should be fewer)
- [ ] Overall stability: Nuitka vs PyInstaller

## Platform-Specific Tests (macOS)

- [ ] Application bundle structure is correct
- [ ] Can create DMG file (if applicable)
- [ ] Application runs on clean macOS system (no venv)
- [ ] Code signing works (if applicable)
- [ ] Gatekeeper doesn't block the app

## Final Validation

- [ ] All critical functionality works
- [ ] No regression from PyInstaller build
- [ ] Performance is acceptable
- [ ] Ready for merge to main branch

## Issues Found

Document any issues encountered during testing:

1. **Issue**: [Description]
   - **Severity**: Critical / High / Medium / Low
   - **Steps to reproduce**: [Steps]
   - **Expected behavior**: [What should happen]
   - **Actual behavior**: [What actually happens]
   - **Workaround**: [If any]

---

## Test Results Summary

- **Date**: ___________
- **Tester**: ___________
- **Build Version**: ___________
- **Overall Status**: ✅ Pass / ❌ Fail / ⚠️ Needs Work
- **Recommendation**: Merge to main / Fix issues first / Needs more testing
