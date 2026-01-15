# PyQt6 Migration Analysis

## Current Status
- **Current Framework**: PyQt5 5.15.10
- **Branch**: `nuitka-pyqt6` (created from `nuitka-support`)
- **Migration Target**: PyQt6 (latest stable)

## PyQt6 Advantages

### 1. **Better macOS Support**
- Improved compatibility with newer macOS versions (macOS 13+)
- Better handling of native macOS features (dark mode, native dialogs)
- Improved code signing and notarization support
- Better performance on Apple Silicon (M1/M2/M3)

### 2. **Active Development**
- PyQt5 is in **maintenance mode** (only bug fixes, no new features)
- PyQt6 is actively developed with new features and improvements
- Better long-term support and security updates

### 3. **Performance Improvements**
- Better rendering performance
- Improved memory management
- Faster startup times
- Better multi-threading support

### 4. **Modern Features**
- Better HiDPI/Retina display support
- Improved accessibility features
- Better support for modern UI patterns
- Enhanced styling capabilities

### 5. **Nuitka Compatibility**
- PyQt6 has better support in Nuitka
- More reliable plugin detection
- Better code signing integration

## Migration Effort Assessment

### Code Changes Required

#### 1. **Import Statements** (Low effort - ~5 minutes)
```python
# PyQt5
from PyQt5.QtWidgets import ...
from PyQt5.QtCore import ...
from PyQt5.QtGui import ...

# PyQt6
from PyQt6.QtWidgets import ...
from PyQt6.QtCore import ...
from PyQt6.QtGui import ...
```

#### 2. **Deprecated APIs** (Medium effort - ~30 minutes)

**QDesktopWidget → QScreen**
```python
# PyQt5
desktop = QDesktopWidget()
screen_geometry = desktop.screenGeometry()

# PyQt6
screen = QApplication.primaryScreen()
screen_geometry = screen.geometry()
```

**app.exec_() → app.exec()**
```python
# PyQt5
sys.exit(app.exec_())

# PyQt6
sys.exit(app.exec())
```

#### 3. **Signal/Slot Syntax** (Low effort - already compatible)
- `pyqtSignal` → `pyqtSignal` (same in PyQt6)
- No changes needed if using modern syntax

#### 4. **Enum Changes** (Low-Medium effort - ~15 minutes)
```python
# PyQt5
Qt.AlignCenter | Qt.AlignVCenter

# PyQt6 (same, but some enums moved)
from PyQt6.QtCore import Qt
# Most enums are the same, but some moved between modules
```

#### 5. **Build Scripts** (Low effort - ~10 minutes)
- Update `requirements.txt`: `PyQt6==6.6.0` (or latest)
- Update `build_nuitka_macos.sh`: `--include-package=PyQt6`
- Update `launch_gui.sh`: PyQt6 plugin paths
- Update `repair_pyqt5.sh` → `repair_pyqt6.sh` (or rename)

### Estimated Total Effort

| Task | Time Estimate |
|------|---------------|
| Update imports | 5 minutes |
| Fix deprecated APIs | 30 minutes |
| Update enum usage | 15 minutes |
| Update build scripts | 10 minutes |
| Update launch scripts | 10 minutes |
| Testing & debugging | 2-4 hours |
| **Total** | **3-5 hours** |

## Breaking Changes to Watch For

1. **QDesktopWidget removed** - Need to use QScreen/QApplication.primaryScreen()
2. **exec_() → exec()** - Simple find/replace
3. **Some enums moved** - Qt.AlignCenter, etc. may need different imports
4. **Plugin paths** - Slightly different directory structure
5. **Some deprecated methods** - Check PyQt6 migration guide

## Migration Strategy

### Phase 1: Preparation (15 minutes)
1. ✅ Create `nuitka-pyqt6` branch (DONE)
2. Update `requirements.txt` to PyQt6
3. Install PyQt6 in venv

### Phase 2: Code Updates (1 hour)
1. Update all import statements
2. Replace QDesktopWidget with QScreen
3. Replace exec_() with exec()
4. Fix any enum issues

### Phase 3: Build Scripts (30 minutes)
1. Update Nuitka build script
2. Update launch script
3. Update/rename repair script

### Phase 4: Testing (2-3 hours)
1. Test GUI launches correctly
2. Test all tabs and functionality
3. Test file operations
4. Test Nuitka build
5. Test app bundle execution

## Risk Assessment

**Low Risk:**
- Most PyQt5 code is compatible with PyQt6
- Standard widgets work the same
- Signals/slots unchanged

**Medium Risk:**
- Some edge cases with deprecated APIs
- Plugin path differences
- Potential enum import issues

**Mitigation:**
- Keep `nuitka-support` branch as fallback
- Test thoroughly before merging
- Document all changes

## Recommendation

**YES, worth migrating** because:
1. ✅ Better macOS support (important for your use case)
2. ✅ Active development vs maintenance mode
3. ✅ Low migration effort (3-5 hours)
4. ✅ Better Nuitka compatibility
5. ✅ Future-proofing

**Timeline:**
- Can be done in one focused session (half day)
- Or spread over 2-3 shorter sessions

## Next Steps

1. Update requirements.txt
2. Install PyQt6
3. Update code systematically
4. Test incrementally
5. Update build scripts
6. Full testing
7. Merge to nuitka-support when stable
