# BactoMate

Integrated microscopy analysis platform: file conversion, segmentation (Omnipose, U-Net, Cellpose-SAM), mask analysis, tracking, **lineage QC & export**, training, and reporting.

**v2.5 is source-first (developers & researchers):** clone this repo, run the install script for your OS, then launch locally. macOS `.app`/DMG packaging can also be built locally from the installed `venv`; Windows standalone packaging is still deferred.

## Quick Install

For most users, open a terminal window and run the commands for your operating system.

### macOS

```bash
git clone https://github.com/poppphil/BactoMate.git
cd BactoMate
./setup/install_mac.sh
./launch_gui.sh
```

If macOS says the script is not executable, run:

```bash
chmod +x setup/install_mac.sh setup/setup_fresh_env.sh launch_gui.sh
```

### Windows

Open Command Prompt, then run:

```bat
git clone https://github.com/poppphil/BactoMate.git
cd BactoMate
setup\install_windows.bat
launch_windows.bat
```

Or download the repository as a ZIP from GitHub, extract it, open a terminal in the extracted folder, and start from the install command for your OS.

## macOS — install and launch

**Requires:** macOS 10.15+, Python 3.11, Terminal.

```bash
chmod +x setup/install_mac.sh setup/setup_fresh_env.sh launch_gui.sh
./setup/install_mac.sh
```

Verify:

```bash
source venv/bin/activate
python main.py --version    # BactoMate 2.6.0
python main.py --smoke-test # Smoke test OK
```

Launch:

```bash
./launch_gui.sh
```

Build a macOS app/DMG after installation:

```bash
./build/build_app_bundle.sh
./build/create_dmg.sh
```

## Windows — install and launch

**Requires:** Windows 10/11 64-bit, Python 3.11 64-bit, [Visual C++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

Open Command Prompt in the project folder:

```bat
setup\install_windows.bat
```

Then double-click `launch_windows.bat`, run it from Command Prompt, or launch the installed console command:

```bat
venv\Scripts\activate
bactomate
```

Full Windows troubleshooting: **[docs/INSTALL_AND_LAUNCH_windows.md](docs/INSTALL_AND_LAUNCH_windows.md)**

## Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Full user guide, CLI, and workflows |
| [docs/BACTOMATE_USER_MANUAL.md](docs/BACTOMATE_USER_MANUAL.md) | Tab-by-tab reference |
| [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md) | Version history |
| [docs/INSTALL_AND_LAUNCH_windows.md](docs/INSTALL_AND_LAUNCH_windows.md) | Windows install details |

## Quick troubleshooting

| Issue | Fix |
|-------|-----|
| macOS Qt / PyQt6 errors | Re-run `./setup/install_mac.sh` or `./setup/repair_pyqt6.sh` |
| Windows DLL / torch errors | Install VC++ Redistributable x64, use Python 3.11 64-bit, recreate `venv` |
| Segmentation Cellpose errors | `python setup/apply_all_patches.py` (with venv active) |
| GPU optional | CPU works; CUDA PyTorch is offered during install |

## License

MIT — see [LICENSE](LICENSE).
