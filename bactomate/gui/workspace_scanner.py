"""
Scan a workspace folder for paths relevant to each pipeline module.

Non-recursive listing of the workspace root, plus common subfolders (e.g. Tif_files).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_RAW_EXTS = (
    ".nd2", ".ND2", ".lif", ".LIF", ".czi", ".CZI",
    ".tif", ".tiff", ".TIF", ".TIFF",
)


_CONVERTIBLE_SUFFIXES = frozenset({".nd2", ".lif", ".czi", ".tif", ".tiff"})

_SKIP_WALK_DIR_NAMES = frozenset({
    ".git", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", ".idea", ".vscode", "build", "dist",
})


_TIFF_SUFFIXES = frozenset({".tif", ".tiff"})


def list_tiff_files(
    folder: Path,
    *,
    max_files: int = 2500,
    max_depth: int = 40,
) -> list[Path]:
    """Recursively find ``.tif`` / ``.tiff`` files under ``folder`` (bounded scan).

    Same traversal policy as ``list_convertible_microscopy_files`` but TIFF-only
    (for segmentation / post-TIFF workflows).
    """
    folder = folder.resolve()
    if not folder.is_dir():
        return []
    found: list[Path] = []
    seen: set[Path] = set()

    for dirpath, dirnames, filenames in os.walk(folder, topdown=True):
        cur = Path(dirpath)
        try:
            rel_depth = len(cur.relative_to(folder).parts)
        except ValueError:
            rel_depth = 0
        if rel_depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d not in _SKIP_WALK_DIR_NAMES
        )
        for fn in filenames:
            if len(found) >= max_files:
                return sorted(found, key=lambda p: p.name.lower())
            suf = Path(fn).suffix.lower()
            if suf not in _TIFF_SUFFIXES:
                continue
            p = cur / fn
            try:
                if not p.is_file():
                    continue
                r = p.resolve()
            except OSError:
                continue
            if r in seen:
                continue
            seen.add(r)
            found.append(r)

    return sorted(found, key=lambda p: p.name.lower())[:max_files]


def list_convertible_microscopy_files(
    folder: Path,
    *,
    max_files: int = 2000,
    max_depth: int = 40,
) -> list[Path]:
    """Recursively find ND2 / LIF / CZI / TIFF under ``folder`` (bounded by count and depth).

    Skips hidden directories and common non-data folders. Intended for workspace / browse roots,
    not arbitrary huge trees — adjust ``max_files`` / ``max_depth`` if needed.
    """
    folder = folder.resolve()
    if not folder.is_dir():
        return []
    found: list[Path] = []
    seen: set[Path] = set()

    for dirpath, dirnames, filenames in os.walk(folder, topdown=True):
        cur = Path(dirpath)
        try:
            rel_depth = len(cur.relative_to(folder).parts)
        except ValueError:
            rel_depth = 0
        if rel_depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d not in _SKIP_WALK_DIR_NAMES
        )
        for fn in filenames:
            if len(found) >= max_files:
                return sorted(found, key=lambda p: p.name.lower())
            suf = Path(fn).suffix.lower()
            if suf not in _CONVERTIBLE_SUFFIXES:
                continue
            p = (cur / fn)
            try:
                if not p.is_file():
                    continue
                r = p.resolve()
            except OSError:
                continue
            if r in seen:
                continue
            seen.add(r)
            found.append(r)

    return sorted(found, key=lambda p: p.name.lower())[:max_files]


def _first_sorted(paths: list[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.name.lower())[0]


@dataclass
class WorkspaceScanResult:
    """Best-effort paths discovered under a workspace."""

    root: Path
    raw_microscopy: Optional[Path] = None  # single file at root (preferred over TIFF for conversion)
    raw_folder: Optional[Path] = None  # workspace itself if it looks like a data folder
    tif_files_dir: Optional[Path] = None
    segmentation_input: Optional[Path] = None  # Tif_files or raw file / folder
    mask_or_project_dir: Optional[Path] = None  # _segmentation_* or folder with mask-like tifs
    tracking_masks_dir: Optional[Path] = None
    post_input_dir: Optional[Path] = None
    fi_example: Optional[Path] = None
    foci_example: Optional[Path] = None

    hints: dict[str, str] = field(default_factory=dict)


def scan_workspace(root: Path) -> WorkspaceScanResult:
    root = root.resolve()
    out = WorkspaceScanResult(root=root)
    if not root.is_dir():
        return out

    files_at_root = [p for p in root.iterdir() if p.is_file()]
    dirs_at_root = [p for p in root.iterdir() if p.is_dir()]

    raw_files = [p for p in files_at_root if p.suffix in _RAW_EXTS]
    out.raw_microscopy = _first_sorted(raw_files)

    tif_dir = root / "Tif_files"
    if tif_dir.is_dir():
        out.tif_files_dir = tif_dir
        out.segmentation_input = tif_dir
        out.hints["segmentation"] = str(tif_dir)
        out.hints["file_conversion"] = str(tif_dir)  # batch TIFF in folder
    elif out.raw_microscopy is not None:
        out.segmentation_input = out.raw_microscopy
        out.hints["segmentation"] = str(out.raw_microscopy)
        out.hints["file_conversion"] = str(out.raw_microscopy)
    else:
        # folder of misc files — still use workspace as browse root
        out.raw_folder = root
        out.segmentation_input = root
        out.hints["segmentation"] = str(root)
        out.hints["file_conversion"] = str(root)

    # Omnipose / masks: _segmentation_* next to Tif parent or in workspace
    seg_dirs = sorted(
        [p for p in dirs_at_root if p.name.startswith("_segmentation")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if seg_dirs:
        out.mask_or_project_dir = seg_dirs[0]
        out.tracking_masks_dir = seg_dirs[0]
        out.hints["tracking"] = str(seg_dirs[0])
        out.hints["filtering"] = str(seg_dirs[0])
        out.hints["analyze_masks"] = str(seg_dirs[0])
        # FI / Foci naming hints (first matching file in mask folder)
        for pat in ("*_fi*.tif", "*_fi*.tiff", "*_FI*.tif"):
            found = list(seg_dirs[0].glob(pat))
            if found:
                out.fi_example = sorted(found)[0]
                break
        for pat in ("*_foci*.tif", "*_foci*.tiff", "*_FOCI*.tif"):
            found = list(seg_dirs[0].glob(pat))
            if found:
                out.foci_example = sorted(found)[0]
                break

    # If no seg dir but Tif_files has tifs, point tracking at Tif_files (phase stacks)
    if out.tracking_masks_dir is None and out.tif_files_dir is not None:
        tifs = list(out.tif_files_dir.glob("*.tif")) + list(out.tif_files_dir.glob("*.tiff"))
        if tifs:
            out.tracking_masks_dir = out.tif_files_dir
            out.hints["tracking"] = str(out.tif_files_dir)

    # Post-processing: prefer workspace or Tif_files for images
    out.post_input_dir = out.tif_files_dir or root
    out.hints["post_processing"] = str(out.post_input_dir)

    # Training: workspace as default project/dataset root
    out.hints["training"] = str(root)

    # File conversion primary: prefer raw file at root, else first raw in Tif_files, else folder
    if out.raw_microscopy is not None:
        out.hints["file_conversion"] = str(out.raw_microscopy)
    elif out.tif_files_dir is not None:
        tifs = sorted(
            list(out.tif_files_dir.glob("*.tif")) + list(out.tif_files_dir.glob("*.tiff")),
            key=lambda p: p.name.lower(),
        )
        if tifs:
            out.hints["file_conversion"] = str(tifs[0])

    return out
