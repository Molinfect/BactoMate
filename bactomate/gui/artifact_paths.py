"""
Stage artifact naming + discovery helpers.

These helpers are intentionally GUI-agnostic: they encode filename conventions for
Segmentation → Filtering → Analyze so both UI code and CLI/batch utilities can
write/read the same artifacts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from bactomate.gui.analysis_paths import artifact_search_dirs_for_mask, base_dir_for_mask, stem_from_mask_path

# Segmentation / filtering foci CSV layout under each `_segmentation_*` (or mask parent) folder.
FOCI_DETECTION_DIRNAME = "foci_detection"
FOCI_FILTERED_DIRNAME = "foci_filtered"
FILTERED_MASKS_DIRNAME = "filtered_masks"

_CHANNEL_SUFFIX_TOKENS = {
    "mcherry",
    "gfp",
    "rfp",
    "cfp",
    "yfp",
    "dsred",
    "tdtomato",
    "mng",
    "scarlet",
    "mscarlet",
    "cy5",
    "cy3",
    "dapi",
}

_ROLE_MARKERS = ("_fi_foci", "_phase", "_foci", "_fi")


def stem_has_role_marker(stem: str) -> bool:
    """True when a TIFF stem already carries a BactoMate role tag."""
    low = f"_{(stem or '').strip('_').lower()}_"
    return any(marker in low for marker in _ROLE_MARKERS)


def longest_common_underscore_prefix(stem_a: str, stem_b: str) -> str:
    """Longest shared ``_``-delimited prefix (casefold token compare)."""
    pa = [p for p in (stem_a or "").split("_") if p != ""]
    pb = [p for p in (stem_b or "").split("_") if p != ""]
    out: list[str] = []
    for a, b in zip(pa, pb):
        if a.casefold() != b.casefold():
            break
        out.append(a)
    return "_".join(out)


def foci_detection_dir(pipeline_dir: Path) -> Path:
    """Folder for raw Spotiflow peak CSVs (+ optional QC overlays) after segmentation."""
    return Path(pipeline_dir) / FOCI_DETECTION_DIRNAME


def foci_filtered_dir(pipeline_dir: Path) -> Path:
    """Folder for peak CSVs after the Filtering stage."""
    return Path(pipeline_dir) / FOCI_FILTERED_DIRNAME


def filtered_masks_dir(pipeline_dir: Path) -> Path:
    """Folder for filtered cell mask TIFFs (Filtering stage)."""
    return Path(pipeline_dir) / FILTERED_MASKS_DIRNAME


@dataclass(frozen=True)
class PipelineArtifacts:
    """Computed artifact paths for one mask + (optional) foci source."""

    pipeline_dir: Path
    mask_path: Path
    filtered_mask_path: Path
    foci_csv_path: Optional[Path] = None
    raw_foci_csv_path: Optional[Path] = None  # legacy name (kept for migration)
    filtered_foci_csv_path: Optional[Path] = None


def segmentation_pipeline_dir_from_input(
    input_path: Path,
    *,
    pipeline_folder_name: Optional[str] = None,
) -> Path:
    """
    Return the pipeline folder for outputs.

    Convention:
    - If `input_path` is a file: pipeline dir is next to the file.
    - If `input_path` is a folder: pipeline dir is inside it.
    - If `pipeline_folder_name` is provided, it is appended accordingly.

    This keeps the helper generic; specific backends can pass their existing
    `_segmentation_*` folder name.
    """
    p = Path(input_path)
    root = p.parent if p.is_file() else p
    if pipeline_folder_name:
        return root / pipeline_folder_name
    return root


def _filtered_mask_basename(mask_path: Path) -> str:
    stem = stem_from_mask_path(Path(mask_path))
    return f"{stem}_mask_filtered.tif"


def filtered_mask_path_for_mask(mask_path: Path) -> Path:
    """
    Filtering-stage mask artifact path.

    Layout: ``{mask_parent}/filtered_masks/{stem}_mask_filtered.tif``
    """
    m = Path(mask_path)
    return filtered_masks_dir(m.parent) / _filtered_mask_basename(m)


def legacy_filtered_mask_paths_for_raw_mask(mask_path: Path) -> list[Path]:
    """Flat-path filtered masks (pre-``filtered_masks/`` layout) for migration reads."""
    m = Path(mask_path)
    name = _filtered_mask_basename(m)
    return [m.parent / name]


def filtered_mask_read_candidates(mask_path: Path) -> list[Path]:
    """Paths to try for a filtered mask paired with a raw ``*_masks.tif`` (canonical first)."""
    m = Path(mask_path)
    ordered = [filtered_mask_path_for_mask(m)] + legacy_filtered_mask_paths_for_raw_mask(m)
    out: list[Path] = []
    seen: set[str] = set()
    for p in ordered:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_filtered_mask_path(mask_path: Path) -> Optional[Path]:
    """First existing filtered mask for the raw mask at ``mask_path``."""
    for cand in filtered_mask_read_candidates(mask_path):
        if cand.is_file():
            return cand
    return None


def is_filtered_mask_tiff(path: Path) -> bool:
    """True when ``path`` is a filtering-stage ``*_mask_filtered`` TIFF."""
    low = Path(path).name.lower()
    return low.endswith("_mask_filtered.tif") or low.endswith("_mask_filtered.tiff")


def mask_sample_identity_key(mask_path: Path) -> str:
    """Pair raw ``*_masks`` and ``*_mask_filtered`` TIFFs for the same acquisition."""
    return _base_id_stem_from_mask(Path(mask_path)).strip().lower()


def _token_looks_like_channel(token: str) -> bool:
    """True for known dye/channel tokens (``GFP``, ``TRITC``, ``mScarlet``, …)."""
    t = (token or "").strip("_").lower()
    if not t:
        return False
    if t in _CHANNEL_SUFFIX_TOKENS:
        return True
    if t in ("tritc", "fitc", "texasred", "alexa488", "alexa594"):
        return True
    # Vendor labels like Channel_2 / Ch3 — not bare replicate numbers (001, 002, …).
    if re.fullmatch(r"ch(annel)?_?\d+", t):
        return True
    if t.endswith("red") or t.endswith("green") or t.startswith("cy"):
        return True
    return False


def foci_acquisition_ids_align(base_id: str, candidate_id: str) -> bool:
    """
    True when two role-stripped acquisition IDs refer to the same sample/replicate.

    Exact match, or ``candidate_id`` is ``base_id`` plus one optional channel token
    (e.g. ``sample_001`` ↔ ``sample_001_GFP`` for ``sample_001_GFP_fi_foci``).
    """
    base = (base_id or "").strip("_")
    cand = (candidate_id or "").strip("_")
    if not base or not cand:
        return False
    if base.casefold() == cand.casefold():
        return True
    bl, cl = base.casefold(), cand.casefold()
    if not cl.startswith(bl + "_"):
        return False
    extra = cand[len(base) :].lstrip("_")
    if not extra or "_" in extra:
        return False
    return _token_looks_like_channel(extra)


def sample_identity_stem(stem: str) -> str:
    """
    Canonical sample/replicate stem shared by phase/FI/foci/mask exports.

    Primary rule: strip at role markers (``_phase``, ``_fi``, ``_foci``, ``_fi_foci``).
    Also strips mask suffixes and double-underscore replicate cuts.

    Dye-token stripping (``_GFP``, ``_mCherry``, …) remains as a weak legacy aid for
    older exports without ``_fi`` tags. Free-form names like ``_DIA`` are *not* in that
    list — use :func:`foci_pairs_with_input` LCP fallback for those.

    Examples:
      ``EM20140_NoPhage_001_mCherry`` -> ``EM20140_NoPhage_001``
      ``EM20140_NoPhage_001_foci_GFP`` -> ``EM20140_NoPhage_001``
      ``EM20140_NoPhage__mCherry`` -> ``EM20140_NoPhage``
      ``sample_Phase60X`` -> ``sample``
      ``EM20285_003_phase`` -> ``EM20285_003``
    """
    s = (stem or "").strip("_")
    low = s.lower()
    for suffix in ("_mask_filtered", "_masks", "_mask"):
        if low.endswith(suffix):
            s = s[: -len(suffix)].strip("_")
            low = s.lower()
            break
    for marker in _ROLE_MARKERS:
        idx = low.find(marker)
        if idx >= 0:
            return s[:idx].strip("_")
    if "__" in s:
        return s.split("__", 1)[0].strip("_")
    parts = s.split("_")
    if len(parts) > 1 and parts[-1].lower() in _CHANNEL_SUFFIX_TOKENS:
        return "_".join(parts[:-1]).strip("_")
    return s.strip("_")


def foci_pairs_with_input(input_stem: str, foci_stem: str) -> bool:
    """Pair foci by acquisition identity, including legacy free-form phase names."""
    return "_foci" in (foci_stem or "").lower() and _channel_pairs_with_input(input_stem, foci_stem)


def _channel_pairs_with_input(input_stem: str, channel_stem: str) -> bool:
    inp = (input_stem or "").strip()
    candidate = (channel_stem or "").strip()
    if not inp or not candidate:
        return False
    inp = re.sub(r"_(mask_filtered|masks|mask)$", "", inp, flags=re.IGNORECASE)
    # Pure foci-as-input keeps exact identity (no legacy prefix fallback).
    if "_foci" in inp.lower() and "_fi_foci" not in inp.lower():
        return sample_identity_stem(inp).casefold() == sample_identity_stem(candidate).casefold()
    inp_id = sample_identity_stem(inp)
    candidate_id = sample_identity_stem(candidate)
    if foci_acquisition_ids_align(inp_id, candidate_id):
        return True
    if stem_has_role_marker(inp):
        return False
    lcp = longest_common_underscore_prefix(inp, candidate)
    if not lcp or not foci_acquisition_ids_align(lcp, candidate_id):
        return False
    # Only a final legacy channel name may follow the acquisition, never another replicate/frame.
    return "_" not in inp[len(lcp) :].strip("_")


def dedupe_mask_paths_prefer_filtered(paths: Sequence[Path]) -> list[Path]:
    """
    One mask per sample: prefer ``*_mask_filtered`` when both filtered and raw exist.

    Falls back to ``*_masks`` only when no filtered sibling is in ``paths``.
    """
    groups: dict[str, list[Path]] = {}
    for raw in paths:
        p = Path(raw)
        if not is_cell_mask_tiff(p):
            name = p.name.lower()
            if not (
                name.endswith((".tif", ".tiff"))
                and ("_masks" in name or "_mask_filtered" in name)
                and "_trajectories" not in name
            ):
                continue
        key = mask_sample_identity_key(p)
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        groups.setdefault(key, []).append(resolved)

    out: list[Path] = []
    for key in sorted(groups.keys()):
        items = groups[key]
        filtered = [p for p in items if is_filtered_mask_tiff(p)]
        if filtered:

            def _rank(fp: Path) -> tuple[int, str]:
                parts = str(fp).replace("\\", "/").lower()
                in_sub = 1 if f"/{FILTERED_MASKS_DIRNAME.lower()}/" in parts else 0
                return (in_sub, fp.name.lower())

            out.append(sorted(filtered, key=_rank)[0])
            continue
        raw_masks = [p for p in items if not is_filtered_mask_tiff(p)]
        pool = raw_masks if raw_masks else items
        out.append(sorted(pool, key=lambda fp: fp.name.lower())[0])
    return sorted(out, key=lambda p: p.name.lower())


def is_cell_mask_tiff(path: Path) -> bool:
    """True for cell segmentation / filtered mask stacks (not raw phase or tracking outputs)."""
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() not in (".tif", ".tiff"):
        return False
    name = p.name.lower()
    if any(
        tag in name
        for tag in (
            "_trajectories",
            "_overlay",
            "_filter_visualization",
            "masks_tracked",
            "_lineage",
        )
    ):
        return False
    return "_masks" in name or "_mask_filtered" in name


def _add_mask_tiffs_from_dir(mask_set: set[Path], directory: Path) -> None:
    """Add raw/filtered mask TIFFs from one directory (no recursion)."""
    d = Path(directory)
    if not d.is_dir():
        return
    for p in collect_filtered_mask_files_in_folder(d):
        if is_cell_mask_tiff(p):
            try:
                mask_set.add(p.resolve())
            except OSError:
                mask_set.add(p)
    for pat in ("*_masks.tif", "*_masks.tiff", "*_mask_filtered.tif", "*_mask_filtered.tiff"):
        for p in d.glob(pat):
            if p.is_file() and is_cell_mask_tiff(p):
                try:
                    mask_set.add(p.resolve())
                except OSError:
                    mask_set.add(p)


def discover_mask_tiffs_for_workspace(path: Path) -> list[Path]:
    """
    Find segmentation mask TIFFs for Analyze / Tracking / Filtering pickers.

    - **Folder** (e.g. ``Tif_files/`` or ``_segmentation_cellpose_sam/``): scans the folder,
      ``filtered_masks/``, and ``_segmentation_*`` subfolders (with a recursive fallback).
    - **Phase / original TIFF**: scans the file's parent folder the same way.
    - **Mask TIFF**: returns that file only (deduped with filtered preference when applicable).
    """
    path = Path(path)
    if not path.exists():
        return []
    if path.is_file():
        if is_cell_mask_tiff(path):
            resolved = path.resolve()
            # If the user pointed at raw *_masks.tif, prefer the filtered sibling when present.
            if not is_filtered_mask_tiff(resolved):
                filt = resolve_filtered_mask_path(resolved)
                if filt is not None:
                    return [Path(filt).resolve()]
            return dedupe_mask_paths_prefer_filtered([resolved])
        root = path.parent.resolve()
    else:
        root = path.resolve()

    mask_set: set[Path] = set()
    _add_mask_tiffs_from_dir(mask_set, root)
    filt_dir = root / FILTERED_MASKS_DIRNAME
    if filt_dir.is_dir():
        _add_mask_tiffs_from_dir(mask_set, filt_dir)
    for seg in sorted(root.glob("_segmentation*")):
        if seg.is_dir():
            _add_mask_tiffs_from_dir(mask_set, seg)
            seg_filt = seg / FILTERED_MASKS_DIRNAME
            if seg_filt.is_dir():
                _add_mask_tiffs_from_dir(mask_set, seg_filt)
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        _add_mask_tiffs_from_dir(mask_set, sub)
        for seg in sorted(sub.glob("_segmentation*")):
            if seg.is_dir():
                _add_mask_tiffs_from_dir(mask_set, seg)
    for seg in root.rglob("_segmentation*"):
        if seg.is_dir():
            _add_mask_tiffs_from_dir(mask_set, seg)
    if not mask_set:
        for pat in (
            "**/*_masks.tif",
            "**/*_masks.tiff",
            "**/*_mask_filtered.tif",
            "**/*_mask_filtered.tiff",
        ):
            for p in root.glob(pat):
                if p.is_file() and is_cell_mask_tiff(p):
                    try:
                        mask_set.add(p.resolve())
                    except OSError:
                        mask_set.add(p)
    return dedupe_mask_paths_prefer_filtered(sorted(mask_set, key=lambda x: x.name.lower()))


def collect_filtered_mask_files_in_folder(folder: Path) -> list[Path]:
    """All filtered mask TIFFs under ``folder`` (``filtered_masks/`` + legacy flat)."""
    root = Path(folder)
    patterns = (
        f"{FILTERED_MASKS_DIRNAME}/*_mask_filtered.tif",
        f"{FILTERED_MASKS_DIRNAME}/*_mask_filtered.tiff",
        f"**/{FILTERED_MASKS_DIRNAME}/*_mask_filtered.tif",
        f"**/{FILTERED_MASKS_DIRNAME}/*_mask_filtered.tiff",
        "*_mask_filtered.tif",
        "*_mask_filtered.tiff",
        "**/*_mask_filtered.tif",
        "**/*_mask_filtered.tiff",
    )
    out: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        for p in root.glob(pat):
            if not p.is_file():
                continue
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return sorted(out, key=lambda p: p.name.lower())


def _sanitize_channel_id(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _base_id_stem_from_mask(mask_path: Path) -> str:
    """
    Short stem used for foci artifacts.

    Example:
      G183A-004_Phase60x_mask_filtered.tif -> G183A-004
      G183A-004_Phase60x_masks.tif         -> G183A-004
    """
    stem = sample_identity_stem(stem_from_mask_path(Path(mask_path)))
    return stem.strip("_- .") or stem_from_mask_path(Path(mask_path))


def foci_channel_id_from_source(mask_path: Path, foci_source_path: Path) -> str:
    """
    Compute a stable channel id for naming foci CSV artifacts.

    Strategy:
    - Start with the foci source filename stem.
    - If it starts with the mask stem (with `_masks` stripped), remove that prefix.
    - Strip leading separators and sanitize for filenames.
    """
    src = Path(foci_source_path).stem
    # Prefer extracting the channel for common naming: <base>_foci_<channel>.tif
    m = re.match(r"^(?P<prefix>.+?)_foci_(?P<channel>.+)$", src, flags=re.IGNORECASE)
    if m:
        return _sanitize_channel_id(m.group("channel"))

    # Otherwise, strip known prefixes using both the full mask stem and the shortened base-id stem.
    base_full = stem_from_mask_path(Path(mask_path))
    base_short = _base_id_stem_from_mask(Path(mask_path))
    for base in (base_full, base_short):
        if base and src.startswith(base):
            src = src[len(base) :]
            break
    src = src.lstrip("_- .")
    return _sanitize_channel_id(src)


def _foci_artifact_basename(stem: str, ch: str, *, kind: str) -> str:
    """Build `{stem}_foci…csv` basename; ``kind`` is ``foci``, ``foci_raw``, or ``foci_filtered``."""
    suffix = f"_{ch}" if ch else ""
    if kind == "foci":
        return f"{stem}_foci{suffix}.csv"
    if kind == "foci_raw":
        return f"{stem}_foci_raw{suffix}.csv"
    if kind == "foci_filtered":
        return f"{stem}_foci_filtered{suffix}.csv"
    raise ValueError(f"unknown foci artifact kind: {kind!r}")


def foci_csv_path(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> Path:
    """
    Segmentation-stage canonical foci detection CSV path.

    Layout: ``{pipeline_dir}/foci_detection/{stem}_foci[_<channel>].csv``
    """
    stem = _base_id_stem_from_mask(Path(mask_path))
    ch = foci_channel_id_from_source(mask_path, foci_source_path)
    name = _foci_artifact_basename(stem, ch, kind="foci")
    return foci_detection_dir(pipeline_dir) / name


def raw_foci_csv_path(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> Path:
    """Legacy segmentation-stage foci CSV name (`*_foci_raw*.csv`) under ``foci_detection/``."""
    stem = _base_id_stem_from_mask(Path(mask_path))
    ch = foci_channel_id_from_source(mask_path, foci_source_path)
    name = _foci_artifact_basename(stem, ch, kind="foci_raw")
    return foci_detection_dir(pipeline_dir) / name


def filtered_foci_csv_path(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> Path:
    """
    Filtering-stage filtered foci CSV path.

    Layout: ``{pipeline_dir}/foci_filtered/{stem}_foci_filtered[_<channel>].csv``
    """
    stem = _base_id_stem_from_mask(Path(mask_path))
    ch = foci_channel_id_from_source(mask_path, foci_source_path)
    name = _foci_artifact_basename(stem, ch, kind="foci_filtered")
    return foci_filtered_dir(pipeline_dir) / name


def legacy_foci_csv_candidates(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> list[Path]:
    """
    Return legacy flat-path foci CSV locations (pre-``foci_detection/`` / ``foci_filtered/`` layout).

    Used only for reading older projects; new writes use subfolders.
    """
    full_stem = stem_from_mask_path(Path(mask_path))
    channel_stem = full_stem
    channel_low = channel_stem.lower()
    for suffix in ("_mask_filtered", "_masks", "_mask"):
        if channel_low.endswith(suffix):
            channel_stem = channel_stem[: -len(suffix)].strip("_")
            channel_low = channel_stem.lower()
            break
    base_stem = _base_id_stem_from_mask(Path(mask_path))
    ch = foci_channel_id_from_source(mask_path, foci_source_path)
    suffix = f"_{ch}" if ch else ""
    p = Path(pipeline_dir)
    names = [
        _foci_artifact_basename(base_stem, ch, kind="foci"),
        _foci_artifact_basename(base_stem, ch, kind="foci_raw"),
        _foci_artifact_basename(base_stem, ch, kind="foci_filtered"),
        f"{full_stem}_foci{suffix}.csv",
        f"{full_stem}_foci_raw{suffix}.csv",
        f"{full_stem}_foci_filtered{suffix}.csv",
        f"{channel_stem}_foci{suffix}.csv",
        f"{channel_stem}_foci_raw{suffix}.csv",
        f"{channel_stem}_foci_filtered{suffix}.csv",
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for name in names:
        for parent in (p, foci_detection_dir(p), foci_filtered_dir(p)):
            cand = parent / name
            key = str(cand)
            if key not in seen:
                seen.add(key)
                out.append(cand)
    return out


def segmentation_foci_csv_read_candidates(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> list[Path]:
    """Paths to try when loading raw segmentation foci CSV (canonical first, then legacy)."""
    ordered = [
        foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path),
        raw_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path),
    ]
    for cand in legacy_foci_csv_candidates(
        pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path
    ):
        if cand.name.endswith("_foci_filtered.csv") or "_foci_filtered_" in cand.name:
            continue
        ordered.append(cand)
    out: list[Path] = []
    seen: set[str] = set()
    for p in ordered:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_segmentation_foci_csv(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> Optional[Path]:
    """First existing raw/detection foci CSV for this mask + foci source."""
    for cand in segmentation_foci_csv_read_candidates(
        pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path
    ):
        if cand.is_file():
            return cand
    return None


def filtered_foci_csv_read_candidates(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> list[Path]:
    """Paths to try when loading filtered foci CSV (canonical first, then legacy flat)."""
    stem = _base_id_stem_from_mask(Path(mask_path))
    ch = foci_channel_id_from_source(mask_path, foci_source_path)
    name = _foci_artifact_basename(stem, ch, kind="foci_filtered")
    p = Path(pipeline_dir)
    ordered = [
        filtered_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path),
        p / name,
    ]
    for cand in legacy_foci_csv_candidates(
        pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path
    ):
        if "_foci_filtered" in cand.name:
            ordered.append(cand)
    out: list[Path] = []
    seen: set[str] = set()
    for path in ordered:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def resolve_filtered_foci_csv(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> Optional[Path]:
    """First existing filtered foci CSV for this mask + foci source."""
    for cand in filtered_foci_csv_read_candidates(
        pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path
    ):
        if cand.is_file():
            return cand
    return None


def pipeline_artifacts_for_mask(
    mask_path: Path,
    *,
    pipeline_dir: Optional[Path] = None,
    foci_source_path: Optional[Path] = None,
) -> PipelineArtifacts:
    """
    Compute all stage artifact paths for a single mask.

    - `pipeline_dir` defaults to the mask's parent directory.
    - `foci_source_path` optionally yields raw/filtered CSV paths.
    """
    mask_path = Path(mask_path)
    pipeline_dir = Path(pipeline_dir) if pipeline_dir is not None else mask_path.parent
    filt = filtered_mask_path_for_mask(mask_path)
    foci_csv = None
    raw_csv = None
    filt_csv = None
    if foci_source_path is not None:
        foci_csv = foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=Path(foci_source_path))
        raw_csv = raw_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=Path(foci_source_path))
        filt_csv = filtered_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=Path(foci_source_path))
    return PipelineArtifacts(
        pipeline_dir=pipeline_dir,
        mask_path=mask_path,
        filtered_mask_path=filt,
        foci_csv_path=foci_csv,
        raw_foci_csv_path=raw_csv,
        filtered_foci_csv_path=filt_csv,
    )


def stem_pairing_prefix(stem: str) -> str:
    """Acquisition key shared by phase / FI+foci exports (underscore-delimited stems)."""
    return sample_identity_stem(stem)


def foci_tiff_stem_matches_prefix(fstem: str, prefix: str) -> bool:
    """
    True when a TIFF stem is a foci stack for sample prefix ``prefix``.

    Handles file-conversion names like ``sample__fi_foci_GFP`` (double underscore).
    """
    fl, pl = fstem.lower(), prefix.lower()
    if fl == pl or not fl.startswith(pl):
        return False
    suffix = fl[len(pl) :]
    if suffix and suffix[0] != "_":
        return False
    return "_foci" in fl


def _looks_non_channel_token(token: str) -> bool:
    t = (token or "").strip().lower()
    if not t:
        return True
    if t in {"fi", "foci", "mask", "masks", "phase", "bf", "dic", "seg", "segmentation"}:
        return True
    if t.startswith("t=") or t.startswith("z="):
        return True
    if t.endswith("x") and t[:-1].isdigit():
        return True
    return False


# Settings key for foci stacks whose name carries no channel token (``sample_foci.tif``).
FOCI_CHANNEL_FALLBACK_TAG = "foci"


def _channel_tag_or_fallback(token: str, *, explicit: bool) -> str:
    """``explicit``: token follows ``_foci_`` (digits allowed); else a heuristic suffix token."""
    t = (token or "").strip("_- .")
    if t.lower().startswith("overlay_"):
        t = t[len("overlay_") :].strip("_- .")
    if not t or _looks_non_channel_token(t) or (not explicit and t.isdigit()):
        return FOCI_CHANNEL_FALLBACK_TAG
    return t


def foci_channel_tag_from_path(foci_source_path: Path) -> str:
    """
    Readable, sample-independent channel tag of a foci (or FI) stack; original case, never truncated.

    Rules (first match wins, separators matched case-insensitively):
      ``<sample>_fi_foci_<ch>`` / ``<sample>_foci_<ch>`` -> ``<ch>`` (``overlay_`` stripped)
      ``<sample>[-_ .]<ch>_fi_foci`` / ``<sample>[-_ .]<ch>_foci`` -> ``<ch>``
      bare ``<sample>_foci`` (no channel token: single-token prefix, replicate number,
      or role word like ``fi``/``phase``) -> :data:`FOCI_CHANNEL_FALLBACK_TAG`
      ``<sample>_fi_<ch>`` (FI stacks) -> ``<ch>``
      anything else -> full stem.

    Limitation: ``<sample>_<word>_foci`` cannot tell a channel from a sample word, so the
    last token is used (same heuristic as :func:`extract_channel_tag_for_mask_stem`).
    """
    stem = Path(foci_source_path).stem.strip()
    for pat in (r"_fi_foci_(?P<ch>.+)$", r"_foci_(?P<ch>.+)$"):
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            return _channel_tag_or_fallback(m.group("ch"), explicit=True)
    m = re.match(r"^(?P<prefix>.+?)(?:_fi)?_foci$", stem, flags=re.IGNORECASE)
    if m:
        tokens = [t for t in re.split(r"[-_ .]+", m.group("prefix")) if t]
        if len(tokens) < 2:
            return FOCI_CHANNEL_FALLBACK_TAG
        return _channel_tag_or_fallback(tokens[-1], explicit=False)
    m = re.search(r"_fi_(?P<ch>.+)$", stem, flags=re.IGNORECASE)
    if m:
        ch = m.group("ch").strip("_")
        if ch:
            return ch
    return stem


def foci_channel_key_from_path(foci_source_path: Path) -> str:
    """Canonical settings key for a foci stack: :func:`foci_channel_tag_from_path`, casefolded."""
    return foci_channel_tag_from_path(foci_source_path).casefold()


def _candidate_match_stems(stem: str) -> list[str]:
    out: list[str] = []

    def _add(s: str) -> None:
        s = (s or "").strip("_")
        if s and s not in out:
            out.append(s)

    _add(stem)
    # Common pipeline suffixes that shouldn't participate in matching other artifacts.
    # Example dataset:
    #   mask:  G183A-004_Phase60x_mask_filtered.tif
    #   foci:  G183A-004_foci_dsRed.tif
    # so we must consider the shorter "G183A-004" stem as a match candidate.
    low = stem.lower()
    if low.endswith("_mask_filtered"):
        _add(stem[: -len("_mask_filtered")])
    if low.endswith("_mask"):
        _add(stem[: -len("_mask")])
    if "_phase" in low:
        try:
            i = low.index("_phase")
            _add(stem[:i])
        except Exception:
            pass
    if "_" in stem:
        _add(stem.rsplit("_", 1)[0])
    if "__" in stem:
        s2 = stem.split("__", 1)[0]
        _add(s2)
        if "_" in s2:
            _add(s2.rsplit("_", 1)[0])
    _add(sample_identity_stem(stem))
    return out


def _matches_fi_name(name: str, stem: str) -> bool:
    nl, sl = name.lower(), stem.lower()
    if not nl.startswith(sl + "_"):
        return False
    rest = nl[len(sl) :].lstrip("_")
    return rest == "fi" or rest.startswith("fi_") or rest.endswith("_fi") or "_fi_foci" in rest


def extract_channel_tag_for_mask_stem(file_stem: str, mask_stem: str) -> str | None:
    for s in _candidate_match_stems(mask_stem):
        if file_stem in (f"{s}_fi", f"{s}_foci", f"{s}_fi_foci", f"{s}_foci_fi"):
            return None
        if file_stem.startswith(f"{s}_fi_"):
            rem = file_stem[len(s) + 4 :].strip("_")
            if not rem or rem == "foci":
                return None
            if rem.startswith("foci_"):
                rem = rem[len("foci_") :].strip("_")
            if rem.endswith("_foci"):
                rem = rem[: -len("_foci")].strip("_")
            return rem or None
        if file_stem.startswith(f"{s}_foci_"):
            rem = file_stem[len(s) + 6 :].strip("_")
            if rem.lower().startswith("overlay_"):
                rem = rem[8:].strip("_")
            return rem or None
        # File conversion: sample__fi_foci_GFP.tif (double underscore after sample id)
        ls, ss = file_stem.lower(), s.lower()
        if ls.startswith(ss):
            rem2 = file_stem[len(s) :].lstrip("_")
            if rem2.lower().startswith("foci_"):
                tag = rem2[5:].strip("_")
                if tag.lower().startswith("overlay_"):
                    tag = tag[8:].strip("_")
                return tag or None
            m = re.match(r"^(\d+_)?foci_(?P<ch>.+)$", rem2, flags=re.IGNORECASE)
            if m:
                return (m.group("ch") or "").strip("_") or None
            if rem2.lower().startswith("fi_foci_"):
                tag = rem2[8:].strip("_")
                return tag or None
            m = re.match(r"^(\d+_)?fi_foci_(?P<ch>.+)$", rem2, flags=re.IGNORECASE)
            if m:
                return (m.group("ch") or "").strip("_") or None
        if file_stem.startswith(f"{s}_") and file_stem.endswith("_foci"):
            rem = file_stem[len(s) + 1 : -len("_foci")].strip("_")
            return rem or None
        if file_stem.startswith(f"{s}_") and file_stem.endswith("_fi"):
            rem = file_stem[len(s) + 1 : -len("_fi")].strip("_")
            return rem or None
    if file_stem.endswith("_foci"):
        prefix = file_stem[: -len("_foci")].strip("_")
        if "_" in prefix:
            token = prefix.rsplit("_", 1)[-1].strip("_")
            if token and not _looks_non_channel_token(token):
                return token
    return None


def mask_stem_from_path(mask_path: Path) -> str:
    stem = Path(mask_path).stem
    low = stem.lower()
    for suffix in ("_masks", "_mask_filtered"):
        if low.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def file_identity_key(path: Path) -> str:
    """Dedupe key for the same on-disk file (handles .tif vs .TIF on case-insensitive volumes)."""
    p = Path(path)
    try:
        st = p.resolve().stat()
        return f"{int(st.st_dev)}:{int(st.st_ino)}"
    except OSError:
        pass
    try:
        return str(p.resolve()).casefold()
    except OSError:
        return str(p).casefold()


def path_matches_mask_sample(mask_path: Path, file_path: Path) -> bool:
    """Whether a companion TIFF belongs to this mask's sample and replicate."""
    if not file_path.is_file() or not _channel_pairs_with_input(mask_stem_from_path(mask_path), file_path.stem):
        return False
    fs = file_path.stem
    ms = mask_stem_from_path(mask_path)
    if "_foci" in fs.lower():
        return foci_pairs_with_input(ms, fs)
    for stem in _candidate_match_stems(ms):
        if fs.lower() == stem.lower():
            return True
        if _matches_fi_name(fs, stem) or extract_channel_tag_for_mask_stem(fs, stem):
            return True
    return False


def filter_paths_for_mask(mask_path: Path, paths: list[Path]) -> list[Path]:
    """Keep matching companions, sorted and de-duplicated by on-disk file identity."""
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        p = Path(raw)
        key = file_identity_key(p)
        if key not in seen and path_matches_mask_sample(mask_path, p):
            seen.add(key)
            out.append(p)
    return sorted(out, key=lambda p: p.name.lower())


def _companion_tiffs(*search_dirs: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for folder in search_dirs:
        try:
            entries = sorted(Path(folder).iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for p in entries:
            if not p.is_file() or p.suffix.lower() not in (".tif", ".tiff"):
                continue
            stem = p.stem.lower()
            # Acquisition names may contain these words; only generated roles are excluded.
            if stem.endswith((
                "_masks", "_mask_filtered", "_masks_tracked", "_overlay", "_overlay_tracked",
                "_analysis", "_analysis_visualization", "_filter_visualization", "_trajectories",
            )) or re.search(r"_foci_overlay(?:_|$)", stem):
                continue
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    return sorted(out, key=lambda p: p.name.lower())


def discover_foci_source_tiffs(input_stem: str, *search_dirs: Path) -> list[Path]:
    """Foci stacks for one acquisition, including legacy DIA and conversion names."""
    return [p for p in _companion_tiffs(*search_dirs) if foci_pairs_with_input(input_stem, p.stem)]


def discover_foci_companions(input_image_path: Path) -> list[Path]:
    """Segmentation companions next to an input, excluding the input itself."""
    p = Path(input_image_path)
    input_key = file_identity_key(p)
    return [
        fp for fp in discover_foci_source_tiffs(p.stem, p.parent)
        if file_identity_key(fp) != input_key
    ]


def resolve_foci_paths_for_mask(mask_path: Path) -> list[Path]:
    return discover_foci_source_tiffs(
        mask_stem_from_path(mask_path), *artifact_search_dirs_for_mask(mask_path)
    )


def resolve_fi_paths_for_mask(mask_path: Path) -> list[Path]:
    stems = _candidate_match_stems(mask_stem_from_path(mask_path))
    return filter_paths_for_mask(mask_path, [
        p for p in _companion_tiffs(*artifact_search_dirs_for_mask(mask_path))
        if any(_matches_fi_name(p.stem, stem) for stem in stems)
    ])


def collect_background_paths_for_mask(mask_path: Path) -> list[Path]:
    """Preview companions, including base-stem phase images and legacy child folders."""
    mask_path = Path(mask_path)
    out = filter_paths_for_mask(mask_path, _companion_tiffs(*artifact_search_dirs_for_mask(mask_path)))
    if not out and not mask_path.parent.name.startswith("_segmentation"):
        out = filter_paths_for_mask(mask_path, _companion_tiffs(mask_path.parent / "_segmentation"))
    return out


def resolve_fi_inputs_for_mask(mask_path: Path) -> list[Path]:
    """Analyze FI inputs: prefer explicit FI, adding foci-only channels once."""
    explicit = [p for p in resolve_fi_paths_for_mask(mask_path) if "_foci" not in p.stem.lower()]
    mask_stem = mask_stem_from_path(mask_path)

    def channel_key(p: Path) -> str:
        # Normalize comparison keys only; keep historical output channel labels unchanged.
        stem = re.sub(r"_+", "_", p.stem.casefold()).removesuffix("_foci")
        mask = re.sub(r"_+", "_", mask_stem.casefold())
        return extract_channel_tag_for_mask_stem(stem, mask) or ""

    covered = {channel_key(p) for p in explicit} - {""}
    out = explicit + [
        p for p in resolve_foci_paths_for_mask(mask_path)
        if not channel_key(p) or channel_key(p) not in covered
    ]
    return filter_paths_for_mask(mask_path, out)


def infer_pipeline_dir_from_mask(mask_path: Path) -> Path:
    """
    Infer the pipeline directory from a mask path.

    If the mask lives in a `_segmentation_*` folder, return that folder;
    if it lives under ``filtered_masks/`` inside one, return the segmentation folder;
    otherwise return the mask's parent directory.
    """
    m = Path(mask_path)
    p = m.parent
    if p.name == FILTERED_MASKS_DIRNAME and "_segmentation" in p.parent.name:
        return p.parent
    if "_segmentation" in p.name:
        return p
    return p


def infer_base_dir_from_any_mask_artifact(path: Path) -> Path:
    """
    Infer the base dir for a path that could be a mask or filtered mask.

    This reuses `base_dir_for_mask` which already handles `_segmentation_*`.
    """
    return base_dir_for_mask(Path(path))

