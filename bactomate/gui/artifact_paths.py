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

from bactomate.gui.analysis_paths import base_dir_for_mask, stem_from_mask_path

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
    """
    True when ``foci_stem`` is a foci stack for the same acquisition as ``input_stem``.

    1. **Role-marker identity** — ``sample_identity_stem`` equal on both sides.
    2. **Legacy LCP** — when the input has *no* role marker (e.g. ``EM20285_003_DIA``),
       pair if the foci role-stripped ID equals the longest common underscore prefix and
       the foci file matches that prefix as a foci sibling.
    """
    inp = (input_stem or "").strip()
    foc = (foci_stem or "").strip()
    if not inp or not foc:
        return False
    if "_foci" not in foc.lower():
        return False
    # Pure foci-as-input: require exact identity match (no LCP bleed).
    if "_foci" in inp.lower() and "_fi_foci" not in inp.lower():
        return sample_identity_stem(inp).casefold() == sample_identity_stem(foc).casefold()

    inp_id = sample_identity_stem(inp)
    foc_id = sample_identity_stem(foc)
    if inp_id and foc_id and foci_acquisition_ids_align(inp_id, foc_id):
        return True

    # Free-form phase names (DIA, Phase60X, …): LCP when input lacks role tags.
    if stem_has_role_marker(inp):
        return False
    if not foc_id:
        return False
    lcp = longest_common_underscore_prefix(inp, foc)
    if not lcp or not foci_acquisition_ids_align(lcp, foc_id):
        return False
    if not foci_tiff_stem_matches_prefix(foc, lcp):
        return False
    il = inp.casefold()
    pl = lcp.casefold()
    if il == pl:
        return True
    return il.startswith(pl + "_")


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


def discover_foci_source_tiffs(input_stem: str, *search_dirs: Path) -> list[Path]:
    """
    Find ``*_foci*.tif`` stacks for a segmentation or mask stem.

    Mirrors Omnipose folder-mode discovery: prefix globs plus progressively shorter
    underscore-trimmed stems so ``sample_Phase.tif`` / ``sample_Phase_masks.tif`` can
    locate ``sample_foci_GFP.tif`` from file conversion.
    """
    stem = (input_stem or "").strip()
    if not stem:
        return []
    candidates: list[Path] = []

    def _glob_at(folder: Path, prefix: str) -> None:
        if not folder.is_dir():
            return
        for pat in (
            f"{prefix}_foci.tif",
            f"{prefix}_foci.tiff",
            f"{prefix}_foci_*.tif",
            f"{prefix}_foci_*.tiff",
            f"{prefix}_fi_foci.tif",
            f"{prefix}_fi_foci.tiff",
            f"{prefix}_fi_foci_*.tif",
            f"{prefix}_fi_foci_*.tiff",
            f"{prefix}__fi_foci.tif",
            f"{prefix}__fi_foci.tiff",
            f"{prefix}__fi_foci_*.tif",
            f"{prefix}__fi_foci_*.tiff",
            f"{prefix}*_foci*.tif",
            f"{prefix}*_foci*.tiff",
        ):
            candidates.extend(folder.glob(pat))

    dirs = [Path(d) for d in search_dirs if Path(d).exists()]
    for folder in dirs:
        _glob_at(folder, stem)
        parts = stem.split("_")
        for i in range(len(parts) - 1, 0, -1):
            prefix = "_".join(parts[:i])
            if prefix:
                _glob_at(folder, prefix)

    uniq: list[Path] = []
    seen: set[str] = set()
    for c in sorted(candidates, key=lambda x: x.name.lower()):
        if "_foci" not in c.name.lower():
            continue
        try:
            rp = str(c.resolve())
        except Exception:
            rp = str(c)
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(c)

    stem_variants: list[str] = [stem]
    parts3 = stem.split("_")
    for i in range(len(parts3) - 1, 0, -1):
        pfx = "_".join(parts3[:i])
        if pfx:
            stem_variants.append(pfx)
    seen_names = {x.name.lower() for x in uniq}
    for folder in dirs:
        try:
            folder_entries = sorted(folder.iterdir(), key=lambda x: x.name.lower())
        except OSError:
            folder_entries = []
        for fp in folder_entries:
            if not fp.is_file():
                continue
            low = fp.name.lower()
            if not (low.endswith(".tif") or low.endswith(".tiff")):
                continue
            if "_foci" not in low:
                continue
            fstem = fp.stem
            hit = False
            for pfx in stem_variants:
                if foci_tiff_stem_matches_prefix(fstem, pfx):
                    hit = True
                    break
            if not hit:
                continue
            try:
                rp = str(fp.resolve())
            except Exception:
                rp = str(fp)
            if rp in seen:
                continue
            if fp.name.lower() in seen_names:
                continue
            seen.add(rp)
            seen_names.add(fp.name.lower())
            uniq.append(fp)

    inp_key = stem_pairing_prefix(stem)
    paired = [x for x in uniq if foci_pairs_with_input(stem, x.stem)]
    # Prefer strict pairs; if none matched but identity keys equal after strip, keep those.
    if not paired:
        paired = [x for x in uniq if stem_pairing_prefix(x.stem).casefold() == inp_key.casefold()]
    uniq = paired
    uniq.sort(key=lambda x: x.name.lower())
    return uniq


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

