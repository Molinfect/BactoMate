from __future__ import annotations

from pathlib import Path


def test_dedupe_prefers_filtered_over_raw() -> None:
    from bactomate.gui.artifact_paths import dedupe_mask_paths_prefer_filtered

    raw = Path("Hook_maleimide_immonustaining_003_Phase60X_masks.tif")
    filt = Path("filtered_masks/Hook_maleimide_immonustaining_003_Phase60X_mask_filtered.tif")
    out = dedupe_mask_paths_prefer_filtered([raw, filt])
    assert len(out) == 1
    assert out[0].name.endswith("_mask_filtered.tif")


def test_dedupe_keeps_raw_when_no_filtered() -> None:
    from bactomate.gui.artifact_paths import dedupe_mask_paths_prefer_filtered

    raw = Path("sample_005_Phase60X_masks.tif")
    out = dedupe_mask_paths_prefer_filtered([raw])
    assert len(out) == 1
    assert out[0].name == raw.name


def test_dedupe_one_entry_per_sample() -> None:
    from bactomate.gui.artifact_paths import dedupe_mask_paths_prefer_filtered

    paths = [
        Path("a_003_Phase60X_masks.tif"),
        Path("a_003_Phase60X_mask_filtered.tif"),
        Path("a_004_Phase60X_masks.tif"),
        Path("a_004_Phase60X_mask_filtered.tif"),
    ]
    out = dedupe_mask_paths_prefer_filtered(paths)
    assert len(out) == 2
    assert all(p.name.endswith("_mask_filtered.tif") for p in out)
