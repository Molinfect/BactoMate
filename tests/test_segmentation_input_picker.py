"""Segmentation TIFF picker: which files are listed / preferred as cell inputs."""
from __future__ import annotations

from pathlib import Path

from bactomate.gui.tabs.omnipose_seg import OmniposeSegTab


def test_fi_foci_stacks_are_not_segmentation_inputs() -> None:
    assert not OmniposeSegTab._is_segmentation_input_file(Path("sample_fi_foci_GFP.tif"))
    assert not OmniposeSegTab._is_segmentation_input_file(Path("sample_foci_GFP.tif"))
    assert OmniposeSegTab._is_segmentation_input_file(Path("sample_phase.tif"))
    assert OmniposeSegTab._is_segmentation_input_file(Path("sample_fi_mCherry.tif"))
    assert not OmniposeSegTab._is_segmentation_input_file(Path("sample_phase_masks.tif"))


def test_default_selection_prefers_phase_over_fi() -> None:
    files = [
        Path("sample_phase.tif"),
        Path("sample_fi_mCherry.tif"),
    ]
    pred = OmniposeSegTab._default_segmentation_selection_predicate(files)
    assert pred is not None
    assert pred(Path("sample_phase.tif")) is True
    assert pred(Path("sample_fi_mCherry.tif")) is False


def test_default_selection_allows_all_when_only_fi_present() -> None:
    files = [Path("sample_fi_mCherry.tif")]
    pred = OmniposeSegTab._default_segmentation_selection_predicate(files)
    assert pred is None


def test_segmentation_companions_include_input_and_spotiflow_tiffs() -> None:
    tab = OmniposeSegTab.__new__(OmniposeSegTab)
    input_tiff = Path("sample_phase.tif")
    foci_tiffs = [Path("sample_foci_GFP.tif"), Path("sample_fi_foci_mCherry.tif")]
    tab._find_foci_sources_for_input = lambda _path: foci_tiffs

    companions = tab._companions_for_segmentation_input(input_tiff)

    assert companions["mask"] == [(input_tiff, "segmentation input")]
    assert companions["fi"] == [
        (Path("sample_foci_GFP.tif"), "spotiflow companion"),
        (Path("sample_fi_foci_mCherry.tif"), "spotiflow companion"),
    ]
