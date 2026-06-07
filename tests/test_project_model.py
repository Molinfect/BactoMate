"""Project file schema migration and path helpers."""
from __future__ import annotations

from pathlib import Path

from bactomate.gui.project_model import (
    SCHEMA_VERSION,
    default_project_dict,
    load_project,
    migrate_project,
    save_project,
    store_module_path,
)


def test_migrate_v1_to_v2_preserves_raw_input(tmp_path: Path) -> None:
    legacy = {
        "schema_version": 1,
        "paths": {"raw_input": "data/sample.nd2"},
        "ui": {"last_module": "file_conversion"},
    }
    out = migrate_project(legacy)
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["paths"]["raw_input"] == "data/sample.nd2"
    assert isinstance(out["paths"]["segmentation"], dict)
    assert out["paths"]["training"]["dataset_dir"] == ""


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    data = default_project_dict(display_name="demo")
    data["paths"]["raw_input"] = "inputs"
    data["paths"]["segmentation"]["backend"] = "unet"
    save_project(tmp_path, data)
    loaded = load_project(tmp_path)
    assert loaded is not None
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["paths"]["segmentation"]["backend"] == "unet"
    assert loaded["display_name"] == "demo"


def test_store_module_path_relative(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    target = ws / "Tif_files"
    target.mkdir()
    paths: dict = {}
    store_module_path(paths, ws, "segmentation", "input", str(target))
    assert paths["segmentation"]["input"] == "Tif_files"
