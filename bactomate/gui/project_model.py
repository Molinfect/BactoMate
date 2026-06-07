"""
Workspace project file: bactomate_project.json at the workspace root.

Keeps project-scoped paths (relative when possible) and light UI state.
Global preferences remain in QSettings (see preferences.py).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PROJECT_FILENAME = "bactomate_project.json"
SCHEMA_VERSION = 2

MODULE_DASHBOARD = "dashboard"
MODULE_FILE_CONVERSION = "file_conversion"
MODULE_SEGMENTATION = "segmentation"
MODULE_TRACKING = "tracking"
MODULE_POST_PROCESSING = "post_processing"
MODULE_TRAINING = "training"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_project_dict(*, display_name: str = "") -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "display_name": display_name,
        "paths": default_module_paths(),
        "ui": {
            "last_module": MODULE_DASHBOARD,
            "splitter_states": {},
        },
    }


def default_module_paths() -> dict[str, Any]:
    return {
        "raw_input": "",
        "segmentation": {"backend": "omnipose", "input": ""},
        "filtering": {"masks": ""},
        "analyze": {"masks": ""},
        "tracking": {"masks": "", "source": ""},
        "post_processing": {"video": "", "montage": "", "excel": ""},
        "training": {"dataset_dir": ""},
    }


def migrate_project(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade older project JSON to the current schema."""
    if not isinstance(data, dict):
        return default_project_dict()
    out = dict(data)
    version = int(out.get("schema_version", 1) or 1)
    paths = dict(out.get("paths") or {})
    defaults = default_module_paths()
    if version < 2:
        raw = str(paths.get("raw_input") or "")
        merged = dict(defaults)
        if raw:
            merged["raw_input"] = raw
        out["paths"] = merged
        out["schema_version"] = SCHEMA_VERSION
    else:
        merged = dict(defaults)
        merged.update({k: v for k, v in paths.items() if k in defaults})
        for key in ("segmentation", "filtering", "analyze", "tracking", "post_processing", "training"):
            if not isinstance(merged.get(key), dict):
                merged[key] = dict(defaults[key])
            else:
                merged[key] = {**defaults[key], **merged[key]}
        out["paths"] = merged
        out["schema_version"] = SCHEMA_VERSION
    ui = out.get("ui")
    if not isinstance(ui, dict):
        out["ui"] = {"last_module": MODULE_DASHBOARD, "splitter_states": {}}
    return out


def project_file_path(workspace: Path) -> Path:
    return workspace.resolve() / PROJECT_FILENAME


def load_project(workspace: Path) -> Optional[dict[str, Any]]:
    """Load, migrate, and return project dict, or None if missing or invalid."""
    path = project_file_path(workspace)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return migrate_project(data)
    except (OSError, json.JSONDecodeError):
        return None


def save_project(workspace: Path, data: dict[str, Any]) -> None:
    """Write project JSON; sets schema_version and updated_at."""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    out = migrate_project(dict(data))
    out["schema_version"] = SCHEMA_VERSION
    out["updated_at"] = utc_now_iso()
    path = project_file_path(workspace)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def path_relative_to_workspace(workspace: Path, target: Path) -> str:
    """Return posix relative path from workspace to target, or absolute if not under workspace."""
    workspace = workspace.resolve()
    try:
        target = target.resolve()
        rel = target.relative_to(workspace)
        return rel.as_posix()
    except ValueError:
        return target.resolve().as_posix()


def resolve_maybe_relative(workspace: Path, stored: str) -> Path:
    """Resolve a path stored in JSON (relative to workspace or absolute)."""
    if not stored:
        return Path()
    p = Path(stored)
    if p.is_absolute():
        return p
    return (workspace.resolve() / p).resolve()


def store_module_path(
    paths: dict[str, Any],
    workspace: Path,
    section: str,
    field: str,
    path_str: str,
) -> None:
    """Store one module path under paths[section][field], relative when possible."""
    text = (path_str or "").strip()
    section_dict = paths.setdefault(section, {})
    if not isinstance(section_dict, dict):
        section_dict = {}
        paths[section] = section_dict
    if not text:
        section_dict[field] = ""
        return
    try:
        section_dict[field] = path_relative_to_workspace(workspace, Path(text))
    except Exception:
        section_dict[field] = text
