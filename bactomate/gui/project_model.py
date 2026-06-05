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
SCHEMA_VERSION = 1

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
        "paths": {},
        "ui": {
            "last_module": MODULE_DASHBOARD,
            "splitter_states": {},
        },
    }


def project_file_path(workspace: Path) -> Path:
    return workspace.resolve() / PROJECT_FILENAME


def load_project(workspace: Path) -> Optional[dict[str, Any]]:
    """Load and return project dict, or None if missing or invalid."""
    path = project_file_path(workspace)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def save_project(workspace: Path, data: dict[str, Any]) -> None:
    """Write project JSON; sets schema_version and updated_at."""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    out = dict(data)
    out["schema_version"] = int(out.get("schema_version", SCHEMA_VERSION))
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
