"""
Shared rules for public distributions (DMG, Windows zip, main-branch source).

Developer-only paths stay in the git ``dev`` branch under ``docs/internal/``.
Release bundles and ``strip_dev_tree_for_main.sh`` omit them so end users only
receive user-facing docs and an empty ``Logs/`` folder.
"""
from __future__ import annotations

from pathlib import Path

# Top-level directory names never copied into public bundles.
PUBLIC_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".cursor",
        ".github",
        "BactoMate.app",
        "dist",
        "venv.old",
        "Logs",
        "paper_data",
    }
)

# Under docs/, only internal/ is omitted from public trees; user manuals stay.
PUBLIC_SKIP_DOC_PARTS: tuple[str, ...] = ("docs/internal",)

# Root-level dev artifacts (optional cleanup on main).
PUBLIC_DEV_ROOT_FILES: frozenset[str] = frozenset(
    {
        "BactoMate_Mns_NatureMethods_v2.docx",
    }
)


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_public_doc_path(rel_posix: str) -> bool:
    """True when a docs/ path should ship to end users."""
    if not rel_posix.startswith("docs/"):
        return True
    return not any(rel_posix == part or rel_posix.startswith(part + "/") for part in PUBLIC_SKIP_DOC_PARTS)


def should_skip_public_tree_entry(name: str, *, is_dir: bool) -> bool:
    if name.endswith(".pyc") or name.endswith(".pyo"):
        return True
    if name == ".DS_Store":
        return True
    if is_dir and name in PUBLIC_SKIP_DIR_NAMES:
        return True
    return False


def should_skip_public_copy(src: Path, root: Path) -> bool:
    rel = relative_posix(src, root)
    if should_skip_public_tree_entry(src.name, is_dir=src.is_dir()):
        return True
    if src.is_dir() and src.name == "docs":
        return False
    if not is_public_doc_path(rel):
        return True
    if src.is_file() and src.name in PUBLIC_DEV_ROOT_FILES:
        return True
    return False


def copytree_ignore_public(_dir: str, names: list[str], *, root: Path) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        child = Path(_dir) / name
        try:
            rel_root = relative_posix(child, root)
        except ValueError:
            ignored.add(name)
            continue
        if should_skip_public_tree_entry(name, is_dir=child.is_dir()):
            ignored.add(name)
            continue
        if not is_public_doc_path(rel_root):
            ignored.add(name)
    return ignored


def ensure_empty_logs_dir(logs_dir: Path) -> None:
    """Create a writable, empty Logs folder for fresh installs."""
    if logs_dir.exists():
        for child in logs_dir.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        logs_dir.mkdir(parents=True, exist_ok=True)
    keep = logs_dir / ".gitkeep"
    if not keep.exists():
        keep.write_text("", encoding="utf-8")
