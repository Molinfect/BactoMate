"""Tests for public release packaging rules."""
from __future__ import annotations

from pathlib import Path

from build.public_packaging import (
    copytree_ignore_public,
    ensure_empty_logs_dir,
    is_public_doc_path,
    should_skip_public_copy,
)


def test_internal_docs_are_not_public() -> None:
    assert not is_public_doc_path("docs/internal/PLAN.md")
    assert not is_public_doc_path("docs/internal/history/foo.md")
    assert is_public_doc_path("docs/BACTOMATE_USER_MANUAL.md")
    assert is_public_doc_path("docs/RELEASE_NOTES.md")


def test_copytree_ignore_skips_internal_docs(tmp_path: Path) -> None:
    root = tmp_path
    docs = root / "docs"
    (docs / "internal").mkdir(parents=True)
    (docs / "internal" / "PLAN.md").write_text("secret", encoding="utf-8")
    (docs / "README.md").write_text("public", encoding="utf-8")

    ignored = copytree_ignore_public(str(docs), ["internal", "README.md"], root=root)
    assert "internal" in ignored
    assert "README.md" not in ignored


def test_should_skip_public_copy_dev_root_file(tmp_path: Path) -> None:
    root = tmp_path
    draft = root / "BactoMate_Mns_NatureMethods_v2.docx"
    draft.write_bytes(b"x")
    assert should_skip_public_copy(draft, root)


def test_ensure_empty_logs_dir_clears_runtime_logs(tmp_path: Path) -> None:
    logs = tmp_path / "Logs"
    logs.mkdir()
    (logs / "bactomate_gui_2026.log").write_text("old", encoding="utf-8")
    ensure_empty_logs_dir(logs)
    assert (logs / ".gitkeep").is_file()
    assert list(logs.glob("*.log")) == []
