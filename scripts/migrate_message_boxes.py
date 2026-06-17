#!/usr/bin/env python3
"""One-shot migration: QMessageBox -> BactoMate styled dialog helpers."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

GUI_ROOT = Path(__file__).resolve().parents[1] / "bactomate" / "gui"

KIND_BY_ATTR = {
    "warning": "warning",
    "information": "info",
    "critical": "error",
}

HELPER_IMPORTS = {
    "show_message_dialog",
    "show_confirm_dialog",
}


def _is_qmessagebox(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "QMessageBox"


def _is_std_button(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "QMessageBox"
        and node.value.attr == "StandardButton"
        and node.attr == name
    )


def _default_confirm_from_arg(node: ast.AST | None) -> bool:
    if node is None:
        return True
    if _is_std_button(node, "Yes") or _is_std_button(node, "Ok"):
        return True
    if _is_std_button(node, "No") or _is_std_button(node, "Cancel"):
        return False
    return True


def _labels_from_buttons(node: ast.AST | None) -> tuple[str, str]:
    text = ast.unparse(node) if node is not None else ""
    if "Ok" in text and "Cancel" in text:
        return "OK", "Cancel"
    return "Yes", "No"


def _confirm_call_from_question(call: ast.Call) -> ast.Call:
    parent = call.args[0] if len(call.args) >= 1 else ast.Constant(None)
    title = call.args[1] if len(call.args) >= 2 else ast.Constant("")
    message = call.args[2] if len(call.args) >= 3 else ast.Constant("")
    buttons = call.args[3] if len(call.args) >= 4 else None
    default_btn = call.args[4] if len(call.args) >= 5 else None
    confirm_label, cancel_label = _labels_from_buttons(buttons)
    default_confirm = _default_confirm_from_arg(default_btn)
    return ast.Call(
        func=ast.Name(id="show_confirm_dialog", ctx=ast.Load()),
        args=[parent],
        keywords=[
            ast.keyword(arg="title", value=title),
            ast.keyword(arg="message", value=message),
            ast.keyword(arg="confirm_label", value=ast.Constant(confirm_label)),
            ast.keyword(arg="cancel_label", value=ast.Constant(cancel_label)),
            ast.keyword(arg="default_confirm", value=ast.Constant(default_confirm)),
        ],
    )


def _message_call(call: ast.Call, kind: str) -> ast.Call:
    args = list(call.args[:3])
    return ast.Call(
        func=ast.Name(id="show_message_dialog", ctx=ast.Load()),
        args=args,
        keywords=[ast.keyword(arg="kind", value=ast.Constant(kind))],
    )


class MessageBoxMigrator(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False
        self.needs_show_message = False
        self.needs_show_confirm = False
        self._question_assigns: dict[str, ast.Call] = {}

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        if not isinstance(node.value, ast.Call):
            return node
        call = node.value
        if not (
            isinstance(call.func, ast.Attribute)
            and _is_qmessagebox(call.func.value)
            and call.func.attr == "question"
        ):
            return node
        target = node.targets[0].id
        self._question_assigns[target] = call
        self.changed = True
        self.needs_show_confirm = True
        new_call = _confirm_call_from_question(call)
        return ast.Assign(targets=[node.targets[0]], value=new_call)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node
        op = node.ops[0]
        if not isinstance(op, (ast.Eq, ast.NotEq)):
            return node
        left = node.left
        right = node.comparators[0]

        question_call: ast.Call | None = None
        positive_button: str | None = None
        negate = False

        if isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute):
            if _is_qmessagebox(left.func.value) and left.func.attr == "question":
                question_call = left
                if _is_std_button(right, "Yes") or _is_std_button(right, "Ok"):
                    positive_button = right.attr if isinstance(right, ast.Attribute) else "Yes"
                    negate = isinstance(op, ast.NotEq)
                elif _is_std_button(right, "No") or _is_std_button(right, "Cancel"):
                    positive_button = "Yes" if _is_std_button(right, "No") else "Ok"
                    negate = not isinstance(op, ast.NotEq)
        elif isinstance(left, ast.Name) and left.id in self._question_assigns:
            question_call = self._question_assigns[left.id]
            if _is_std_button(right, "Yes") or _is_std_button(right, "Ok"):
                negate = isinstance(op, ast.NotEq)
            elif _is_std_button(right, "No") or _is_std_button(right, "Cancel"):
                negate = not isinstance(op, ast.NotEq)

        if question_call is None:
            return node

        self.changed = True
        self.needs_show_confirm = True
        confirm = _confirm_call_from_question(question_call)
        if negate:
            return ast.UnaryOp(op=ast.Not(), operand=confirm)
        return confirm

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        test = node.test
        if isinstance(test, ast.Compare):
            return node
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = test.operand
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                if inner.func.id == "show_confirm_dialog":
                    return node
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute):
            return node
        if not _is_qmessagebox(node.func.value):
            return node
        attr = node.func.attr
        if attr in KIND_BY_ATTR:
            self.changed = True
            self.needs_show_message = True
            return _message_call(node, KIND_BY_ATTR[attr])
        return node


def _remove_qmessagebox_import(tree: ast.Module) -> bool:
    removed = False
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module != "PyQt6.QtWidgets":
            continue
        new_names = [alias for alias in node.names if alias.name != "QMessageBox"]
        if len(new_names) != len(node.names):
            node.names = new_names
            removed = True
    return removed


def _ensure_widget_imports(tree: ast.Module, needs_message: bool, needs_confirm: bool) -> None:
    wanted = set()
    if needs_message:
        wanted.add("show_message_dialog")
    if needs_confirm:
        wanted.add("show_confirm_dialog")

    existing_from_widgets = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "bactomate.gui.widgets":
            existing_from_widgets = node
            break

    if not wanted:
        return

    if existing_from_widgets is None:
        names = [ast.alias(name=n) for n in sorted(wanted)]
        insert_at = 0
        for i, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                insert_at = i + 1
        tree.body.insert(
            insert_at,
            ast.ImportFrom(
                module="bactomate.gui.widgets",
                names=names,
                level=0,
            ),
        )
        return

    present = {alias.name for alias in existing_from_widgets.names}
    for name in sorted(wanted):
        if name not in present:
            existing_from_widgets.names.append(ast.alias(name=name))


def migrate_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if "QMessageBox" not in source:
        return False
    tree = ast.parse(source, filename=str(path))
    migrator = MessageBoxMigrator()
    new_tree = migrator.visit(tree)
    if not migrator.changed:
        return False
    ast.fix_missing_locations(new_tree)
    _remove_qmessagebox_import(new_tree)
    _ensure_widget_imports(new_tree, migrator.needs_show_message, migrator.needs_show_confirm)
    new_source = ast.unparse(new_tree) + "\n"
    path.write_text(new_source, encoding="utf-8")
    return True


def main() -> int:
    changed_files = []
    for path in sorted(GUI_ROOT.rglob("*.py")):
        if migrate_file(path):
            changed_files.append(path)
    print(f"Migrated {len(changed_files)} files:")
    for path in changed_files:
        print(f"  {path.relative_to(GUI_ROOT.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
