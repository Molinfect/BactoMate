"""
Helpers for tab session params: get dict from widgets and apply dict to widgets.
Reduces duplication across get_session_params() / set_session_params() in tabs.
"""
from __future__ import annotations

from typing import Any, List, Union

# Widget type: (key, widget) or (key, widget, "combo"|"spin"|"check").
# If not specified, type is inferred from widget (combo: currentText/findText; spin: value/setValue; check: isChecked/setChecked).
ParamSpec = Union[tuple[str, Any], tuple[str, Any, str]]


def get_from_widgets(specs: List[ParamSpec]) -> dict[str, Any]:
    """Build a session param dict from a list of (key, widget) or (key, widget, type)."""
    out: dict[str, Any] = {}
    for item in specs:
        if len(item) == 3:
            key, widget, kind = item
        else:
            key, widget = item
            kind = _widget_kind(widget)
        if kind == "combo":
            if hasattr(widget, "currentText"):
                out[key] = widget.currentText()
        elif kind == "spin":
            if hasattr(widget, "value"):
                out[key] = widget.value()
        elif kind == "check":
            if hasattr(widget, "isChecked"):
                out[key] = widget.isChecked()
    return out


def apply_to_widgets(data: dict[str, Any], specs: List[ParamSpec]) -> None:
    """Apply a session param dict to widgets. Ignores missing keys and invalid values."""
    if not isinstance(data, dict):
        return
    for item in specs:
        if len(item) == 3:
            key, widget, kind = item
        else:
            key, widget = item
            kind = _widget_kind(widget)
        if key not in data:
            continue
        val = data[key]
        try:
            if kind == "combo" and hasattr(widget, "findText") and hasattr(widget, "setCurrentIndex"):
                idx = widget.findText(str(val))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif kind == "spin" and hasattr(widget, "setValue"):
                widget.setValue(float(val))
            elif kind == "check" and hasattr(widget, "setChecked"):
                widget.setChecked(bool(val))
        except (TypeError, ValueError):
            pass


def _widget_kind(widget: Any) -> str:
    """Infer combo / spin / check from widget methods."""
    if hasattr(widget, "currentText") and hasattr(widget, "findText"):
        return "combo"
    if hasattr(widget, "value") and hasattr(widget, "setValue"):
        return "spin"
    if hasattr(widget, "isChecked") and hasattr(widget, "setChecked"):
        return "check"
    return "combo"
