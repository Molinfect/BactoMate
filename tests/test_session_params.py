"""
Tests for gui.session_params (get_from_widgets / apply_to_widgets).
Uses mock widgets to avoid Qt dependency. Run with: pytest tests/ or python -m tests.test_session_params
"""
import sys
from pathlib import Path

# Add project root so we can import bactomate
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bactomate.gui.session_params import get_from_widgets, apply_to_widgets


class MockCombo:
    def __init__(self, items=None, current=0):
        self._items = items or ["a", "b", "c"]
        self._current = current

    def currentText(self):
        return self._items[self._current]

    def findText(self, text):
        try:
            return self._items.index(text)
        except ValueError:
            return -1

    def setCurrentIndex(self, idx):
        if 0 <= idx < len(self._items):
            self._current = idx


class MockSpin:
    def __init__(self, value=0.0):
        self._value = float(value)

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = float(v)


class MockCheck:
    def __init__(self, checked=False):
        self._checked = bool(checked)

    def isChecked(self):
        return self._checked

    def setChecked(self, v):
        self._checked = bool(v)


def test_get_from_widgets():
    combo = MockCombo(current=1)
    spin = MockSpin(42.5)
    check = MockCheck(True)
    out = get_from_widgets([
        ("model", combo),
        ("diameter", spin),
        ("use_gpu", check),
    ])
    assert out["model"] == "b"
    assert out["diameter"] == 42.5
    assert out["use_gpu"] is True


def test_apply_to_widgets():
    combo = MockCombo(current=0)
    spin = MockSpin(0)
    check = MockCheck(False)
    data = {"model": "c", "diameter": 10.0, "use_gpu": True}
    apply_to_widgets(data, [
        ("model", combo),
        ("diameter", spin),
        ("use_gpu", check),
    ])
    assert combo.currentText() == "c"
    assert spin.value() == 10.0
    assert check.isChecked() is True


def test_round_trip():
    combo = MockCombo(current=2)
    spin = MockSpin(3.14)
    check = MockCheck(True)
    specs = [("x", combo), ("y", spin), ("z", check)]
    data = get_from_widgets(specs)
    combo2 = MockCombo(current=0)
    spin2 = MockSpin(0)
    check2 = MockCheck(False)
    apply_to_widgets(data, [("x", combo2), ("y", spin2), ("z", check2)])
    assert combo2.currentText() == combo.currentText()
    assert spin2.value() == spin.value()
    assert check2.isChecked() == check.isChecked()


def test_apply_ignores_missing_keys():
    spin = MockSpin(5.0)
    apply_to_widgets({"other_key": 99}, [("diameter", spin)])
    assert spin.value() == 5.0


def test_apply_handles_invalid_values():
    spin = MockSpin(1.0)
    check = MockCheck(False)
    apply_to_widgets({"diameter": "not_a_number", "use_gpu": True}, [
        ("diameter", spin),
        ("use_gpu", check),
    ])
    assert spin.value() == 1.0
    assert check.isChecked() is True


if __name__ == "__main__":
    test_get_from_widgets()
    test_apply_to_widgets()
    test_round_trip()
    test_apply_ignores_missing_keys()
    test_apply_handles_invalid_values()
    print("All session_params tests passed.")
