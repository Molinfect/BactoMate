"""Stable fingerprints for folder-preview settings (stale detection / cache keys)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def settings_fingerprint(params: Mapping[str, Any] | None) -> str:
    """Deterministic short hash of a settings mapping (JSON-sorted)."""
    payload = _normalize(params or {})
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _normalize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _normalize(obj[k]) for k in sorted(obj.keys(), key=lambda x: str(x))}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)
