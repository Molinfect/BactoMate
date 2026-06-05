"""Trackastra model availability checks for the Tracking tab."""
from __future__ import annotations

TRACKASTRA_SAM2_MODEL = "general_2d_w_SAM2_features"
TRACKASTRA_DEFAULT_MODEL = "general_2d"

_INSTALL_SAM2_FEATURES = (
    "The model «general_2d_w_SAM2_features» needs optional pretrained features.\n\n"
    "In your BactoMate environment, run:\n"
    "  pip install 'trackastra[etultra]'\n\n"
    "Or choose Model: general_2d — it works without this extra."
)


def trackastra_pretrained_feats_installed() -> bool:
    try:
        import trackastra_pretrained_feats  # noqa: F401

        return True
    except ImportError:
        return False


def model_needs_pretrained_feats(model_name: str) -> bool:
    name = (model_name or "").strip()
    return name == TRACKASTRA_SAM2_MODEL or "sam2" in name.lower()


def validate_trackastra_model_choice(model_name: str) -> tuple[bool, str | None]:
    """Return (ok, user_message) before starting a Trackastra worker."""
    if model_needs_pretrained_feats(model_name) and not trackastra_pretrained_feats_installed():
        return False, _INSTALL_SAM2_FEATURES
    return True, None


def format_trackastra_model_error(exc: BaseException) -> str:
    """Short, actionable message for common model-load failures."""
    msg = str(exc)
    if "trackastra_pretrained_feats" in msg or "pretrained features" in msg.lower():
        return _INSTALL_SAM2_FEATURES
    return msg
