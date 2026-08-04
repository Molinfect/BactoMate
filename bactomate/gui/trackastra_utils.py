"""Trackastra availability checks for the Tracking tab."""
from __future__ import annotations

TRACKASTRA_SAM2_MODEL = "general_2d_w_SAM2_features"
TRACKASTRA_DEFAULT_MODEL = "general_2d"

_INSTALL_SAM2_FEATURES = (
    "The model «general_2d_w_SAM2_features» needs optional pretrained features.\n\n"
    "In your BactoMate environment, run:\n"
    "  pip install 'trackastra[etultra]'\n\n"
    "Or choose Model: general_2d — it works without this extra."
)

_INSTALL_ILP_DEPS = (
    "ILP tracking needs a Trackastra-compatible motile solver package.\n\n"
    "In your BactoMate environment, run:\n"
    "  pip install \"motile<1.0\"\n\n"
    "Standard BactoMate installs now include this automatically. "
    "If this environment is older, re-run:\n"
    "  pip install -r requirements.txt\n\n"
    "Or choose Mode: greedy / greedy_nodiv."
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


def ilp_dependencies_installed() -> bool:
    try:
        import motile

        return bool(
            hasattr(motile, "costs")
            and hasattr(motile.costs, "NodeSelection")
            and hasattr(motile.costs, "EdgeSelection")
        )
    except ImportError:
        return False


def validate_trackastra_model_choice(
    model_name: str,
    *,
    linking_mode: str | None = None,
) -> tuple[bool, str | None]:
    """Return (ok, user_message) before starting a Trackastra worker."""
    if model_needs_pretrained_feats(model_name) and not trackastra_pretrained_feats_installed():
        return False, _INSTALL_SAM2_FEATURES
    if (linking_mode or "").strip().lower() == "ilp" and not ilp_dependencies_installed():
        return False, _INSTALL_ILP_DEPS
    return True, None


def format_trackastra_model_error(exc: BaseException) -> str:
    """Short, actionable message for common model-load failures."""
    msg = str(exc)
    if "trackastra_pretrained_feats" in msg or "pretrained features" in msg.lower():
        return _INSTALL_SAM2_FEATURES
    msg_lower = msg.lower()
    if "motile" in msg_lower and (
        "module" in msg_lower
        or "nodeselection" in msg_lower
        or "edgeselection" in msg_lower
    ):
        return _INSTALL_ILP_DEPS
    return msg
