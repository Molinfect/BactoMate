"""Regression tests for Spotiflow wandb/protobuf import error messaging."""
from __future__ import annotations

from bactomate.modules.foci_utils import (
    format_spotiflow_import_error,
    is_spotiflow_wandb_protobuf_error,
)


def test_is_spotiflow_wandb_protobuf_error_detects_imports_failure() -> None:
    exc = ImportError(
        "cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'"
    )
    assert is_spotiflow_wandb_protobuf_error(exc) is True


def test_is_spotiflow_wandb_protobuf_error_walks_cause_chain() -> None:
    root = ImportError(
        "cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'"
    )
    wrapped = RuntimeError("Spotiflow load failed")
    wrapped.__cause__ = root
    assert is_spotiflow_wandb_protobuf_error(wrapped) is True


def test_is_spotiflow_wandb_protobuf_error_ignores_unrelated() -> None:
    assert is_spotiflow_wandb_protobuf_error(ImportError("No module named 'spotiflow'")) is False


def test_format_spotiflow_import_error_includes_repair_command() -> None:
    exc = ImportError(
        "cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'"
    )
    msg = format_spotiflow_import_error(exc)
    assert "wandb" in msg.lower()
    assert "protobuf" in msg.lower()
    assert "pip install" in msg
    assert "smoke-test" in msg
