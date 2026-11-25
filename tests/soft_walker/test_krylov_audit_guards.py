"""Benchmark checks reject finite rollback adjoints and mismatched values."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from soft_paths import SCRIPTS, SOFT_ROOT

if not SCRIPTS.is_dir():
    pytest.skip("soft_walker command-line tools are unavailable", allow_module_level=True)
_spec = importlib.util.spec_from_file_location(
    "audit_krylov_physics_guards",
    SOFT_ROOT / "diagnostics" / "audit_krylov_physics.py",
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def test_gradient_audit_rejects_finite_rollback_adjoint() -> None:
    # Failed physical steps can roll back to the initial state with a finite,
    # constant loss. Agreement of two zero gradients must not validate this.
    with pytest.raises(RuntimeError, match="gradient interval failed"):
        _audit.validate_gradient_result("fused", 0.0, 1, np.zeros(3), 0.0)


def test_gradient_audit_rejects_different_forward_value() -> None:
    with pytest.raises(AssertionError, match="differs from physical reference"):
        _audit.validate_gradient_result("fused", 0.25, 0, np.ones(3), 0.5)


@pytest.mark.parametrize(
    "value,gradient",
    [
        (np.nan, np.ones(3)),
        (0.25, np.array([1.0, np.inf, 2.0])),
    ],
)
def test_gradient_audit_rejects_nonfinite_adjoint(value: float, gradient: bool) -> None:
    with pytest.raises(RuntimeError, match="nonfinite interval adjoint"):
        _audit.validate_gradient_result("fused", value, 0, gradient, 0.25)


def test_gradient_audit_accepts_valid_matching_result() -> None:
    _audit.validate_gradient_result("fused", 0.25, 0, np.ones(3), 0.25)
