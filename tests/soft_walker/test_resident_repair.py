"""Targeted regressions for the failed resident gate; no full historical suite."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import solve_triangular
from soft_paths import needs_scripts

from envs.soft_walker.diagnostics.residency.probes import PROBES, capture_probe
from envs.soft_walker.engine.execution.compiled import (
    CompiledProgram,
    _jaxpr_audit,
)
from envs.soft_walker.engine.execution.eligibility import capture_eligibility
from envs.soft_walker.engine.execution.policy import ResidencyError, ResidentPolicy
from envs.soft_walker.engine.numerics.krylov import fused_gmres, triangular_backsolve


@pytest.mark.parametrize("size", [1, 2, 7, 20, 40])
def test_array_backsolve_matches_scipy_and_ignores_lower_triangle(
    size: int, dtype: np.dtype[np.floating]
) -> None:
    rng = np.random.default_rng(210 + size)
    upper = np.triu(rng.normal(0, 0.025, (size, size)))
    upper += np.diag(np.linspace(0.75, 1.5, size))
    rhs = rng.normal(size=size)
    upper, rhs = upper.astype(dtype), rhs.astype(dtype)
    upper[np.tril_indices(size, -1)] = np.nan  # Unused storage is never read.
    expected = solve_triangular(upper, rhs, lower=False, check_finite=False)
    got = jax.jit(triangular_backsolve)(jnp.asarray(upper), jnp.asarray(rhs))
    np.testing.assert_allclose(
        got,
        expected,
        rtol=2e-6 if dtype == np.float32 else 5e-14,
        atol=3e-7 if dtype == np.float32 else 5e-15,
    )
    safe = np.triu(upper)
    residual = np.linalg.norm(safe @ np.asarray(got) - rhs) / np.linalg.norm(rhs)
    assert residual < (2e-6 if dtype == np.float32 else 1e-13)


def test_array_backsolve_batched_derivatives_and_invalid_diagonal(
    dtype: np.dtype[np.floating],
) -> None:
    rng = np.random.default_rng(42)
    n = 7
    base = np.triu(rng.normal(0, 0.03, (n, n))) + np.diag(np.linspace(1.0, 2.0, n))
    delta = np.triu(rng.normal(0, 0.02, (n, n)))
    rhs = rng.normal(size=n)
    drhs = rng.normal(size=n)
    weights = np.linspace(0.2, 1.0, n)

    def loss(p: jax.Array) -> jax.Array:
        x = triangular_backsolve(
            jnp.asarray(base, dtype) + p * jnp.asarray(delta, dtype),
            jnp.asarray(rhs, dtype) + p * jnp.asarray(drhs, dtype),
        )
        return jnp.sum(jnp.asarray(weights, dtype) * x**2)

    parameters = jnp.asarray([-0.2, 0.0, 0.13], dtype)
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(loss)))(parameters)
    for i, p in enumerate(np.asarray(parameters, dtype=np.float64)):
        a = base + p * delta
        x = np.linalg.solve(a, rhs + p * drhs)
        dx = np.linalg.solve(a, drhs - delta @ x)
        np.testing.assert_allclose(
            [values[i], gradients[i]],
            [weights @ x**2, 2 * (weights * x) @ dx],
            rtol=1e-5 if dtype == np.float32 else 2e-12,
            atol=1e-7,
        )
    bad = jax.jit(triangular_backsolve)(jnp.zeros((2, 2), dtype), jnp.ones(2, dtype))
    assert not np.isfinite(bad).all()  # No singularity-masking regularization.


def test_backsolve_jaxpr_and_reverse_have_no_library_solve(dtype: np.dtype[np.floating]) -> None:
    a, b = jnp.eye(5, dtype=dtype), jnp.ones(5, dtype)
    forward = _jaxpr_audit(jax.make_jaxpr(triangular_backsolve)(a, b))
    reverse = _jaxpr_audit(
        jax.make_jaxpr(jax.grad(lambda a, b: jnp.sum(triangular_backsolve(a, b)), argnums=(0, 1)))(
            a, b
        )
    )
    for audit in (forward, reverse):
        counts = audit["primitive_counts"]
        assert audit["passed"]
        assert counts.get("while", 0) == counts.get("scan", 0) == 0
        assert capture_eligibility(counts)["passed"]
        assert counts.get("triangular_solve", 0) == 0


@pytest.mark.parametrize("size", [1, 8])
def test_backsolve_full_matrix_and_rhs_adjoint(size: int, dtype: np.dtype[np.floating]) -> None:
    """Check all input sensitivities, including zero unused-lower gradients."""
    rng = np.random.default_rng(470 + size)
    matrix = np.triu(rng.normal(0.0, 0.05, (size, size)))
    matrix += np.diag(np.linspace(0.8, 1.7, size))
    rhs, weights = rng.normal(size=(2, size))
    matrix, rhs, weights = (a.astype(dtype) for a in (matrix, rhs, weights))
    solution = solve_triangular(matrix, rhs, lower=False)
    adjoint = solve_triangular(matrix.T, weights, lower=True)
    expected_matrix = np.triu(-np.outer(adjoint, solution))
    # Invalid unused storage must affect neither values nor their derivatives.
    matrix[np.tril_indices(size, -1)] = np.nan
    objective = lambda a, b: jnp.vdot(jnp.asarray(weights), triangular_backsolve(a, b))
    value, (matrix_grad, rhs_grad) = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        jnp.asarray(matrix), jnp.asarray(rhs)
    )
    tolerance = dict(rtol=3e-6, atol=3e-7) if dtype == np.float32 else dict(rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(value, weights @ solution, **tolerance)
    np.testing.assert_allclose(matrix_grad, expected_matrix, **tolerance)
    np.testing.assert_allclose(rhs_grad, adjoint, **tolerance)


@pytest.mark.parametrize("name", PROBES)
def test_capture_probes_match_independent_numpy(name: str, dtype: np.dtype[np.floating]) -> None:
    raw, reference, _ = capture_probe(name, dtype)
    fn = jax.jit(raw)
    for p in (0.23, -0.41, 0.71):
        parameter = jnp.asarray(p, dtype)
        got, expected = fn(parameter), reference(np.asarray(p, dtype))
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
            np.testing.assert_allclose(
                a,
                b,
                rtol=2e-5 if dtype == np.float32 else 5e-10,
                atol=2e-7 if dtype == np.float32 else 5e-12,
            )
        audit = _jaxpr_audit(jax.make_jaxpr(raw)(parameter))
        assert audit["passed"] and capture_eligibility(audit["primitive_counts"])["passed"]


def test_static_screen_detects_original_triangular_primitive_and_custom_call(
    dtype: np.dtype[np.floating],
) -> None:
    from jax.scipy.linalg import solve_triangular as original

    counts = _jaxpr_audit(jax.make_jaxpr(original)(jnp.eye(4, dtype=dtype), jnp.ones(4, dtype)))[
        "primitive_counts"
    ]
    assert not capture_eligibility(counts)["passed"]
    hlo = 'a = custom-call(), custom_call_target="__cublas$triangularSolve"'
    screen = capture_eligibility({}, hlo)
    assert not screen["passed"]
    assert screen["custom_call_targets"]["__cublas$triangularSolve"] == 1
    good = capture_eligibility({}, 'a = custom-call(), custom_call_target="__cublas$gemm"')
    assert good["passed"] and not good["runtime_residency_proven"]


def test_small_fused_gmres_has_no_hidden_triangular_primitive(
    dtype: np.dtype[np.floating],
) -> None:
    raw = lambda b: fused_gmres(
        lambda v: 2 * v + 0.01 * jnp.sum(v), b, rtol=1e-6, restart=6, max_restarts=3
    )[0]
    audit = _jaxpr_audit(jax.make_jaxpr(raw)(jnp.ones(9, dtype)))
    assert capture_eligibility(audit["primitive_counts"])["passed"]
    assert audit["primitive_counts"].get("while", 0) == 2


def test_first_warmup_nonfinite_is_saved_before_repeat(
    tmp_path: Path, dtype: np.dtype[np.floating]
) -> None:
    device = jax.devices()[0]
    program = CompiledProgram(None, device, ResidentPolicy("cpu"), tmp_path, {})
    calls = []

    def invoke(arguments: object) -> jax.Array:
        calls.append(1)
        return jnp.asarray([jnp.nan, jnp.inf, -jnp.inf], dtype)

    program._invoke = invoke  # Isolate evidence logic; not a residency test.
    with pytest.raises(ResidencyError, match="warmup_first"):
        program.certify(jnp.asarray(0.2, dtype))
    assert len(calls) == 1 and not program.certified
    report = json.loads((tmp_path / "certificate.json").read_text())
    assert report["phase"] == "warmup_first" and not report["passed"]
    leaf = json.loads((tmp_path / "warmup_first.outputs.json").read_text())["leaves"][0]
    assert (leaf["nan_count"], leaf["positive_inf_count"], leaf["negative_inf_count"]) == (1, 1, 1)
    assert leaf["nonfinite_flat_indices"] == [0, 1, 2]
    assert (tmp_path / "warmup_first.outputs.npz").is_file()


def test_independent_reference_mismatch_cannot_certify(
    tmp_path: Path, dtype: np.dtype[np.floating]
) -> None:
    program = CompiledProgram(None, jax.devices()[0], ResidentPolicy("cpu"), tmp_path, {})
    program._invoke = lambda args: jnp.asarray(1.0, dtype)
    with pytest.raises(ResidencyError, match="independent reference"):
        program.certify(jnp.asarray(0.1, dtype), expected_host=np.asarray(2.0, dtype))
    assert not program.certified


def test_pending_phase_survives_execution_exception(
    tmp_path: Path, dtype: np.dtype[np.floating]
) -> None:
    program = CompiledProgram(None, jax.devices()[0], ResidentPolicy("cpu"), tmp_path, {})

    def fail(arguments: object) -> None:
        assert json.loads((tmp_path / "progress.json").read_text())["name"] == "warmup_first"
        raise RuntimeError("injected execution failure")

    program._invoke = fail
    with pytest.raises(RuntimeError, match="injected"):
        program.certify(jnp.asarray(0.1, dtype))
    saved = json.loads((tmp_path / "certificate.json").read_text())
    assert saved["phases"][-1]["state"] == "started"
    assert "injected execution failure" in saved["exception"]


@needs_scripts
def test_runner_preflight_selection_precedes_units_and_physics(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from envs.soft_walker.diagnostics.run_resident_tests import planned_stages

    args = SimpleNamespace(
        trace=False,
        preflight_timeout=90.0,
        preflight_only=False,
        skip_unit=False,
        drivers=["mpm2"],
        timeout=300.0,
    )
    names = [x[0] for x in planned_stages(args, tmp_path)]
    assert names == [
        "preflight_control",
        "preflight_backsolve",
        "preflight_implicit_gradient",
        "unit",
        "mpm2",
    ]
    args.preflight_only = True
    assert len(planned_stages(args, tmp_path)) == 3
    args.preflight_only, args.skip_unit = False, True
    assert [x[0] for x in planned_stages(args, tmp_path)][-1] == "mpm2"
    assert "unit" not in [x[0] for x in planned_stages(args, tmp_path)]


@needs_scripts
def test_runner_timeout_is_not_a_numerical_failure(tmp_path: Path) -> None:
    import os
    import sys

    from envs.soft_walker.diagnostics.run_resident_tests import run_child

    row = run_child(
        [sys.executable, "-c", "import time;time.sleep(10)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=tmp_path / "timeout.log",
        timeout=0.05,
    )
    assert row["timed_out"] and not row["completed"] and row["returncode"] != 0
    row = run_child(
        [sys.executable, "-c", "raise SystemExit(3)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=tmp_path / "failure.log",
        timeout=2.0,
    )
    assert not row["timed_out"] and row["completed"] and row["returncode"] == 3
