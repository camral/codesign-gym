"""Reuse only independently evaluated Krylov residuals without weakening acceptance."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.numerics.krylov import fused_gmres_with_residual
from envs.soft_walker.engine.numerics.solvers import NewtonConfig, linear_solve


@pytest.mark.parametrize("zero_rhs", [False, True])
def test_retained_residual_is_physical_not_preconditioned(
    zero_rhs: bool, dtype: np.dtype[np.floating]
) -> None:
    """The retained vector equals b-Ax even when the preconditioner changes its norm."""
    matrix = jnp.asarray([[2.0, 0.3], [-0.1, 3.0]], dtype)
    rhs = jnp.zeros(2, dtype) if zero_rhs else jnp.asarray([0.2, -0.1], dtype)
    scale = jnp.asarray([0.1, 4.0], dtype)
    value, info, residual = jax.jit(
        lambda: fused_gmres_with_residual(
            lambda x: matrix @ x,
            rhs,
            rtol=2e-5 if dtype.name == "float32" else 2e-10,
            restart=2,
            max_restarts=3,
            precondition=lambda x: x * scale,
        )
    )()
    assert bool(info.finite)
    tolerance = 3e-5 if dtype.name == "float32" else 2e-10
    np.testing.assert_allclose(
        residual, rhs - matrix @ value, rtol=tolerance, atol=tolerance * 1e-2
    )
    np.testing.assert_allclose(
        info.residual_norm, jnp.linalg.norm(scale * residual), atol=tolerance
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_residual_reuse_preserves_solution_and_implicit_derivative(
    side: str, dtype: np.dtype[np.floating]
) -> None:
    """The optimization changes operator work, not the accepted primal or adjoint equation."""
    matrix = jnp.asarray([[4.0, 0.3, 0], [-0.2, 2.0, 0.1], [0.2, 0.4, 3.0]], dtype)
    rhs = jnp.asarray([0.1, -0.2, 0.3], dtype)
    config = NewtonConfig(gmres_method="fused", preconditioner_side=side)

    def solve(value: jax.Array, reuse: bool) -> tuple[jax.Array, jax.Array]:
        """Use identical preconditioners and residual thresholds in both work modes."""
        return linear_solve(
            lambda x: matrix @ x,
            value,
            jnp.diag(matrix),
            replace(config, linear_work_reuse=reuse),
        )

    expected, reference_info = jax.jit(lambda x: solve(x, False))(rhs)
    actual, info = jax.jit(lambda x: solve(x, True))(rhs)
    assert float(info[1]) == float(reference_info[1]) == 1.0
    reference_gradient = jax.jit(jax.grad(lambda x: jnp.sum(solve(x, False)[0] ** 2)))(rhs)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(solve(x, True)[0] ** 2)))(rhs)
    tolerance = 3e-5 if dtype.name == "float32" else 2e-9
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * 1e-3)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=tolerance, atol=tolerance * 1e-3)


def test_refinement_audits_the_updated_solution(dtype: np.dtype[np.floating]) -> None:
    """A deliberately short Krylov solve needs refinement; reported error follows its new x."""
    matrix = jnp.asarray([[1.0, 0.0], [0.0, 1.1]], dtype)
    rhs = jnp.ones(2, dtype)
    # This tiny linear fixture intentionally limits Krylov work so one explicit
    # refinement is required. These settings do not alter any physics defaults.
    config = NewtonConfig(
        gmres_method="fused",
        restart=1,
        max_restarts=1,
        linear_rtol64=1e-3,
        linear_rtol32=1e-3,
        linear_refinements=1,
        linear_work_reuse=True,
    )
    _, before = jax.jit(
        lambda: linear_solve(
            lambda x: matrix @ x, rhs, jnp.ones(2, dtype), replace(config, linear_refinements=0)
        )
    )()
    value, after = jax.jit(
        lambda: linear_solve(lambda x: matrix @ x, rhs, jnp.ones(2, dtype), config)
    )()
    assert float(before[1]) == 0.0 and float(after[1]) == 1.0
    actual_error = jnp.linalg.norm(matrix @ value - rhs) / jnp.linalg.norm(rhs)
    tolerance = 3e-5 if dtype.name == "float32" else 2e-9
    np.testing.assert_allclose(after[0], actual_error, rtol=tolerance, atol=tolerance * 1e-3)
    assert float(after[0]) < float(before[0])


def test_work_reuse_rejects_unimplemented_krylov_modes() -> None:
    """A requested work optimization cannot silently disappear on another solver path."""
    with pytest.raises(ValueError, match="requires gmres_method='fused'"):
        NewtonConfig(linear_work_reuse=True)
