"""Focused numerical and structural tests for the loop-free Arnoldi internals.

These fixtures are independent of robot calibration. Full physical validation
is performed separately by diagnostics/audit_krylov_physics.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.numerics.solvers import RootInfo


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.numerics.krylov import apply_rotations, fused_gmres
from envs.soft_walker.engine.numerics.solvers import NewtonConfig, linear_solve, solve_root


def cfg(**kw: object) -> NewtonConfig:
    return NewtonConfig(gmres_method="fused", **kw)


def assert_close(
    a: ArrayLike,
    b: ArrayLike,
    dtype: np.dtype[np.floating] | type[np.floating],
    *,
    rtol32: float = 2e-4,
    rtol64: float = 4e-9,
) -> None:
    rtol = rtol32 if dtype == np.dtype("float32") else rtol64
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * 1e-2)


@pytest.mark.parametrize("size", [1, 2, 3, 17, 40])
def test_fused_rotation_prefix_matches_independent_serial(
    size: int, dtype: np.dtype[np.floating]
) -> None:
    rng = np.random.default_rng(51 + size)
    h = rng.normal(size=size + 1).astype(dtype)
    angle = rng.normal(size=size).astype(dtype)
    c, s = np.cos(angle), np.sin(angle)
    # Unused future rotations must act exactly as identity rotations.
    c[size // 2 :] = 1
    s[size // 2 :] = 0
    expected = h.copy()
    for i in range(size):
        a, b = expected[i : i + 2].copy()
        expected[i] = c[i] * a - s[i] * b
        expected[i + 1] = s[i] * a + c[i] * b
    got = jax.jit(apply_rotations)(jnp.asarray(h), jnp.asarray(c), jnp.asarray(s))
    np.testing.assert_allclose(
        got, expected, atol=8 * np.finfo(dtype).eps, rtol=8 * np.finfo(dtype).eps
    )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("n", [1, 7, 33])
def test_fused_nonsymmetric_linear_solve_and_transpose(
    side: str, n: int, dtype: np.dtype[np.floating]
) -> None:
    rng = np.random.default_rng(n)
    dense = np.diag(np.linspace(1, 4, n)) + 0.04 * rng.normal(size=(n, n))
    b = rng.normal(size=n)
    a, rhs = jnp.asarray(dense, dtype), jnp.asarray(b, dtype)
    config = cfg(preconditioner_side=side, restart=min(n, 20))

    def fn(rhs: jax.Array) -> tuple[jax.Array, jax.Array]:
        return linear_solve(lambda x: a @ x, rhs, jnp.diag(a), config)

    x, info = jax.jit(fn)(rhs)
    assert bool(info[1]) and np.isfinite(x).all()
    assert_close(x, np.linalg.solve(dense, b), dtype)
    target = jnp.cos(jnp.arange(n, dtype=dtype))
    _, pullback = jax.vjp(lambda b: fn(b)[0], rhs)
    (adjoint,) = pullback(target)
    assert_close(adjoint, np.linalg.solve(dense.T, np.asarray(target)), dtype)
    direction = jnp.sin(jnp.arange(n, dtype=dtype) + 0.2)
    _, tangent = jax.jvp(lambda b: fn(b)[0], (rhs,), (direction,))
    assert_close(jnp.vdot(target, tangent), jnp.vdot(adjoint, direction), dtype)


def test_fused_zero_rhs_happy_breakdown_and_one_unknown(dtype: np.dtype[np.floating]) -> None:
    for n in (1, 16):
        for scale in (0.0, 1.0):
            b = jnp.full(n, scale, dtype)
            x, info = jax.jit(
                lambda b: fused_gmres(
                    lambda v: 3 * v,
                    b,
                    rtol=1e-10 if dtype == np.float64 else 1e-5,
                    restart=20,
                    max_restarts=3,
                )
            )(b)
            assert_close(x, np.asarray(b) / 3, dtype)
            assert bool(info.finite)
            assert int(info.iterations) == (0 if scale == 0 else 1)
            assert int(info.restarts) == (0 if scale == 0 else 1)


def test_fused_failure_is_visible_not_dense_or_legacy_fallback(
    dtype: np.dtype[np.floating],
) -> None:
    b = jnp.arange(1, 10, dtype=dtype)
    # No solution for a nonzero RHS. The caller's unchanged audit must fail.
    x, info = jax.jit(
        lambda b: linear_solve(lambda v: 0 * v, b, jnp.ones_like(b), cfg(max_restarts=1))
    )(b)
    assert not bool(info[1])
    assert np.isnan(x).all()
    bad = b.at[2].set(jnp.nan)
    x, info = linear_solve(lambda v: v, bad, jnp.ones_like(b), cfg())
    assert not bool(info[1]) and np.isnan(x).all()


def test_fused_restart_limit_and_true_residual(dtype: np.dtype[np.floating]) -> None:
    n = 25
    d = jnp.linspace(0.1, 8, n, dtype=dtype)
    b = jnp.ones(n, dtype)
    x, info = linear_solve(
        lambda v: d * v + 0.2 * jnp.roll(v, 1), b, jnp.ones_like(b), cfg(restart=1, max_restarts=1)
    )
    assert not bool(info[1]) and np.isnan(x).all()


def test_fused_operator_parameter_gradient_has_independent_reference(
    dtype: np.dtype[np.floating],
) -> None:
    n = 13
    diag = np.linspace(1.0, 2.3, n)
    skew = np.diag(np.full(n - 1, 0.2), 1) - np.diag(np.full(n - 1, 0.13), -1)
    b0 = np.linspace(0.2, 1.1, n)
    theta = jnp.asarray([0.15, 0.4, -0.1], dtype)
    dj, sj, bj = map(lambda a: jnp.asarray(a, dtype), (diag, skew, b0))

    def loss(t: jax.Array) -> jax.Array:
        diagonal = jnp.exp(t[0]) * dj
        x, _ = linear_solve(
            lambda v: diagonal * v + t[1] * (sj @ v),
            (1 + t[2]) * bj,
            diagonal,
            cfg(preconditioner_side="right"),
        )
        return 0.5 * jnp.vdot(x, x)

    value, gradient = jax.jit(jax.value_and_grad(loss))(theta)
    t = np.asarray(theta, np.float64)
    a = np.diag(np.exp(t[0]) * diag) + t[1] * skew
    x = np.linalg.solve(a, (1 + t[2]) * b0)
    lam = np.linalg.solve(a.T, x)
    expected = np.array([-lam @ (np.exp(t[0]) * diag * x), -lam @ (skew @ x), lam @ b0])
    assert_close(value, 0.5 * x @ x, dtype)
    assert_close(gradient, expected, dtype)


def test_fused_root_implicit_gradient(dtype: np.dtype[np.floating]) -> None:
    b = jnp.linspace(0.3, 1.2, 9, dtype=dtype)
    theta = jnp.asarray(0.6, dtype)

    def scalar(t: jax.Array) -> tuple[jax.Array, RootInfo]:
        root, info = solve_root(lambda z: z + t * z**3 - b, b, cfg())
        return jnp.sum(root), info

    (value, info), grad = jax.jit(jax.value_and_grad(scalar, has_aux=True))(theta)
    assert bool(info.accepted)
    # Independent scalar cubic roots, not another JAX solve.
    from scipy.optimize import brentq

    roots = np.array(
        [
            brentq(lambda z: z + float(theta) * z**3 - float(v), 0, 2, xtol=1e-14)
            for v in np.asarray(b)
        ]
    )
    expected = np.sum(-(roots**3) / (1 + 3 * float(theta) * roots**2))
    assert_close(value, roots.sum(), dtype)
    assert_close(grad, expected, dtype)


def test_fused_independent_population_vmap(dtype: np.dtype[np.floating]) -> None:
    diagonal = jnp.linspace(1, 2, 17, dtype=dtype)
    controls = jnp.asarray([0.0, 0.3, -0.4, 1.2], dtype)

    def objective(t: jax.Array) -> jax.Array:
        x, _ = linear_solve(
            lambda v: diagonal * v + 0.1 * jnp.roll(v, 1),
            t * jnp.ones_like(diagonal),
            diagonal,
            cfg(),
        )
        return jnp.vdot(x, x)

    batched = jax.jit(jax.vmap(jax.value_and_grad(objective)))(controls)
    separate = [jax.jit(jax.value_and_grad(objective))(v) for v in controls]
    assert_close(batched[0], jnp.stack([a[0] for a in separate]), dtype)
    assert_close(batched[1], jnp.stack([a[1] for a in separate]), dtype)
    assert np.isfinite(batched[1]).all()


def test_fused_raw_has_only_restart_and_arnoldi_loops(dtype: np.dtype[np.floating]) -> None:
    b = jnp.ones(257, dtype)
    lowered = jax.jit(
        lambda b: fused_gmres(
            lambda v: 1.3 * v + 0.1 * jnp.roll(v, 1),
            b,
            rtol=1e-8,
            restart=8,
            max_restarts=3,
        )
    ).lower(b)
    mlir = str(lowered.compiler_ir(dialect="stablehlo"))
    assert mlir.count("stablehlo.while") == 2
    assert "257x257" not in mlir
    assert "callback" not in mlir.lower()


def test_fused_configuration_is_explicit() -> None:
    from envs.soft_walker.presets.locomotion import DemoConfig

    assert NewtonConfig().gmres_method == "incremental"
    assert DemoConfig(gmres_method="fused").gmres_method == "fused"
    with pytest.raises(ValueError):
        NewtonConfig(gmres_method="automatic_hardware_fallback")


def test_fused_reverse_program_does_not_construct_a_dense_tangent(
    dtype: np.dtype[np.floating],
) -> None:
    diagonal = jnp.linspace(1, 2, 257, dtype=dtype)

    def loss(t: jax.Array) -> jax.Array:
        x, _ = linear_solve(
            lambda v: (diagonal + t) * v + 0.1 * jnp.roll(v, 1),
            jnp.ones_like(diagonal),
            diagonal + t,
            cfg(restart=8, max_restarts=5),
        )
        return jnp.mean(x * x)

    value = jnp.asarray(0.2, dtype)
    lowered = jax.jit(jax.value_and_grad(loss)).lower(value)
    ir = str(lowered.compiler_ir(dialect="stablehlo"))
    assert "257x257" not in ir
    assert "callback" not in ir.lower()
    result = lowered.compile()(value)
    assert np.isfinite(np.asarray(result)).all()
