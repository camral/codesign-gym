"""Nonlinear roots, adaptive integration, and implicit differentiation contracts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as j
import numpy as np
import pytest

from envs.soft_walker.engine.numerics.integrators import A, C, kvaerno3_step
from envs.soft_walker.engine.numerics.solvers import (
    NewtonConfig,
    adjoint_diagnostics,
    linear_solve,
    solve_root,
)
from envs.soft_walker.engine.physics.common import DynamicsConfig, StepInfo, advance_retry


def test_m2_root_nonsymmetric_gradient_and_jvp(dtype: np.dtype[np.floating]) -> None:
    n = 17
    diag = j.linspace(1.5, 2.5, n, dtype=dtype)
    op = lambda y: diag * y + 0.17 * j.roll(y, 1) - 0.08 * j.roll(y, -1)

    def fun(p: jax.Array) -> jax.Array | tuple[jax.Array, StepInfo]:
        return solve_root(lambda y: op(y) + 0.2 * y**3 - p, j.zeros(n, dtype), diagonal=diag)[0]

    p = j.linspace(0.1, 0.5, n, dtype=dtype)
    y = jax.jit(fun)(p)
    M = (
        np.diag(np.asarray(diag))
        + np.roll(np.eye(n), -1, axis=1) * 0.17
        - np.roll(np.eye(n), 1, axis=1) * 0.08
    )
    jac = M + np.diag(0.6 * np.asarray(y) ** 2)
    dr = j.linspace(0.2, 1.0, n, dtype=dtype)
    jvp = jax.jvp(fun, (p,), (dr,))[1]
    vjp = jax.grad(lambda p: j.vdot(fun(p), dr))(p)
    tol = 1e-4 if dtype == np.dtype("float32") else 1e-8
    np.testing.assert_allclose(jvp, np.linalg.solve(jac, np.asarray(dr)), rtol=tol, atol=tol)
    np.testing.assert_allclose(vjp, np.linalg.solve(jac.T, np.asarray(dr)), rtol=tol, atol=tol)
    lam, info = adjoint_diagnostics(lambda y: op(y) + 0.2 * y**3 - p, y, dr, diagonal=diag)
    assert info[1] == 1
    np.testing.assert_allclose(lam, vjp, rtol=tol, atol=tol)


def test_m2_failure_not_silent(dtype: np.dtype[np.floating]) -> None:
    b = j.ones(8, dtype)
    out, info = linear_solve(lambda x: j.zeros_like(x), b, b)
    assert info[1] == 0
    assert np.isnan(out).all()
    y, info = solve_root(lambda x: j.ones_like(x), b)
    assert info.accepted == 0 and info.code == 3


def test_m2_padding_infinity_norm(dtype: np.dtype[np.floating]) -> None:
    cfg = NewtonConfig()
    f = lambda n: solve_root(
        lambda y: y + y**3 - j.concatenate((j.ones(3, dtype) * 0.2, j.zeros(n - 3, dtype))),
        j.zeros(n, dtype),
        cfg,
    )
    a, ia = jax.jit(lambda: f(3))()
    b, ib = jax.jit(lambda: f(37))()
    np.testing.assert_allclose(a, b[:3], rtol=1e-5, atol=1e-7)
    assert ia.accepted == ib.accepted == 1
    assert ia.iterations == ib.iterations


def test_m2_invalid_initial_and_backtracking(dtype: np.dtype[np.floating]) -> None:
    x, info = solve_root(
        lambda x: j.log(x) - 0.1, j.array([-1.0], dtype), admissible=lambda x: j.all(x > 0)
    )
    assert info.code == 4 and info.accepted == 0
    x, info = solve_root(
        lambda x: j.log(x) - 0.1, j.array([10.0], dtype), admissible=lambda x: j.all(x > 0)
    )
    assert info.accepted == 1 and info.backtracks > 0
    np.testing.assert_allclose(x, np.exp(0.1), rtol=2e-5)


def test_m2_tableau_is_exact_diffrax() -> None:
    diffrax = pytest.importorskip("diffrax")  # optional: not a codesign-gym dependency
    tab = diffrax.Kvaerno3.tableau
    np.testing.assert_allclose(np.array(A)[-1], tab.b_sol, rtol=0, atol=3e-16)
    for i in range(1, 4):
        np.testing.assert_allclose(np.array(A)[i, :i], tab.a_lower[i - 1], rtol=0, atol=3e-16)
    np.testing.assert_allclose(C[1:], tab.c, rtol=0, atol=1e-15)


def test_m2_native_kvaerno_matches_diffrax_and_gradient(dtype: np.dtype[np.floating]) -> None:
    dfx = pytest.importorskip("diffrax")
    ox = pytest.importorskip("optimistix")  # optional: not a codesign-gym dependency
    tol = 2e-11 if dtype == np.dtype("float64") else 2e-6
    cfg = NewtonConfig(atol64=tol, atol32=tol)
    y = j.array([0.4, -0.15], dtype)

    def rhs(t: jax.Array, y: jax.Array, p: jax.Array) -> jax.Array:
        return j.stack((y[1], -p * y[0] - 1.3 * y[1] + 0.2 * j.cos(t)))

    def native(p: jax.Array) -> jax.Array:
        return kvaerno3_step(rhs, j.array(0.1, dtype), y, j.array(0.01, dtype), p, cfg)[0]

    def ref(p: jax.Array) -> jax.Array:
        r = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            dfx.Kvaerno3(root_finder=ox.Newton(rtol=tol, atol=tol)),
            t0=j.array(0.1, dtype),
            t1=j.array(0.11, dtype),
            dt0=j.array(0.01, dtype),
            y0=y,
            args=p,
            max_steps=2,
        )
        return r.ys[0]

    p = j.array(30.0, dtype)
    eps = 1e-4 if dtype == np.dtype("float32") else 2e-9
    np.testing.assert_allclose(jax.jit(native)(p), jax.jit(ref)(p), rtol=eps, atol=eps)
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(native))(p), jax.jit(jax.jacrev(ref))(p), rtol=eps, atol=eps
    )


def test_m2_retry_transaction_and_replay(dtype: np.dtype[np.floating]) -> None:
    class S(NamedTuple):
        v: jax.Array
        time: jax.Array
        status: jax.Array

    s = S(j.zeros(1, dtype), j.array(0.0, dtype), j.array(0, j.int32))

    def trial(s: S, a: jax.Array, h: jax.Array) -> tuple[S, StepInfo]:
        good = h <= 0.26
        z = j.array(0.0, dtype)
        one = j.array(1.0, dtype)
        out = S(s.v + h * a, s.time + h, j.where(good, 0, 1).astype(j.int32))
        info = StepInfo(
            good.astype(dtype),
            (~good).astype(dtype),
            one,
            one,
            z,
            z,
            z,
            one,
            one,
            z,
            j.where(good, z, one),
        )
        return out, info

    cfg = DynamicsConfig(max_halvings=2)
    out, info, trace = jax.jit(lambda h: advance_retry(trial, s, j.ones(1, dtype), h, cfg))(
        j.array(1.0, dtype)
    )
    np.testing.assert_allclose(out.time, 1.0)
    np.testing.assert_allclose(trace, j.ones(4) * 0.25)
    assert info.rejected == 3 and info.accepted == 4 and out.status == 0
    out, info, trace = jax.jit(lambda h: advance_retry(trial, s, j.ones(1, dtype), h, cfg))(
        j.array(2.0, dtype)
    )
    assert out.status != 0 and out.time == 0 and np.all(np.asarray(trace) == 0)


def test_m2_native_stage_failure_is_not_a_valid_result(dtype: np.dtype[np.floating]) -> None:
    cfg = NewtonConfig(max_steps=1, atol64=1e-14, atol32=1e-8)

    def fun(p: jax.Array) -> jax.Array | tuple[jax.Array, StepInfo]:
        out, info = kvaerno3_step(
            lambda t, y, p: -p * y**3,
            j.array(0.0, dtype),
            j.ones(3, dtype),
            j.array(1.0, dtype),
            p,
            cfg,
        )
        return out, info

    out, info = jax.jit(fun)(j.array(10.0, dtype))
    assert not bool(j.all(info.accepted > 0))
    assert not np.isfinite(out).all()


def test_m2_implicit_gradient_matches_unrolled_small_reference(
    dtype: np.dtype[np.floating],
) -> None:
    """Dense 3x3 Newton is an independent TEST ONLY trajectory reference."""
    matrix = j.asarray([[2.0, 0.15, -0.04], [-0.08, 1.6, 0.1], [0.03, -0.1, 2.2]], dtype)
    p = j.asarray([0.3, -0.2, 0.4], dtype)
    weight = j.asarray([0.4, 0.7, -0.2], dtype)

    def residual(x: jax.Array, p: jax.Array) -> jax.Array:
        return matrix @ x + 0.12 * x**3 - p

    def implicit(p: jax.Array) -> jax.Array:
        return solve_root(lambda x: residual(x, p), j.zeros(3, dtype), diagonal=j.diag(matrix))[0]

    def unrolled(p: jax.Array) -> jax.Array:
        def body(_: int, x: jax.Array) -> jax.Array:
            # Fixed Newton iteration count means this reference differentiates
            # the iterations, unlike the production implicit-function rule.
            tangent = matrix + j.diag(0.36 * x * x)
            return x - j.linalg.solve(tangent, residual(x, p))

        return jax.lax.fori_loop(0, 8, body, j.zeros(3, dtype))

    a, ga = jax.jit(jax.value_and_grad(lambda p: j.vdot(implicit(p), weight)))(p)
    b, gb = jax.jit(jax.value_and_grad(lambda p: j.vdot(unrolled(p), weight)))(p)
    tol = 5e-5 if dtype == np.dtype("float32") else 2e-9
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)
    np.testing.assert_allclose(ga, gb, rtol=tol, atol=tol)


def test_m2_configuration_rejects_nonfinite() -> None:
    from envs.soft_walker.engine.physics.contact import PlaneContact

    with pytest.raises(ValueError):
        NewtonConfig(atol32=float("nan"))
    with pytest.raises(ValueError):
        PlaneContact(width=float("inf"))
