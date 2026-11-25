"""Constitutive energy derivatives and admissible physical material ranges."""

from __future__ import annotations

import jax
import jax.numpy as j
import numpy as np
import pytest

from envs.soft_walker.engine.physics import material as m


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_m2_constitutive_independent_numpy(
    dim: int, kind: int, dtype: np.dtype[np.floating]
) -> None:
    rng = np.random.default_rng(12)
    F = np.eye(dim) + rng.normal(size=(dim, dim)) * 0.12
    n = rng.normal(size=dim)
    n /= np.linalg.norm(n)
    E = 23000.0
    nu = 0.36
    a = 0.11
    Q = np.outer(n, n) if kind == 1 else np.eye(dim) / dim if kind == 2 else np.zeros((dim, dim))
    import scipy.linalg

    A = scipy.linalg.expm(-a * Q)
    Fe = F @ A
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    lj = np.log(np.linalg.det(Fe))
    energy = 0.5 * mu * (np.sum(Fe * Fe) - dim) - mu * lj + 0.5 * lam * lj * lj
    P = (mu * Fe + (lam * lj - mu) * np.linalg.inv(Fe).T) @ A.T
    args = (
        j.array(E, dtype),
        j.array(nu, dtype),
        j.array(a, dtype),
        j.array(n, dtype),
        j.array(kind),
    )
    F = j.array(F, dtype)
    eps = 2e-5 if dtype == np.dtype("float32") else 1e-11
    np.testing.assert_allclose(m.energy_density(F, *args), energy, rtol=eps, atol=eps)
    np.testing.assert_allclose(m.stress(F, *args), P, rtol=eps, atol=eps * E)
    np.testing.assert_allclose(jax.grad(m.energy_density)(F, *args), P, rtol=eps, atol=eps * E)
    H = j.array(rng.normal(size=(dim, dim)), dtype)
    analytic = m.tangent_action(F, H, *args)
    auto = jax.jvp(lambda F: m.stress(F, *args), (F,), (H,))[1]
    np.testing.assert_allclose(analytic, auto, rtol=eps, atol=eps * E)
    np.testing.assert_allclose(
        m.inverse_transpose(F), np.linalg.inv(np.asarray(F)).T, rtol=eps, atol=eps
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_m2_energy_objectivity_and_zero(dim: int, dtype: np.dtype[np.floating]) -> None:
    F = j.eye(dim, dtype=dtype)
    n = F[0]
    args = (j.array(2e4, dtype), j.array(0.35, dtype), j.array(0.0, dtype), n, j.array(1))
    assert float(m.energy_density(F, *args)) == 0.0
    np.testing.assert_array_equal(m.stress(F, *args), j.zeros_like(F))
    theta = 0.43
    R = np.eye(dim)
    R[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    R = j.array(R, dtype)
    F = F.at[0, 1].set(0.2)
    tol = 5e-5 if dtype == np.dtype("float32") else 1e-11
    np.testing.assert_allclose(m.energy_density(R @ F, *args), m.energy_density(F, *args), rtol=tol)
    np.testing.assert_allclose(
        m.stress(R @ F, *args), R @ m.stress(F, *args), rtol=tol, atol=tol * 2e4
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_m2_work_exact_zero_and_difference(
    dim: int, kind: int, dtype: np.dtype[np.floating]
) -> None:
    F = j.eye(dim, dtype=dtype).at[0, 0].set(1.13)
    n = j.eye(dim, dtype=dtype)[0]
    E = j.array(2e4, dtype)
    nu = j.array(0.35, dtype)
    k = j.array(kind)
    old = j.array(0.12, dtype)
    new = j.array(0.16, dtype)
    assert float(m.work_density(F, E, nu, old, old, n, k)) == 0.0
    ref = m.energy_density(F, E, nu, new, n, k) - m.energy_density(F, E, nu, old, n, k)
    tol = 2e-3 if dtype == np.dtype("float32") else 1e-10
    np.testing.assert_allclose(m.work_density(F, E, nu, old, new, n, k), ref, rtol=tol, atol=tol)


def test_m2_plane_strain_matches_3d(dtype: np.dtype[np.floating]) -> None:
    F = j.array([[1.13, 0.11], [-0.07, 0.95]], dtype)
    F3 = j.eye(3, dtype=dtype).at[:2, :2].set(F)
    E = j.array(20000.0, dtype)
    nu = j.array(0.35, dtype)
    a = j.array(0.1, dtype)
    # Fiber direction stays in plane; isotropic 2D deliberately uses trace Q=1
    # in its in-plane block, not unconstrained 3D isotropic actuation.
    args = (E, nu, a, j.array([1.0, 0.0], dtype), j.array(1))
    args3 = (E, nu, a, j.array([1.0, 0.0, 0.0], dtype), j.array(1))
    np.testing.assert_allclose(m.energy_density(F, *args), m.energy_density(F3, *args3), rtol=2e-6)
    np.testing.assert_allclose(m.stress(F, *args), m.stress(F3, *args3)[:2, :2], rtol=2e-6)


def test_m2_no_inversion_clamp(dtype: np.dtype[np.floating]) -> None:
    F = j.diag(j.array([-1.0, 1.0], dtype))
    assert not np.isfinite(
        m.energy_density(
            F,
            j.array(2e4, dtype),
            j.array(0.35, dtype),
            j.array(0.0, dtype),
            j.array([1.0, 0.0], dtype),
            j.array(1),
        )
    )


def test_m2_small_strain_delta_accuracy_and_zero_direction(dtype: np.dtype[np.floating]) -> None:
    # The delta API intentionally preserves strains too small to survive I+D
    # in float32. Independent linear elasticity is the first-order reference.
    D = j.diag(j.array([2e-8, -1e-8], dtype))
    E = j.array(2e4, dtype)
    nu = j.array(0.35, dtype)
    n = j.array([1.0, 0.0], dtype)
    kind = j.array(1)
    z = j.array(0.0, dtype)
    mu = 20000 / (2 * 1.35)
    lam = 20000 * 0.35 / (1.35 * 0.3)
    ref = 2 * mu * np.asarray(D) + lam * np.trace(np.asarray(D)) * np.eye(2)
    got = m.stress_from_delta(D, E, nu, z, n, kind)
    np.testing.assert_allclose(got, ref, rtol=4e-7, atol=1e-10)
    np.testing.assert_allclose(
        jax.grad(m.energy_from_delta)(D, E, nu, z, n, kind), got, rtol=4e-7, atol=1e-10
    )
    deriv = jax.jacrev(lambda n: m.projector(n, j.array(1)))(j.zeros(2, dtype))
    assert np.isfinite(deriv).all()
    np.testing.assert_array_equal(deriv, 0.0)
