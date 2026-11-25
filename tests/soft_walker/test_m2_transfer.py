"""MPM interpolation partition, transfer conservation, and round trips."""

from __future__ import annotations

from itertools import product

import jax
import jax.numpy as j
import numpy as np
import pytest

from envs.soft_walker.engine.physics import contact, mpm
from envs.soft_walker.engine.physics.common import surface_exposure


@pytest.mark.parametrize("dim", [2, 3])
def test_m2_quadratic_identities_and_affine(dim: int, dtype: np.dtype[np.floating]) -> None:
    grid = mpm.Grid(shape=(8,) * dim, spacing=0.1, origin=(0.0,) * dim)
    x = j.array(np.random.default_rng(9).uniform(0.16, 0.48, size=(7, dim)), dtype)
    st = mpm.stencil(x, grid)
    eps = 3e-5 if dtype == np.dtype("float32") else 2e-12
    assert np.all(st.valid)
    np.testing.assert_allclose(st.weight.sum(-1), 1.0, atol=eps)
    np.testing.assert_allclose(st.gradient.sum(1), 0.0, atol=eps)
    np.testing.assert_allclose(j.sum(st.weight[..., None] * st.offset, 1), 0.0, atol=eps)
    D = j.einsum("pk,pki,pkj->pij", st.weight, st.offset, st.offset)
    np.testing.assert_allclose(
        D, j.broadcast_to(j.eye(dim) * grid.spacing**2 / 4, D.shape), atol=eps * grid.spacing**2
    )
    moment = j.einsum("pki,pkj->pij", st.offset, st.gradient)
    np.testing.assert_allclose(moment, j.broadcast_to(j.eye(dim), moment.shape), atol=eps)
    gx = j.array(list(product(*[np.arange(8) * 0.1] * dim)), dtype)
    A = j.array(np.random.default_rng(4).normal(size=(dim, dim)) * 0.12, dtype)
    b = j.arange(dim, dtype=dtype) * 0.2 + 0.1
    gv = gx @ A.T + b
    v, C, L = mpm.g2p(gv, st, grid)
    np.testing.assert_allclose(v, x @ A.T + b, rtol=eps, atol=eps)
    np.testing.assert_allclose(C, j.broadcast_to(A, C.shape), rtol=eps, atol=eps)
    np.testing.assert_allclose(L, j.broadcast_to(A, L.shape), rtol=eps, atol=eps)
    # Independent slow NumPy scatter, including affine velocity contributions.
    masses = np.linspace(0.5, 1.1, len(x)).astype(dtype)
    Cs = np.broadcast_to(np.asarray(A), (len(x), dim, dim))
    gm, vg = mpm.p2g(j.array(masses), v, j.array(Cs), st, grid)
    nm = np.zeros(grid.count)
    mom = np.zeros((grid.count, dim))
    for p in range(len(x)):
        for k in range(st.ids.shape[1]):
            idx = int(st.ids[p, k])
            w = float(st.weight[p, k])
            nm[idx] += masses[p] * w
            mom[idx] += masses[p] * w * (np.asarray(v[p]) + Cs[p] @ np.asarray(st.offset[p, k]))
    np.testing.assert_allclose(gm, nm, rtol=eps, atol=eps)
    np.testing.assert_allclose(gm[:, None] * vg, mom, rtol=eps, atol=eps)
    np.testing.assert_allclose(
        j.sum(gm[:, None] * vg, 0), j.sum(j.array(masses)[:, None] * v, 0), rtol=eps, atol=eps
    )


def test_m2_apic_angular_momentum_and_energy(dtype: np.dtype[np.floating]) -> None:
    grid = mpm.Grid(shape=(8, 8), spacing=0.1, origin=(0.0, 0.0))
    x = j.array(np.random.default_rng(11).uniform(0.15, 0.45, (12, 2)), dtype)
    v = j.array(np.random.default_rng(22).normal(size=(12, 2)), dtype)
    C = j.array(np.random.default_rng(33).normal(size=(12, 2, 2)), dtype)
    mass = j.linspace(0.7, 1.3, 12, dtype=dtype)
    st = mpm.stencil(x, grid)
    gm, gv = mpm.p2g(mass, v, C, st, grid)
    D = grid.spacing**2 / 4
    gx = j.array(list(product(np.arange(8) * 0.1, repeat=2)), dtype)
    lp = j.sum(mass * (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0] + D * (C[:, 1, 0] - C[:, 0, 1])))
    lg = j.sum(gm * (gx[:, 0] * gv[:, 1] - gx[:, 1] * gv[:, 0]))
    np.testing.assert_allclose(lp, lg, rtol=2e-5, atol=2e-6)
    vp, Cp, _ = mpm.g2p(gv, st, grid)
    lp2 = j.sum(mass * (x[:, 0] * vp[:, 1] - x[:, 1] * vp[:, 0] + D * (Cp[:, 1, 0] - Cp[:, 0, 1])))
    np.testing.assert_allclose(lp2, lg, rtol=2e-5, atol=2e-6)
    E0 = 0.5 * j.sum(mass * (j.sum(v * v, -1) + D * j.sum(C * C, (-2, -1))))
    Eg = 0.5 * j.sum(gm[:, None] * gv * gv)
    Ep = 0.5 * j.sum(mass * (j.sum(vp * vp, -1) + D * j.sum(Cp * Cp, (-2, -1))))
    assert Eg <= E0 + 1e-5 and Ep <= Eg + 1e-5


@pytest.mark.parametrize("dim", [2, 3])
def test_m2_uniform_cloud_volume_surface(dim: int, dtype: np.dtype[np.floating]) -> None:
    cloud = mpm.uniform_cloud((2,) * dim, spacing=0.04, origin=(0.0,) * dim, dtype=dtype)
    factor = 0.01 if dim == 2 else 1.0
    np.testing.assert_allclose(cloud.volume.sum(), 0.08**dim * factor, rtol=1e-6)
    alpha = j.ones(cloud.reference.shape[0], dtype)
    exposure = surface_exposure(alpha, cloud.neighbors, cloud.has_neighbor)
    np.testing.assert_allclose(
        j.sum(exposure * cloud.face_area), 2 * dim * 0.08 ** (dim - 1) * factor, rtol=1e-6
    )
    np.testing.assert_allclose(
        cloud.volume, j.full_like(cloud.volume, cloud.volume[0]), rtol=0, atol=0
    )
    gray = surface_exposure(alpha * 0.5, cloud.neighbors, cloud.has_neighbor)
    np.testing.assert_allclose(
        j.sum(gray * cloud.face_area), dim * 0.08 ** (dim - 1) * factor, rtol=1e-6
    )
    interior = j.all(cloud.has_neighbor, axis=-1)
    assert np.all(np.asarray(gray)[np.asarray(interior)] == 0)


def test_m2_support_exit_rejected_by_geometry(dtype: np.dtype[np.floating]) -> None:
    grid = mpm.Grid(shape=(6, 6), spacing=0.1, origin=(0.0, 0.0))
    x = j.array([[0.01, 0.3], [0.22, 0.28]], dtype)
    assert not bool(mpm.stencil(x, grid).valid[0])
    assert bool(mpm.stencil(x, grid).valid[1])


def test_m2_contact_gradient_and_nonnegative_dissipation(dtype: np.dtype[np.floating]) -> None:
    cfg = contact.PlaneContact(enabled=True)
    x = j.array([[0.2, -0.001], [0.3, 0.002]], dtype)
    v = j.array([[0.2, -0.1], [-0.3, 0]], dtype)
    area = j.array([0.01, 0.02], dtype)
    force, E, d = contact.response(x, v, area, cfg)
    grad = jax.grad(lambda x: j.sum(contact.response(x, j.zeros_like(v), area, cfg)[1]))(x)
    np.testing.assert_allclose(force[:, 1], -grad[:, 1], rtol=2e-6)
    assert np.all(d >= 0)
    f2, E2, d2 = contact.response(j.repeat(x, 2, 0), j.repeat(v, 2, 0), j.repeat(area / 2, 2), cfg)
    np.testing.assert_allclose(f2.reshape(2, 2, 2).sum(1), force, rtol=2e-6)
