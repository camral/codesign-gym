"""Continuum assembly and constitutive mechanics across MPM and FEM."""

from __future__ import annotations

import jax
import jax.numpy as j
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import Fields
from envs.soft_walker.engine.physics import contact, fem, mpm
from envs.soft_walker.engine.physics.common import DynamicsConfig


def fields(points: jax.Array, alpha: float = 1.0) -> Fields:
    n, dim = points.shape
    one = j.ones(n, points.dtype)
    direction = j.broadcast_to(j.eye(dim, dtype=points.dtype)[0], (n, dim))
    return Fields(
        one * alpha,
        one * 20000,
        one * 1000,
        one * 0.5,
        direction,
        j.ones(n, j.int32),
        (points[:, 0] > 0).astype(j.int32),
    )


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 2), (3, 2, 1)])
def test_m2_hex_affine_patch_volume_and_force(
    shape: tuple[int, int, int], dtype: np.dtype[np.floating]
) -> None:
    mesh = fem.structured_mesh(shape=shape, dtype=dtype)
    m = fem.instantiate(mesh, fields(mesh.points))
    np.testing.assert_allclose(j.sum(mesh.weights), 0.1 * 0.05 * 0.05, rtol=1e-6)
    np.testing.assert_allclose(mesh.grad.sum(1), 0.0, atol=2e-5)
    H = j.array([[0.08, 0.12, 0.01], [-0.04, -0.03, 0.02], [0.01, 0.03, 0.04]], dtype)
    u = mesh.reference @ H.T
    F = fem.deformation(m, u)
    target = j.broadcast_to(j.eye(3, dtype=dtype) + H, F.shape)
    tol = 3e-5 if dtype == np.dtype("float32") else 2e-11
    np.testing.assert_allclose(F, target, rtol=tol, atol=tol)
    cfg = DynamicsConfig(gravity=0)
    a = j.array([0.1, -0.05], dtype)
    force = fem.internal_force(m, u, a, cfg)
    grad = jax.grad(fem.elastic_energy, argnums=1)(m, u, a, cfg)
    np.testing.assert_allclose(force, -grad, rtol=tol, atol=tol)
    np.testing.assert_allclose(force.sum(0), 0.0, atol=tol)
    torque = j.sum(j.cross(mesh.reference + u, force), 0)
    np.testing.assert_allclose(torque, 0.0, atol=tol)
    # Uniform passive affine stress: fully interior nodes must balance.
    fp = fem.internal_force(m, u, j.zeros_like(a), cfg)
    x = np.asarray(mesh.reference)
    inside = np.all((x > x.min(0) + 1e-8) & (x < x.max(0) - 1e-8), axis=1)
    if np.any(inside):
        np.testing.assert_allclose(fp[inside], 0.0, atol=tol)


def test_m2_hex_affine_refinement_and_rigid_rotation(dtype: np.dtype[np.floating]) -> None:
    cfg = DynamicsConfig(gravity=0)
    energies = []
    for shape in [(1, 1, 1), (2, 2, 2), (3, 2, 2)]:
        mesh = fem.structured_mesh(shape=shape, dtype=dtype)
        m = fem.instantiate(mesh, fields(mesh.points))
        u = mesh.reference * j.array([0.1, -0.03, 0.04], dtype)
        energies.append(fem.elastic_energy(m, u, j.zeros(2, dtype), cfg))
        ang = 0.37
        R = j.array(
            [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]], dtype
        )
        u = mesh.reference @ (R - j.eye(3, dtype=dtype)).T
        f = fem.internal_force(m, u, j.zeros(2, dtype), cfg)
        np.testing.assert_allclose(f, 0.0, atol=1e-4 if dtype == np.dtype("float32") else 1e-10)
    np.testing.assert_allclose(energies, np.repeat(energies[0], len(energies)), rtol=2e-5)


def test_m2_hex_freeflight_and_fixed_nodes(dtype: np.dtype[np.floating]) -> None:
    mesh = fem.structured_mesh(shape=(1, 1, 1), dtype=dtype)
    m = fem.instantiate(mesh, fields(mesh.points))
    s = fem.initial_state(m)
    cfg = DynamicsConfig(gravity=9.81)
    out, info = jax.jit(lambda m, s: fem.trial(m, s, j.zeros(2, dtype), 0.001, cfg))(m, s)
    assert info.accepted == 1
    np.testing.assert_allclose(out.v[:, 1], -0.00981, rtol=2e-5)
    np.testing.assert_allclose(out.u[:, 1], -0.00000981, rtol=2e-5)
    assert out.work_positive == 0
    fixed = j.broadcast_to(
        (mesh.reference[:, 0] == j.min(mesh.reference[:, 0]))[:, None], s.u.shape
    )
    mf = m._replace(fixed=fixed)
    out, info = jax.jit(lambda m, s: fem.trial(m, s, j.array([0.2, -0.1], dtype), 0.001, cfg))(
        mf, s
    )
    assert info.accepted == 1
    np.testing.assert_array_equal(out.u[fixed], 0.0)
    np.testing.assert_array_equal(out.v[fixed], 0.0)


def test_m2_hex_empty_nodes_nan_safety(dtype: np.dtype[np.floating]) -> None:
    mesh = fem.structured_mesh(shape=(2, 1, 1), dtype=dtype)
    f = fields(mesh.points)._replace(occupancy=j.array([1.0, 0.0], dtype))
    m = fem.instantiate(mesh, f)
    s = fem.initial_state(m)
    sn = s._replace(
        u=j.where(m.active[:, None], s.u, j.nan), v=j.where(m.active[:, None], s.v, j.nan)
    )
    cfg = DynamicsConfig(gravity=0)
    out, info = jax.jit(lambda m, s: fem.trial(m, s, j.zeros(2, dtype), 0.001, cfg))(m, sn)
    assert info.accepted == 1
    assert np.isfinite(out.u).all() and np.isfinite(out.v).all()


def test_m2_mpm_internal_force_variational_and_contact(dtype: np.dtype[np.floating]) -> None:
    cloud = mpm.uniform_cloud(cell_shape=(2, 1), origin=(0.025, 0.001), dtype=dtype)
    m = mpm.instantiate(cloud, fields(cloud.points))
    s = mpm.initial_state(m)
    grid = mpm.Grid(shape=(8, 7))
    st = mpm.stencil(s.x, grid)
    cfg = DynamicsConfig(gravity=0, contact=contact.PlaneContact(enabled=True, friction=0))
    rng = np.random.default_rng(48)
    gv = j.array(rng.normal(size=(grid.count, 2)) * 0.01, dtype)
    dt = 0.001
    a = j.array([0.2, -0.1], dtype)

    def en(gv: jax.Array) -> jax.Array:
        _, _, L = mpm.g2p(gv, st, grid)
        return mpm.elastic_energy(m, mpm.deformation_next(s.F, L, dt), a, cfg)

    _, _, L = mpm.g2p(gv, st, grid)
    F = mpm.deformation_next(s.F, L, dt)
    fi = mpm.internal_grid_force(m, F, s.F, a, st, grid, cfg)
    tol = 4e-5 if dtype == np.dtype("float32") else 2e-10
    np.testing.assert_allclose(jax.grad(en)(gv), -dt * fi, rtol=tol, atol=tol * 1e-4)
    q, H = mpm.surface_mapping(m, s, st)

    def ce(gv: jax.Array) -> jax.Array:
        return mpm.grid_contact(m, q, H, gv, st, grid, dt, cfg)[1]

    fc = mpm.grid_contact(m, q, H, gv, st, grid, dt, cfg)[0]
    np.testing.assert_allclose(jax.grad(ce)(gv), -dt * fc, rtol=tol, atol=tol * 1e-5)


def test_m2_mpm_freeflight(dtype: np.dtype[np.floating]) -> None:
    cloud = mpm.uniform_cloud(cell_shape=(2, 1), dtype=dtype)
    m = mpm.instantiate(cloud, fields(cloud.points))
    s = mpm.initial_state(m)
    grid = mpm.Grid(shape=(8, 7))
    cfg = DynamicsConfig(gravity=9.81)
    out, info = jax.jit(lambda m, s: mpm.trial(m, s, j.zeros(2, dtype), 0.001, grid, cfg))(m, s)
    assert info.accepted == 1
    np.testing.assert_allclose(out.v[:, 1], -0.00981, rtol=3e-5)
    np.testing.assert_allclose(out.F, s.F, rtol=2e-5, atol=2e-6)
    assert out.work_positive == 0


def test_m2_mpm_inactive_nan_and_exit_rollback(dtype: np.dtype[np.floating]) -> None:
    cloud = mpm.uniform_cloud(cell_shape=(2, 1), dtype=dtype)
    f = fields(cloud.points)
    f = f._replace(occupancy=f.occupancy.at[0].set(0.0))
    m = mpm.instantiate(cloud, f)
    s = mpm.initial_state(m)
    grid = mpm.Grid(shape=(8, 7))
    cfg = DynamicsConfig(gravity=0)
    sn = s._replace(
        x=s.x.at[0].set(j.nan),
        F=s.F.at[0].set(j.nan),
        C=s.C.at[0].set(j.nan),
        v=s.v.at[0].set(j.nan),
    )
    out, info = jax.jit(lambda m, s: mpm.trial(m, s, j.zeros(2, dtype), 0.001, grid, cfg))(m, sn)
    assert info.accepted == 1
    assert np.isfinite(out.x).all()
    sn = s._replace(x=s.x.at[1, 0].set(-0.048))
    out, info = jax.jit(lambda m, s: mpm.trial(m, s, j.zeros(2, dtype), 0.001, grid, cfg))(m, sn)
    assert out.status != 0 and info.accepted == 0 and info.support_valid == 0
    np.testing.assert_array_equal(out.x, sn.x)
    assert out.time == 0


def test_m2_mpm_timestep_local_explicit_comparison(dtype: np.dtype[np.floating]) -> None:
    if dtype == np.dtype("float32"):
        # Still executes at useful step sizes rather than demanding convergence
        # below the single-precision deformation-gradient roundoff floor.
        hs = [0.001, 0.0005, 0.00025]
    else:
        hs = [0.001, 0.0005, 0.00025, 0.000125]
    cloud = mpm.uniform_cloud(cell_shape=(2, 1), dtype=dtype)
    m = mpm.instantiate(cloud, fields(cloud.points))
    s = mpm.initial_state(m)
    s = s._replace(F=s.F.at[:, 0, 0].set(1.05))
    grid = mpm.Grid(shape=(8, 7))
    cfg = DynamicsConfig(gravity=0)
    fn = jax.jit(lambda h: mpm.trial(m, s, j.zeros(2, dtype), h, grid, cfg))
    explicit = jax.jit(lambda h: mpm.explicit_trial(m, s, j.zeros(2, dtype), h, grid, cfg))
    errors = []
    for h in hs:
        sn, info = fn(j.array(h, dtype))
        ex = explicit(j.array(h, dtype))
        assert info.accepted == 1
        errors.append(float(j.linalg.norm(sn.v - ex.v)))
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1)), errors
    assert errors[-2] / errors[-1] > 2.5, errors
