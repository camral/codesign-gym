"""Nonflat contact virtual work, including the selected branch at box ties."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from envs.soft_walker.engine.physics.terrain import Terrain


from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import Fields
from envs.soft_walker.engine.physics import beam, fem, mpm, spring, terrain
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact, response
from envs.soft_walker.engine.physics.spring_config import SpringConfig


def nonflat(kind: str, dim: int, dtype: np.dtype[np.floating] | type[np.floating]) -> Terrain:
    if kind == "slope":
        config = terrain.TerrainConfig(
            kind="slopes", blocks=0, height=0.017, fixed_slopes=(0.22, 0.11 if dim == 3 else 0.0)
        )
    else:
        config = terrain.TerrainConfig(
            kind="blocks",
            blocks=1,
            rounding=0.003,
            fixed_boxes=((0.035, 0.02, 0.025, 0.025, 0.012, 0.04, 0.003),),
        )
    return terrain.generate(config, dim=dim, dtype=dtype)


def fields(count: int, dim: int, dtype: np.dtype[np.floating] | type[np.floating]) -> Fields:
    return Fields(
        jnp.ones(count, dtype),
        jnp.full(count, 2.0e4, dtype),
        jnp.full(count, 1.0e3, dtype),
        jnp.zeros(count, dtype),
        jnp.broadcast_to(jnp.eye(dim, dtype=dtype)[0], (count, dim)),
        jnp.ones(count, jnp.int32),
        jnp.zeros(count, jnp.int32),
    )


def assert_force(
    actual: ArrayLike, expected: ArrayLike, dtype: np.dtype[np.floating] | type[np.floating]
) -> None:
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=3.0e-5 if dtype == np.dtype("float32") else 3.0e-11,
        atol=2.0e-6 if dtype == np.dtype("float32") else 2.0e-12,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_box_medial_tie_uses_same_face_for_gap_normal_and_energy(
    dim: int, dtype: np.dtype[np.floating]
) -> None:
    # Equal x/y distances select the first face, as the native terrain routine
    # does. The point is nonsmooth: check the declared branch, not a central FD.
    config = terrain.TerrainConfig(
        kind="blocks", blocks=1, fixed_boxes=((0.0, 0.04, 0.0, 0.02, 0.02, 0.02, 0.002),)
    )
    t = terrain.generate(config, dim=dim, dtype=dtype)
    point = jnp.asarray([0.005, 0.045, 0.0][:dim], dtype)
    # Construct the tie in floating point too; subtracting .04 from .045 may
    # otherwise put the nominally equal components on opposite sides of a tie.
    t = t._replace(centers=jnp.zeros_like(t.centers), offset=jnp.asarray(-1.0, dtype))
    point = jnp.asarray([0.005, 0.005, 0.0][:dim], dtype)
    _, normal = terrain.gap_normal(point, t)
    derivative = jax.grad(lambda x: terrain.gap_normal(x, t)[0])(point)
    np.testing.assert_array_equal(normal, np.eye(dim, dtype=dtype)[0])
    np.testing.assert_array_equal(derivative, normal)
    cfg = PlaneContact(enabled=True, friction=0.0)
    force = response(point, jnp.zeros_like(point), jnp.asarray(0.001, dtype), cfg, t)[0]
    energy_grad = jax.grad(
        lambda x: response(x, jnp.zeros_like(x), jnp.asarray(0.001, dtype), cfg, t)[1]
    )(point)
    assert_force(force, -energy_grad, dtype)


@pytest.mark.parametrize("kind", ["slope", "box"])
@pytest.mark.parametrize("physics", ["spring", "fem", "beam"])
def test_nonflat_solid_contact_force_and_torque_are_energy_derivatives(
    physics: str, kind: str, dtype: np.dtype[np.floating]
) -> None:
    dim = 2 if physics == "spring" else 3
    t = nonflat(kind, dim, dtype)
    cfg = DynamicsConfig(gravity=0.0, contact=PlaneContact(enabled=True, friction=0.0))
    if physics == "spring":
        sc = SpringConfig(
            grid_shape=(1, 1),
            size=(0.05, 0.05),
            gravity=0.0,
            ground=True,
            initial_clearance=0.02,
            friction=0.0,
        )
        topo = spring.topology(sc, dtype)
        m = spring.instantiate(fields(1, dim, dtype), topo, sc, 1)._replace(terrain=t)
        x = m.reference
        zero = jnp.zeros_like(x)
        force, energy, _ = spring.contact(m, x, zero, sc)
        grad = jax.grad(lambda y: spring.contact(m, y, zero, sc)[1])(x)
        assert_force(force, -grad, dtype)
        np.testing.assert_allclose(
            spring.admissibility(m, x, sc)[-1],
            jnp.max(
                jnp.where(m.contact_k > 0, jnp.maximum(-terrain.gap_normal(x, t)[0], 0.0), 0.0)
            ),
            rtol=2.0e-6,
            atol=1.0e-9,
        )
    elif physics == "fem":
        mesh = fem.structured_mesh((1, 1, 1), (0.05, 0.05, 0.05), (0.0, 0.02, 0.0), dtype)
        m = fem.instantiate(mesh, fields(1, dim, dtype), channels=1)._replace(terrain=t)
        s = fem.initial_state(m)
        force, _ = fem.force(m, s.u, s.v, s.actuator, cfg)
        off = replace(cfg, contact=replace(cfg.contact, enabled=False))
        force = force - fem.force(m, s.u, s.v, s.actuator, off)[0]
        energy = fem.energy_budget(m, s, cfg)[3]
        grad = jax.grad(lambda u: fem.energy_budget(m, s._replace(u=u), cfg)[3])(s.u)
        assert_force(force, -grad, dtype)
    else:
        topo = beam.topology((2, 1, 1), (0.1, 0.05, 0.05), (0.0, 0.02, 0.0), dtype)
        m = beam.instantiate(topo, fields(2, dim, dtype), channels=1)._replace(terrain=t)
        s = beam.initial_state(m)
        # A rotated marker makes nonflat contact torque observable.
        R = beam.exp_so3(jnp.full_like(s.u, 0.07)) @ s.rotation
        force, torque, energy, _, _ = beam.surface(m, s.u, R, s.v, s.omega, cfg)

        def potential(u: jax.Array, theta: jax.Array) -> jax.Array:
            return beam.surface(m, u, beam.exp_so3(theta) @ R, s.v, s.omega, cfg)[2]

        du, dr = jax.grad(potential, argnums=(0, 1))(s.u, jnp.zeros_like(s.u))
        assert_force(force, -du, dtype)
        assert_force(torque, -dr, dtype)
        assert np.linalg.norm(np.asarray(torque)) > 1.0e-5
    assert float(energy) > 1.0e-8, "Fixture must actually touch the terrain"
    assert np.linalg.norm(np.asarray(force)) > 1.0e-4


@pytest.mark.parametrize("kind", ["slope", "box"])
@pytest.mark.parametrize("dim", [2, 3])
def test_nonflat_mpm_grid_contact_obeys_marker_virtual_work(
    kind: str, dim: int, dtype: np.dtype[np.floating]
) -> None:
    cloud = mpm.uniform_cloud(
        (1,) * dim, 0.05, (0.0, 0.02, 0.0)[:dim], particles_per_axis=1, dtype=dtype
    )
    m = mpm.instantiate(cloud, fields(1, dim, dtype), channels=1)._replace(
        terrain=nonflat(kind, dim, dtype)
    )
    s = mpm.initial_state(m)
    # A nonidentity F checks that the marker-to-grid pullback includes its offset.
    F = s.F.at[0, 0, 1].set(0.08)
    s = s._replace(F=F)
    grid = mpm.Grid((7,) * dim, 0.05, (-0.1,) * dim)
    st = mpm.stencil(s.x, grid)
    q0, H = mpm.surface_mapping(m, s, st)
    gv = jnp.zeros((grid.count, dim), dtype)
    dt = jnp.asarray(0.001, dtype)
    cfg = DynamicsConfig(contact=PlaneContact(enabled=True, friction=0.0))
    force, energy, _, _ = mpm.grid_contact(m, q0, H, gv, st, grid, dt, cfg)
    gradient = jax.grad(lambda v: mpm.grid_contact(m, q0, H, v, st, grid, dt, cfg)[1])(gv)
    assert_force(force, -gradient / dt, dtype)
    assert float(energy) > 1.0e-8, "Fixture must actually touch the terrain"
    assert np.linalg.norm(np.asarray(force)) > 1.0e-4
