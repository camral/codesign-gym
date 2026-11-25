"""Beam objectivity, force/torque balance, and terrain derivatives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import topology_checks
from envs.soft_walker.engine.embodiments.decoders import FieldSpec, cast_embedding, make_decoder
from envs.soft_walker.engine.physics import beam, spring, terrain
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact, response
from envs.soft_walker.engine.physics.spring_config import SpringConfig


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", ["flat", "slopes", "blocks"])
def test_m3_terrain_replay_and_geometry(dim: int, kind: str, dtype: np.dtype[np.floating]) -> None:
    cfg = terrain.TerrainConfig(kind=kind)
    a = terrain.generate(cfg, 77, dim, dtype)
    b = terrain.from_dict(terrain.serialize(a), dtype)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    c = terrain.generate(cfg, 77, dim, dtype)
    for x, y in zip(a, c):
        np.testing.assert_array_equal(x, y)
    pts = np.random.default_rng(7).normal(size=(8, dim)) * 0.01
    pts[:, 0] += 0.11
    pts[:, 1] += 0.02
    gap, n = jax.jit(terrain.gap_normal)(jnp.asarray(pts, dtype), a)
    automatic = jax.vmap(jax.grad(lambda x: terrain.gap_normal(x, a)[0]))(jnp.asarray(pts, dtype))
    np.testing.assert_allclose(n, automatic, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(jnp.linalg.norm(n, axis=-1), 1, atol=3e-7)
    if kind != "flat":
        other = terrain.generate(cfg, 78, dim, dtype)
        assert any(not np.array_equal(x, y) for x, y in zip(a, other))


@pytest.mark.parametrize("dim", [2, 3])
def test_m3_contact_gradient_and_dissipation(dim: int, dtype: np.dtype[np.floating]) -> None:
    t = terrain.generate(terrain.TerrainConfig(kind="slopes"), 33, dim, dtype)
    x = (
        jnp.zeros((3, dim), dtype)
        .at[:, 0]
        .set(jnp.asarray([0.015, 0.035, 0.055], dtype))
        .at[:, 1]
        .set(-0.0004)
    )
    v = jnp.asarray(np.random.default_rng(4).normal(size=(3, dim)) * 0.03, dtype)
    area = jnp.ones(3, dtype) * 0.001
    cfg = PlaneContact(enabled=True)
    f, u, d = response(x, v, area, cfg, t)
    conservative = response(x, jnp.zeros_like(v), area, cfg, t)[0]
    g = jax.grad(lambda q: jnp.sum(response(q, jnp.zeros_like(v), area, cfg, t)[1]))(x)
    np.testing.assert_allclose(conservative, -g, rtol=8e-6, atol=2e-7)
    assert bool(jnp.all(d >= -1e-12))
    normalforce = jnp.sum(f * t.normal, axis=-1)
    tang = f - normalforce[:, None] * t.normal
    assert bool(jnp.all(jnp.linalg.norm(tang, axis=-1) <= cfg.friction * normalforce + 1e-6))


def test_m3_block_top_and_side_contact(dtype: np.dtype[np.floating]) -> None:
    t = terrain.generate(terrain.TerrainConfig(kind="blocks", blocks=1), 3, 3, dtype)
    c, h = np.asarray(t.centers[0]), np.asarray(t.half_sizes[0])
    for axis in [0, 1, 2]:
        p = c.copy()
        p[1] += h[1] * 0.5
        p[axis] = c[axis] + h[axis] + 0.001
        if axis == 1:
            p = c + np.array([0.0, h[1] + 0.001, 0.0])
        gap, n = terrain.gap_normal(jnp.asarray(p, dtype), t)
        if axis == 2 and gap < 0:
            continue
        np.testing.assert_allclose(gap, 0.001, rtol=1e-4, atol=3e-8)
        np.testing.assert_allclose(n, np.eye(3)[axis], atol=2e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_m3_nd_topology_host_compiled_agree(dim: int) -> None:
    # Exhaust every local occupancy pattern, not a handful of handpicked designs.
    n = 2**dim
    masks = np.asarray([[(i >> k) & 1 for k in range(n)] for i in range(2**n)], dtype=float)
    actual = np.asarray(
        jax.jit(jax.vmap(lambda a: topology_checks.valid(a, (2,) * dim)))(jnp.asarray(masks))
    )
    expected = np.asarray([not topology_checks.errors(a, (2,) * dim) for a in masks])
    np.testing.assert_array_equal(actual, expected)


def beam_model(
    dtype: np.dtype[np.floating] | type[np.floating],
    shape: tuple[int, int, int] = (2, 1, 1),
    size: tuple[float, float, float] = (0.1, 0.05, 0.05),
) -> tuple[beam.Model, DynamicsConfig]:
    t = beam.topology(shape, size, (0.0, 0.01, 0.0), dtype)
    d = make_decoder("direct", spec=FieldSpec(dim=3, design_mode="relaxed"), dtype=dtype)
    f = d.decode(cast_embedding(d, d.base()), t.points)
    return beam.instantiate(t, f), DynamicsConfig(gravity=0.0)


def test_m3_beam_axial_reference_and_mass(dtype: np.dtype[np.floating]) -> None:
    m, cfg = beam_model(dtype)
    s = beam.initial_state(m)
    eps = 0.03
    u = s.u.at[1, 0].set(float(m.topology.h) * eps)
    energy = beam.elastic_energy(m, u, s.rotation, s.actuator, cfg)
    expected = 0.5 * float(m.topology.h) * float(m.extension[0, 0, 0]) * eps**2
    np.testing.assert_allclose(energy, expected, rtol=5e-6)
    fine, _ = beam_model(dtype, (4, 2, 2), (0.1, 0.05, 0.05))
    np.testing.assert_allclose(jnp.sum(m.mass), jnp.sum(fine.mass), rtol=2e-6)
    # A finite common rotation/translation cannot create elastic energy.
    Q = beam.exp_so3(jnp.asarray([0.5, -0.6, 0.2], dtype))
    translation = jnp.asarray([0.12, 0.03, -0.01], dtype)
    u = (m.topology.reference @ Q.T) + translation - m.topology.reference
    R = jnp.broadcast_to(Q, s.rotation.shape)
    assert float(jnp.abs(beam.elastic_energy(m, u, R, s.actuator, cfg))) < (
        2e-11 if dtype == np.float32 else 1e-24
    )


def test_m3_beam_energy_objectivity_and_force_torque(dtype: np.dtype[np.floating]) -> None:
    m, cfg = beam_model(dtype, (2, 2, 1), (0.1, 0.1, 0.05))
    s = beam.initial_state(m)
    rng = np.random.default_rng(44)
    u = jnp.asarray(rng.normal(size=s.u.shape) * 0.001, dtype)
    R = beam.exp_so3(jnp.asarray(rng.normal(size=s.u.shape) * 0.05, dtype))
    a = jnp.asarray([0.22, -0.13], dtype)
    Q = beam.exp_so3(jnp.asarray([0.6, 0.3, -0.4], dtype))
    e = beam.elastic_energy(m, u, R, a, cfg)
    rotated = (m.topology.reference + u) @ Q.T - m.topology.reference
    er = beam.elastic_energy(m, rotated, Q @ R, a, cfg)
    np.testing.assert_allclose(e, er, rtol=3e-5 if dtype == np.float32 else 2e-12)
    f, t = beam.elastic_force_torque(m, u, R, a, cfg)
    np.testing.assert_allclose(jnp.sum(f, axis=0), 0, atol=3e-6 if dtype == np.float32 else 1e-13)
    torque = jnp.sum(jnp.cross(m.topology.reference + u, f) + t, axis=0)
    np.testing.assert_allclose(torque, 0, atol=3e-6 if dtype == np.float32 else 2e-13)
    # Independent NumPy central differences, retaining explicit epsilons.
    dr = jnp.asarray(rng.normal(size=u.shape) * 0.01, dtype)
    h = 0.003 if dtype == np.float32 else 1e-4
    fd = (
        float(beam.elastic_energy(m, u + h * dr, R, a, cfg))
        - float(beam.elastic_energy(m, u - h * dr, R, a, cfg))
    ) / (2 * h)
    np.testing.assert_allclose(
        fd, -jnp.sum(f * dr), rtol=3e-3 if dtype == np.float32 else 2e-7, atol=2e-7
    )


def test_m3_so3_zero_gradient_and_roundtrip(dtype: np.dtype[np.floating]) -> None:
    v = jnp.asarray([0.13, -0.21, 0.08], dtype)
    R = beam.exp_so3(v)
    np.testing.assert_allclose(beam.log_so3(R), v, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(R.T @ R, jnp.eye(3), atol=3e-7)
    z = jnp.zeros(3, dtype)
    a = jnp.asarray([0.3, -0.2, 0.1], dtype)
    jac = jax.jvp(beam.exp_so3, (z,), (a,))[1]
    np.testing.assert_allclose(jac, beam.skew(a), rtol=1e-6, atol=2e-7)
    assert np.all(np.isfinite(jax.jacfwd(jax.jacrev(lambda t: jnp.sum(beam.exp_so3(t))))(z)))


def test_m3_gray_spring_contact_surface_refinement(dtype: np.dtype[np.floating]) -> None:
    totals = []
    for shape in [(2, 1), (4, 2), (8, 4)]:
        cfg = SpringConfig(grid_shape=shape, linear_solver="gmres")
        topo = spring.topology(cfg, dtype)
        d = make_decoder("direct", spec=FieldSpec(design_mode="relaxed"), dtype=dtype)
        f = d.decode(cast_embedding(d, d.base()), topo.points)
        m = spring.instantiate(f, topo, cfg, 2)
        totals.append(float(jnp.sum(m.contact_k)))
    np.testing.assert_allclose(totals, totals[0], rtol=2e-6)


def test_m3_boolean_topology_does_not_block_shape_gradient(dtype: np.dtype[np.floating]) -> None:
    alpha = jnp.full((8,), 0.7, dtype)

    def f(a: jax.Array) -> jax.Array:
        return jnp.where(topology_checks.valid(a, (2, 2, 2)), jnp.sum(a * a), jnp.nan)

    np.testing.assert_allclose(jax.jit(jax.grad(f))(alpha), 2 * alpha, rtol=1e-6)


def test_m3_beam_zero_link_action_mask(dtype: np.dtype[np.floating]) -> None:
    from envs.soft_walker.engine.drivers import environment as drivers

    model, _ = beam_model(dtype, (1, 1, 1), (0.05, 0.05, 0.05))
    assert not np.any(drivers.channel_mask(model))


def test_m3_beam_poisson_shear_analytic(dtype: np.dtype[np.floating]) -> None:
    m, cfg = beam_model(dtype)
    s = beam.initial_state(m)
    u = s.u.at[1, 1].set(0.002)
    nu = jnp.asarray(0.35, dtype)

    # Replacing the shear stiffness by its exact nu-dependent factor isolates
    # the independent constitutive identity G=E/[2(1+nu)].
    def energy(q: jax.Array) -> jax.Array:
        factor = (1 + nu) / (1 + q)
        ext = m.extension.at[:, 1, 1].multiply(factor).at[:, 2, 2].multiply(factor)
        return beam.elastic_energy(m._replace(extension=ext), u, s.rotation, s.actuator, cfg)

    value = energy(nu)
    derivative = jax.grad(energy)(nu)
    np.testing.assert_allclose(derivative, -value / (1 + nu), rtol=3e-6, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 2])
def test_m3_beam_independent_torsion_bending_energy(
    axis: int, dtype: np.dtype[np.floating]
) -> None:
    m, cfg = beam_model(dtype)
    s = beam.initial_state(m)
    theta = jnp.asarray(0.12, dtype)
    direction = jnp.eye(3, dtype=dtype)[axis]

    def energy(q: jax.Array) -> jax.Array:
        rotations = jnp.stack(
            (beam.exp_so3(-0.5 * q * direction), beam.exp_so3(0.5 * q * direction))
        )
        return beam.elastic_energy(m, s.u, rotations, s.actuator, cfg)

    E = float(2 / (1 / m.young[0] + 1 / m.young[1]))
    nu = float(m.poisson[0])
    h = float(m.topology.h)
    area_moment = h**4 / 12
    weight = float(jnp.minimum(m.alpha[0], m.alpha[1]))
    stiffness = weight * (E / (2 * (1 + nu)) * 2 * area_moment if axis == 0 else E * area_moment)
    np.testing.assert_allclose(
        energy(theta),
        0.5 * stiffness / h * float(theta) ** 2,
        rtol=4e-6 if dtype == np.float32 else 1e-12,
    )
    np.testing.assert_allclose(
        jax.grad(energy)(theta),
        stiffness / h * float(theta),
        rtol=5e-6 if dtype == np.float32 else 2e-12,
    )


def test_m3_free_voxel_rotation_and_translation(dtype: np.dtype[np.floating]) -> None:
    m, cfg = beam_model(dtype, (1, 1, 1), (0.05, 0.05, 0.05))
    s = beam.initial_state(m)
    v = jnp.asarray([[0.1, -0.05, 0.02]], dtype)
    w = jnp.asarray([[0.2, -0.1, 0.3]], dtype)
    s = s._replace(v=v, omega=w)
    dt = jnp.asarray(0.001, dtype)
    n, info = jax.jit(lambda state: beam.trial(m, state, m.channels, dt, cfg))(s)
    assert int(n.status) == 0
    np.testing.assert_allclose(n.u, dt * v, rtol=3e-6, atol=1e-10)
    np.testing.assert_allclose(n.v, v, rtol=3e-6, atol=1e-10)
    np.testing.assert_allclose(n.omega, w, rtol=3e-6, atol=1e-10)
    np.testing.assert_allclose(n.rotation, beam.exp_so3(dt * w), rtol=3e-6, atol=1e-10)
