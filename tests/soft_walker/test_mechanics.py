"""Spring energy, force, integration, and implicit-gradient numerical contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.types import Embedding


from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import make
from envs.soft_walker.engine.differentiation import bind_rollout
from envs.soft_walker.engine.embodiments.decoders import FieldSpec, GridDecoder, cast_embedding
from envs.soft_walker.engine.physics import spring
from envs.soft_walker.engine.physics.spring_config import SpringConfig


def fixture(
    dtype: np.dtype[np.floating] | type[np.floating],
    cfg: SpringConfig | None = None,
    mode: str = "hard",
) -> tuple[SpringConfig, GridDecoder, spring.Topology, Embedding, spring.Model, spring.State]:
    cfg = cfg or SpringConfig(
        grid_shape=(2, 1), ground=False, gravity=0, damping_ratio=0, settle_steps=0
    )
    d = GridDecoder(FieldSpec(design_mode=mode), dtype=dtype)
    topo = spring.topology(cfg, dtype)
    emb = cast_embedding(d, d.base())
    f = d.decode(emb, topo.points)
    m = spring.instantiate(f, topo, cfg, 2)
    return cfg, d, topo, emb, m, spring.initial_state(m)


def test_free_rigid_motions(dtype: np.dtype[np.floating]) -> None:
    cfg, d, t, b, m, s = fixture(dtype)
    angle = 0.713
    rot = jnp.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=dtype
    )
    x = s.x @ rot.T + jnp.asarray([0.31, -0.42], dtype=dtype)
    energy = spring.elastic_energy(m, x, s.actuator)
    force, _ = spring.forces(m, x, s.v, s.actuator, cfg)
    assert abs(float(energy)) < (1e-10 if dtype == np.float32 else 1e-24)
    np.testing.assert_allclose(force, 0, atol=3e-5 if dtype == np.float32 else 1e-12)


def test_independent_force_energy_torque(dtype: np.dtype[np.floating]) -> None:
    cfg, d, t, b, m, s = fixture(dtype)
    x = np.asarray(s.x) + np.random.default_rng(7).normal(size=s.x.shape) * 0.001
    x = jnp.asarray(x, dtype=dtype)
    a = jnp.asarray([0.2, -0.1], dtype=dtype)
    f, _ = spring.forces(m, x, s.v, a, cfg)
    # Independent scalar NumPy edge calculation.
    expected = np.zeros_like(np.asarray(x))
    energy = 0.0
    for edge, k, rest_length, row in zip(
        np.asarray(m.edges),
        np.asarray(m.stiffness),
        np.asarray(m.length),
        np.asarray(m.logstrain_map),
    ):
        i, j = edge
        v = np.asarray(x)[j] - np.asarray(x)[i]
        length = np.linalg.norm(v)
        preferred = rest_length * np.exp(row @ np.asarray(a))
        local = k * (length - preferred) * v / length
        expected[i] += local
        expected[j] -= local
        energy += 0.5 * k * (length - preferred) ** 2
    np.testing.assert_allclose(
        f,
        expected,
        rtol=2e-5 if dtype == np.float32 else 1e-12,
        atol=1e-6 if dtype == np.float32 else 1e-12,
    )
    np.testing.assert_allclose(
        spring.elastic_energy(m, x, a), energy, rtol=3e-5 if dtype == np.float32 else 1e-12
    )
    np.testing.assert_allclose(
        np.asarray(f).sum(axis=0), 0, atol=2e-6 if dtype == np.float32 else 1e-12
    )
    torque = jnp.sum(x[:, 0] * f[:, 1] - x[:, 1] * f[:, 0])
    assert abs(float(torque)) < (1e-6 if dtype == np.float32 else 1e-12)
    grad = jax.grad(lambda q: spring.elastic_energy(m, q, a))(x)
    np.testing.assert_allclose(
        -grad,
        f,
        rtol=2e-5 if dtype == np.float32 else 2e-12,
        atol=2e-6 if dtype == np.float32 else 1e-12,
    )


def test_elastic_tangent_symmetry(dtype: np.dtype[np.floating]) -> None:
    cfg, d, t, b, m, s = fixture(dtype)
    h = jax.hessian(lambda x: spring.elastic_energy(m, x.reshape(s.x.shape), s.actuator))(
        s.x.reshape(-1)
    )
    np.testing.assert_allclose(h, h.T, rtol=2e-6, atol=2e-4 if dtype == np.float32 else 1e-11)


def test_resolution_mass_and_affine_energy_scaling(dtype: np.dtype[np.floating]) -> None:
    masses = []
    energies = []
    for shape in ((2, 1), (4, 2), (6, 3)):
        c = SpringConfig(grid_shape=shape, ground=False, gravity=0, settle_steps=0)
        cfg, d, t, b, m, s = fixture(dtype, c)
        masses.append(float(m.mass.sum()))
        energies.append(float(spring.elastic_energy(m, s.x * 1.04, s.actuator)))
    np.testing.assert_allclose(masses, 0.2, rtol=1e-6)
    np.testing.assert_allclose(energies, energies[0], rtol=3e-5 if dtype == np.float32 else 1e-12)


def test_compliance_weighted_shared_link(dtype: np.dtype[np.floating]) -> None:
    cfg, d, t, b, m, s = fixture(dtype)
    fields = d.decode(b, t.points)._replace(young=jnp.asarray([10000.0, 80000.0], dtype=dtype))
    m = spring.instantiate(fields, t, cfg, 2)
    shared = np.where(np.asarray(t.owner_mask).sum(axis=1) == 2)[0]
    assert len(shared) == 1
    k = int(shared[0])
    harmonic = 2 / (1 / 10000 + 1 / 80000)
    expected = (
        harmonic
        * 2
        * cfg.thickness
        * (cfg.size[0] * cfg.size[1] / 2)
        / (2 * float(t.lengths[k]) ** 2)
    )
    np.testing.assert_allclose(m.stiffness[k], expected, rtol=2e-6)


def test_preferred_strain_bounds(dtype: np.dtype[np.floating]) -> None:
    cfg, d, t, b, m, s = fixture(dtype)
    max_log = np.asarray(m.logstrain_map).sum(axis=-1)
    assert (max_log >= 0).all() and (max_log <= cfg.max_log_stretch + 1e-7).all()
    for a in (jnp.ones(2, dtype=dtype), -jnp.ones(2, dtype=dtype)):
        _, _, preferred = spring.edge_geometry(m, s.x, a)
        assert np.all(preferred > 0)
        assert np.max(np.abs(np.log(preferred / m.length))) <= cfg.max_log_stretch + 1e-6


def test_inactive_nan_sanitization(dtype: np.dtype[np.floating]) -> None:
    c = SpringConfig(grid_shape=(2, 1), ground=False, gravity=0, settle_steps=0, substeps=2)
    d = GridDecoder(dtype=dtype, shape=(2, 2))
    b = d.base()
    b["values"][0, :, 0] = -6
    b["values"][1, :, 0] = 6
    t = spring.topology(c, dtype)
    f = d.decode(cast_embedding(d, b), t.points)
    m = spring.instantiate(f, t, c, 2)
    s = spring.initial_state(m)
    assert np.any(~np.asarray(m.active)) and np.any(np.asarray(m.active))
    bad = s._replace(
        x=jnp.where(m.active[:, None], s.x, jnp.nan), v=jnp.where(m.active[:, None], s.v, jnp.nan)
    )
    f, _ = spring.forces(m, bad.x, bad.v, bad.actuator, c)
    assert np.isfinite(f).all()
    n, diag = jax.jit(lambda s: spring.advance(m, s, jnp.zeros(2, dtype=dtype), c))(bad)
    assert int(n.status) == 0 and np.isfinite(n.x).all()


def test_backward_euler_energy_damping(dtype: np.dtype[np.floating]) -> None:
    cfg, d, t, b, m, s = fixture(dtype)
    cfg = replace(cfg, substeps=10)
    x = s.x.at[-1, 0].add(0.002)
    s = s._replace(x=x)
    e0 = float(spring.energy_budget(m, s, cfg).sum())
    fn = jax.jit(lambda s: spring.advance(m, s, jnp.zeros(2, dtype=dtype), cfg))
    n, di = fn(s)
    assert int(n.status) == 0
    e1 = float(spring.energy_budget(m, n, cfg).sum())
    assert e1 < e0 and float(n.work_positive) == 0


def test_contact_energy_force_and_friction(dtype: np.dtype[np.floating]) -> None:
    c = SpringConfig(grid_shape=(1, 1), settle_steps=0)
    cfg, d, t, b, m, s = fixture(dtype, c)
    x = s.x.at[:, 1].add(-0.003)
    v = s.v.at[:, 0].set(0.05)
    f, u, diss = spring.contact(m, x, v, c)
    g = jax.grad(lambda p: spring.contact(m, p, v, c)[1])(x)
    np.testing.assert_allclose(f[:, 1], -g[:, 1], rtol=2e-6)
    assert float(diss) >= 0 and np.all(np.asarray(f)[:, 0] <= 0)
    assert np.all(np.asarray(f)[:, 1] >= 0)


def test_root_implicit_derivative(dtype: np.dtype[np.floating]) -> None:
    c = SpringConfig(settle_steps=0)

    def f(p: jax.Array) -> jax.Array:
        r, aux = spring.solve_root(lambda y: y + y**3 - p, jnp.zeros((1,), dtype=dtype), c)
        return r[0]

    p = jnp.asarray([0.2], dtype=dtype)
    y = jax.jit(f)(p)
    g = jax.jit(jax.grad(f))(p)
    np.testing.assert_allclose(g, 1 / (1 + 3 * y * y), rtol=2e-6 if dtype == np.float32 else 2e-10)


def test_rollout_matches_api_including_settling(
    dtype: np.dtype[np.floating], small_config: SpringConfig, tolerance: float
) -> None:
    e, b = make(dtype=dtype, config=small_config, max_observation_space_size=128)
    e.reset(embodiment=b)
    u = np.array([[0.2, -0.1, 0, 0], [0.1, 0.2, 0.4, -0.3]], dtype=dtype)
    observations = []
    for a in u:
        observations.append(e.step(a)[0])
    out = jax.jit(bind_rollout(e))(cast_embedding(e.decoder, b), jnp.asarray(u))
    assert bool(out["metrics"]["valid"])
    np.testing.assert_allclose(
        out["observations"], np.asarray(observations), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(out["state"].x, e._state.x, rtol=tolerance, atol=tolerance)
    g = jax.jit(
        jax.grad(
            lambda acts: bind_rollout(e)(cast_embedding(e.decoder, b), acts)["metrics"][
                "work_positive_j"
            ]
        )
    )(jnp.asarray(u))
    assert np.isfinite(g).all() and np.linalg.norm(np.asarray(g)[:, :2]) > 0
    np.testing.assert_array_equal(np.asarray(g)[:, 2:], 0)


def test_invalid_pure_rollout_has_nan_objectives(dtype: np.dtype[np.floating]) -> None:
    e, b = make(dtype=dtype, config=SpringConfig(grid_shape=(2, 1), settle_steps=0, substeps=2))
    b["values"][..., 0] = -6
    out = jax.jit(bind_rollout(e))(cast_embedding(e.decoder, b), jnp.zeros((1, 4), dtype=dtype))
    assert not bool(out["metrics"]["valid"]) and np.isnan(out["metrics"]["objective"]).all()


def test_small_kvaerno_driver_and_control_gradient(dtype: np.dtype[np.floating]) -> None:
    pytest.importorskip(
        "diffrax"
    )  # the kvaerno3 integrator uses diffrax, not a codesign-gym dependency
    c = SpringConfig(grid_shape=(1, 1), settle_steps=0, substeps=2, integrator="kvaerno3")
    e, b = make(dtype=dtype, config=c)
    e.reset(embodiment=b)
    result = e.step(np.array([0.2, -0.1, 0, 0], dtype=dtype))
    assert not result[2] and result[4]["solver"]["max_residual"] == -1
    r = bind_rollout(e)
    p = cast_embedding(e.decoder, b)
    fun = jax.jit(jax.value_and_grad(lambda a: r(p, a)["metrics"]["work_positive_j"]))
    val, g = fun(jnp.asarray([[0.2, -0.1]], dtype=dtype))
    assert np.isfinite(val) and np.isfinite(g).all() and np.linalg.norm(g) > 0


@pytest.mark.parametrize(
    "kw",
    [
        {"grid_shape": (20, 20)},
        {"substeps": 0},
        {"settle_steps": -1},
        {"contact_width": 0},
        {"control_dt": float("nan")},
        {"integrator": "explicit"},
        {"max_dense_dofs": 100000},
        {"damping_ratio": -1},
    ],
)
def test_numerical_config_guards(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SpringConfig(**kw)
