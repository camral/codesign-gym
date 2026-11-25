"""Regression fixtures for mutation, cancellation, batching and runtime semantics."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import make
from envs.soft_walker.engine.device import resolve_dtype
from envs.soft_walker.engine.differentiation import bind_rollout
from envs.soft_walker.engine.embodiments.decoders import cast_embedding
from envs.soft_walker.engine.embodiments.precedents import evogym_category_scores, evogym_fixture
from envs.soft_walker.engine.physics import spring
from envs.soft_walker.engine.physics.spring_config import SpringConfig


@pytest.mark.parametrize("decoder", ["direct", "cppn", "primitives", "bspline"])
def test_every_decoder_drives_spring(dtype: np.dtype[np.floating], decoder: str) -> None:
    cfg = SpringConfig(grid_shape=(2, 1), settle_steps=0, substeps=4, horizon=2)
    e, b = make(dtype=dtype, decoder=decoder, config=cfg)
    obs, reject, info = e.reset(embodiment=b)
    assert not reject[0] and info["mass_kg"] > 0
    obs, reward, term, trunc, info = e.step(np.array([0.15, -0.1, 0, 0], dtype=dtype))
    assert not term and info["valid"] and np.isfinite(obs).all()


def test_original_cppn_architecture_against_scalar_evaluation(
    dtype: np.dtype[np.floating],
) -> None:
    p = evogym_fixture(dtype)
    points = np.array([[-0.4, 0.3], [0.8, -0.2], [0, 0]], dtype=dtype)
    got = jax.jit(evogym_category_scores)(
        {k: jnp.asarray(v) for k, v in p.items()}, jnp.asarray(points)
    )
    expected = []
    for x, y in points:
        nodes = [float(x), float(y), math.sqrt(float(x) ** 2 + float(y) ** 2)] + [0.0] * 6
        for i in range(6):
            z = sum(
                nodes[j] * float(p["weights"][i, j]) * int(p["edge_mask"][i, j]) for j in range(9)
            ) + float(p["bias"][i])
            nodes[3 + i] = 1 / (1 + math.exp(-max(-60, min(60, 5 * z))))
        expected.append(nodes[4:])
    np.testing.assert_allclose(
        got,
        expected,
        rtol=3e-6 if dtype == np.float32 else 2e-14,
        atol=1e-7 if dtype == np.float32 else 2e-15,
    )
    assert p["edge_mask"].sum() == 12


def test_vmap_gradient_matches_individual_rollouts(dtype: np.dtype[np.floating]) -> None:
    cfg = SpringConfig(grid_shape=(1, 1), settle_steps=0, substeps=4, horizon=2)
    e, b = make(dtype=dtype, config=cfg, design_mode="relaxed")
    p = cast_embedding(e.decoder, b)
    r = bind_rollout(e)

    def loss(v: jax.Array, a: jax.Array) -> jax.Array:
        return r({"values": v}, a)["metrics"]["work_signed_j"] * 1e4

    params = jnp.stack([p["values"], p["values"].at[..., 1].add(0.12)])
    actions = jnp.asarray([[[0.2, -0.1]], [[0.15, -0.08]]], dtype=dtype)
    vm = jax.jit(jax.vmap(jax.grad(loss, argnums=(0, 1))))(params, actions)
    single = jax.jit(jax.grad(loss, argnums=(0, 1)))
    for i in range(2):
        a, b = single(params[i], actions[i])
        np.testing.assert_allclose(
            vm[0][i],
            a,
            rtol=2e-3 if dtype == np.float32 else 2e-8,
            atol=2e-5 if dtype == np.float32 else 1e-10,
        )
        np.testing.assert_allclose(
            vm[1][i],
            b,
            rtol=2e-3 if dtype == np.float32 else 2e-8,
            atol=2e-5 if dtype == np.float32 else 1e-10,
        )


def test_material_values_are_not_compilation_constants(dtype: np.dtype[np.floating]) -> None:
    c = SpringConfig(grid_shape=(1, 1), settle_steps=0)
    e, b = make(dtype=dtype, config=c)
    p = cast_embedding(e.decoder, b)
    traces = []

    def f(values: jax.Array) -> jax.Array:
        traces.append(1)
        return e.decoder.decode({"values": values}, e._topology.points).young

    fn = jax.jit(f)
    a = fn(p["values"])
    b = fn(p["values"].at[..., 1].add(0.3))
    jax.block_until_ready(b)
    assert len(traces) == 1 and not np.allclose(a, b)


def test_zero_command_cannot_create_roundoff_work(dtype: np.dtype[np.floating]) -> None:
    c = SpringConfig(grid_shape=(1, 1), settle_steps=0)
    e, b = make(dtype=dtype, config=c)
    e.reset(embodiment=b)
    s = e._state
    m = e._model
    deformed = s.x.at[-1, 0].add(0.01)
    np.testing.assert_array_equal(spring.actuation_work(m, deformed, s.actuator, s.actuator), 0)


@pytest.mark.parametrize("integrator", ["backward_euler", "kvaerno3"])
def test_displacement_uses_integrator_increment(
    dtype: np.dtype[np.floating], integrator: str
) -> None:
    if integrator == "kvaerno3":
        pytest.importorskip(
            "diffrax"
        )  # the kvaerno3 integrator uses diffrax, not a codesign-gym dependency
    c = SpringConfig(
        grid_shape=(1, 1),
        settle_steps=0,
        substeps=4,
        gravity=0,
        ground=False,
        damping_ratio=0,
        integrator=integrator,
    )
    e, b = make(dtype=dtype, config=c)
    e.reset(embodiment=b)
    m, s = e._model, e._state
    speed = jnp.asarray([0.002, 0.003], dtype=dtype)
    s = s._replace(v=jnp.broadcast_to(speed, s.v.shape))
    n, di = jax.jit(lambda s: spring.advance(m, s, jnp.zeros(2, dtype=dtype), c))(s)
    assert int(n.status) == 0
    np.testing.assert_allclose(
        n.com_displacement,
        c.control_dt * speed,
        rtol=4e-5 if dtype == np.float32 else 2e-10,
        atol=2e-9 if dtype == np.float32 else 1e-13,
    )
    np.testing.assert_allclose(
        n.com_displacement,
        spring.com(m, n.x) - spring.com(m, s.x),
        atol=4e-8 if dtype == np.float32 else 1e-14,
    )


def test_precision_request_cannot_silently_truncate(dtype: np.dtype[np.floating]) -> None:
    assert resolve_dtype(dtype) == dtype
    if not jax.config.x64_enabled:
        with pytest.raises(ValueError, match="silent truncation"):
            resolve_dtype("float64")
    with pytest.raises(ValueError):
        resolve_dtype("float16")


def test_rgb_preview_shape_without_gui(dtype: np.dtype[np.floating]) -> None:
    pytest.importorskip("PIL")
    e, b = make(
        dtype=dtype, config=SpringConfig(grid_shape=(1, 1), settle_steps=0), render_mode="rgb_array"
    )
    e.reset(embodiment=b)
    image = e.render()
    assert image.shape == (320, 640, 3) and image.dtype == np.uint8
    e.close()
