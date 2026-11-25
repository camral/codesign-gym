"""Same fixed API/model, independently selected dense and matrix-free solvers."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as j
import numpy as np
import pytest

from envs.soft_walker.engine import make
from envs.soft_walker.engine.physics import spring
from envs.soft_walker.engine.physics.spring_config import SpringConfig


def test_m2_spring_matrixfree_matches_dense(dtype: np.dtype[np.floating]) -> None:
    from envs.soft_walker.engine.embodiments.decoders import FieldSpec, cast_embedding, make_decoder

    cfg = SpringConfig(
        grid_shape=(2, 1), settle_steps=0, ground=False, substeps=2, control_dt=0.004
    )
    decoder = make_decoder("direct", spec=FieldSpec(design_mode="relaxed"), dtype=dtype)
    e = cast_embedding(decoder, decoder.base())
    topo = spring.topology(cfg, dtype)
    model = spring.instantiate(decoder.decode(e, topo.points), topo, cfg, decoder.spec.channels)
    state = spring.initial_state(model)
    a = j.array([0.2, -0.1], dtype)

    def run(a: jax.Array, c: SpringConfig) -> tuple[jax.Array, spring.State, spring.Diagnostics]:
        s, info = spring.advance(model, state, a, c)
        return j.sum(s.v**2), s, info

    ref = jax.jit(lambda a: run(a, cfg))(a)
    mf = replace(cfg, linear_solver="gmres")
    got = jax.jit(lambda a: run(a, mf))(a)
    assert got[1].status == ref[1].status == 0
    tol = 2e-4 if dtype == np.dtype("float32") else 1e-7
    np.testing.assert_allclose(got[1].v, ref[1].v, rtol=tol, atol=tol * 1e-2)
    ga = jax.jit(jax.grad(lambda a: run(a, cfg)[0]))(a)
    gb = jax.jit(jax.grad(lambda a: run(a, mf)[0]))(a)
    np.testing.assert_allclose(ga, gb, rtol=tol, atol=tol * 1e-4)


def test_m2_native_kvaerno_driver_api(dtype: np.dtype[np.floating]) -> None:
    cfg = SpringConfig(
        grid_shape=(1, 1),
        ground=False,
        settle_steps=0,
        substeps=2,
        control_dt=0.002,
        integrator="kvaerno3_native",
        linear_solver="gmres",
    )
    env, base = make(
        "SoftSpring2D-v0",
        dtype=dtype,
        config=cfg,
        max_action_space_size=2,
        max_observation_space_size=128,
        is_eval_env=False,
    )
    obs, rejected, info = env.reset(embodiment=base)
    assert not rejected
    obs, reward, terminated, truncated, info = env.step(np.array([0.2, -0.1], dtype))
    assert np.isfinite(obs).all() and np.isfinite(reward) and not terminated
    env.close()


def test_m2_no_silent_solver_switch() -> None:
    with pytest.raises(ValueError, match="dense"):
        SpringConfig(integrator="kvaerno3", linear_solver="gmres")
    with pytest.raises(ValueError, match="gmres"):
        SpringConfig(integrator="kvaerno3_native", linear_solver="dense")
    SpringConfig(grid_shape=(10, 10), linear_solver="gmres")
    with pytest.raises(ValueError, match="capacity"):
        SpringConfig(grid_shape=(10, 10), linear_solver="dense")
