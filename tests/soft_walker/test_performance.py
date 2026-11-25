"""Focused regressions for batching and the opt-in GMRES execution method.

No tolerances or physical timesteps are loosened. These tests do not replace
long-horizon gait checks or the deferred full historical suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.types import DriverState


from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import make, make_vec
from envs.soft_walker.engine.batching import reporting_kernel, vector_step_kernel
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.numerics.solvers import NewtonConfig, linear_solve, solve_root
from envs.soft_walker.engine.physics.spring_config import SpringConfig

IDS = [
    "SoftSpring2D-v0",
    "SoftHexFEM3D-v0",
    "SoftMPM2D-v0",
    "SoftMPM3D-v0",
    "SoftBeam3D-v0",
]


def small_profile(env_id: str, *, horizon: int = 3) -> SpringConfig | BodyConfig:
    shape = (2, 1) if env_id in (IDS[0], IDS[2]) else (2, 1, 1)
    args = dict(
        grid_shape=shape,
        size=tuple(0.05 * n for n in shape),
        control_dt=0.00025,
        substeps=1,
        settle_steps=0,
        horizon=horizon,
    )
    if env_id == IDS[0]:
        return SpringConfig(**args)
    return BodyConfig(**args, grid_margin=1)


def assert_tree_close(
    left: object, right: object, dtype: np.dtype[np.floating] | type[np.floating]
) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    rtol, atol = (4e-4, 3e-6) if dtype == np.dtype("float32") else (2e-9, 2e-11)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("env_id", IDS)
def test_performance_batched_api_matches_scalar_and_gradients(
    env_id: str, dtype: np.dtype[np.floating]
) -> None:
    cfg = small_profile(env_id)
    vec, base = make_vec(
        env_id,
        num_envs=2,
        dtype=dtype,
        config=cfg,
        max_observation_space_size=4096,
    )
    try:
        _, rejected, _ = vec.reset(seed=17, embodiment=base)
        assert not rejected.any()
        controls = np.asarray([[0.21, -0.13, 0, 0], [-0.11, 0.17, 0, 0]], dtype)
        before = tuple(e._state for e in vec.envs)
        models = tuple(e._model for e in vec.envs)
        first = vec.envs[0]

        expected = []
        for env, state, action in zip(vec.envs, before, controls):
            new, diag = env._advance(env._model, state, jnp.asarray(action[:2]))
            payload = reporting_kernel(env._observe, env._summary, env._reward)(
                env._model, state, new
            )
            expected.append(jax.device_get((new, diag, payload)))

        obs, rewards, terminated, truncated, info = vec.step(controls)
        assert not terminated.any() and not truncated.any()
        for i, (state, diag, payload) in enumerate(expected):
            assert_tree_close(vec.envs[i]._state, state, dtype)
            np.testing.assert_allclose(obs[i], payload[0], rtol=4e-4, atol=3e-6)
            np.testing.assert_array_equal(info["observation_mask"][i], payload[1])
            np.testing.assert_allclose(rewards[i], payload[3], rtol=4e-4, atol=1e-10)
            assert "batch_synchronized_pipeline_s" in info["timing"]

        kernel = vector_step_kernel(first._advance, first._observe, first._summary, first._reward)

        def feature(state: DriverState) -> jax.Array:
            physical = state if env_id == IDS[0] else state.body
            return 20 * jnp.sum(state.v**2) + 1e5 * physical.work_signed

        def batch_loss(actions: jax.Array) -> jax.Array:
            states, _ = kernel(models, before, actions)
            return sum(feature(state) for state in states)

        def sequential_loss(actions: jax.Array) -> jax.Array:
            return sum(
                feature(first._advance(model, state, action)[0])
                for model, state, action in zip(models, before, actions)
            )

        actions = jnp.asarray(controls[:, :2])
        batched_gradient = jax.jit(jax.grad(batch_loss))(actions)
        scalar_gradient = jax.jit(jax.grad(sequential_loss))(actions)
        assert np.all(np.isfinite(batched_gradient))
        assert_tree_close(batched_gradient, scalar_gradient, dtype)
    finally:
        vec.close()
        # Isolate the five unrelated compiled model families on small hosts.
        # This suite is change-focused; it does not rerun old model sweeps.
        jax.clear_caches()


def test_performance_warm_vector_has_no_eager_jax_stack(
    dtype: np.dtype[np.floating], monkeypatch: pytest.MonkeyPatch
) -> None:
    vec, base = make_vec(num_envs=2, dtype=dtype, config=small_profile(IDS[0]))
    try:
        vec.reset(seed=3, embodiment=base)
        actions = np.zeros((2, 4), dtype)
        vec.step(actions)  # Compile the exact signature.

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("Warm vector step issued an eager jnp.stack")

        monkeypatch.setattr(jnp, "stack", forbidden)
        vec.step(actions)
    finally:
        vec.close()


def test_performance_all_autoreset_skips_dummy_physics(
    dtype: np.dtype[np.floating], monkeypatch: pytest.MonkeyPatch
) -> None:
    import envs.soft_walker.engine.batching as batching

    vec, base = make_vec(num_envs=2, dtype=dtype, config=small_profile(IDS[0], horizon=1))
    try:
        vec.reset(seed=4, embodiment=base)
        actions = np.zeros((2, 4), dtype)
        assert vec.step(actions)[3].all()

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("All-autoreset batch requested a physics kernel")

        monkeypatch.setattr(batching, "vector_step_kernel", forbidden)
        _, reward, terminated, truncated, info = vec.step(actions)
        assert not terminated.any() and not truncated.any()
        assert not reward.any() and info["autoreset"].all()
    finally:
        vec.close()


def test_performance_invalid_batch_is_atomic(dtype: np.dtype[np.floating]) -> None:
    vec, base = make_vec(num_envs=2, dtype=dtype, config=small_profile(IDS[0]))
    try:
        vec.reset(seed=4, embodiment=base)
        before = tuple(e._state for e in vec.envs)
        for value in (np.nan, 2.0):
            actions = np.zeros((2, 4), dtype)
            actions[1, 0] = value
            with pytest.raises(ValueError):
                vec.step(actions)
            assert all(e._state is s for e, s in zip(vec.envs, before))
    finally:
        vec.close()


def test_performance_mixed_autoreset_and_external_lane_reset(
    dtype: np.dtype[np.floating],
) -> None:
    vec, base = make_vec(num_envs=2, dtype=dtype, config=small_profile(IDS[0], horizon=2))
    try:
        vec.reset(seed=4, embodiment=base)
        actions = np.zeros((2, 4), dtype)
        vec.step(actions)
        # A reset through an exposed scalar lane must not leave a stale batch
        # state cache. The new implementation does not maintain such a cache.
        vec.envs[1].reset(seed=9)
        out = vec.step(actions)
        np.testing.assert_array_equal(out[3], [True, False])
        out = vec.step(actions)
        np.testing.assert_array_equal(out[4]["autoreset"], [True, False])
        np.testing.assert_array_equal(out[3], [False, True])
    finally:
        vec.close()


def test_performance_spring_views_share_compilations(dtype: np.dtype[np.floating]) -> None:
    a, _ = make(dtype=dtype, config=small_profile(IDS[0]))
    b, _ = make(dtype=dtype, config=small_profile(IDS[0]))
    try:
        assert a._observe is b._observe
        assert a._summary is b._summary
        assert a._reward is b._reward
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("method", ["incremental", "batched"])
def test_performance_gmres_value_gradient_and_failure(
    method: str, dtype: np.dtype[np.floating]
) -> None:
    cfg = NewtonConfig(gmres_method=method)
    n = 12
    indices = jnp.arange(n, dtype=dtype)
    rhs = jnp.sin(indices + 0.3)
    weight = jnp.cos(indices * 0.2)
    theta = jnp.asarray(0.2, dtype)

    def matvec(t: jax.Array, x: jax.Array) -> jax.Array:
        # Nonsymmetric operator exercises the transpose solve as well.
        diagonal = 2.0 + jnp.exp(t) + 0.03 * indices
        return diagonal * x + 0.17 * jnp.roll(x, 1) - 0.08 * jnp.roll(x, -1)

    def loss(t: jax.Array) -> jax.Array:
        op = lambda x: matvec(t, x)
        x, _ = linear_solve(op, rhs, jnp.ones_like(rhs), cfg)
        return jnp.sum(weight * x**2)

    value, gradient = jax.jit(jax.value_and_grad(loss))(theta)
    eye = np.eye(n)
    matrix = np.column_stack([np.asarray(matvec(theta, jnp.asarray(x, dtype))) for x in eye])
    x = np.linalg.solve(matrix.astype(np.float64), np.asarray(rhs, np.float64))
    dx = np.linalg.solve(matrix.astype(np.float64), -np.exp(float(theta)) * x)
    expected = np.sum(np.asarray(weight) * x**2)
    derivative = 2 * np.sum(np.asarray(weight) * x * dx)
    np.testing.assert_allclose(value, expected, rtol=3e-5, atol=1e-8)
    np.testing.assert_allclose(gradient, derivative, rtol=5e-5, atol=1e-8)

    # Failed Krylov convergence stays visible; opting into a different method
    # must never cause an unchecked or silently dense fallback.
    insufficient = replace(cfg, restart=1, max_restarts=1)
    bad, info = jax.jit(
        lambda b: linear_solve(lambda x: (1 + indices) * x, b, jnp.ones_like(b), insufficient)
    )(jnp.ones(n, dtype))
    assert float(info[1]) == 0 and np.isnan(bad).all()


@pytest.mark.parametrize("method", ["incremental", "batched"])
def test_performance_gmres_zero_rhs_and_nonlinear_root(
    method: str, dtype: np.dtype[np.floating]
) -> None:
    cfg = NewtonConfig(gmres_method=method)
    z, info = jax.jit(lambda b: linear_solve(lambda x: 2 * x, b, jnp.ones_like(b), cfg))(
        jnp.zeros(5, dtype)
    )
    np.testing.assert_array_equal(z, np.zeros(5))
    assert float(info[1]) == 1

    def root(p: jax.Array) -> jax.Array:
        return solve_root(lambda x: x + 0.1 * x**3 - p, p, cfg)[0]

    p = jnp.asarray([0.2, 0.5, 0.8], dtype)
    x, grad = jax.jit(jax.value_and_grad(lambda p: jnp.sum(root(p))))(p)
    r = jax.jit(root)(p)
    np.testing.assert_allclose(grad, 1 / (1 + 0.3 * np.asarray(r) ** 2), rtol=4e-5, atol=1e-7)


def test_performance_invalid_gmres_method() -> None:
    with pytest.raises(ValueError, match="gmres_method"):
        NewtonConfig(gmres_method="automatic_silent_fallback")
