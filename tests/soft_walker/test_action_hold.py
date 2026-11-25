"""Action hold changes API cadence, preserving physics dt and reset settling.

    CODESIGN_TEST_DEVICE=cpu/cuda:0 CODESIGN_TEST_DTYPE=float64/float32

The numerical k=10 checks use identical commands at the same physical times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.drivers.environment import BodyEnv
    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.engine.warp_native.env import NativeEnv


import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import make, make_gymnasium_vec, make_vec
from envs.soft_walker.engine.action_timing import native_action_hold, with_action_hold
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact
from envs.soft_walker.engine.physics.spring_config import SpringConfig

DEVICE = os.environ.get("CODESIGN_TEST_DEVICE", os.environ.get("WARP_TEST_DEVICE", "cpu"))
DT = 1e-4
IDS = {
    "spring": "SoftSpring2D-v0",
    "beam": "SoftBeam3D-v0",
    "fem": "SoftHexFEM3D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
}


def jax_config(
    kind: str, *, settling: int = 0, horizon: int = 3, substeps: int = 1
) -> SpringConfig | BodyConfig:
    """Keep the integration step at DT while varying control and settling budgets."""
    if kind == "spring":
        return SpringConfig(
            grid_shape=(2, 1),
            size=(0.1, 0.05),
            control_dt=DT * substeps,
            substeps=substeps,
            settle_steps=settling,
            horizon=horizon,
            gravity=0.5,
            ground=False,
        )
    shape = (2, 1) if kind == "mpm2" else ((1, 2, 1) if kind == "beam" else (1, 1, 1))
    return BodyConfig(
        grid_shape=shape,
        size=tuple(0.05 * n for n in shape),
        control_dt=DT * substeps,
        substeps=substeps,
        settle_steps=settling,
        horizon=horizon,
        particles_per_axis=2,
        dynamics=DynamicsConfig(gravity=0.5, max_halvings=0, contact=PlaneContact(enabled=False)),
    )


def options(
    backend: str,
    kind: str,
    dtype: np.dtype[np.floating] | type[np.floating],
    *,
    settling: int = 0,
    horizon: int = 3,
) -> dict[str, object]:
    """Build comparable tiny JAX or native bodies without changing solver tolerances."""
    common = dict(backend=backend, dtype=dtype, device=DEVICE)
    if backend == "jax":
        return dict(common, config=jax_config(kind, settling=settling, horizon=horizon))
    pytest.importorskip("warp")
    from envs.soft_walker.engine.warp_native.model import uniform_body
    from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

    if kind.startswith("mpm"):
        bundle = uniform_body(int(kind[-1]), precision=str(np.dtype(dtype)))
    else:
        shape = {"spring": (1, 1), "beam": (1, 2, 1), "fem": (1, 1, 1)}[kind]
        bundle = regular_solid(kind, shape=shape, precision=str(np.dtype(dtype)))
    bundle.config = replace(bundle.config, dt=DT, gravity=0.5, contact_enabled=False)
    return dict(common, bundle=bundle, settling_steps=settling, horizon=horizon)


def state_arrays(env: Spring2DEnv | BodyEnv | NativeEnv, backend: str) -> list[np.ndarray]:
    """Copy complete physical state to host arrays at a shared action boundary."""
    if backend == "jax":
        return [np.asarray(array) for array in jax.tree.leaves(env._state)]
    fields = (
        ("x", "v", "F", "C", "actuator", "ledger", "status")
        if env.kind.startswith("mpm")
        else ("u", "v", "rotation", "omega", "actuator", "ledger", "status")
    )
    return [getattr(env.owner.initial, field).numpy().copy() for field in fields]


def assert_states_equal(
    actual: list[np.ndarray],
    expected: list[np.ndarray],
    dtype: np.dtype[np.floating] | type[np.floating],
) -> None:
    """Compare every physical state leaf using the suite's established tolerances."""
    rtol, atol = (8e-4, 4e-6) if np.dtype(dtype) == np.float32 else (3e-8, 2e-10)
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)


@pytest.mark.parametrize("bad", [0, -1, True, False, 10.0, np.int64(10), "10", np.nan])
def test_action_hold_requires_positive_python_integer(bad: object) -> None:
    config = jax_config("spring")
    for factory in (
        lambda: with_action_hold(config, bad),
        lambda: native_action_hold(physics_steps_per_action=bad),
        lambda: native_action_hold(substeps=bad),
    ):
        with pytest.raises(ValueError, match="positive Python integer"):
            factory()


def test_action_hold_defaults_and_explicit_native_alias_conflicts() -> None:
    config = jax_config("spring", substeps=3, settling=2)
    assert with_action_hold(config) is config
    assert with_action_hold(config, 3) is config
    held = with_action_hold(config, 10)
    assert held.substeps == 10 and held.horizon == config.horizon
    assert held.settle_steps == config.settle_steps
    assert held.dt == pytest.approx(config.dt, rel=1e-15)
    assert held.control_dt == pytest.approx(10 * config.dt, rel=1e-15)
    assert native_action_hold() == 1
    assert native_action_hold(substeps=3) == 3
    assert native_action_hold(physics_steps_per_action=10) == 10
    assert native_action_hold(substeps=10, physics_steps_per_action=10) == 10
    with pytest.raises(ValueError, match="must agree"):
        native_action_hold(substeps=1, physics_steps_per_action=10)


@pytest.mark.parametrize("backend", ["jax", "warp"])
@pytest.mark.parametrize("kind", list(IDS))
def test_all_driver_action_hold_configuration_wiring(
    backend: str, kind: str, dtype: np.dtype[np.floating]
) -> None:
    env, _ = make(
        IDS[kind], physics_steps_per_action=10, **options(backend, kind, dtype, settling=2)
    )
    try:
        assert env.physics_steps_per_action == 10
        assert env.physics_dt == pytest.approx(DT, rel=1e-15)
        assert env.control_dt == pytest.approx(10 * DT, rel=1e-15)
        if backend == "jax":
            assert env.config.horizon == 3
            assert env.config.substeps == 10
            assert env._initialization_config.substeps == 1
            assert env._initialization_config.control_dt == DT
            assert env._initialization_config.settle_steps == 2
        else:
            assert env.substeps == 10 and env.horizon == 3
            assert env.settling_steps == 2 and env.template.config.dt == DT
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["jax", "warp"])
def test_vector_and_gymnasium_publish_action_timing(
    backend: str, dtype: np.dtype[np.floating]
) -> None:
    for factory in (make_vec, make_gymnasium_vec):
        created = factory(
            IDS["spring"],
            num_envs=2,
            physics_steps_per_action=10,
            **options(backend, "spring", dtype),
        )
        env = created[0] if factory is make_vec else created
        try:
            assert env.physics_steps_per_action == 10
            assert env.physics_dt == pytest.approx(DT, rel=1e-15)
            assert env.control_dt == pytest.approx(10 * DT, rel=1e-15)
        finally:
            env.close()


@pytest.mark.parametrize("backend", ["jax", "warp"])
@pytest.mark.parametrize("kind", ["spring", "mpm2"])
@pytest.mark.parametrize("settling", [0, 2])
def test_k10_matches_ten_k1_transitions(
    backend: str, kind: str, settling: int, dtype: np.dtype[np.floating]
) -> None:
    held, _ = make(
        IDS[kind],
        physics_steps_per_action=10,
        **options(backend, kind, dtype, settling=settling, horizon=3),
    )
    single, _ = make(
        IDS[kind],
        physics_steps_per_action=1,
        **options(backend, kind, dtype, settling=settling, horizon=30),
    )
    try:
        held_initial, held_reject, _ = held.reset(seed=4)
        single_initial, single_reject, _ = single.reset(seed=4)
        assert not held_reject[0] and not single_reject[0]
        # Gravity makes nonzero settling observable: both reset velocities and
        # positions must agree despite the different policy-action cadence.
        assert_states_equal(state_arrays(held, backend), state_arrays(single, backend), dtype)
        np.testing.assert_allclose(held_initial, single_initial, rtol=8e-4, atol=4e-6)
        action = np.zeros(held.action_space.shape, dtype=dtype)
        action[: held.channels] = np.linspace(0.16, -0.08, held.channels)
        result = held.step(action)
        rewards = []
        for _ in range(10):
            reference = single.step(action)
            assert not reference[2] and not reference[3]
            rewards.append(reference[1])
        assert not result[2] and not result[3]
        assert_states_equal(state_arrays(held, backend), state_arrays(single, backend), dtype)
        np.testing.assert_allclose(result[0], reference[0], rtol=8e-4, atol=4e-6)
        np.testing.assert_allclose(
            result[1],
            sum(rewards),
            rtol=8e-4,
            atol=4e-7 if np.dtype(dtype) == np.float32 else 2e-11,
        )
        assert held._steps == 1 and single._steps == 10
        if backend == "jax":
            assert float(held._state.time) == pytest.approx(10 * DT, rel=3e-6)
        else:
            assert held._episode.steps == 10
    finally:
        held.close()
        single.close()


@pytest.mark.parametrize("kind,mode", [("spring", "open"), ("spring", "policy"), ("mpm2", "open")])
def test_k10_pure_rollout_matches_api_with_original_settling(
    kind: str, mode: str, dtype: np.dtype[np.floating]
) -> None:
    from envs.soft_walker.engine.differentiation import bind_policy, bind_rollout

    env, design = make(
        IDS[kind], physics_steps_per_action=10, **options("jax", kind, dtype, settling=2, horizon=2)
    )
    try:
        assert not env.reset(seed=8)[1][0]
        action = np.zeros(env.action_space.shape, dtype=dtype)
        action[: env.channels] = np.linspace(0.13, -0.07, env.channels)
        actions = np.stack([action, action])
        if mode == "open":
            function = bind_rollout(env)
            result = jax.jit(function)(jax.tree.map(jnp.asarray, design), jnp.asarray(actions))
        else:

            def policy(
                parameters: jax.Array, observation: jax.Array, carry: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                return parameters, carry

            function = bind_policy(env, policy, steps=2)
            result = jax.jit(function)(jax.tree.map(jnp.asarray, design), jnp.asarray(action), ())
        result = jax.block_until_ready(result)
        for action in actions:
            observed = env.step(action)
            assert not observed[2]
        assert observed[3]
        assert_states_equal(jax.tree.leaves(result["state"]), state_arrays(env, "jax"), dtype)
    finally:
        env.close()


def test_k10_resident_initializer_preserves_original_settling(
    dtype: np.dtype[np.floating],
) -> None:
    from envs.soft_walker.engine.execution.program import prepare

    env, design = make(
        IDS["spring"], physics_steps_per_action=10, **options("jax", "spring", dtype, settling=2)
    )
    try:
        assert not env.reset(seed=8)[1][0]
        _, initial = jax.jit(lambda embedding: prepare(env, embedding))(
            jax.tree.map(jnp.asarray, design)
        )
        initial = jax.block_until_ready(initial)
        assert_states_equal(jax.tree.leaves(initial), state_arrays(env, "jax"), dtype)
    finally:
        env.close()
