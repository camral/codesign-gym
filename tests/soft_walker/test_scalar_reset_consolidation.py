"""Shared reset lifecycle retains driver-specific failure and randomness contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.physics.terrain import TerrainConfig
from envs.soft_walker.engine.spaces import ResetNeeded


def _environment(kind: str, dtype: np.dtype[np.floating]) -> tuple[Any, Any]:
    """Construct a small real environment; settling failures are injected explicitly."""
    terrain = TerrainConfig(kind="slopes", blocks=0)
    if kind.startswith("spring"):
        return make(
            dtype=dtype,
            config=SpringConfig(
                grid_shape=(2, 1),
                substeps=1,
                settle_steps=0,
                horizon=2,
                terrain=None if kind == "spring_legacy" else terrain,
            ),
        )
    return make(
        "SoftMPM2D-v0",
        dtype=dtype,
        config=BodyConfig(
            grid_shape=(2, 1),
            size=(0.1, 0.05),
            substeps=1,
            settle_steps=0,
            horizon=2,
            terrain=terrain,
        ),
    )


@pytest.mark.parametrize("kind", ["spring_legacy", "spring_terrain", "mpm2"])
def test_reset_seed_consumption_and_invalid_call_are_stable(
    kind: str, dtype: np.dtype[np.floating]
) -> None:
    """Reset draws the documented terrain seed and malformed calls preserve a live episode."""
    env, body = _environment(kind, dtype)
    try:
        expected_rng = np.random.default_rng(41)
        for index in range(2):
            _, rejected, info = env.reset(seed=41 if index == 0 else None, embodiment=body)
            assert not rejected[0]
            if kind == "spring_legacy":
                assert env._terrain_seed is None
            else:
                assert env._terrain_seed == int(expected_rng.integers(0, 2**31))
            assert env.np_random.bit_generator.state == expected_rng.bit_generator.state
            expected_timing = {"prepare_s", "total_reset_s"}
            if kind.startswith("spring"):
                expected_timing.add("settle_and_observe_s")
            assert set(info["timing"]) == expected_timing
        accepted_state = env._state
        rng_state = deepcopy(env.np_random.bit_generator.state)
        for arguments in ({"seed": -1}, {"seed": 1.5}, {"options": {"unknown": True}}):
            with pytest.raises(ValueError):
                env.reset(**arguments)
            assert env._state is accepted_state
            assert env.np_random.bit_generator.state == rng_state
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["spring_terrain", "mpm2"])
def test_failed_settling_invalidates_previous_episode_and_preserves_error_schema(
    kind: str, dtype: np.dtype[np.floating], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real accepted episode cannot survive rejection during the next settling phase."""
    env, body = _environment(kind, dtype)
    try:
        assert not env.reset(seed=19, embodiment=body)[1][0]
        initialize = env._initialize
        failed_state = env._state._replace(status=jnp.asarray(2, jnp.int32))
        monkeypatch.setattr(env, "_initialize", lambda model: failed_state)
        obs, rejected, info = env.reset(seed=23, embodiment=body)
        assert rejected.dtype == np.bool_ and rejected.tolist() == [True]
        assert np.count_nonzero(obs) == 0 and not info["valid"]
        assert env._state is None and env._model is None and env._initial is None
        assert set(info["timing"]) == {"prepare_s", "total_reset_s"}
        expected = "SETTLING_GEOMETRY_INVALID" if kind.startswith("spring") else "SETTLING_FAILED"
        assert info["embodiment_errors"] == (expected,)
        assert ("terrain_seed" in info) == (kind == "mpm2")
        with pytest.raises(ResetNeeded):
            env.step(np.zeros(env.action_space.shape, dtype))
        monkeypatch.setattr(env, "_initialize", initialize)
        assert not env.reset(seed=23, embodiment=body)[1][0]
        saved = np.asarray(env._embodiment["values"]).copy()
        body["values"].flat[0] += 0.25
        np.testing.assert_array_equal(env._embodiment["values"], saved)
    finally:
        env.close()
