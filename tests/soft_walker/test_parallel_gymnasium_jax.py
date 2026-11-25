"""Real JAX batched transitions through the standard Gymnasium contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.gymnasium_adapter import GymnasiumVectorAdapter


from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from envs.soft_walker.engine import make, make_gymnasium_vec
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.spaces import ResetNeeded


def config(**updates: object) -> SpringConfig:
    result = SpringConfig(
        grid_shape=(2, 1),
        size=(0.1, 0.05),
        settle_steps=0,
        substeps=1,
        control_dt=0.00025,
        horizon=3,
        gravity=0.0,
        ground=False,
    )
    return replace(result, **updates)


def factory(
    dtype: np.dtype[np.floating] | type[np.floating], **updates: object
) -> GymnasiumVectorAdapter:
    return make_gymnasium_vec(
        "SoftSpring2D-v0",
        num_envs=2,
        dtype=dtype,
        config=config(**updates),
        max_observation_space_size=128,
    )


@pytest.mark.parametrize(
    ("seed", "mask"),
    [
        ([100, -1], [True, True]),
        ([100, 1.5], [True, True]),
        ([100, "invalid"], [True, True]),
        ([100, -1], [True, False]),
        (-1, [True, True]),
    ],
)
def test_invalid_seed_batch_does_not_reset_any_jax_lane(
    dtype: np.dtype[np.floating], seed: object, mask: list[bool]
) -> None:
    """A malformed later seed cannot reset an earlier lane or consume its RNG.

    Finished episodes retain their pending next-step autoresets after the failed
    request, including when the malformed seed belongs to an unselected lane.
    """
    env = factory(dtype, horizon=1)
    try:
        env.reset(seed=[5, 6])
        terminal = env.step(np.zeros(env.action_space.shape, dtype=dtype))
        assert terminal[3].all()
        lanes = env.inner.envs
        states = [lane._state for lane in lanes]
        models = [lane._model for lane in lanes]
        initial = [lane._initial for lane in lanes]
        embodiments = [lane._embodiment for lane in lanes]
        generators = [lane.np_random for lane in lanes]
        rng_states = [deepcopy(generator.bit_generator.state) for generator in generators]
        steps = [lane._steps for lane in lanes]
        done = [lane._done for lane in lanes]
        with pytest.raises(ValueError):
            env.reset(seed=seed, options={"reset_mask": np.asarray(mask)})
        assert env.np_random_seed == (5, 6)
        assert [lane._steps for lane in lanes] == steps
        assert [lane._done for lane in lanes] == done
        for index, lane in enumerate(lanes):
            assert lane._state is states[index]
            assert lane._model is models[index]
            assert lane._initial is initial[index]
            assert lane._embodiment is embodiments[index]
            assert lane.np_random is generators[index]
            assert lane.np_random.bit_generator.state == rng_states[index]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("seed", "expected"),
    [(31, (31, 32)), (np.int64(31), (31, 32)), ([31, None], (31, 6)), ([31, 47], (31, 47))],
)
def test_jax_vector_seed_expansion_and_optional_lane_seed(
    dtype: np.dtype[np.floating], seed: object, expected: tuple[int, int]
) -> None:
    """Valid scalar expansion and sequence None entries preserve their public meanings."""
    env = factory(dtype)
    try:
        env.reset(seed=[5, 6])
        observation, _ = env.reset(seed=seed)
        assert env.np_random_seed == expected
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_gymnasium_vector_statistics_and_next_step_autoreset(
    dtype: np.dtype[np.floating],
) -> None:
    adapter = factory(dtype, horizon=2)
    env = RecordEpisodeStatistics(adapter)
    try:
        initial, info = env.reset(seed=[4, 9])
        assert adapter.metadata["autoreset_mode"] == AutoresetMode.NEXT_STEP
        assert env.observation_space.contains(initial)
        assert isinstance(info, dict) and initial.shape == (2, 128)
        actions = np.asarray([[0.2, -0.1, 0.8, -0.6], [-0.12, 0.14, -0.7, 0.9]], dtype=dtype)
        first = env.step(actions)
        assert not first[2].any() and not first[3].any()
        terminal = env.step(actions)
        assert not terminal[2].any() and terminal[3].all()
        np.testing.assert_array_equal(terminal[4]["_episode"], [True, True])
        np.testing.assert_array_equal(terminal[4]["episode"]["l"], [2, 2])
        np.testing.assert_allclose(terminal[4]["episode"]["r"], first[1] + terminal[1])
        # A valid nonzero action on the autoreset call must not advance the new
        # episode or affect the recorded length of its next transition.
        reset = env.step(np.full(actions.shape, 0.9, dtype=dtype))
        np.testing.assert_array_equal(reset[0], initial)
        np.testing.assert_array_equal(reset[1], 0.0)
        assert not reset[2].any() and not reset[3].any()
        assert reset[4]["autoreset"].all()
        assert all(lane._steps == 0 for lane in adapter.inner.envs)
        replay = env.step(actions)
        np.testing.assert_allclose(replay[0], first[0], rtol=5e-4, atol=3e-6)
    finally:
        env.close()


def test_jax_partial_reset_preserves_healthy_lane_and_scalar_parity(
    dtype: np.dtype[np.floating],
) -> None:
    env = factory(dtype)
    scalar, _ = make(dtype=dtype, config=config(), max_observation_space_size=128)
    try:
        designs = deepcopy(env.base_embodiment)
        designs["values"][1, ..., 1] += 0.2
        env.reset(seed=[3, 7], options={"embodiment": designs})
        scalar.reset(seed=7, embodiment={key: value[1] for key, value in designs.items()})
        actions = np.asarray([[0.2, -0.1, 0.0, 0.0], [-0.12, 0.16, 0.0, 0.0]], dtype=dtype)
        first = env.step(actions)
        scalar.step(actions[1])
        preserved = env.inner.envs[1]._state
        initial, info = env.reset(options={"reset_mask": np.asarray([True, False])})
        assert env.inner.envs[1]._state is preserved
        assert env.inner.envs[1]._steps == 1 and env.inner.envs[0]._steps == 0
        np.testing.assert_array_equal(initial[1], first[0][1])
        next_step = env.step(actions)
        expected = scalar.step(actions[1])
        np.testing.assert_allclose(next_step[0][1], expected[0], rtol=5e-4, atol=3e-6)
        np.testing.assert_allclose(next_step[1][1], expected[1], rtol=5e-4, atol=3e-6)
        assert not next_step[2].any() and not next_step[3].any()
    finally:
        env.close()
        scalar.close()


def test_jax_real_solver_failure_is_per_lane_and_autoresets(dtype: np.dtype[np.floating]) -> None:
    # One Newton update cannot solve the large actuated change. The other lane
    # remains at exact unloaded equilibrium and should succeed in the same vmap.
    env = factory(dtype, newton_max_steps=1, control_dt=0.02, horizon=4)
    try:
        initial, _ = env.reset(seed=5)
        actions = np.asarray([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=dtype)
        failed = env.step(actions)
        np.testing.assert_array_equal(failed[2], [True, False])
        assert not failed[3].any()
        np.testing.assert_array_equal(failed[4]["valid"], [False, True])
        assert failed[4]["status_name"][0] == "NONLINEAR_SOLVE_FAILED"
        healthy_time = float(env.inner.envs[1]._state.time)
        reset = env.step(actions)
        assert not reset[2].any() and not reset[3].any()
        np.testing.assert_array_equal(reset[4]["autoreset"], [True, False])
        np.testing.assert_array_equal(reset[0][0], initial[0])
        assert reset[1][0] == 0.0
        assert env.inner.envs[0]._steps == 0 and env.inner.envs[1]._steps == 2
        assert float(env.inner.envs[1]._state.time) > healthy_time
    finally:
        env.close()


def test_invalid_actions_do_not_advance_any_jax_lane(dtype: np.dtype[np.floating]) -> None:
    env = factory(dtype)
    try:
        env.reset()
        states = [lane._state for lane in env.inner.envs]
        actions = np.zeros(env.action_space.shape, dtype=dtype)
        actions[1, 0] = np.nan
        with pytest.raises(ValueError):
            env.step(actions)
        integer_actions = np.zeros(env.action_space.shape, dtype=np.int64)
        integer_actions[1, 0] = np.iinfo(np.int64).min
        with pytest.raises(ValueError):
            env.step(integer_actions)
        assert all(lane._state is state for lane, state in zip(env.inner.envs, states))
        assert all(lane._steps == 0 for lane in env.inner.envs)
    finally:
        env.close()


def test_gymnasium_rejected_partial_reset_requires_explicit_recovery(
    dtype: np.dtype[np.floating],
) -> None:
    env = factory(dtype)
    try:
        env.reset()
        healthy = env.inner.envs[1]._state
        invalid = deepcopy(env.base_embodiment)
        invalid["values"][0, ..., 0] = -6.0
        options = {"embodiment": invalid, "reset_mask": np.asarray([True, False])}
        with pytest.raises(ValueError, match="rejected"):
            env.reset(options=options)
        assert env.inner.envs[1]._state is healthy
        with pytest.raises(ResetNeeded):
            env.step(np.zeros(env.action_space.shape, dtype=dtype))
        env.reset(
            options={"embodiment": env.base_embodiment, "reset_mask": np.asarray([True, False])}
        )
        assert env.inner.envs[1]._state is healthy
        assert not env.step(np.zeros(env.action_space.shape, dtype=dtype))[2].any()
    finally:
        env.close()
