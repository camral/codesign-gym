"""Selective native resets retain untouched solver owners and lane lifecycles.

Use WARP_TEST_DEVICE=cpu/cuda:0 and WARP_TEST_DTYPE=float64/float32. The real
physics fixtures keep the configured production tolerances and a small dt.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("warp")
from envs.soft_walker.engine import make, make_gymnasium_vec, make_vec
from envs.soft_walker.engine.spaces import ResetNeeded
from envs.soft_walker.engine.warp_native.model import uniform_body
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

if TYPE_CHECKING:
    from envs.soft_walker.engine.warp_native.host_types import FieldArrays
    from envs.soft_walker.engine.warp_native.model import Bundle
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle
    from envs.soft_walker.engine.warp_native.vector_env import NativeVectorEnv

DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")
DTYPE = os.environ.get("WARP_TEST_DTYPE", "float64")
RTOL, ATOL = (6e-4, 3e-6) if DTYPE == "float32" else (2e-8, 2e-11)
IDS = {
    "spring": "SoftSpring2D-v0",
    "beam": "SoftBeam3D-v0",
    "fem": "SoftHexFEM3D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
}


def _bundle(kind: str) -> Bundle | SolidBundle:
    """Build one tiny actuated body with no settling, gravity or terrain contact."""
    if kind.startswith("mpm"):
        result = uniform_body(int(kind[-1]), particles_per_axis=1, precision=DTYPE)
    else:
        shape = {"spring": (1, 1), "beam": (1, 2, 1), "fem": (1, 1, 1)}[kind]
        result = regular_solid(kind, shape=shape, precision=DTYPE)
    result.config = replace(result.config, dt=1e-4, gravity=0.0, contact_enabled=False)
    return result


def _vector(kind: str = "spring", horizon: int = 8) -> tuple[NativeVectorEnv, FieldArrays]:
    """Construct two independently owned native lanes with padded actions."""
    return make_vec(
        IDS[kind],
        num_envs=2,
        backend="warp",
        bundle=_bundle(kind),
        dtype=DTYPE,
        device=DEVICE,
        max_action_space_size=3,
        horizon=horizon,
    )


@pytest.mark.parametrize("kind", list(IDS))
def test_partial_reset_retains_owner_rng_graph_and_scalar_trajectory(kind: str) -> None:
    vector, fields = _vector(kind)
    scalar, _ = make(
        IDS[kind],
        backend="warp",
        bundle=_bundle(kind),
        dtype=DTYPE,
        device=DEVICE,
        max_action_space_size=3,
        horizon=8,
    )
    try:
        vector.reset(seed=7, embodiment=fields)
        scalar.reset(seed=8, embodiment={key: value[1] for key, value in fields.items()})
        actions = np.asarray([[0.14, -0.07, 0.0], [-0.11, 0.16, 0.0]], dtype=DTYPE)
        first = vector.step(actions)
        scalar.step(actions[1])
        retained = vector.envs[1]
        owner, episode, rng = retained.owner, retained._episode, retained.np_random
        rng_state = deepcopy(rng.bit_generator.state)
        graphs = dict(episode.graphs)
        accepted = deepcopy(vector._embodiments[1])
        changed = deepcopy(fields)
        mass_key = "density" if kind.startswith("mpm") else "mass"
        changed[mass_key][0] *= 1.3
        changed[mass_key][1].fill(-1.0)  # An unselected candidate must not be installed.
        result = vector.reset(
            seed=41, embodiment=changed, options={"reset_mask": np.array([True, False])}
        )
        assert not result[1].any()
        np.testing.assert_array_equal(result[0][1], first[0][1])
        assert vector.envs[0]._steps == 0 and retained._steps == 1
        assert vector.envs[0].np_random_seed == 41 and retained.np_random_seed == 8
        assert retained.owner is owner and retained._episode is episode
        assert retained.np_random is rng and rng.bit_generator.state == rng_state
        assert episode.graphs == graphs
        for key, expected in accepted.items():
            np.testing.assert_array_equal(vector._embodiments[1][key], expected)
        actual, expected = vector.step(actions), scalar.step(actions[1])
        np.testing.assert_allclose(actual[0][1], expected[0], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(actual[1][1], expected[1], rtol=RTOL, atol=ATOL)
        assert actual[2][1] == expected[2] and actual[3][1] == expected[3]
    finally:
        vector.close()
        scalar.close()


@pytest.mark.parametrize(
    "options",
    [
        {"reset_mask": [1, 0]},
        {"reset_mask": [True]},
        {"reset_mask": [[True, False]]},
        {"reset_mask": True},
        {"reset_mask": [True, False], "unknown": 1},
        [True, False],
    ],
)
def test_invalid_mask_is_atomic(options: object) -> None:
    vector, _ = _vector()
    try:
        vector.reset(seed=[2, 9])
        owners = [env.owner for env in vector.envs]
        episodes = [env._episode for env in vector.envs]
        vector._next_reset[:] = [True, False]
        with pytest.raises(ValueError):
            vector.reset(seed=33, options=options)
        np.testing.assert_array_equal(vector._next_reset, [True, False])
        assert all(env.owner is owner for env, owner in zip(vector.envs, owners))
        assert all(env._episode is episode for env, episode in zip(vector.envs, episodes))
        assert [env.np_random_seed for env in vector.envs] == [2, 9]
    finally:
        vector.close()


@pytest.mark.parametrize("seed", [[2], [2, -1], [2, 1.5], -1])
def test_invalid_seed_is_atomic(seed: object) -> None:
    vector, _ = _vector()
    try:
        vector.reset(seed=5)
        owners = [env.owner for env in vector.envs]
        with pytest.raises(ValueError):
            vector.reset(seed=seed, options={"reset_mask": [True, False]})
        assert all(env.owner is owner for env, owner in zip(vector.envs, owners))
        assert [env.np_random_seed for env in vector.envs] == [5, 6]
    finally:
        vector.close()


def test_all_false_mask_does_not_initialize_or_capture() -> None:
    vector, _ = _vector()
    try:
        result = vector.reset(options={"reset_mask": [False, False]})
        assert result[1].all()
        np.testing.assert_array_equal(result[0], 0.0)
        assert all(env.owner is None and env._episode is None for env in vector.envs)
        assert all(
            info["embodiment_errors"] == ["LANE_UNINITIALIZED"] for info in result[2]["per_env"]
        )
        result = vector.reset(seed=[None, 17], options={"reset_mask": [False, True]})
        np.testing.assert_array_equal(result[1], [True, False])
        assert vector.envs[0].owner is None and vector.envs[1].np_random_seed == 17
        owner, episode = vector.envs[1].owner, vector.envs[1]._episode
        assert not episode.graphs
        vector.reset(seed=99, options={"reset_mask": [False, False]})
        assert vector.envs[1].owner is owner and vector.envs[1]._episode is episode
        assert not episode.graphs and vector.envs[1].np_random_seed == 17
        with pytest.raises(ResetNeeded):
            vector.step(np.zeros(vector.action_space.shape, dtype=DTYPE))
    finally:
        vector.close()


def test_unselected_truncated_lane_keeps_pending_autoreset_and_latest_design() -> None:
    vector, fields = _vector(horizon=1)
    try:
        fields["mass"][1] *= 1.7
        initial, rejected, _ = vector.reset(embodiment=fields)
        assert not rejected.any()
        actions = np.full(vector.action_space.shape, 0.12, dtype=DTYPE)
        terminal = vector.step(actions)
        assert terminal[3].all()
        result = vector.reset(options={"reset_mask": [True, False]})
        np.testing.assert_array_equal(vector._next_reset, [False, True])
        np.testing.assert_array_equal(result[0][1], terminal[0][1])
        result = vector.step(actions)
        assert result[3][0] and not result[3][1]
        assert not result[4]["per_env"][0]["autoreset"]
        assert result[4]["per_env"][1]["autoreset"] and result[1][1] == 0
        np.testing.assert_array_equal(result[0][1], initial[1])
        np.testing.assert_array_equal(vector.envs[1].owner.model.mass.numpy(), fields["mass"][1])
    finally:
        vector.close()


def test_unselected_failed_lane_still_autoresets() -> None:
    vector, _ = _vector()
    try:
        vector.reset()
        vector.envs[0].owner.initial.status.fill_(7)
        actions = np.full(vector.action_space.shape, 0.12, dtype=DTYPE)
        failed = vector.step(actions)
        assert failed[2][0] and not failed[2][1]
        owner, episode = vector.envs[0].owner, vector.envs[0]._episode
        result = vector.reset(options={"reset_mask": [False, True]})
        assert not result[1].any()
        np.testing.assert_array_equal(result[0][0], failed[0][0])
        assert vector.envs[0].owner is owner and vector.envs[0]._episode is episode
        np.testing.assert_array_equal(vector._next_reset, [True, False])
        result = vector.step(actions)
        assert result[4]["per_env"][0]["autoreset"] and result[1][0] == 0
        assert not result[2].any() and not result[3].any()
    finally:
        vector.close()


def test_rejected_selected_lane_requires_recovery_without_clearing_other_pending_reset() -> None:
    vector, fields = _vector(horizon=1)
    try:
        vector.reset(embodiment=fields)
        actions = np.full(vector.action_space.shape, 0.12, dtype=DTYPE)
        vector.step(actions)
        bad = deepcopy(fields)
        bad["mass"][0].fill(-1)
        result = vector.reset(embodiment=bad, options={"reset_mask": [True, False]})
        np.testing.assert_array_equal(result[1], [True, False])
        np.testing.assert_array_equal(vector._next_reset, [False, True])
        with pytest.raises(ResetNeeded):
            vector.step(actions)
        result = vector.reset(embodiment=fields, options={"reset_mask": [True, False]})
        assert not result[1].any()
        assert vector.step(actions)[4]["per_env"][1]["autoreset"]
    finally:
        vector.close()


def test_gymnasium_adapter_preserves_unselected_native_lane() -> None:
    pytest.importorskip("gymnasium")
    env = make_gymnasium_vec(
        IDS["spring"],
        num_envs=2,
        backend="warp",
        bundle=_bundle("spring"),
        dtype=DTYPE,
        device=DEVICE,
    )
    try:
        env.reset(seed=5)
        first = env.step(np.full(env.action_space.shape, 0.12, dtype=DTYPE))
        owner = env.inner.envs[1].owner
        result, _ = env.reset(seed=21, options={"reset_mask": [True, False]})
        np.testing.assert_array_equal(result[1], first[0][1])
        assert env.inner.envs[1].owner is owner
        assert env.np_random_seed == (21, 6)
    finally:
        env.close()
