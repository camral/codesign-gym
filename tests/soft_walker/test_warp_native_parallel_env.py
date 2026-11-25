"""Real native lane isolation and scalar parity, including CUDA graph replay.

Select WARP_TEST_DEVICE=cpu/cuda:0 and WARP_TEST_DTYPE=float64/float32.
These use tiny actuated bodies and the production convergence tolerances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.warp_native.model import Bundle
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle


import os
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from envs.soft_walker.engine import make, make_vec
from envs.soft_walker.engine.spaces import ResetNeeded
from envs.soft_walker.engine.warp_native.model import uniform_body
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

DTYPE = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")
IDS = {
    "spring": "SoftSpring2D-v0",
    "beam": "SoftBeam3D-v0",
    "fem": "SoftHexFEM3D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
}
RTOL, ATOL = (6e-4, 3e-6) if DTYPE == "float32" else (2e-8, 2e-11)


def tiny_bundle(kind: str) -> Bundle | SolidBundle:
    """Create actuated fixtures with fixed integration dt and no gravity/contact forces."""
    if kind.startswith("mpm"):
        body = uniform_body(int(kind[-1]), particles_per_axis=1, precision=DTYPE)
    else:
        shape = {"spring": (1, 1), "beam": (1, 2, 1), "fem": (1, 1, 1)}[kind]
        body = regular_solid(kind, shape=shape, precision=DTYPE)
    body.config = replace(body.config, dt=1e-4, gravity=0.0, contact_enabled=False)
    return body


def options(kind: str, **updates: object) -> dict[str, object]:
    result = dict(
        backend="warp",
        bundle=tiny_bundle(kind),
        dtype=DTYPE,
        device=DEVICE,
        max_action_space_size=3,
        horizon=8,
    )
    result.update(updates)
    return result


def check_step(actual: tuple[object, ...], expected: tuple[object, ...]) -> None:
    """Compare observations, rewards, termination, and physical metrics per lane."""
    np.testing.assert_allclose(actual[0], expected[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(actual[1], expected[1], rtol=RTOL, atol=ATOL)
    assert actual[2:4] == expected[2:4]
    assert actual[4]["valid"] == expected[4]["valid"]
    for key in ("displacement", "positive_work"):
        np.testing.assert_allclose(
            actual[4]["metrics"][key], expected[4]["metrics"][key], rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("kind", list(IDS))
def test_vector_lanes_match_independent_scalars_across_reset(kind: str) -> None:
    vector, fields = make_vec(IDS[kind], num_envs=2, execution="auto", **options(kind))
    scalars = [make(IDS[kind], **options(kind))[0] for _ in range(2)]
    try:
        # Same capacities, different physical values; each lane must use its
        # own model and command buffers even after graph capture and reset.
        fields["density" if kind.startswith("mpm") else "mass"][1] *= 1.35
        if "young" in fields and kind.startswith("mpm"):
            fields["young"][1] *= 0.8
        for generation in range(2):
            if generation:
                fields["density" if kind.startswith("mpm") else "mass"][0] *= 0.7
            observations, rejected, _ = vector.reset(embodiment=fields)
            assert not rejected.any()
            for lane, scalar in enumerate(scalars):
                observation, reject, _ = scalar.reset(
                    embodiment={key: value[lane].copy() for key, value in fields.items()}
                )
                assert not reject[0]
                np.testing.assert_array_equal(observations[lane], observation)
            for tick in range(2):
                actions = np.asarray(
                    [[0.24 - 0.03 * tick, -0.17, 0.9], [-0.19 + 0.02 * tick, 0.13, -0.7]],
                    dtype=DTYPE,
                )
                result = vector.step(actions)
                assert not result[2].any()
                for lane, scalar in enumerate(scalars):
                    actual = (*[value[lane] for value in result[:4]], result[4]["per_env"][lane])
                    check_step(actual, scalar.step(actions[lane]))
                    assert not actual[4]["autoreset"]
            assert not np.allclose(
                vector.envs[0].owner.initial.actuator.numpy(),
                vector.envs[1].owner.initial.actuator.numpy(),
                atol=1e-7,
            )
    finally:
        vector.close()
        for scalar in scalars:
            scalar.close()


def test_masked_actions_and_invalid_batch_are_atomic() -> None:
    vector, fields = make_vec(IDS["spring"], num_envs=2, execution="auto", **options("spring"))
    scalar, _ = make(IDS["spring"], **options("spring"))
    try:
        vector.reset(embodiment=fields)
        scalar.reset()
        snapshots = [lane._observation(lane.owner.initial) for lane in vector.envs]
        for invalid in (np.nan, np.inf, 1.01):
            actions = np.zeros(vector.action_space.shape, dtype=DTYPE)
            actions[-1, 0] = invalid
            with pytest.raises(ValueError):
                vector.step(actions)
            assert all(lane._steps == 0 for lane in vector.envs)
            for lane, snapshot in zip(vector.envs, snapshots):
                np.testing.assert_array_equal(lane._observation(lane.owner.initial), snapshot)
        integer_actions = np.zeros(vector.action_space.shape, dtype=np.int64)
        integer_actions[-1, 0] = np.iinfo(np.int64).min
        with pytest.raises(ValueError):
            vector.step(integer_actions)
        assert all(lane._steps == 0 for lane in vector.envs)
        # The single spring cell only uses channel 0. Both the unused physical
        # channel and the padded channel must be masked before simulation.
        actions = np.asarray([[0.2, 1.0, -1.0], [0.2, -0.9, 0.8]], dtype=DTYPE)
        result = vector.step(actions)
        reference = scalar.step(np.asarray([0.2, 0.0, 0.0], dtype=DTYPE))
        for lane in range(2):
            check_step((*[v[lane] for v in result[:4]], result[4]["per_env"][lane]), reference)
    finally:
        vector.close()
        scalar.close()


def test_numerical_failure_isolated_and_next_step_resets_only_failed_lane() -> None:
    vector, _ = make_vec(IDS["spring"], num_envs=2, execution="auto", **options("spring"))
    scalar, _ = make(IDS["spring"], **options("spring"))
    try:
        vector.reset()
        scalar.reset()
        actions = np.full(vector.action_space.shape, 0.12, dtype=DTYPE)
        # Inject a solver failure status, without changing another lane's
        # model, to verify the production early-exit/next-reset path.
        vector.envs[0].owner.initial.status.fill_(7)
        result = vector.step(actions)
        np.testing.assert_array_equal(result[2], [True, False])
        assert not result[3].any() and result[1][0] == 0.0
        check_step((*[v[1] for v in result[:4]], result[4]["per_env"][1]), scalar.step(actions[1]))
        result = vector.step(actions)
        assert not result[2].any() and not result[3].any()
        assert result[1][0] == 0.0 and result[4]["per_env"][0]["autoreset"]
        assert not result[4]["per_env"][1]["autoreset"]
        assert vector.envs[0]._steps == 0 and vector.envs[1]._steps == 2
        check_step((*[v[1] for v in result[:4]], result[4]["per_env"][1]), scalar.step(actions[1]))
        assert not vector.step(actions)[2].any()
    finally:
        vector.close()
        scalar.close()


def test_horizon_autoreset_uses_latest_accepted_embodiment() -> None:
    vector, fields = make_vec(
        IDS["spring"], num_envs=2, execution="auto", **options("spring", horizon=1)
    )
    try:
        fields["mass"][1] *= 1.7
        reset_observation, rejected, _ = vector.reset(embodiment=fields)
        assert not rejected.any()
        actions = np.full(vector.action_space.shape, 0.16, dtype=DTYPE)
        first = vector.step(actions)
        assert first[3].all() and not first[2].any()
        reset = vector.step(actions)
        np.testing.assert_array_equal(reset[0], reset_observation)
        np.testing.assert_array_equal(reset[1], 0.0)
        assert not reset[2].any() and not reset[3].any()
        assert all(info["autoreset"] for info in reset[4]["per_env"])
        for lane, env in enumerate(vector.envs):
            np.testing.assert_array_equal(env.owner.model.mass.numpy(), fields["mass"][lane])
        replay = vector.step(actions)
        np.testing.assert_allclose(replay[0], first[0], rtol=RTOL, atol=ATOL)
    finally:
        vector.close()


def test_rejected_design_requires_explicit_reset_and_never_advances_good_lane() -> None:
    vector, fields = make_vec(IDS["spring"], num_envs=2, execution="auto", **options("spring"))
    try:
        vector.reset()
        bad = deepcopy(fields)
        bad["mass"][0].fill(-1.0)
        _, rejected, _ = vector.reset(embodiment=bad)
        np.testing.assert_array_equal(rejected, [True, False])
        with pytest.raises(ResetNeeded):
            vector.step(np.zeros(vector.action_space.shape, dtype=DTYPE))
        assert vector.envs[1]._steps == 0
        assert not vector.reset(embodiment=fields)[1].any()
    finally:
        vector.close()


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_explicit_cuda_execution_modes_match_scalar(mode: str) -> None:
    if not DEVICE.startswith("cuda"):
        pytest.skip("Explicit CUDA concurrency check; CPU auto mode is covered above")
    vector, _ = make_vec(IDS["spring"], num_envs=2, execution=mode, **options("spring"))
    scalar, _ = make(IDS["spring"], **options("spring"))
    try:
        vector.reset()
        scalar.reset()
        actions = np.full(vector.action_space.shape, 0.18, dtype=DTYPE)
        result = vector.step(actions)
        reference = scalar.step(actions[0])
        for lane in range(2):
            check_step((*[v[lane] for v in result[:4]], result[4]["per_env"][lane]), reference)
    finally:
        vector.close()
        scalar.close()


def test_explicit_parallel_cpu_is_rejected() -> None:
    with pytest.raises(ValueError, match="CUDA|cuda|parallel"):
        make_vec(IDS["spring"], num_envs=2, execution="parallel", **options("spring", device="cpu"))


def test_native_gymnasium_vector_wrapper_contract() -> None:
    pytest.importorskip("gymnasium")
    from gymnasium.vector import AutoresetMode
    from gymnasium.wrappers.vector import RecordEpisodeStatistics

    from envs.soft_walker.engine import make_gymnasium_vec

    adapter = make_gymnasium_vec(
        IDS["spring"], num_envs=2, execution="auto", **options("spring", horizon=1)
    )
    env = RecordEpisodeStatistics(adapter)
    try:
        initial, info = env.reset(seed=4)
        assert adapter.metadata["autoreset_mode"] == AutoresetMode.NEXT_STEP
        assert env.observation_space.contains(initial)
        result = env.step(np.full(env.action_space.shape, 0.12, dtype=DTYPE))
        assert not result[2].any() and result[3].all()
        assert result[4]["_episode"].all()
        np.testing.assert_array_equal(result[4]["episode"]["l"], [1, 1])
        assert result[4]["_valid"].all() and result[4]["valid"].all()
        reset = env.step(np.zeros(env.action_space.shape, dtype=DTYPE))
        np.testing.assert_array_equal(reset[0], initial)
        assert reset[4]["autoreset"].all() and reset[4]["_autoreset"].all()
    finally:
        env.close()
