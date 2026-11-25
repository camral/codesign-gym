"""Autoreset restores private device state without rebuilding physics or graphs."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("warp")

from envs.soft_walker.engine.spaces import ResetNeeded
from envs.soft_walker.engine.warp_native.env import NativeEnv
from envs.soft_walker.engine.warp_native.model import uniform_body
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

if TYPE_CHECKING:
    from envs.soft_walker.engine.warp_native.model import Bundle
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle

DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")
DTYPE = os.environ.get("WARP_TEST_DTYPE", "float64")
IDS = {
    "spring": "SoftSpring2D-v0",
    "beam": "SoftBeam3D-v0",
    "fem": "SoftHexFEM3D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
}


def lane(kind: str) -> NativeEnv:
    """Build a tiny real device owner without settling or stepping the solver."""
    body: Bundle | SolidBundle
    if kind.startswith("mpm"):
        body = uniform_body(int(kind[-1]), particles_per_axis=1, precision=DTYPE)
    else:
        shape = {"spring": (1, 1), "beam": (1, 2, 1), "fem": (1, 1, 1)}[kind]
        body = regular_solid(kind, shape=shape, precision=DTYPE)
    return NativeEnv(IDS[kind], bundle=body, dtype=DTYPE, device=DEVICE)


@pytest.mark.parametrize("kind", IDS)
def test_autoreset_restores_every_state_buffer_and_preserves_owner(kind: str) -> None:
    env = lane(kind)
    try:
        initial, rejected, initial_info = env.reset(seed=14)
        assert not rejected.any() and env._reset_state is not None
        owner, episode = env.owner, env._episode
        expected = {
            name: getattr(owner.initial, name).numpy().copy() for name in env._state_fields()
        }
        for name, saved in env._reset_state.buffers:
            current = getattr(owner.initial, name)
            assert saved.ptr != current.ptr
            current.fill_(1)
        env._steps, env._travel = env.horizon, 17.0
        env._last_work, env._work_origin = 31.0, 4.0
        env.initialized = False  # Numerical termination is also eligible for autoreset.
        restored, rejected, info = env._autoreset()
        np.testing.assert_array_equal(restored, initial)
        assert not rejected.any() and env.initialized
        assert env.owner is owner and env._episode is episode
        assert env._steps == 0 and env._travel == 0.0
        assert info["metrics"] == initial_info["metrics"]
        np.testing.assert_array_equal(info["objectives"], initial_info["objectives"])
        for name, value in expected.items():
            np.testing.assert_array_equal(getattr(owner.initial, name).numpy(), value)
    finally:
        env.close()
    assert env._reset_state is None
    with pytest.raises(ResetNeeded):
        env._autoreset()


@pytest.mark.parametrize("kind", IDS)
def test_explicit_design_reset_replaces_snapshot_and_rejection_invalidates_it(kind: str) -> None:
    env = lane(kind)
    try:
        env.reset(seed=3)
        old_owner, old_snapshot = env.owner, env._reset_state
        changed = deepcopy(env._embodiment)
        mass_key = "density" if kind.startswith("mpm") else "mass"
        changed[mass_key] *= 1.2
        initial, rejected, _ = env.reset(embodiment=changed)
        assert not rejected.any()
        assert env.owner is not old_owner and env._reset_state is not old_snapshot
        np.testing.assert_array_equal(env._embodiment[mass_key], changed[mass_key])
        np.testing.assert_array_equal(env._autoreset()[0], initial)
        changed[mass_key].fill(-1)
        _, rejected, _ = env.reset(embodiment=changed)
        assert rejected.all() and env._reset_state is None
        with pytest.raises(ResetNeeded):
            env._autoreset()
    finally:
        env.close()
