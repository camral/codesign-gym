"""The standard adapter exposes the RNGs of the actual physics lanes."""

from __future__ import annotations

import numpy as np
import pytest

from envs.soft_walker.engine import make_gymnasium_vec
from envs.soft_walker.engine.physics.spring_config import SpringConfig


@pytest.mark.parametrize("backend", ["jax", "warp"])
def test_adapter_reports_and_preserves_lane_seeds(
    backend: str, dtype: np.dtype[np.floating]
) -> None:
    if backend == "warp":
        pytest.importorskip("warp")
        from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

        kwargs = {"bundle": regular_solid("spring", shape=(1, 1), precision=str(dtype))}
    else:
        kwargs = {
            "config": SpringConfig(
                grid_shape=(2, 1),
                size=(0.1, 0.05),
                settle_steps=0,
                substeps=1,
                gravity=0.0,
                ground=False,
            )
        }
    env = make_gymnasium_vec(
        num_envs=2,
        backend=backend,
        device="cpu",
        dtype=dtype,
        **kwargs,
    )
    try:
        env.reset(seed=12)
        assert env.np_random_seed == (12, 13)
        rngs = env.np_random
        env.reset(seed=None)
        assert all(a is b for a, b in zip(rngs, env.np_random))
        assert env.np_random_seed == (12, 13)
        env.reset(seed=[21, 37])
        assert env.np_random_seed == (21, 37)
        for rng, seed in zip(env.np_random, (21, 37)):
            np.testing.assert_array_equal(rng.random(4), np.random.default_rng(seed).random(4))
    finally:
        env.close()
