"""NeroGraspAlt seeding: reset(seed) seeds the env's own RNG (washer sizes), never the
caller's process-wide random / numpy / torch state."""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

import codesign_gym as cg  # noqa: E402


@pytest.fixture(scope="module")
def thin():
    env, base = cg.make_vec("NeroGrasp-Thin", 2, use_cpu=True)
    yield env
    env.close()


def _washers(env):
    return tuple(np.asarray(a).copy() for a in env._sample_washer_dimensions())


def test_seeded_reset_leaves_global_rngs_alone(thin):
    import torch
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    expected = (random.random(), np.random.rand(), float(torch.rand(1)))
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    thin.reset(seed=7)
    assert (random.random(), np.random.rand(), float(torch.rand(1))) == expected


def test_washer_sizes_follow_the_seed(thin):
    thin.reset(seed=7)
    a = _washers(thin)
    thin.reset(seed=7)
    b = _washers(thin)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    thin.reset(seed=[7, 9])
    c = _washers(thin)
    thin.reset(seed=[7, 9])
    for x, y in zip(c, _washers(thin)):
        np.testing.assert_array_equal(x, y)
    thin.reset(seed=np.int64(7))
    for x, y in zip(a, _washers(thin)):
        np.testing.assert_array_equal(x, y)
