"""Canonical package boundaries around native vector coordination."""

from __future__ import annotations

import pickle
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("warp")

from envs.soft_walker.engine.warp_native import env, vector_env
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.solid_model import SolidBundle


@pytest.mark.parametrize("name", ("NativeVectorEnv", "make_vec", "step_lanes"))
def test_native_vector_objects_use_their_defining_module(name: str) -> None:
    """New class/function references round-trip through the vector module."""
    value = getattr(vector_env, name)
    assert value.__module__ == vector_env.__name__
    assert pickle.loads(pickle.dumps(value)) is value
    assert not hasattr(env, name)


def test_scalar_native_api_retains_its_own_factory_and_type() -> None:
    """Removing vector aliases preserves the scalar module's public entrypoints."""
    assert env.NativeEnv.__module__ == env.__name__
    assert env.make.__module__ == env.__name__


def test_engine_vector_factory_dispatches_to_the_canonical_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine API reaches canonical vector dispatch without creating devices."""
    from envs.soft_walker import engine

    result = object()
    calls = []

    def factory(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(vector_env, "make_vec", factory)
    assert engine.make_vec("SoftSpring2D-v0", num_envs=3, backend="warp") is result
    assert calls == [(("SoftSpring2D-v0",), {"num_envs": 3, "backend": "warp"})]


def test_native_vector_constructor_keeps_independent_design_ownership() -> None:
    """Extraction must preserve copies without initializing device physics."""
    bundle = regular_solid("spring", shape=(1, 1))
    original = deepcopy(bundle.fields)
    vector, base = vector_env.make_vec(
        "SoftSpring2D-v0", num_envs=2, bundle=bundle, execution="serial", device="cpu"
    )
    try:
        base["mass"][0] *= 2
        np.testing.assert_array_equal(bundle.fields["mass"], original["mass"])
        np.testing.assert_array_equal(vector.envs[0]._base["mass"], original["mass"])
        assert not np.shares_memory(vector.envs[0]._base["mass"], vector.envs[1]._base["mass"])
        assert all(isinstance(lane.template, SolidBundle) for lane in vector.envs)
    finally:
        vector.close()
