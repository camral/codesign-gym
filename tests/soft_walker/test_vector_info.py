"""Gymnasium-compatible nested info, including mixed autoreset lanes."""

from __future__ import annotations

import numpy as np
import pytest

from envs.soft_walker.engine.drivers.vector_lifecycle import _stack_info


def _assert_same(actual: object, expected: object) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, dict):
            assert isinstance(actual[key], dict)
            _assert_same(actual[key], value)
        else:
            assert actual[key].dtype == value.dtype
            assert actual[key].shape == value.shape
            if value.dtype == object:
                for got, want in zip(actual[key], value):
                    if isinstance(want, np.ndarray):
                        np.testing.assert_array_equal(got, want)
                    else:
                        assert got == want
            else:
                np.testing.assert_array_equal(actual[key], value)


@pytest.mark.parametrize("reverse", [False, True])
def test_info_matches_gymnasium_for_missing_nested_fields_and_present_none(reverse: bool) -> None:
    gym = pytest.importorskip("gymnasium")
    infos = [
        {
            "autoreset": True,
            "timing": {"prepare_s": 0.02, "total_reset_s": 0.03},
            "warning": None,
            "empty": {},
            "final_obs": np.asarray([1.0, 2.0]),
        },
        {
            "autoreset": False,
            "timing": {"batch_synchronized_pipeline_s": 0.01},
            "solver": {"iterations": 3.0, "details": {"residual": np.float32(0.001)}},
            "mask": np.asarray([True, False]),
            "warning": None,
            "final_obs": np.asarray([3.0, 4.0, 5.0]),
        },
        {
            "autoreset": False,
            "solver": {"iterations": 4.0, "details": {"condition": "accepted"}},
            "mask": np.asarray([False, True]),
        },
    ]
    if reverse:
        infos.reverse()
    reference = gym.vector.VectorEnv()
    reference.num_envs = len(infos)
    expected = {}
    for lane, info in enumerate(infos):
        expected = reference._add_info(expected, info, lane)
    result = _stack_info(infos)
    _assert_same(result, expected)
    assert isinstance(result["solver"], dict)
    np.testing.assert_array_equal(
        result["_warning"], [True, True, False][::-1] if reverse else [True, True, False]
    )
    np.testing.assert_array_equal(result["solver"]["_details"], result["_solver"])


def test_empty_info_batch_and_explicit_none_presence() -> None:
    assert _stack_info([]) == {}
    result = _stack_info([{"value": None}, {}, {"value": None}])
    np.testing.assert_array_equal(result["value"], [None, None, None])
    np.testing.assert_array_equal(result["_value"], [True, False, True])


def test_real_mixed_autoreset_preserves_nested_solver_and_timing_info(
    dtype: np.dtype[np.floating],
) -> None:
    from envs.soft_walker.engine import make_gymnasium_vec
    from envs.soft_walker.engine.physics.spring_config import SpringConfig

    config = SpringConfig(
        grid_shape=(2, 1),
        size=(0.1, 0.05),
        settle_steps=0,
        substeps=1,
        control_dt=0.00025,
        horizon=2,
        gravity=0.0,
        ground=False,
    )
    env = make_gymnasium_vec(
        "SoftSpring2D-v0",
        num_envs=2,
        device="cpu",
        dtype=dtype,
        config=config,
        max_observation_space_size=128,
    )
    actions = np.zeros(env.action_space.shape, dtype=dtype)
    try:
        env.reset(seed=[2, 5])
        env.step(actions)
        # Stagger horizons through the public partial-reset API. On the third
        # step, lane 0 advances while lane 1 performs next-step autoreset.
        env.reset(options={"reset_mask": np.asarray([True, False])})
        second = env.step(actions)
        np.testing.assert_array_equal(second[3], [False, True])
        mixed = env.step(actions)
        info = mixed[4]
        np.testing.assert_array_equal(info["autoreset"], [False, True])
        np.testing.assert_array_equal(info["_solver"], [True, False])
        assert isinstance(info["solver"], dict)
        for key in info["solver"]:
            if key.startswith("_"):
                np.testing.assert_array_equal(info["solver"][key], [True, False])
        assert isinstance(info["timing"], dict)
        np.testing.assert_array_equal(info["_timing"], [True, True])
        np.testing.assert_array_equal(
            info["timing"]["_batch_synchronized_pipeline_s"], [True, False]
        )
        np.testing.assert_array_equal(info["timing"]["_total_reset_s"], [False, True])
        assert info["timing"]["batch_synchronized_pipeline_s"].dtype.kind == "f"
        assert info["timing"]["total_reset_s"].dtype.kind == "f"
        assert info["timing"]["batch_synchronized_pipeline_s"][1] == 0.0
        assert info["timing"]["total_reset_s"][0] == 0.0
    finally:
        env.close()
