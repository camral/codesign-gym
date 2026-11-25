"""Contracts between the cleanup's extracted responsibilities."""

from __future__ import annotations

import pickle

import numpy as np

from envs.soft_walker.engine.drivers import (
    environment,
    model_access,
    reporting,
    spring_environment as env,
)
from envs.soft_walker.engine.drivers.vector_lifecycle import _stack_info
from envs.soft_walker.engine.embodiments import decoders, fields


def test_responsibilities_use_shared_implementations() -> None:
    """Model access and field interpretation have one implementation across consumers."""
    for name in ("interpret", "baseline_raw", "grid_points", "interpolate"):
        assert getattr(decoders, name) is getattr(fields, name)
    assert env.field_errors is fields.field_errors
    assert env._stack_info is _stack_info
    for name in ("reference", "positions", "channel_mask", "module"):
        assert getattr(environment, name) is getattr(model_access, name)
    for name in ("observe", "mechanics_diagnostics", "_mpm_energy"):
        assert getattr(environment, name) is getattr(reporting, name)


def test_persisted_class_identities_use_canonical_modules() -> None:
    """New Python records resolve their defining modules without legacy aliases."""
    spec = decoders.FieldSpec(dim=3, channels=4)
    assert pickle.loads(pickle.dumps(spec)) == spec
    for value in (decoders.Fields, decoders.FieldSpec, decoders.GridDecoder, decoders.CPPNDecoder):
        assert value.__module__ == "envs.soft_walker.engine.embodiments.decoders"
    assert env.Spring2DVectorEnv.__module__ == "envs.soft_walker.engine.drivers.spring_environment"


def test_nested_vector_info_distinguishes_absent_keys_from_present_none() -> None:
    """Mixed autoreset lanes retain nested masks and terminal observation ownership."""
    terminal = np.asarray([1.0, 2.0])
    result = _stack_info(
        [
            {"solver": {"iterations": np.asarray(3)}, "final_obs": terminal, "note": None},
            {"solver": {"residual": 0.1}},
            {"autoreset": True},
        ]
    )
    np.testing.assert_array_equal(result["_solver"], [True, True, False])
    np.testing.assert_array_equal(result["solver"]["_iterations"], [True, False, False])
    np.testing.assert_array_equal(result["solver"]["iterations"], [3, 0, 0])
    np.testing.assert_array_equal(result["solver"]["_residual"], [False, True, False])
    np.testing.assert_array_equal(result["_note"], [True, False, False])
    np.testing.assert_array_equal(result["_final_obs"], [True, False, False])
    assert result["final_obs"].dtype == object
    assert result["final_obs"][0] is terminal
    assert result["final_obs"][1] is None
