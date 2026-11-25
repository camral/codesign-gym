"""Persisted status values keep each backend's physical-step meaning."""

from __future__ import annotations

import os

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from envs.soft_walker.contracts.metrics import WORK_INCREMENT_WIDTH
from envs.soft_walker.engine.warp_native.model import DeviceModel, uniform_body
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel


@pytest.mark.parametrize("physics", ["mpm", "spring"])
@pytest.mark.parametrize("root_code", [1, 3, 4])
def test_native_failed_commit_preserves_binary_vs_root_status(physics: str, root_code: int) -> None:
    """MPM reports binary failure; solids retain the specific root failure code."""
    device = os.environ.get("WARP_TEST_DEVICE", "cpu")
    precision = os.environ.get("WARP_TEST_DTYPE", "float64")
    owner = (
        DeviceModel(uniform_body(2, precision=precision), device)
        if physics == "mpm"
        else SolidDeviceModel(regular_solid("spring", shape=(1, 1), precision=precision), device)
    )
    ledger = np.asarray([0.2, 0.3, 0.4, 0.5], dtype=precision)
    actuator = np.full(owner.channels, 0.125, dtype=precision)
    owner.initial.ledger.assign(ledger)
    owner.initial.actuator.assign(actuator)
    output = owner.empty_state()
    code = wp.array([root_code], dtype=int, device=device)
    work = wp.ones(WORK_INCREMENT_WIDTH, dtype=owner.k.real, device=device)
    command = wp.ones(owner.channels, dtype=owner.k.real, device=device)
    wp.launch(
        owner.k.finish_state,
        dim=1,
        inputs=[owner.initial, command, owner.parameters, code, work, output],
        device=device,
    )
    wp.synchronize_device(device)
    # These integers are the historical archive/API contract, independently of
    # the constant names used to implement the two physical-step commit paths.
    assert int(output.status.numpy()[0]) == (1 if physics == "mpm" else root_code)
    np.testing.assert_array_equal(output.ledger.numpy(), ledger)
    np.testing.assert_array_equal(output.actuator.numpy(), actuator)
