"""Focused native tests; selected separately from historical JAX validation.

The runner starts a fresh process per precision and device. CUDA calls below
capture each complete test computation; they never execute the step's CPU loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from envs.soft_walker.engine.warp_native.config import MPMConfig
from envs.soft_walker.engine.warp_native.episode import Episode
from envs.soft_walker.engine.warp_native.model import Bundle, DeviceModel, uniform_body
from envs.soft_walker.engine.warp_native.step import ImplicitStep

DTYPE = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")
RTOL = 3e-4 if DTYPE == "float32" else 3e-9


def execute(program: Callable[[], None], device: wp.context.Device) -> None:
    if device.is_cuda:
        if not wp.is_conditional_graph_supported():
            raise RuntimeError("CUDA conditional graphs are required; no fallback")
        with wp.ScopedCapture(device=device, force_module_load=True) as capture:
            program()
        wp.capture_launch(capture.graph)
    else:
        program()
    wp.synchronize_device(device)


def fixture(dim: int = 2, **changes: object) -> Bundle:
    bundle = uniform_body(dim, precision=DTYPE)
    bundle.config = replace(bundle.config, **changes)
    return bundle


@pytest.mark.parametrize(
    "change",
    [
        dict(dt=0),
        dict(grid_shape=(3, 8)),
        dict(actuator_tau=-1),
        dict(actuator_tau=0),
        dict(restart=0),
        dict(min_J=float("nan")),
        dict(friction=-1),
        dict(origin=(0,)),
        dict(actuation_model="unknown"),
        dict(linear_refinements=4),
    ],
)
def test_config_rejects(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(MPMConfig(), **change)


@pytest.mark.parametrize("dim", [2, 3])
def test_portable_round_trip(tmp_path: Path, dim: int) -> None:
    bundle = fixture(dim)
    bundle.save(tmp_path / "model.npz")
    other = Bundle.load(tmp_path / "model.npz")
    for key in bundle.fields:
        np.testing.assert_array_equal(bundle.fields[key], other.fields[key])
    assert bundle.config == other.config


@pytest.mark.parametrize(
    "field,value",
    [
        ("alpha", -1),
        ("alpha", 2),
        ("young", 0),
        ("density", 0),
        ("poisson", 0.5),
        ("strength", 2),
        ("channel", 1),
        ("kind", 3),
        ("contact_area", -1),
        ("reference", float("nan")),
    ],
)
def test_model_rejects(field: str, value: float | int) -> None:
    bundle = fixture()
    bundle.fields[field].fill(value)
    with pytest.raises(ValueError):
        bundle.validate()


@pytest.mark.parametrize("dim", [2, 3])
def test_transfer_partition_and_mass(dim: int) -> None:
    bundle = fixture(dim)
    model = DeviceModel(bundle, DEVICE)
    step = ImplicitStep(model)
    execute(step.prepare, model.device)
    np.testing.assert_allclose(
        step.st.weight.numpy().sum(axis=1), 1, rtol=0, atol=10 * np.finfo(DTYPE).eps
    )
    np.testing.assert_allclose(
        step.st.gradient.numpy().sum(axis=1), 0, atol=2e-5 if DTYPE == "float32" else 1e-12
    )
    expected = np.sum(bundle.fields["alpha"] * bundle.fields["density"] * bundle.fields["volume"])
    np.testing.assert_allclose(step.st.mass.numpy().sum(), expected, rtol=RTOL)
    assert 0 < step.st.live.numpy()[0] <= step.capacity


@pytest.mark.parametrize("dim", [2, 3])
def test_zero_equilibrium_and_actuator_ledger_adjoint(dim: int) -> None:
    bundle = fixture(dim, gravity=0)
    model = DeviceModel(bundle, DEVICE)
    step = ImplicitStep(model)
    actuator_seed = wp.ones_like(step.output.actuator)
    ledger_seed = wp.ones_like(step.output.ledger)

    def program() -> None:
        step.forward(record_derivatives=True)
        step.backward({step.output.actuator: actuator_seed, step.output.ledger: ledger_seed})

    execute(program, model.device)
    assert step.diagnostics()["code"] == 0
    np.testing.assert_array_equal(step.output.x.numpy(), bundle.fields["reference"])
    np.testing.assert_array_equal(step.output.v.numpy(), np.zeros_like(bundle.fields["reference"]))
    np.testing.assert_allclose(
        step.command.grad.numpy(),
        -np.expm1(-bundle.config.dt / bundle.config.actuator_tau),
        rtol=3e-5,
    )
    np.testing.assert_allclose(model.initial.ledger.grad.numpy(), 1, rtol=RTOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_tangent_transpose_pair(dim: int) -> None:
    model = DeviceModel(fixture(dim, contact_enabled=True, contact_height=0.0252), DEVICE)
    step = ImplicitStep(model)
    step.command.fill_(0.4)
    rng = np.random.default_rng(67)
    x = wp.array(
        rng.normal(size=(step.capacity, dim)).astype(DTYPE), dtype=model.k.vec, device=DEVICE
    )
    y = wp.array(
        rng.normal(size=(step.capacity, dim)).astype(DTYPE), dtype=model.k.vec, device=DEVICE
    )
    ax, aty = wp.zeros_like(x), wp.zeros_like(y)

    def program() -> None:
        step.forward()
        step.matvec(x, ax, ax)
        step.transpose_matvec(y, aty, aty)

    execute(program, model.device)
    assert step.diagnostics()["code"] == 0
    np.testing.assert_allclose(
        np.sum(ax.numpy() * y.numpy()), np.sum(x.numpy() * aty.numpy()), rtol=RTOL, atol=RTOL
    )


def test_empty_travel_grid_does_not_expand_solver() -> None:
    a = DeviceModel(fixture(), DEVICE)
    b = DeviceModel(fixture(grid_shape=(800, 800)), DEVICE)
    sa, sb = ImplicitStep(a), ImplicitStep(b)
    assert sa.capacity == sb.capacity == a.n * 9
    assert sa.keys.size == sb.keys.size
    assert (
        sa.linear._basis.shape == sb.linear._basis.shape if hasattr(sa.linear, "_basis") else True
    )


def test_invalid_step_rolls_back_and_episode_rejects_gradient() -> None:
    bundle = fixture()
    bundle.fields["reference"][:, 0] += 100
    model = DeviceModel(bundle, DEVICE)
    episode = Episode(model, np.zeros((2, 1), DTYPE), checkpoint_stride=1)
    result = episode.run(gradient=True)
    assert result["status"] != 0 and np.isnan(result["loss"])
    np.testing.assert_array_equal(episode.final.x.numpy(), bundle.fields["reference"])
    with pytest.raises(RuntimeError, match="failed"):
        episode.gradients()


def test_checkpoint_and_replay_reuse() -> None:
    bundle = fixture(gravity=0, actuation_model="active_stress", actuator_tau=0)
    commands = np.array([[0.3], [-0.2], [0.1]], DTYPE)
    one = Episode(
        DeviceModel(bundle, DEVICE),
        commands,
        settling_steps=1,
        checkpoint_stride=1,
        work_weight=0.01,
    )
    two = Episode(
        DeviceModel(bundle, DEVICE),
        commands,
        settling_steps=1,
        checkpoint_stride=3,
        work_weight=0.01,
    )
    r1, r2 = one.run(gradient=True), two.run(gradient=True)
    assert r1["status"] == r2["status"] == 0
    np.testing.assert_allclose(
        one.gradients()["commands"], two.gradients()["commands"], rtol=RTOL, atol=1e-9
    )
    original = two.gradients()["commands"].copy()
    two.run(gradient=True)
    np.testing.assert_allclose(two.gradients()["commands"], original, rtol=RTOL, atol=1e-9)
    two.set_commands(-commands)
    with pytest.raises(RuntimeError):
        two.gradients()
    assert two.run(gradient=True)["status"] == 0
    assert not np.allclose(two.gradients()["commands"], original, rtol=1e-3, atol=1e-14)
    two.run()
    with pytest.raises(RuntimeError):
        two.gradients()


def test_command_and_memory_validation() -> None:
    model = DeviceModel(fixture(), DEVICE)
    for commands in (np.zeros(3), np.array([[2.0]]), np.array([[np.nan]])):
        with pytest.raises(ValueError):
            Episode(model, commands)
    with pytest.raises(MemoryError):
        Episode(model, np.zeros((100, 1)), memory_mib=0)


def test_no_jax_dependency() -> None:
    code = """import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('jax','jaxlib','diffrax','equinox'):
   raise ImportError('Native runtime attempted to import '+fullname)
sys.meta_path.insert(0,Block())
import numpy as np
from envs.soft_walker.engine.execution.cupti import DriverAudit
from envs.soft_walker.engine.warp_native.model import uniform_body,DeviceModel
from envs.soft_walker.engine.warp_native.episode import Episode
e=Episode(DeviceModel(uniform_body()),np.zeros((1,1)))
assert e.run()['status']==0
assert not any(name=='jax' or name.startswith('jax.') for name in sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_backward_replay_failure_never_reads_stale_checkpoint() -> None:
    """A valid forward pass does not excuse a failed reverse replay."""
    episode = Episode(
        DeviceModel(fixture(), DEVICE),
        np.zeros((3, 1), DTYPE),
        checkpoint_stride=2,
        velocity_weight=0.1,
    )
    original = episode.replay_body

    def fail_replay() -> None:
        original()
        episode.working.status.fill_(1)
        episode.inner_condition.zero_()

    episode.replay_body = fail_replay
    result = episode.run(gradient=True)
    assert result["status"] == 0
    assert result["adjoint_status"] == 20
    with pytest.raises(RuntimeError, match="replay"):
        episode.gradients()


def test_accumulated_displacement_matches_constant_velocity_and_float64_positions() -> None:
    bundle = fixture(gravity=0, actuation_model="active_stress", actuator_tau=0)
    n, dim = bundle.fields["reference"].shape
    speed = np.zeros((n, dim), DTYPE)
    speed[:, 0] = 0.0123
    bundle.initial = dict(
        x=bundle.fields["reference"].copy(),
        v=speed,
        F=np.tile(np.eye(dim, dtype=DTYPE), (n, 1, 1)),
        C=np.zeros((n, dim, dim), DTYPE),
        actuator=np.zeros(1, DTYPE),
        ledger=np.zeros(4, DTYPE),
        status=np.zeros(1, np.int32),
    )
    episode = Episode(DeviceModel(bundle, DEVICE), np.zeros((3, 1), DTYPE), checkpoint_stride=2)
    result = episode.run(gradient=True)
    assert result["status"] == result["adjoint_status"] == 0
    expected = 3 * bundle.config.dt * speed
    np.testing.assert_allclose(episode.travel.numpy(), expected, rtol=RTOL, atol=1e-13)
    if DTYPE == "float64":
        np.testing.assert_allclose(
            episode.travel.numpy(),
            episode.final.x.numpy() - bundle.initial["x"],
            rtol=1e-9,
            atol=1e-16,
        )
    episode.run(gradient=True)
    np.testing.assert_allclose(episode.travel.numpy(), expected, rtol=RTOL, atol=1e-13)
