"""Rejected discrete embodiment updates preserve native episode inputs and derivatives."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp
from test_warp_embodiment_bridge import _fixture

from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers import environment as drivers
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact
from envs.soft_walker.engine.types import HostArray, HostEmbedding
from envs.soft_walker.engine.warp_native.embodiment_bridge import EmbodimentEpisode
from envs.soft_walker.engine.warp_native.host_types import EpisodeGradients


@dataclass(frozen=True)
class EpisodeSnapshot:
    """Host copies and device addresses from the most recent successful adjoint run."""

    arrays: dict[str, HostArray]
    pointers: dict[str, int]
    derivatives: EpisodeGradients


def _storage(bridge: EmbodimentEpisode) -> dict[str, wp.array]:
    """Collect all resolved model and initial-state buffers plus supplied command storage."""
    assert bridge.bundle.initial is not None
    arrays = {f"model.{name}": getattr(bridge.owner.model, name) for name in bridge.bundle.fields}
    arrays.update(
        {f"initial.{name}": getattr(bridge.owner.initial, name) for name in bridge.bundle.initial}
    )
    arrays["commands"] = bridge.episode.commands
    arrays["loss"] = bridge.episode.loss
    return arrays


def _snapshot(bridge: EmbodimentEpisode) -> EpisodeSnapshot:
    """Download comparison evidence after success; gradient availability itself is checked."""
    arrays = _storage(bridge)
    assert bridge.episode.completed and bridge.episode.last_was_gradient
    return EpisodeSnapshot(
        arrays={name: value.numpy().copy() for name, value in arrays.items()},
        pointers={name: value.ptr for name, value in arrays.items()},
        derivatives=bridge.episode.gradients(),
    )


def _assert_unchanged(bridge: EmbodimentEpisode, before: EpisodeSnapshot) -> None:
    """A rejected transaction must not overwrite storage or invalidate prior valid gradients."""
    arrays = _storage(bridge)
    assert {name: value.ptr for name, value in arrays.items()} == before.pointers
    for name, value in arrays.items():
        np.testing.assert_array_equal(value.numpy(), before.arrays[name], err_msg=name)
    assert bridge.episode.completed and bridge.episode.last_was_gradient
    derivatives = bridge.episode.gradients()
    assert derivatives.keys() == before.derivatives.keys()
    np.testing.assert_array_equal(derivatives["commands"], before.derivatives["commands"])
    for group in ("model", "initial"):
        assert derivatives[group].keys() == before.derivatives[group].keys()
        for name, value in derivatives[group].items():
            np.testing.assert_array_equal(
                value, before.derivatives[group][name], err_msg=f"{group}.{name}"
            )


def _changed_label(body: HostEmbedding, dim: int, channels: int, label: str) -> HostEmbedding:
    """Cross a discrete argmax using valid floating direct-decoder coefficients."""
    candidate = {name: value.copy() for name, value in body.items()}
    start, width, selected = (
        (7 + dim, 3, 2) if label == "kind" else (10 + dim, channels, channels - 1)
    )
    candidate["values"][..., start : start + width] = -3
    candidate["values"][..., start + selected] = 3
    return candidate


@pytest.mark.parametrize("kind", ("spring", "beam", "fem", "mpm2", "mpm3"))
@pytest.mark.parametrize("label", ("kind", "channel"))
def test_discrete_label_change_is_atomic(
    kind: str, label: str, dtype: np.dtype[np.floating]
) -> None:
    """Labels remain fixed even when native spring/beam exports fold them into float routing."""
    env, body, commands, bridge = _fixture(kind, dtype.name)
    try:
        report, body_gradient, command_gradient = bridge.evaluate(body, commands, gradient=True)
        assert report["status"] == report["adjoint_status"] == 0
        assert body_gradient is not None and command_gradient is not None
        before = _snapshot(bridge)
        candidate = _changed_label(body, env.decoder.spec.dim, env.channels, label)
        # This remains a valid design for a fresh environment. Only the bound
        # episode's fixed discrete structure makes the update inadmissible.
        assert env.validate_embodiment(candidate)
        original_fields = bridge._decoded_fields(body)
        candidate_fields = bridge._decoded_fields(candidate)
        attribute = "actuator_type" if label == "kind" else "channel"
        assert not np.array_equal(
            getattr(original_fields, attribute), getattr(candidate_fields, attribute)
        )
        with pytest.raises(ValueError, match="Sampled active masks and discrete actuator routing"):
            bridge.evaluate(candidate, commands * 0.5, gradient=True)
        _assert_unchanged(bridge, before)
    finally:
        env.close()


def _hard_mpm(
    dim: int, dtype: np.dtype[np.floating]
) -> tuple[drivers.BodyEnv, HostEmbedding, HostArray, EmbodimentEpisode]:
    """Construct hard geometry locally without changing the shared relaxed test fixture."""
    config = BodyConfig(
        grid_shape=(2,) + (1,) * (dim - 1),
        size=(0.1,) + (0.05,) * (dim - 1),
        control_dt=0.00025,
        substeps=1,
        settle_steps=0,
        horizon=3,
        dynamics=DynamicsConfig(
            max_halvings=0,
            gravity=3.0,
            contact=PlaneContact(enabled=False),
            newton=NewtonConfig(preconditioner_side="right"),
        ),
    )
    env, body = make(
        f"SoftMPM{dim}D-v0",
        config=config,
        dtype=dtype.name,
        device="cpu",
        decoder="direct",
        design_mode="hard",
        max_observation_space_size=10000,
    )
    commands = np.full((2, env.channels), 0.15, dtype)
    bridge = EmbodimentEpisode(
        env,
        body,
        commands,
        device=os.environ.get("WARP_TEST_DEVICE", "cpu"),
        velocity_weight=0.7,
        checkpoint_stride=2,
    )
    return env, body, commands, bridge


@pytest.mark.parametrize("dim", (2, 3))
def test_hard_mpm_active_mask_change_is_atomic(dim: int, dtype: np.dtype[np.floating]) -> None:
    """Hard occupancy changes must be rejected although MPM has no native free-mask array."""
    env, body, commands, bridge = _hard_mpm(dim, dtype)
    try:
        report, body_gradient, command_gradient = bridge.evaluate(body, commands, gradient=True)
        assert report["status"] == report["adjoint_status"] == 0
        assert body_gradient is not None and command_gradient is not None
        before = _snapshot(bridge)
        candidate = {name: value.copy() for name, value in body.items()}
        values = candidate["values"]
        # Leave the left half occupied and remove the right half. The resulting
        # body is connected and valid; the episode may not change its active set.
        values[..., 0] = np.linspace(2, -2, values.shape[0], dtype=dtype).reshape(
            (values.shape[0],) + (1,) * (dim - 1)
        )
        assert env.validate_embodiment(candidate)
        old_active = np.asarray(bridge._decoded_fields(body).occupancy) > 0
        new_active = np.asarray(bridge._decoded_fields(candidate).occupancy) > 0
        assert old_active.all() and new_active.any() and not new_active.all()
        with pytest.raises(ValueError, match="Sampled active masks and discrete actuator routing"):
            bridge.evaluate(candidate, commands * 0.5, gradient=True)
        _assert_unchanged(bridge, before)
    finally:
        env.close()
