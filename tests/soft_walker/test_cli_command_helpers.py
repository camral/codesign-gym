"""Contracts at CLI configuration and legacy replay artifact boundaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
from envs.soft_walker.engine.types import PhysicsModel, PhysicsState
from envs.soft_walker.presets.locomotion import DemoConfig
from envs.soft_walker.presets.scenarios.config import Scenario
from envs.soft_walker.scripts._locomotion_commands import build_config, controller_parameters
from envs.soft_walker.scripts._scenario_commands import replay_chunks


def _arguments(config: DemoConfig, **overrides: object) -> argparse.Namespace:
    """Supply physical options and explicit controller source choices."""
    fields: dict[str, object] = asdict(config)
    fields.update(
        amplitude=-0.8,
        lift=1.2,
        phase=0.3,
        gait="pronk",
        harmonics=2,
        passive=False,
        resume=None,
        controller=None,
    )
    fields.update(overrides)
    return argparse.Namespace(**fields)


def test_cli_config_preserves_nondefault_physical_protocol() -> None:
    """The extracted builder retains device-independent material/solver choices."""
    expected = DemoConfig(
        driver="mpm3",
        dtype="float32",
        cycles=3,
        period=0.12,
        control_dt=0.01,
        substeps=8,
        settle_seconds=0.04,
        refinement=2,
        grid_margin=7,
        travel_body_lengths=5.0,
        gmres_method="fused",
        spring_solver="gmres",
        friction=0.6,
        friction_velocity=0.004,
        gravity=4.0,
        max_log_stretch=0.2,
        actuator_tau=0.0,
        seed=19,
        actuation_model="active_stress",
        active_stress_ratio=0.6,
    )
    assert build_config(_arguments(expected)) == expected
    with pytest.raises(ValueError, match="MPM drivers only"):
        build_config(_arguments(expected, driver="beam"))


@pytest.mark.parametrize("driver,channels", [("mpm2", 4), ("mpm3", 16)])
def test_controller_seed_and_passive_modes(driver: str, channels: int) -> None:
    """Channel dimensions and passivity stay consistent for bipeds and quadrupeds."""
    config = DemoConfig(driver=driver, dtype="float32")
    active = np.asarray(controller_parameters(_arguments(config), config))
    passive = np.asarray(controller_parameters(_arguments(config, passive=True), config))
    assert active.shape == (channels, 5)
    assert active.dtype == np.float32
    assert np.any(active != 0)
    np.testing.assert_array_equal(passive, np.zeros_like(active))


def test_saved_controller_precedence_and_dtype(tmp_path: Path) -> None:
    """Resume coefficients take precedence over a seed, passive flag and controller file."""
    config = DemoConfig(dtype="float32")
    resumed = np.arange(20, dtype=np.float64).reshape(4, 5) / 10
    controller = tmp_path / "controller.npz"
    resume = tmp_path / "last.npz"
    np.savez(controller, parameters=np.ones((4, 5)))
    np.savez(resume, parameters=resumed)
    args = _arguments(config, controller=controller, resume=resume, passive=True)
    actual = np.asarray(controller_parameters(args, config))
    np.testing.assert_array_equal(actual, resumed.astype(np.float32))
    assert actual.dtype == np.float32
    args.resume = None
    np.testing.assert_array_equal(controller_parameters(args, config), np.ones((4, 5)))


@pytest.mark.parametrize("invalid", ["channels", "nan", "infinity"])
def test_saved_controller_rejects_incompatible_values(tmp_path: Path, invalid: str) -> None:
    """Invalid persisted coefficients fail before compilation or simulation."""
    config = DemoConfig()
    coefficients = np.zeros((3 if invalid == "channels" else 4, 5))
    if invalid != "channels":
        coefficients[0, 0] = np.nan if invalid == "nan" else np.inf
    controller = tmp_path / "controller.npz"
    np.savez(controller, parameters=coefficients)
    with pytest.raises(ValueError, match="channels/values"):
        controller_parameters(_arguments(config, controller=controller), config)


class _RecordingState(NamedTuple):
    """Small JAX pytree for exercising recording without changing physics fixtures."""

    count: jax.Array
    status: jax.Array


@pytest.mark.parametrize("fail_after,expected_steps", [(None, 5), (3, 4)])
def test_scenario_chunks_keep_tail_and_failure_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_after: int | None, expected_steps: int
) -> None:
    """One compiled chunk records only the requested tail and retains a failed prefix."""
    from envs.soft_walker.engine.drivers import model_access
    from envs.soft_walker.experiments.scenarios import rollout
    from envs.soft_walker.viewer import trajectory_animation as animation

    case = Scenario(cycles=1, period=0.05, control_dt=0.01, settle_seconds=0.0)
    bindings: list[int] = []

    def bind_chunk(env: object, scenario: Scenario, size: int) -> object:
        """Stand in for the physical transition while retaining a real JAX executable."""
        bindings.append(size)

        def advance(
            model: jax.Array, state: _RecordingState, parameters: jax.Array, start: jax.Array
        ) -> tuple[_RecordingState, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
            intervals = start + jnp.arange(1, size + 1)
            statuses = (
                jnp.zeros_like(intervals)
                if fail_after is None
                else (intervals >= fail_after).astype(intervals.dtype)
            )
            rows = jnp.stack((intervals * scenario.control_dt, statuses), axis=1)
            commands = jnp.broadcast_to(parameters, (size, len(parameters)))
            positions = jnp.stack((intervals, jnp.zeros_like(intervals)), axis=1)[:, None]
            solver = statuses[:, None]
            return _RecordingState(intervals[-1], statuses[-1]), (rows, commands, positions, solver)

        return advance

    def measure(env: object, model: object, state: _RecordingState) -> jax.Array:
        return jnp.asarray([state.count * case.control_dt, state.status])

    def positions(env: object, model: object, state: _RecordingState) -> jax.Array:
        return jnp.asarray([[state.count, 0]])

    def topology_record(env: object, model: object) -> dict[str, np.ndarray]:
        return {"edges": np.empty((0, 2), dtype=np.int32)}

    monkeypatch.setattr(rollout, "bind_chunk", bind_chunk)
    monkeypatch.setattr(rollout, "measure", measure)
    monkeypatch.setattr(model_access, "task_positions", positions)
    monkeypatch.setattr(animation, "topology_record", topology_record)
    stages: list[str] = []

    def stage(name: str, **details: object) -> None:
        stages.append(name)

    costs: dict[str, float] = {}
    # Only the recorder's borrowed interface matters here; fixture methods are patched above.
    data = replay_chunks(
        cast(Spring2DEnv, None),
        case,
        argparse.Namespace(chunk_size=2),
        cast(PhysicsModel, jnp.asarray(0)),
        cast(PhysicsState, _RecordingState(jnp.asarray(0), jnp.asarray(0))),
        jnp.asarray([0.25, -0.5]),
        costs,
        stage,
        tmp_path,
    )
    assert bindings == [2]
    assert stages.count("replay_chunk_compile") == 1
    assert stages.count("replay_chunk") == (expected_steps + 1) // 2
    assert costs["replay_chunk_compile_s"] >= 0
    np.testing.assert_array_equal(data.positions[:, 0, 0], np.arange(expected_steps + 1))
    assert data.rows.shape == (expected_steps + 1, 2)
    assert data.commands.shape == (expected_steps, 2)
    assert data.solver.shape == (expected_steps, 1)
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["positions"], data.positions)
        np.testing.assert_array_equal(saved["solver"], data.solver)
        assert set(saved.files) == {
            "positions",
            "metrics",
            "actions",
            "solver",
            "parameters",
            "edges",
        }
    timing = json.loads((tmp_path / "timings.json").read_text())
    assert sum(chunk["intervals"] for chunk in timing["chunks"]) == expected_steps
    if fail_after is None:
        with np.load(tmp_path / "chunks/000004.npz", allow_pickle=False) as tail:
            assert tail["positions"].shape[0] == 1
    else:
        assert data.solver[-1, 0] == 1
        assert not (tmp_path / "chunks/000004.npz").exists()
