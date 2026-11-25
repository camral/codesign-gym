"""Cheap controller, objective and saved-flight checks; no physical rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


import importlib.util
import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from soft_paths import SOFT_ROOT, needs_scripts

from envs.soft_walker.engine.physics.terrain import TerrainConfig, generate
from envs.soft_walker.experiments.locomotion.rollout import (
    ObjectiveConfig,
    flight_assessment,
    flight_columns,
    hop_sample_terms,
    objective_components,
    objective_record,
    sample_flight_metrics,
)
from envs.soft_walker.presets.gaits import seed_controller
from envs.soft_walker.shared import periodic_action


@pytest.mark.parametrize("dim", [2, 3])
def test_pronk_seed_is_synchronous_and_has_no_bending_at_zero_amplitude(
    dim: int, dtype: np.dtype[np.floating]
) -> None:
    parameters = seed_controller(dim, gait="pronk", amplitude=0.0, lift=1.2, dtype=dtype)
    np.testing.assert_array_equal(
        parameters, np.tile([0.0, 0.0, 1.2], (len(parameters), 1)).astype(dtype)
    )
    for time in (0.0, 0.125, 0.25, 0.375):
        commands = np.asarray(periodic_action(jnp.asarray(parameters), time, 0.5))
        np.testing.assert_allclose(
            commands, np.tanh(1.2 * np.cos(2 * np.pi * time / 0.5)), atol=2e-7
        )
    assert float(periodic_action(jnp.asarray(parameters), 0.0, 0.5)[0]) > 0
    assert float(periodic_action(jnp.asarray(parameters), 0.25, 0.5)[0]) < 0


def test_legacy_seed_default_unchanged(dtype: np.dtype[np.floating]) -> None:
    p = seed_controller(2, dtype=dtype)
    expected = np.array(
        [[0.0, -0.8, 0.5], [0.0, 0.8, 0.5], [0.0, 0.8, -0.5], [0.0, -0.8, -0.5]], dtype=dtype
    )
    np.testing.assert_allclose(p, expected, atol=2e-7)


def _demo(steps: int = 9, cycles: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        steps=steps, cycles=cycles, steps_per_cycle=steps // cycles, size=(0.15, 0.15), dim=2
    )


def _flight(
    force_ratios: Sequence[float], gaps: Sequence[float], *, weight: float = 10.0
) -> np.ndarray:
    rows = np.zeros((len(force_ratios), len(flight_columns())))
    rows[:, 0] = np.arange(len(rows)) * 0.02
    rows[:, 4] = weight
    rows[:, 5] = force_ratios
    rows[:, 6] = gaps
    rows[:, 9] = 1.0
    return rows


def test_gap_threshold_interruptions_do_not_invent_two_flights() -> None:
    # Two qualifying clearance spans, but a single force-free episode/landing.
    rows = _flight(
        [1, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0.004, 0.004, 0.0029, 0.004, 0.004, 0, 0, 0, 0]
    )
    result = flight_assessment(rows, _demo(), valid_rollout=True)
    assert len(result["observed_airborne_intervals"]) == 2
    assert result["completed_flights_at_least_20ms"] == 1
    assert not result["repeated_airborne_observed"]


def test_repeated_flight_requires_force_recontact_and_valid_physics() -> None:
    rows = _flight([1, 0, 0, 1, 0, 0, 1, 1, 1, 1], [0, 0.004, 0.004, 0, 0.004, 0.004, 0, 0, 0, 0])
    result = flight_assessment(rows, _demo(), valid_rollout=True)
    assert result["completed_flights_at_least_20ms"] == 2
    assert result["repeated_airborne_observed"]
    assert not flight_assessment(rows, _demo(), valid_rollout=False)["repeated_airborne_observed"]
    rows[:, 4] = 0
    assert (
        flight_assessment(rows, _demo(), valid_rollout=True)["completed_flights_at_least_20ms"] == 0
    )


def test_unlanded_or_single_sample_clearance_is_not_completed_flight() -> None:
    rows = _flight([1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0.004, 0, 0, 0, 0, 0, 0.004, 0.004, 0.004])
    result = flight_assessment(rows, _demo(), valid_rollout=True)
    assert result["completed_flights_at_least_20ms"] == 0


@pytest.mark.parametrize("reference", [0.0, -1.0, float("nan"), float("inf")])
def test_explicit_work_reference_must_be_positive_and_finite(reference: float) -> None:
    """Invalid recorded normalizers fail before loss reconstruction rather than falling back."""
    with pytest.raises(ValueError, match="work_ref must be finite and positive"):
        objective_components(
            np.zeros((2, 13)), np.zeros((2, 10)), _demo(), ObjectiveConfig(), work_ref=reference
        )


def test_contact_measure_uses_transformed_sites_actual_weight_and_finite_orientation_gradient(
    dtype: np.dtype[np.floating],
) -> None:
    ref = jnp.asarray([[-0.05, -0.01], [-0.05, 0.01], [0.05, -0.01], [0.05, 0.01]], dtype)
    offsets = jnp.asarray([[0.0, -0.015], [0.0, 0.015]], dtype)
    model = SimpleNamespace(
        cloud=SimpleNamespace(reference=ref, face_offset=offsets),
        mass=jnp.full(4, 2.0, dtype),
        active=jnp.ones(4, bool),
        contact_area=jnp.ones((4, 2), dtype),
        terrain=generate(TerrainConfig(), dtype=dtype),
    )
    env = SimpleNamespace(
        kind="mpm2",
        config=SimpleNamespace(
            dynamics=SimpleNamespace(
                gravity=3.0, contact=SimpleNamespace(stiffness=20.0, width=0.002)
            )
        ),
    )

    def evaluate(angle: jax.Array, shift: jax.Array) -> jax.Array:
        c, s = jnp.cos(angle), jnp.sin(angle)
        rotation = jnp.stack((jnp.stack((c, -s)), jnp.stack((s, c))))
        body = SimpleNamespace(
            x=ref @ rotation.T + jnp.stack((jnp.asarray(0.0, dtype), shift)),
            v=jnp.tile(jnp.asarray([0.0, 0.3], dtype), (4, 1)),
            F=jnp.tile(rotation, (4, 1, 1)),
            time=jnp.asarray(0.2, dtype),
        )
        return sample_flight_metrics(env, model, SimpleNamespace(body=body), jnp.ones(4, bool))

    value = np.asarray(evaluate(jnp.asarray(0.4, dtype), jnp.asarray(0.02, dtype)))
    assert value[4] == 24.0
    np.testing.assert_allclose(value[5], value[3] / 24.0, rtol=1e-6)
    np.testing.assert_allclose(value[[1, 2, 8, 9]], [0.02, 0.3, 0.4, np.cos(0.4)], atol=1e-6)
    rotation = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
    sites = (
        np.asarray(ref)[:, None, :] @ rotation.T
        + np.asarray(offsets)[None, :, :] @ rotation.T
        + [0, 0.02]
    )
    np.testing.assert_allclose(value[6], sites[..., 1].min(), atol=1e-7)
    fun = lambda z: jnp.sum(
        hop_sample_terms(evaluate(z[0], z[1]), 0.02, 0.15, ObjectiveConfig(name="hop"))
    )
    gradient = np.asarray(jax.grad(fun)(jnp.asarray([0.0, 0.02], dtype)))
    assert np.isfinite(gradient).all()
    assert gradient[1] < 0  # Grounded seed has a smooth useful lift gradient.


def test_hop_cost_prefers_upright_lift_and_excludes_startup() -> None:
    demo = _demo(steps=6, cycles=3)
    metrics = np.zeros((7, 13))
    metrics[:, 0] = np.arange(7) * 0.02
    metrics[:, 1] = [0, 0.3, 0.6, 0.61, 0.62, 0.63, 0.64]
    flight = _flight(np.ones(7), np.full(7, 0.01))
    flight[:, 1] = 0.1
    flight[:, 7] = 0.04
    objective = ObjectiveConfig(name="hop")
    cost = objective_components(metrics, flight, demo, objective)
    assert cost["forward"] == pytest.approx(-0.25 * 0.04 / 0.15)
    lifted = flight.copy()
    lifted[3:, 1] += 0.01
    assert objective_components(metrics, lifted, demo, objective)["total"] < cost["total"]
    tipped = flight.copy()
    tipped[3:, 9] = 0.0
    assert objective_components(metrics, tipped, demo, objective)["upright"] == 2.0
    grounded = flight.copy()
    grounded[3:, 7] = 0
    assert objective_components(metrics, grounded, demo, objective)["torso_ground"] > 0.4


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("work_ref_scale", [1.0, 2.3])
def test_custom_hop_objective_agrees_in_legacy_resident_and_saved_metrics(
    monkeypatch: pytest.MonkeyPatch, dtype: np.dtype[np.floating], dim: int, work_ref_scale: float
) -> None:
    """Exercise both rollout loss paths with a known, nontrivial trajectory.

    A synthetic transition keeps this objective regression independent of solver
    cost. It includes startup displacement, lift, tilt, work and lateral drift.
    """
    from envs.soft_walker.engine.drivers import environment as drivers
    from envs.soft_walker.experiments.locomotion import rollout
    from envs.soft_walker.experiments.locomotion.resident import bind_demo_episode
    from envs.soft_walker.presets.locomotion import DemoConfig

    demo = DemoConfig(
        driver="mpm2" if dim == 2 else "mpm3",
        cycles=3,
        period=0.04,
        control_dt=0.02,
        substeps=1,
        settle_seconds=0.0,
    )
    objective = ObjectiveConfig(
        name="hop",
        hop_progress_weight=4.0,
        hop_lift_weight=1.0,
        hop_upright_weight=1.3,
        hop_torso_ground_weight=0.4,
        work_weight=0.003,
        lateral_weight=0.7,
    )
    rows = np.zeros((demo.steps + 1, 9 + (4 if dim == 2 else 8)), dtype=dtype)
    rows[:, 0] = np.arange(len(rows)) * demo.control_dt
    rows[:, 1] = [0.0, 0.03, 0.06, 0.061, 0.064, 0.068, 0.075]
    rows[:, 3] = np.linspace(0.0, 0.03 if dim == 3 else 0.0, len(rows))
    rows[:, 4] = np.linspace(0.0, 0.12, len(rows))
    flight = _flight(np.ones(len(rows)), np.full(len(rows), 0.01)).astype(dtype)
    flight[:, 1] = np.linspace(0.1, 0.115, len(rows))
    flight[:, 7] = 0.008
    flight[:, 9] = np.linspace(1.0, 0.97, len(rows))
    metric_values, flight_values = jnp.asarray(rows), jnp.asarray(flight)
    Body = namedtuple("SyntheticBody", "time")
    State = namedtuple("SyntheticState", "body status displacement index")
    initial = State(
        Body(jnp.asarray(0.0, dtype)), jnp.int32(0), jnp.zeros(dim, dtype), jnp.int32(0)
    )
    env = SimpleNamespace(
        kind=demo.driver,
        channels=demo.channels,
        _grid=None,
        config=SimpleNamespace(
            control_dt=demo.control_dt,
            horizon=demo.steps,
            substeps=1,
            settle_steps=0,
            length_ref=demo.size[0],
            size=demo.size,
            work_ref=(4.0 if dim == 3 else 0.4) * work_ref_scale,
            dynamics=SimpleNamespace(newton=SimpleNamespace(debug_linear=False)),
        ),
    )
    info = SimpleNamespace(
        code=jnp.int32(0),
        max_residual=jnp.asarray(0.0, dtype),
        max_linear_residual=jnp.asarray(0.0, dtype),
        min_J=jnp.asarray(1.0, dtype),
        support_valid=True,
        max_penetration=jnp.asarray(0.0, dtype),
        nonlinear_iterations=1,
    )

    def advance(
        kind: str,
        model: SimpleNamespace,
        state: tuple,
        action: jax.Array,
        config: SimpleNamespace,
        grid: object,
    ) -> tuple[tuple, tuple[SimpleNamespace]]:
        index = state.index + 1
        row = metric_values[index]
        return State(Body(row[0]), state.status, row[1 : 1 + dim], index), (info,)

    monkeypatch.setattr(drivers, "advance", advance)
    monkeypatch.setattr(rollout, "region_tags", lambda env, model: (None, None))
    monkeypatch.setattr(
        rollout, "sample_metrics", lambda env, model, state, torso, feet: metric_values[state.index]
    )
    monkeypatch.setattr(
        rollout,
        "sample_flight_metrics",
        lambda env, model, state, torso=None: flight_values[state.index],
    )
    controller = jnp.zeros((demo.channels, 3), dtype)
    model = jnp.asarray(0.0, dtype)
    legacy = rollout.bind_experiment(env, demo, objective=objective, block_size=2)
    resident = bind_demo_episode(env, demo, objective=objective, checkpoint_block=2)
    expected = objective_components(rows, flight, demo, objective, work_ref=env.config.work_ref)
    historical = objective_components(rows, flight, demo, objective)
    assert expected["work"] == pytest.approx(historical["work"] / work_ref_scale)
    assert expected["lateral"] == pytest.approx(0.7 * (0.03 / 0.15) ** 2 if dim == 3 else 0.0)
    for loss in (
        legacy(model, initial, controller)["loss"],
        resident.fixed_loss(model, initial, controller),
    ):
        np.testing.assert_allclose(loss, expected["total"], rtol=2e-6, atol=2e-7)
    description = objective_record(objective, demo)
    assert "hop_progress_weight" in description["formula"]
    assert ("lateral_weight" in description["formula"]) == (dim == 3)


@needs_scripts
def test_objective_resume_metadata_rejects_loss_changes(tmp_path: Path) -> None:
    script = SOFT_ROOT / "scripts" / "locomotion.py"
    spec = importlib.util.spec_from_file_location("locomotion_cli_hop_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "objective.json"
    travel = objective_record(ObjectiveConfig(), _demo())
    hop = objective_record(ObjectiveConfig(name="hop"), _demo())
    module.validate_resume_objective(path, travel)
    with pytest.raises(ValueError, match="travel-only"):
        module.validate_resume_objective(path, hop)
    path.write_text(json.dumps(hop))
    module.validate_resume_objective(path, hop)
    with pytest.raises(ValueError, match="objective changed"):
        module.validate_resume_objective(path, travel)
