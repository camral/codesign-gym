"""Fixed field/observation/ledger formats stay compatible with existing consumers."""

from __future__ import annotations

from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.contracts.metrics import (
    FLIGHT_COLUMNS,
    LEDGER_COLUMNS,
    LOCOMOTION_SOLVER_COLUMNS,
    MPM_WORK_COUNT_COLUMNS,
    NATIVE_DIAGNOSTIC_COLUMNS,
    SOLID_WORK_COUNT_COLUMNS,
    FlightMetric,
    LocomotionMetricLayout,
    LocomotionSolver,
    locomotion_metric_columns,
    scenario_metric_layout,
)
from envs.soft_walker.contracts.observations import (
    body_observation_layout,
    resolved_observation_layout,
    spring_observation_layout,
)
from envs.soft_walker.engine.drivers.spring_environment import observation as spring_observation
from envs.soft_walker.engine.embodiments.decoders import FieldSpec
from envs.soft_walker.engine.physics import spring
from envs.soft_walker.engine.physics.spring_config import SpringConfig


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("channels", (1, 4, 16))
def test_raw_layout_keeps_historical_scores_and_descriptor_fields(dim: int, channels: int) -> None:
    """The new view does not serialize extra fields or shift old semantic heads."""
    spec = FieldSpec(dim=dim, channels=channels)
    layout = spec.raw_layout
    assert "raw_layout" not in asdict(spec)
    row = np.arange(spec.width)
    np.testing.assert_array_equal(row[layout.direction], np.arange(4, 4 + dim))
    np.testing.assert_array_equal(row[layout.material_logits], np.arange(4 + dim, 7 + dim))
    np.testing.assert_array_equal(row[layout.actuator_logits], np.arange(7 + dim, 10 + dim))
    np.testing.assert_array_equal(
        row[layout.channel_logits], np.arange(10 + dim, 10 + dim + channels)
    )
    assert spec.poisson_index == layout.poisson == 10 + dim + channels
    assert spec.width == 11 + dim + channels
    assert row[layout.properties][layout.primitive_poisson] == layout.poisson


@pytest.mark.parametrize(
    ("kind", "dim", "profile", "expected"),
    (
        ("mpm2", 2, "compact", 80),
        ("mpm2", 2, "full", 96),
        ("mpm3", 3, "compact", 95),
        ("mpm3", 3, "full", 131),
        ("fem", 3, "full", 95),
        ("beam", 3, "full", 119),
    ),
)
def test_body_profile_widths_and_block_order(
    kind: str, dim: int, profile: str, expected: int
) -> None:
    """Public profiles retain distinct mechanics blocks before actuator/terrain tails."""
    layout = body_observation_layout(kind, dim, 2, 3, 4, profile=profile)
    assert layout.width == expected
    assert [name for name, _ in layout.blocks][:3] == ["global", "nodes", "fields"]
    assert [name for name, _ in layout.blocks][-2:] == ["actuators", "terrain"]
    assert layout.block("terrain").stop == expected


def marked_spring_records() -> tuple[spring.Model, spring.State]:
    """Packing-only records with unique markers; unused mechanics leaves are inert."""
    model = {name: jnp.asarray(0.0) for name in spring.Model._fields}
    model.update(
        reference=jnp.zeros((2, 2)),
        mass=jnp.array([1.0, 0.0]),
        active=jnp.array([True, False]),
        occupancy=jnp.array([1.0]),
        young=jnp.array([40000.0]),
        density=jnp.array([2000.0]),
        strength=jnp.array([0.25]),
        channel_mask=jnp.array([True, False]),
        terrain=None,
    )
    state = {name: jnp.asarray(0.0) for name in spring.State._fields}
    state.update(
        x=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        v=jnp.array([[5.0, 6.0], [7.0, 8.0]]),
        actuator=jnp.array([9.0, 10.0]),
        work_signed=jnp.asarray(11.0),
        work_positive=jnp.asarray(12.0),
        dissipated=jnp.asarray(13.0),
        time=jnp.asarray(14.0),
    )
    return spring.Model(**model), spring.State(**state)


def test_spring_observation_normalization_and_masks_keep_legacy_layout() -> None:
    """A masked node and padded channel keep their established value/mask semantics."""
    model, state = marked_spring_records()
    cfg = SpringConfig(length_ref=2.0, velocity_ref=4.0, horizon=5, control_dt=0.2, settle_steps=0)
    layout = spring_observation_layout(2, 1, 2, terrain=False)
    values, mask = spring_observation(model, state, cfg, layout.width + 2)
    expected = [
        0.5,
        1.0,
        1.25,
        1.5,
        14.0,
        0.5,
        1.0,
        1.25,
        1.5,
        1.0,
        0,
        0,
        0,
        0,
        0,
        1,
        np.log(2.0),
        2.0,
        0.25,
        9,
        1,
        10,
        0,
        0,
        0,
    ]
    np.testing.assert_allclose(values, expected, rtol=2e-7, atol=0)
    np.testing.assert_array_equal(mask, [True] * 10 + [False] * 5 + [True] * 6 + [False] * 4)


def test_native_reporting_names_and_scenario_suffix_remain_stable() -> None:
    """Different counter families retain distinct names for their unused slots."""
    assert LEDGER_COLUMNS == ("work_signed_j", "work_positive_j", "dissipated_j", "time_s")
    assert MPM_WORK_COUNT_COLUMNS[3:5] == ("primal_residual", "support_build")
    assert SOLID_WORK_COUNT_COLUMNS[3:5] == ("unused3", "unused4")
    assert len(MPM_WORK_COUNT_COLUMNS) == len(SOLID_WORK_COUNT_COLUMNS) == 7
    assert NATIVE_DIAGNOSTIC_COLUMNS == (
        "code",
        "newton_iterations",
        "backtracks",
        "root_residual",
        "linear_relative_residual",
        "live_nodes",
    )
    for dim, width in ((2, 18), (3, 22)):
        layout = scenario_metric_layout(dim)
        columns = layout.columns()
        assert len(columns) == layout.width == width
        assert columns[layout.torso_center_gap] == "torso_center_gap_m"
        assert columns[layout.normal_force_over_weight] == "normal_force_over_weight"
        assert columns[layout.contact_energy] == "contact_energy_j"


@pytest.mark.parametrize("dim,expected_width", [(2, 13), (3, 17)])
def test_locomotion_metric_views_match_persisted_column_names(
    dim: int, expected_width: int
) -> None:
    """The shared prefix does not turn absolute torso height into scenario ground gap."""
    layout = LocomotionMetricLayout(dim)
    columns = locomotion_metric_columns(dim)
    assert len(columns) == layout.width == expected_width
    assert columns[layout.torso_height] == "torso_height_m"
    assert columns[layout.work_positive] == "work_positive_j"
    assert columns[layout.foot_gaps] == [f"foot_{k}_clearance_m" for k in range(layout.legs)]
    assert columns[layout.foot_vx] == [f"foot_{k}_vx_m_s" for k in range(layout.legs)]
    assert FLIGHT_COLUMNS[FlightMetric.torso_surface_gap] == "torso_contact_surface_gap_m"
    assert FLIGHT_COLUMNS[FlightMetric.com_vy] == "com_vy_m_s"
    assert FLIGHT_COLUMNS[FlightMetric.torso_upright] == "torso_upright_cosine"
    assert [LOCOMOTION_SOLVER_COLUMNS[k] for k in LocomotionSolver.repeatability_columns] == [
        "code",
        "min_J_or_area_ratio",
        "support_valid",
        "max_penetration_m",
        "newton_iterations",
    ]
