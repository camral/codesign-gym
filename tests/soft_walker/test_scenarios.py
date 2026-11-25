"""New scenario tests only; dynamics/gait qualification is a separate local run."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import check_embedding
from envs.soft_walker.engine.physics.terrain import from_dict, gap_normal, generate, serialize
from envs.soft_walker.experiments.scenarios.controllers import action, seed_parameters
from envs.soft_walker.experiments.scenarios.rollout import assess
from envs.soft_walker.presets.embodiments.chambers import ChamberDecoder
from envs.soft_walker.presets.scenarios.build import build
from envs.soft_walker.presets.scenarios.config import (
    CONTROLLERS,
    DRIVERS,
    REPRESENTATIONS,
    TERRAINS,
    Scenario,
)
from envs.soft_walker.presets.scenarios.suites import suite
from envs.soft_walker.presets.terrain.scenario_courses import resolved_terrain, terrain_config
from envs.soft_walker.shared import difference_ladder, physical_budget, preflight, write_json
from envs.soft_walker.viewer.terrain_geometry import height_at


def save_ladder(name: str, ladder: dict[str, object]) -> None:
    location = os.environ.get("CODESIGN_SCENARIO_EVIDENCE")
    if location:
        write_json(Path(location) / f"{name}.json", ladder)


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_sampled_chambers_connected_and_bounded(
    driver: str, representation: str, dtype: np.dtype[np.floating]
) -> None:
    report = preflight(Scenario(driver=driver, representation=representation, dtype=dtype.name))
    assert report["geometry_errors"] == []
    assert report["mass_kg"] > 0
    assert report["mask_iou_vs_primitive_seed"] >= 0.9
    assert len(report["active_channels"]) == (4 if driver in ("spring", "mpm2") else 16)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_relaxed_and_categorical_fields(
    dim: int, representation: str, dtype: np.dtype[np.floating]
) -> None:
    decoder = ChamberDecoder(dim, representation, dtype.name, "relaxed", "categorical")
    embodiment = decoder.base()
    assert check_embedding(decoder, embodiment) == []
    points = jnp.asarray(np.random.default_rng(4).uniform(-0.9, 0.9, (40, dim)), dtype)
    fields = decoder.decode(embodiment, points)
    assert np.all(np.isfinite(fields.occupancy))
    assert np.all((fields.occupancy >= 0.02) & (fields.occupancy <= 1.0))
    assert set(np.asarray(fields.young)).issubset(set(decoder.spec.palette))


@pytest.mark.parametrize("name", TERRAINS)
def test_terrain_roundtrip_and_surface_normal(name: str, dtype: np.dtype[np.floating]) -> None:
    terrain = resolved_terrain(name, dim=3, seed=9, dtype=dtype.name)
    other = from_dict(serialize(terrain), dtype=dtype)
    for a, b in zip(terrain, other):
        np.testing.assert_array_equal(a, b)
    query = jnp.asarray([[0.025, 0.07, -0.013], [0.401, 0.08, 0.03]], dtype)
    gap, normal = gap_normal(query, terrain)
    gradient = jax.vmap(jax.grad(lambda x: gap_normal(x, terrain)[0]))(query)
    np.testing.assert_allclose(gradient, normal, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1.0, atol=2e-6)
    heights = height_at(np.asarray([0.11, 0.271, 0.52]), np.asarray([0.0, 0.07, -0.03]), terrain)
    on_surface = np.c_[np.asarray([0.11, 0.271, 0.52]), heights, np.asarray([0.0, 0.07, -0.03])]
    np.testing.assert_allclose(
        gap_normal(jnp.asarray(on_surface, dtype), terrain)[0], 0.0, atol=1e-6
    )


def test_terrain_seed_and_explicit_slope(dtype: np.dtype[np.floating]) -> None:
    a = resolved_terrain("rough_blocks", dim=3, seed=3, dtype=dtype.name)
    b = resolved_terrain("rough_blocks", dim=3, seed=4, dtype=dtype.name)
    assert not np.array_equal(a.centers, b.centers)
    config = terrain_config("uphill", dim=3)
    a, b = generate(config, 1, 3, dtype), generate(config, 999, 3, dtype)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert float(a.normal[0]) < 0
    with pytest.raises(ValueError):
        terrain_config("cross_slope", dim=2)


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_controller_vmap_feedback_and_difference_ladder(
    controller: str, dtype: np.dtype[np.floating]
) -> None:
    case = Scenario(controller=controller, dtype=dtype.name)
    rng = np.random.default_rng(11)
    parameters = seed_parameters(case)
    parameters += rng.normal(scale=0.03, size=parameters.shape).astype(dtype)
    observation = jnp.asarray([0.25, 0.5, 0.1, -0.2, 0.0], dtype)
    times = jnp.asarray([0.03, 0.13, 0.23], dtype)
    batched = jax.jit(jax.vmap(lambda t: action(case, jnp.asarray(parameters), t, observation)))(
        times
    )
    scalar = jnp.stack([action(case, jnp.asarray(parameters), t, observation) for t in times])
    np.testing.assert_allclose(batched, scalar, atol=2e-6)
    assert np.all(np.abs(batched) <= 1.0)
    if controller in ("feedback", "neural"):
        changed = action(case, jnp.asarray(parameters), times[0], observation.at[2].add(0.3))
        assert not np.array_equal(changed, scalar[0])
    loss = jax.jit(
        lambda p: jnp.sum(jax.vmap(lambda t: action(case, p, t, observation))(times) ** 2)
    )
    gradient = jax.jit(jax.value_and_grad(loss))
    ladder = difference_ladder(loss, gradient, parameters, seed=19)
    save_ladder("controller_" + controller, ladder)
    assert ladder["agreement_passed"]
    assert ladder["best"]["relative_error"] < (1e-6 if dtype == np.dtype("float64") else 5e-3)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_geometry_score_ad_ladder(representation: str, dtype: np.dtype[np.floating]) -> None:
    decoder = ChamberDecoder(2, representation, dtype.name, "relaxed")
    embodiment = {k: jnp.asarray(v) for k, v in decoder.base().items()}
    key = {
        "primitives": "centers",
        "direct": "shape_values",
        "bspline": "shape_values",
        "cppn": "weights",
    }[representation]
    p = np.asarray(embodiment[key])
    points = jnp.asarray(np.random.default_rng(13).uniform(-0.87, 0.87, (31, 2)), dtype)
    baseline_alpha = jax.lax.stop_gradient(decoder.decode(embodiment, points).occupancy)

    def target(parameters: jax.Array) -> jax.Array:
        candidate = dict(embodiment)
        candidate[key] = parameters
        alpha = decoder.decode(candidate, points).occupancy
        # Same derivative as mean(alpha**2), minus a FIXED reference constant.
        # Factor the difference of squares before reduction to avoid comparing
        # two ~0.6 float32 losses for a nearly cancelling 3e-4 derivative.
        return jnp.mean((alpha - baseline_alpha) * (alpha + baseline_alpha))

    loss = jax.jit(target)
    gradient = jax.jit(jax.value_and_grad(loss))
    ladder = difference_ladder(loss, gradient, p, seed=17)
    save_ladder("geometry_" + representation, ladder)
    assert ladder["agreement_passed"]


def test_budget_matches_independent_linear_model() -> None:
    case = Scenario(controller="pulse", gait="pronk", lift=1.6)
    result = physical_budget(case)
    assert result["mass_kg"] == pytest.approx(0.1152)
    assert result["axial_stiffness_n_m"] == pytest.approx(800.0)
    assert result["target_takeoff_speed_m_s"] ** 2 == pytest.approx(2 * 9.81 * 0.01)
    assert result["energy_margin_at_assumed_20_percent"] > 1.0
    # This checks algebra, not whether any fraction is physically recovered.
    categorical = preflight(replace(case, leg_young=100000.0), material_mode="categorical")
    assert categorical["airborne_screen"]["screening_leg_young_pa"] == pytest.approx(80000.0)


def test_suite_serialization_and_invalid_configs() -> None:
    assert len(suite("drivers")) == 5
    assert len(suite("terrains")) == 8
    assert len(suite("representations")) == 8
    assert len(suite("controllers")) == 4
    assert len(suite("airborne")) == 8
    for case in suite("drivers") + suite("airborne"):
        assert Scenario.from_dict(json.loads(json.dumps(case.to_dict()))) == case
    for data in (
        {"substeps": 0},
        {"period": -0.1},
        {"driver": "unknown"},
        {"terrain": "cross_slope"},
        {"density": 1.0},
    ):
        with pytest.raises(ValueError):
            Scenario(**data)


def test_no_false_flight_from_initial_clearance_or_failure() -> None:
    case = Scenario()
    rows = np.zeros((case.steps + 1, 18))
    rows[:, 0] = np.arange(case.steps + 1) * case.control_dt
    rows[:, 7] = 0.08
    rows[:, 9:11] = 0.0005
    rows[:, 13] = 1.0
    rows[:, 14] = 0.0005
    rows[:, 15] = 0.03
    verdict = assess(rows, case)
    assert verdict["valid_rollout"]
    assert not verdict["walking_gate_passed"]
    assert not verdict["repeated_airborne_observed"]
    # Two synthetic flight intervals are an ASSESSMENT fixture, not physics.
    for a, b in ((60, 70), (120, 130)):
        rows[a:b, 13] = 0.0
        rows[a:b, 14] = 0.004
    verdict = assess(rows, case)
    assert verdict["repeated_airborne_observed"]
    rows[-1, 6] = 1
    assert not assess(rows, case)["repeated_airborne_observed"]


def test_new_factory_observation_capacity_and_channels(dtype: np.dtype[np.floating]) -> None:
    # Construction does not integrate. Physical smoke/replay is an explicit job.
    selected_device = jax.devices()[0]
    for driver in DRIVERS:
        case = Scenario(driver=driver, tier="smoke", dtype=dtype.name)
        env, embodiment = build(case, device=selected_device)
        assert env.device == selected_device
        assert env.embodiment_space.contains(embodiment)
        assert env.channels == case.channels
        assert env.max_observation_space_size == env.observation_size
        env.close()
    jax.clear_caches()
