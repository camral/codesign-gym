"""Tests for the vectorized SoftWalker env (envs.soft_walker, native Warp backend) and its registration.

Construction, spaces, the per-cell embodiment and validation run in seconds. Anything that settles or steps the
physics runs the real task (0.25 ms substeps, 0.2 s settling), which on CPU takes minutes per lane, so those tests
are marked ``slow`` and share one module-scoped two-lane SoftWalker-Spring2D env. The engine's own physics tests live
in tests/soft_walker/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from gymnasium.error import ResetNeeded

pytest.importorskip("warp")

import codesign_gym as cg
from codesign_gym import SingleEnv
from envs.soft_walker import config as C, embodiment as E
from envs.soft_walker.vector_env import SoftWalkerVectorEnv, _stack

IDS = {"SoftWalker-Spring2D": "spring", "SoftWalker-MPM2D": "mpm2", "SoftWalker-MPM3D": "mpm3", "SoftWalker-Beam3D": "beam", "SoftWalker-HexFEM3D": "fem"}
TERRAIN_IDS = {f"SoftWalker-{n}{t.capitalize()}": (d, t) for n, d in (("Spring2D", "spring"), ("MPM2D", "mpm2"), ("MPM3D", "mpm3"), ("Beam3D", "beam"), ("HexFEM3D", "fem"))
               for t in ("stairs", "gaps")}
STEP_ID = "SoftWalker-Spring2D"
# Physics-running (slow) tests use WARP_TEST_DEVICE like the engine's Warp tests: cpu by default, cuda on the GPU box.
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


def _cells(env: SoftWalkerVectorEnv, vector: np.ndarray) -> dict[str, np.ndarray]:
    return E.to_cells(vector.copy(), env.dim)


# --- registry / construction -------------------------------------------------------


def test_registry_lists_all_soft_walker_envs() -> None:
    assert set(IDS) <= set(cg.list_envs())


@pytest.mark.parametrize("env_id", IDS)
def test_all_ids_use_one_builder_with_driver(env_id: str) -> None:
    path, defaults = cg.REGISTRY[env_id]
    assert path == "envs.soft_walker.registration:build_vec"
    assert defaults == {"driver": IDS[env_id]}


@pytest.mark.parametrize("env_id", IDS)
def test_make_vec_returns_per_cell_base_in_space(env_id: str) -> None:
    env, base = cg.make_vec(env_id, num_envs=2, device="cpu")
    try:
        assert isinstance(env, SoftWalkerVectorEnv)
        assert np.array_equal(base, env.base_embodiment)
        assert base.dtype == np.float64 and base.shape == (
            np.prod(C.grid(IDS[env_id])) * E.width(env.dim),
        )
        assert env.embodiment_space.contains(base)
        assert env.single_action_space.shape == (C.CHANNELS[env.dim],)
        assert env.validate_embodiment(base).tolist() == [True, True]
    finally:
        env.close()


def test_warp_path_never_imports_jax():
    code = ("import sys, codesign_gym as cg; env, base = cg.make_vec('SoftWalker-Spring2D', num_envs=1, device='cpu'); "
            "env.validate_embodiment(base); env.close(); assert 'jax' not in sys.modules, 'jax imported'")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_unknown_kwargs_rejected() -> None:
    with pytest.raises(TypeError):
        cg.make_vec(STEP_ID, num_envs=1, friction=0.5)
    with pytest.raises(ValueError):
        cg.make_vec("SoftWalker-MPM2D", num_envs=1, device="cpu", terrain="uphill")


def test_metadata_advertises_partial_reset():
    # Callers (e.g. the baselines adapter) route reset_mask partial resets on this flag.
    assert SoftWalkerVectorEnv.metadata["supports_partial_reset"] is True
    assert SoftWalkerVectorEnv.metadata["supports_transition_embodiment"] is False


def test_float32_option_constructs() -> None:
    env, base = cg.make_vec(STEP_ID, num_envs=1, device="cpu", dtype="float32")
    try:
        assert base.dtype == np.float32 and env.single_observation_space.dtype == np.float32
    finally:
        env.close()


def test_single_env_validate_is_bool() -> None:
    env, base = cg.make(STEP_ID, device="cpu")
    try:
        assert isinstance(env, SingleEnv) and base.shape == env.embodiment_space.shape
        assert env.validate_embodiment(base) is True
    finally:
        env.close()


# --- terrain presets -----------------------------------------------------------------


def test_terrain_ids_registered_with_driver_and_terrain() -> None:
    for env_id, (driver, terrain) in TERRAIN_IDS.items():
        assert cg.REGISTRY[env_id] == (
            "envs.soft_walker.registration:build_vec",
            {"driver": driver, "terrain": terrain},
        )


@pytest.mark.parametrize("env_id", TERRAIN_IDS)
def test_terrain_preset_builds_and_validates(env_id: str) -> None:
    env, base = cg.make_vec(env_id, num_envs=1, device="cpu")
    try:
        template = env.lanes[0].template
        assert template.terrain is not None and env.validate_embodiment(base).tolist() == [True]
    finally:
        env.close()


@pytest.mark.parametrize("terrain", ["stairs", "gaps"])
def test_spring_terrain_routes_through_shared_contact(terrain: str) -> None:
    from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel

    bundle = C.build_bundle("spring", C.base_props("spring"), "float64", terrain)
    owner = SolidDeviceModel(bundle, "cpu")
    assert owner.model.legacy_contact == 0
    assert np.allclose(owner.model.terrain.box_centers.numpy(), bundle.terrain["centers"])
    assert float(owner.model.terrain.plane_offset) == (0.0 if terrain == "stairs" else -C.GAP_DEPTH)
    # The spring builder already includes contact modulus in this array. The
    # general terrain law must retain the unit multiplier used by flat contact.
    flat = C.build_bundle("spring", C.base_props("spring"), "float64")
    assert bundle.config.contact_stiffness == flat.config.contact_stiffness == 1.0
    np.testing.assert_array_equal(bundle.fields["contact_area"], flat.fields["contact_area"])


@pytest.mark.parametrize("dim,driver", [(2, "mpm2"), (3, "fem")])
def test_stairs_match_jax_scenario(dim: int, driver: str) -> None:
    terrains = pytest.importorskip("envs.soft_walker.presets.terrain.scenario_courses")
    jax_boxes = np.asarray(terrains.terrain_config("stairs", dim=dim).fixed_boxes, float)
    t = C.terrain(driver, "stairs", "float64")
    assert np.allclose(t["centers"], jax_boxes[:, :dim]) and np.allclose(
        t["half_sizes"], jax_boxes[:, 3 : 3 + dim]
    )
    assert np.allclose(t["rounding"], jax_boxes[:, 6]) and float(t["offset"]) == 0.0


def test_gap_layout() -> None:
    t = C.terrain("mpm2", "gaps", "float64")
    lo, hi = t["centers"][:, 0] - t["half_sizes"][:, 0], t["centers"][:, 0] + t["half_sizes"][:, 0]
    gaps = np.round(lo[1:] - hi[:-1], 6)
    assert (
        gaps.tolist() == [0.04, 0.20, 0.02, 0.08]
        and np.allclose(hi[0], 0.22)
        and float(t["offset"]) == -0.05
    )
    assert np.allclose(
        t["centers"][:, 1] + t["half_sizes"][:, 1], 0.0
    )  # platform tops at ground level
    shape, origin = C.mpm_grid("mpm2", "gaps")
    assert (
        origin[0] + (shape[0] - 1.5) * C.CELL > hi[-2] + C.GAP_WIDTHS[-1] + C.size("mpm2")[0]
    )  # the body can clear the last gap


# --- per-cell embodiment ------------------------------------------------------------


@pytest.fixture(scope="module")
def spring_env() -> Iterator[SoftWalkerVectorEnv]:
    """Share the costly real two-lane spring environment and release it after the module."""
    env, _ = cg.make_vec(STEP_ID, num_envs=2, device=DEVICE)
    yield env
    env.close()


def test_cells_roundtrip(spring_env: SoftWalkerVectorEnv) -> None:
    base = spring_env.base_embodiment
    assert np.array_equal(E.from_cells(E.to_cells(base, 2), 2, base.dtype), base)


def test_validation_is_structural_and_per_env(spring_env: SoftWalkerVectorEnv) -> None:
    env, base = spring_env, spring_env.base_embodiment
    empty = _cells(env, base)
    empty["occupancy"][:] = 0.0
    split = _cells(env, base)
    split["occupancy"].reshape(env.grid)[4, :] = (
        0.0  # a full column removed: two disconnected halves
    )
    corner = _cells(env, base)
    corner["occupancy"].reshape(env.grid)[:] = 0.0
    corner["occupancy"].reshape(env.grid)[0, 0] = corner["occupancy"].reshape(env.grid)[1, 1] = (
        1.0  # joined only at a corner
    )
    fraction = _cells(env, base)
    fraction["kind"][0] = 1.5
    fibreless = _cells(env, base)
    fibreless["direction"][0] = 0.0
    notched = _cells(env, base)
    notched["occupancy"].reshape(env.grid)[3:5, :2] = 0.0  # legs: still one body
    cases = {
        "empty": (empty, "EMPTY_GEOMETRY"),
        "split": (split, "DISCONNECTED_GEOMETRY"),
        "corner": (corner, "NONMANIFOLD_JUNCTION"),
        "fraction": (fraction, "INTEGER_REQUIRED"),
        "fibreless": (fibreless, "DIRECTIONAL_ACTUATOR_WITHOUT_FIBER"),
    }
    for name, (props, code) in cases.items():
        errors = E.structural_errors(E.from_cells(props, 2, float), env.grid, env.channels)
        assert any(e.startswith(code) for e in errors), (name, errors)
    notched_v = E.from_cells(notched, 2, float)
    assert env.validate_embodiment([base, notched_v]).tolist() == [True, True]
    assert env.validate_embodiment(np.stack([base, E.from_cells(empty, 2, float)])).tolist() == [
        True,
        False,
    ]
    assert env.validate_embodiment(np.zeros(5)).tolist() == [False, False]
    assert env.validate_embodiment(np.full(base.shape, 1e6)).tolist() == [False, False]


@pytest.mark.parametrize("env_id,engine_accepts", [("SoftWalker-MPM2D", True), ("SoftWalker-Spring2D", False), ("SoftWalker-Beam3D", False)])
def test_fibreless_actuator_in_an_empty_cell_follows_the_engine(env_id, engine_accepts):
    """The wrapper's fibre rule covers solid cells. For an empty cell the engine decides, through the build that
    validate_embodiment runs: the MPM builders accept a fibreless directional actuator there, the solid ones refuse."""
    env, base = cg.make_vec(env_id, num_envs=1, device="cpu")
    try:
        props = _cells(env, base)
        props["occupancy"][0] = 0.                     # a corner cell, emptied: still one body
        props["kind"][0], props["direction"][0] = 1., 0.
        vector = E.from_cells(props, env.dim, float)
        assert not any(e.startswith("DIRECTIONAL") for e in E.structural_errors(vector, env.grid, env.channels))
        assert env.validate_embodiment(vector).tolist() == [engine_accepts]
    finally:
        env.close()


def test_beam_accepts_spatial_poisson():
    env, base = cg.make_vec("SoftWalker-Beam3D", num_envs=1, device="cpu")
    try:
        mixed = E.to_cells(base.copy(), 3)
        mixed["poisson"][:] = np.linspace(0.25, 0.45, len(mixed["poisson"]))
        vector = E.from_cells(mixed, 3, float)
        assert not E.structural_errors(vector, env.grid, env.channels)
        assert env.validate_embodiment(vector).tolist() == [True]
        varied = C.build_bundle("beam", mixed, "float64")
        uniform = env.lanes[0].template
        assert np.ptp(varied.fields["poisson"]) > 0.1
        assert not np.allclose(varied.fields["extension"], uniform.fields["extension"])
        assert not np.allclose(varied.fields["bending"], uniform.fields["bending"])
    finally:
        env.close()


def _beam_dead_channel(base):
    """A block whose last (corner) cell is empty and the only user of channel 5."""
    cells = E.to_cells(base.copy(), 3)
    cells["channel"][cells["channel"] == 5] = 0.
    cells["occupancy"][-1], cells["channel"][-1], cells["kind"][-1], cells["strength"][-1] = 0., 5., 2., 1.
    return cells


def test_beam_action_mask_ignores_channels_only_empty_cells_use():
    from envs.soft_walker.vector_env import beam_action_mask
    env, base = cg.make_vec("SoftWalker-Beam3D", num_envs=1, device="cpu")
    try:
        cells = _beam_dead_channel(base)
        bundle = C.build_bundle("beam", cells, "float64")
        lane = env.lanes[0]
        assert lane._mask(bundle)[5]                                        # the engine's own mask counts the dead link
        mask = beam_action_mask(bundle.fields, lane.channels, lane.action_size)
        assert not mask[5] and mask.shape == (lane.action_size,)
        live = sorted({int(c) for c, o, k in zip(cells["channel"], cells["occupancy"], cells["kind"]) if o >= .5 and k != 0})
        assert np.flatnonzero(mask).tolist() == live
    finally:
        env.close()


@pytest.mark.slow
def test_beam_reset_reports_the_wrapper_action_mask():
    env, base = cg.make_vec("SoftWalker-Beam3D", num_envs=1, device=DEVICE)
    try:
        obs, rejected, info = env.reset(seed=0, embodiment=E.from_cells(_beam_dead_channel(base), 3, float)[None])
        lane = env.lanes[0]
        assert rejected.tolist() == [False] and not info["action_mask"][0][5] and not lane.action_mask[5]
        np.testing.assert_array_equal(obs[0, lane.raw_observation_size - lane.action_size:lane.raw_observation_size], lane.action_mask)
    finally:
        env.close()


def test_validation_runs_the_builder(monkeypatch):
    env, base = cg.make_vec("SoftWalker-Beam3D", num_envs=1, device="cpu")
    try:
        calls = []

        def rejected_builder(*args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))
            raise ValueError("injected native builder rejection")

        monkeypatch.setattr(C, "build_bundle", rejected_builder)
        assert not E.structural_errors(base, env.grid, env.channels)
        assert env.validate_embodiment(base).tolist() == [False]
        assert len(calls) == 1
        _, errors = env._resolve(base)
        assert errors == ["BUILD: injected native builder rejection"]
    finally:
        env.close()


@pytest.mark.parametrize("driver", ["mpm2", "mpm3", "fem"])
def test_spatial_poisson_survives_native_cell_conversion(driver: str) -> None:
    props = C.base_props(driver)
    props["poisson"][:] = np.linspace(0.25, 0.45, len(props["poisson"]))
    vector = E.from_cells(props, C.dim_of(driver), np.float64)
    restored = E.to_cells(vector, C.dim_of(driver))
    bundle = C.build_bundle(driver, restored, "float64")
    if driver.startswith("mpm"):
        # Recover owning cell from particle reference coordinates instead of
        # assuming the builder's internal particle flattening order.
        ij = np.floor((bundle.fields["reference"] - C.origin(driver)) / C.CELL).astype(int)
        cells = np.ravel_multi_index(tuple(ij.T), C.grid(driver))
        expected = props["poisson"][cells]
    else:
        expected = props["poisson"]
    np.testing.assert_array_equal(bundle.fields["poisson"], expected)


def test_occupancy_is_a_hard_switch_independent_of_material() -> None:
    props = C.base_props("spring")
    props["occupancy"][:] = 0.6  # above threshold: fully solid, nothing scaled
    half = C.build_bundle("spring", props, "float64")
    full = C.build_bundle("spring", C.base_props("spring"), "float64")
    assert np.allclose(half.fields["mass"], full.fields["mass"]) and np.allclose(
        half.fields["stiffness"], full.fields["stiffness"]
    )


@pytest.mark.parametrize("driver", C.DRIVERS)
def test_builder_schema_fixed_under_design_changes(driver: str) -> None:
    base = C.base_props(driver)
    notched = {k: np.array(v, copy=True) for k, v in base.items()}
    notched["occupancy"].reshape(C.grid(driver))[3:5, :2] = 0.0
    a, b = C.build_bundle(driver, base, "float64"), C.build_bundle(driver, notched, "float64")
    a.validate(), b.validate()
    assert all(
        a.fields[k].shape == b.fields[k].shape and a.fields[k].dtype == b.fields[k].dtype
        for k in a.fields
    )


def test_topology_port_matches_engine() -> None:
    topology_checks = pytest.importorskip("envs.soft_walker.engine.topology_checks")
    rng = np.random.default_rng(0)
    for shape in ((4, 3), (3, 3, 2)):
        for _ in range(200):
            solid = rng.random(shape) < 0.6
            assert E.topology_errors(solid) == topology_checks.errors(solid.astype(float), shape)


def test_stack_info_per_key() -> None:
    stacked = _stack(
        [
            {"valid": True, "action_mask": np.ones(3), "metrics": {"d": 1.0}, "errors": ["x"]},
            {"valid": False, "action_mask": np.zeros(3), "metrics": {"d": 2.0}, "errors": []},
        ]
    )
    assert stacked["valid"].tolist() == [True, False] and stacked["action_mask"].shape == (2, 3)
    assert stacked["metrics"]["d"].tolist() == [1.0, 2.0] and stacked["errors"] == [["x"], []]


# --- reset / step (real task; minutes per lane on CPU) --------------------------------


@pytest.mark.slow
def test_reset_step_shapes_and_frozen_embodiment(spring_env: SoftWalkerVectorEnv) -> None:
    env, base = spring_env, spring_env.base_embodiment
    obs, rejected, info = env.reset(seed=0, embodiment=np.stack([base, base]))
    assert obs.shape == (2,) + env.single_observation_space.shape and rejected.tolist() == [
        False,
        False,
    ]
    assert info["action_mask"].shape == (2, env.single_action_space.shape[0]) and info[
        "valid"
    ].tolist() == [True, True]
    actions = np.zeros(env.action_space.shape)
    actions[:, : env.channels] = 0.5
    obs, reward, term, trunc, info = env.step(actions)
    assert reward.shape == term.shape == trunc.shape == (2,) and np.all(np.isfinite(reward))
    assert info["valid"].tolist() == [True, True] and info["autoreset"].tolist() == [False, False]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env.step(np.zeros(env.action_space.shape), embodiment=[base, base])
    assert any("not permitted" in str(w.message) for w in caught)


@pytest.mark.slow
@pytest.mark.parametrize("env_id", ["SoftWalker-MPM2DStairs", "SoftWalker-MPM2DGaps", "SoftWalker-Spring2DStairs", "SoftWalker-Spring2DGaps"])
def test_base_body_settles_on_terrain(env_id):
    env, base = cg.make_vec(env_id, num_envs=1, device=DEVICE)
    try:
        obs, rejected, info = env.reset(seed=0, embodiment=base)
        assert rejected.tolist() == [False], info.get("embodiment_errors")
        lane = env.lanes[0]
        state = lane.owner.initial
        y = (
            state.x.numpy()[:, 1]
            if env.driver.startswith("mpm")
            else (lane.owner.model.reference.numpy() + state.u.numpy())[:, 1]
        )
        assert (
            y.min() > -0.005
        )  # rests on the ground / start platform (penalty contact penetrates a few mm), not in a pit
    finally:
        env.close()


@pytest.mark.slow
def test_partial_reset_rejection_keeps_other_lane_then_step_raises(
    spring_env: SoftWalkerVectorEnv,
) -> None:
    env, base = spring_env, spring_env.base_embodiment
    env.reset(seed=0, embodiment=[base, base])
    empty = E.to_cells(base.copy(), 2)
    empty["occupancy"][:] = 0.0
    obs, rejected, info = env.reset(
        embodiment=[None, E.from_cells(empty, 2, float)],
        options={"reset_mask": np.array([False, True])},
    )
    assert rejected.tolist() == [False, True] and "EMPTY_GEOMETRY" in info["embodiment_errors"][1]
    assert np.any(obs[0] != 0)  # lane 0 kept its live state
    with pytest.raises(ResetNeeded):
        env.step(np.zeros(env.action_space.shape))


@pytest.mark.slow
def test_partial_reset_keeps_a_lane_pending_autoreset_valid(spring_env):
    """A lane that ended on a physics failure (engine marks it uninitialized) is not
    rejected by a partial reset of the others: its next step autoresets it."""
    env, base = spring_env, spring_env.base_embodiment
    env.reset(seed=0, embodiment=[base, base])
    env._pending[0], env.lanes[0].initialized = True, False                # what a failed step leaves behind
    obs, rejected, info = env.reset(options={"reset_mask": np.array([False, True])})
    assert rejected.tolist() == [False, False] and list(info["valid"]) == [True, True]
    obs, reward, term, trunc, info = env.step(np.zeros(env.action_space.shape))
    assert info["autoreset"].tolist() == [True, False] and env.lanes[0].initialized


@pytest.mark.slow
def test_autoreset_restores_the_settled_state_without_settling_again(spring_env, monkeypatch):
    """The engine keeps the state the last explicit reset accepted; a next-step autoreset restores it exactly."""
    env, base = spring_env, spring_env.base_embodiment
    obs0, _, _ = env.reset(seed=0, embodiment=[base, base])
    env.step(np.zeros(env.action_space.shape))
    env._pending[0] = True
    def no_settling(designs, seeds):
        raise AssertionError("autoreset must not settle again")
    monkeypatch.setattr(env, "_reset_lanes", no_settling)
    obs, reward, term, trunc, info = env.step(np.zeros(env.action_space.shape))
    assert info["autoreset"].tolist() == [True, False] and list(info["valid"]) == [True, True] and reward[0] == 0.
    np.testing.assert_array_equal(obs[0], obs0[0])
    assert not env._pending[0] and np.isfinite(reward[1])


def test_near_grid_edge_matches_the_engine_support_limits():
    from envs.soft_walker.vector_env import near_grid_edge
    origin, h, shape = (0., 0.), .1, (10, 10)                          # support: [0.05, 0.85) per axis
    live = np.ones(1, bool)
    assert not near_grid_edge(np.array([[.45, .45]]), live, origin, h, shape)
    assert near_grid_edge(np.array([[.14, .45]]), live, origin, h, shape)       # within one cell of the low limit
    assert near_grid_edge(np.array([[.45, .76]]), live, origin, h, shape)       # ... of the high limit
    assert near_grid_edge(np.array([[.45, .90]]), live, origin, h, shape)       # already outside
    assert not near_grid_edge(np.array([[.45, .90]]), ~live, origin, h, shape)  # empty particles do not count


@pytest.mark.slow
def test_mpm_step_reports_out_of_domain_only_at_the_grid_edge():
    env, base = cg.make_vec("SoftWalker-MPM2D", num_envs=1, device=DEVICE)
    try:
        env.reset(seed=0, embodiment=base[None])
        info = env.step(np.zeros(env.action_space.shape))[4]
        assert info["out_of_domain"].tolist() == [False] and not env._off_grid(0)
        lane, cfg = env.lanes[0], env.lanes[0]._episode.step.config
        x = lane.owner.initial.x.numpy()
        x[lane._mass > 0, 0] = cfg.origin[0] + (cfg.grid_shape[0] - 1.5) * cfg.spacing - .5 * cfg.spacing   # front at the edge
        lane.owner.initial.x.assign(x)
        assert env._off_grid(0)
    finally:
        env.close()


# --- rendering ----------------------------------------------------------------------

from envs.soft_walker import render as R  # noqa: E402
from envs.soft_walker import render3d as R3  # noqa: E402


def _notched(driver: str) -> np.ndarray:
    """Remove bottom cells to produce visible legs without disconnecting the torso."""
    props = C.base_props(driver)
    props["occupancy"].reshape(C.grid(driver))[3:5, :2] = 0.0  # remove two bottom columns: legs
    return E.from_cells(props, C.dim_of(driver), np.float64)


@pytest.mark.parametrize(
    "driver,terrain",
    [("spring", "flat"), ("mpm2", "gaps"), ("fem", "stairs"), ("beam", "gaps"), ("mpm3", "flat")],
)
def test_draw_frame_body_terrain_and_removed_cells(driver: str, terrain: str) -> None:
    design = _notched(driver)
    bundle = C.build_bundle(driver, E.to_cells(design, C.dim_of(driver)), "float64", terrain)
    frame = R.draw(
        driver, bundle.fields["reference"], bundle.fields, design, bundle.terrain, label="x"
    )
    assert frame.shape == (R.HEIGHT, R.WIDTH, 3) and frame.dtype == np.uint8
    colours = {tuple(c) for c in frame.reshape(-1, 3)[::7]}
    assert (
        R.channel_colours(C.CHANNELS[C.dim_of(driver)])[0] in colours or R.PASSIVE in colours
    )  # body drawn
    if terrain != "flat":
        assert R.TERRAIN in colours
    if (
        C.dim_of(driver) == 2
    ):  # the removed leg gap (x in [0.06, 0.10], y < 0.04) shows background, not body
        centre = bundle.fields["reference"][:, 0].mean()
        x = int((0.08 - (centre - R.VIEW_WIDTH / 2)) * R.WIDTH / R.VIEW_WIDTH)
        y = int(R.HEIGHT - (0.02 - R.VIEW_BOTTOM) * R.WIDTH / R.VIEW_WIDTH)
        assert tuple(frame[y, x]) == R.BACKGROUND


def test_render_mode_off_returns_none_and_invalid_raises() -> None:
    env, _ = cg.make_vec(STEP_ID, num_envs=1, device="cpu")
    try:
        assert env.render() is None
    finally:
        env.close()
    with pytest.raises(ValueError):
        cg.make_vec(STEP_ID, num_envs=1, device="cpu", render_mode="bogus")
    env, _ = cg.make_vec(STEP_ID, num_envs=1, device="cpu", render_mode="human")
    try:
        assert env.render() is None  # no world has been reset yet
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.skipif(not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
                    reason="Newton human viewer requires a display")
def test_human_newton_viewer_refreshes_during_rollout():
    env, base = cg.make(STEP_ID, device=DEVICE, render_mode="human")
    try:
        env.reset(seed=0, embodiment=base)
        renderer = env.vector_env._renderers[0][1]
        assert renderer is not None and renderer.studio.viewer.is_running()
        assert env.render() is None
        env.step(np.zeros(env.action_space.shape, dtype=np.float64))
        assert env.vector_env._renderers[0][1] is renderer
    finally:
        env.close()


@pytest.mark.slow
def test_render_frames_after_reset_vector_and_single() -> None:
    env, base = cg.make_vec(STEP_ID, num_envs=2, device=DEVICE, render_mode="rgb_array")
    try:
        env.reset(seed=0, embodiment=np.stack([base, base]))
        frames = env.render()
        shape = frames[0].shape  # Newton 3D where an OpenGL context opens, else the Pillow side view
        assert shape in ((R3.HEIGHT, R3.WIDTH, 3), (R.HEIGHT, R.WIDTH, 3))
        assert isinstance(frames, tuple) and len(frames) == 2 and all(f.shape == shape and f.dtype == np.uint8 for f in frames)
        assert env.render(env_idx=1).shape == shape
    finally:
        env.close()
    single, base = cg.make(STEP_ID, device=DEVICE, render_mode="rgb_array")
    try:
        single.reset(seed=0, embodiment=base)
        assert single.render().shape == shape
    finally:
        single.close()


def _frame_design(driver):
    """Top row plus four corner legs (a thin frame, most cells empty)."""
    m = np.zeros(C.grid(driver), bool)
    m[:, -1] = True
    m[0], m[-1] = (np.ones_like(m[0]), np.ones_like(m[-1])) if C.dim_of(driver) == 2 else (m[0], m[-1])
    if C.dim_of(driver) == 3:
        m[:, -1] = False
        m[:, -1, 0] = m[:, -1, -1] = m[0, -1, :] = m[-1, -1, :] = True
        for x in (0, -1):
            for z in (0, -1):
                m[x, :, z] = True
    props = C.base_props(driver)
    props["occupancy"] = m.ravel().astype(float)
    return E.from_cells(props, C.dim_of(driver), np.float64), m


@pytest.mark.parametrize("driver", C.DRIVERS)
def test_render3d_topology_follows_occupancy(driver):
    design, mask = _frame_design(driver)
    assert not E.structural_errors(design, C.grid(driver), C.CHANNELS[C.dim_of(driver)])
    fields = C.build_bundle(driver, E.to_cells(design, C.dim_of(driver)), "float64").fields
    topo = R3.topology(driver, fields, design)
    active, ref = topo["active"], np.asarray(fields["reference"])
    assert active.any() and topo["torso"].any() and not (topo["torso"] & ~active).any()
    assert ref[topo["torso"], 1].min() > ref[active, 1].min()  # torso is the top row, legs below it
    if driver == "beam":  # one node per cell; links only between occupied cells
        assert (active == mask.ravel()).all() and active[topo["edges"]].all()
    if driver == "spring":
        assert len(topo["cells"]) == mask.sum()
    if driver == "fem":  # a 1-cell frame: every hex face except the shared ones is boundary
        assert len(topo["faces"]) and active[topo["faces"]].all()


@pytest.mark.parametrize("driver,terrain", [("spring", "flat"), ("mpm2", "gaps"), ("fem", "stairs"), ("beam", "gaps"), ("mpm3", "stairs")])
def test_render3d_frame_without_physics(driver, terrain):
    import warp as wp
    design, _ = _frame_design(driver)
    bundle = C.build_bundle(driver, E.to_cells(design, C.dim_of(driver)), "float64", terrain)
    device = wp.get_device()
    renderer = R3.make_renderer(driver, bundle.fields, design, bundle.terrain)
    if renderer is None:
        pytest.skip(f"no OpenGL context for Newton: {R3._unavailable}")
    try:
        first = renderer.draw(bundle.fields["reference"], "x")
        moved = np.asarray(bundle.fields["reference"], float) + np.eye(C.dim_of(driver))[1] * .01
        second = renderer.draw(moved)
    finally:
        renderer.close()
    assert first.shape == (R3.HEIGHT, R3.WIDTH, 3) and first.dtype == np.uint8 and first.std() > 10
    assert not np.array_equal(first, second)  # frames follow the positions passed in
    assert wp.get_device() == device          # the viewer's set_device("cuda:0") is skipped


def test_render3d_unavailable_warns_once_and_falls_back(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("no EGL display")
    monkeypatch.setattr(R3, "NewtonRenderer", broken)
    monkeypatch.setattr(R3, "_unavailable", None)
    design, _ = _frame_design("spring")
    fields = C.build_bundle("spring", E.to_cells(design, 2), "float64").fields
    with pytest.warns(RuntimeWarning, match="libegl1"):
        assert R3.make_renderer("spring", fields, design, None) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert R3.make_renderer("spring", fields, design, None) is None  # cached: no second warning


# --- episode end, single-env path, determinism (full episodes: run with WARP_TEST_DEVICE=cuda) ---------------------


def _gait(channels: int, action_size: int, steps: int, seed: int = 0) -> np.ndarray:
    """Generate reproducible native-channel commands while leaving padded channels zero."""
    rng = np.random.default_rng(seed)
    amplitude, phase = rng.uniform(0.0, 1.0, channels), rng.uniform(0.0, 2 * np.pi, channels)
    t = np.arange(steps)[:, None] * C.CONTROL_DT
    actions = np.zeros((steps, action_size))
    actions[:, :channels] = amplitude * np.sin(2 * np.pi * t / 0.4 + phase)
    return actions


@pytest.mark.slow
def test_episode_truncates_at_horizon_then_autoresets_with_accepted_design() -> None:
    env, base = cg.make_vec(STEP_ID, num_envs=1, device=DEVICE)
    try:
        design = _notched("spring")
        obs0, rejected, _ = env.reset(seed=0, embodiment=design[None])
        assert rejected.tolist() == [False]
        actions = _gait(env.channels, env.single_action_space.shape[0], C.HORIZON)
        for k in range(C.HORIZON):
            obs, reward, term, trunc, info = env.step(actions[k][None])
            assert not term[0] and info["valid"].tolist() == [True]
            assert trunc[0] == (k == C.HORIZON - 1)  # truncated exactly at the horizon
        obs, reward, term, trunc, info = env.step(np.zeros(env.action_space.shape))  # next step: autoreset
        assert info["autoreset"].tolist() == [True] and reward.tolist() == [0.] and not term[0] and not trunc[0]
        np.testing.assert_array_equal(obs, obs0)  # same accepted (notched) design, same settled state, restored exactly
        obs, reward, term, trunc, info = env.step(actions[0][None])  # and the new episode runs
        assert info["autoreset"].tolist() == [False] and np.isfinite(reward[0])
    finally:
        env.close()


@pytest.mark.slow
def test_single_env_path_is_unbatched() -> None:
    env, base = cg.make(STEP_ID, device=DEVICE)
    try:
        assert isinstance(env, SingleEnv) and base.shape == env.embodiment_space.shape
        obs, rejected, info = env.reset(seed=0, embodiment=base)
        assert rejected is False and obs.shape == env.observation_space.shape
        assert (
            info["action_mask"].shape == env.action_space.shape
            and info["observation_mask"].shape == obs.shape
        )
        action = _gait(env.channels, env.action_space.shape[0], 1)[0]  # unbatched (A,) action
        obs, reward, term, trunc, info = env.step(action)
        assert (
            obs.shape == env.observation_space.shape
            and isinstance(reward, float)
            and np.isfinite(reward)
        )
        assert term is False and trunc is False and info["valid"] is not None
        assert env.validate_embodiment(base) is True
    finally:
        env.close()


@pytest.mark.slow
def test_same_seed_design_and_actions_are_deterministic() -> None:
    """Two envs, same seed, design and actions -> the same trajectory (bitwise on CPU; float atomics may reorder on CUDA)."""
    steps = 20
    runs = []
    for _ in range(2):
        env, base = cg.make_vec(STEP_ID, num_envs=1, device=DEVICE)
        try:
            actions = _gait(env.channels, env.single_action_space.shape[0], steps)
            obs, _, _ = env.reset(seed=3, embodiment=_notched("spring")[None])
            trajectory = [obs]
            rewards = []
            for a in actions:
                obs, reward, _, _, _ = env.step(a[None])
                trajectory.append(obs)
                rewards.append(reward)
            runs.append((np.concatenate(trajectory), np.concatenate(rewards)))
        finally:
            env.close()
    (obs_a, rew_a), (obs_b, rew_b) = runs
    if DEVICE == "cpu":
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(rew_a, rew_b)
    else:
        np.testing.assert_allclose(obs_a, obs_b, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(rew_a, rew_b, rtol=1e-9, atol=1e-12)
