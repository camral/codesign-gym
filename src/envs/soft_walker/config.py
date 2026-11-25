"""SoftWalker task: the JAX engine's coarse flat-walk scenario, rebuilt natively on the Warp backend.

Every number mirrors ``engine/scenarios`` (``Scenario()`` defaults, coarse tier, flat terrain) mapped onto Warp's knobs
the way ``warp_native.oracle.from_jax`` / ``solid_oracle.from_jax`` map an exported JAX body: 0.02 m cells on an
8x6 (2D) or 8x6x6 (3D) grid, 4 / 16 actuation channels, a 0.01 s control step of 40 substeps, 0.2 s of zero-command
settling, a 240-step horizon, plane contact (stiffness 1e6 N/m, width 0.5 mm, friction 0.8, friction velocity 2 mm/s),
actuator time constant 0.02 s, max log-stretch 0.35, and the scenario's MPM grid padding. The reward keeps the energy
term (``energy_weight = 1``). Bodies are built from the per-cell design (``embodiment.py``) with Warp's own NumPy
builders; no JAX is imported.

Terrains (``terrain=``; shared by all physics drivers):

  * ``flat`` - the ground plane.
  * ``stairs`` - the JAX scenario's staircase exactly (``scenarios.terrains.terrain_config("stairs")``): four 7.6 cm
    treads centred at x = 0.28 + 0.075 k m, each 6 mm higher than the last (top 2.4 cm), rounding 1 mm, starting 6 cm
    in front of the body.
  * ``gaps`` - ground-level platforms (boxes) separated by pits 5 cm deep (the ground plane is lowered to -5 cm): gaps of
    4, 20, 2 and 8 cm with 5 cm platforms between, the first 6 cm in front of the body. The 20 cm gap is wider than the
    body, so the whole body can fall in; the narrow ones catch part of it. MPM grids extend forward so a body can clear
    the last gap.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from .contracts.api_types import EnvironmentTiming, HostArray
from .contracts.fields import POISSON_DEFAULT
from .engine.physics.construction_defaults import (
    BODY_DENSITY_KG_M3,
    EARTH_GRAVITY_M_S2,
    PLANAR_THICKNESS_M,
    REGULAR_SOLID_STRENGTH,
    REGULAR_SOLID_YOUNG_PA,
    SOLID_CONTACT_PRESSURE_PER_LENGTH_PA_M,
    SPRING_DAMPING_RATIO,
    UNIFORM_MPM_STRENGTH,
    UNIFORM_MPM_YOUNG_PA,
)
from . import embodiment as E

if TYPE_CHECKING:
    from .contracts.api_types import Fields
    from .engine.warp_native.model import Bundle
    from .engine.warp_native.solid_model import SolidBundle


# --- presets -------------------------------------------------------------------------------------------------------
IDS = {
    "spring": "SoftSpring2D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
    "beam": "SoftBeam3D-v0",
    "fem": "SoftHexFEM3D-v0",
}


DRIVERS = tuple(IDS)


NAMES = {
    "spring": "Spring2D",
    "mpm2": "MPM2D",
    "mpm3": "MPM3D",
    "beam": "Beam3D",
    "fem": "HexFEM3D",
}


def preset_id(driver: str, terrain: str = "flat") -> str:
    """Return the registered environment ID for a driver and terrain preset."""
    return f"SoftWalker-{NAMES[driver]}" + ("" if terrain == "flat" else terrain.capitalize())


# Registered task defaults mirror the original coarse locomotion scenario.
DEFAULT_DTYPE = "float64"  # Numerical precision validated for these presets.
CELL = 0.02  # m, coarse scenario cell width.
BASE_GRID = {2: (8, 6), 3: (8, 6, 6)}  # Cells; fixed embodiment/observation layouts.
CHANNELS = {2: 4, 3: 16}  # Routing capacity, not a physical actuator count.
THICKNESS = PLANAR_THICKNESS_M  # m, physical out-of-plane thickness.
CLEARANCE = 0.001  # m, initial separation from the flat ground plane.
CONTROL_DT = 0.01  # s, one default policy action interval.
SUBSTEPS = 40  # Physics steps per default action: dt = 0.25 ms.
SETTLE_SECONDS = 0.2  # s, zero-command settling before the episode.
HORIZON = 240  # Actions: six 0.4 s nominal gait cycles at CONTROL_DT.
GRAVITY = EARTH_GRAVITY_M_S2
FRICTION = 0.8  # Dimensionless inherited contact coefficient.
FRICTION_VELOCITY = 0.002  # m/s, inherited smooth-friction velocity scale.
CONTACT_STIFFNESS = SOLID_CONTACT_PRESSURE_PER_LENGTH_PA_M  # Pa/m for geometric-area contact.
CONTACT_WIDTH = 0.0005  # m, contact smoothing width.
ACTUATOR_TAU = 0.02  # s, actuator filtering time constant.
MAX_LOG_STRETCH = 0.35  # Dimensionless saturation of commanded log stretch.
DAMPING_RATIO = SPRING_DAMPING_RATIO  # Dimensionless SpringConfig default.
PARTICLES_PER_AXIS = 2  # Particle samples per cell axis, matching BodyConfig.
TRAVEL_BODY_LENGTHS = 3.0  # Planned travel capacity for MPM background-grid padding.
ENERGY_WEIGHT = 1.0  # Dimensionless reward weight after work normalization.
WORK_REF = {2: 0.4, 3: 4.0}  # J, inherited task work-normalization scales.

# Default resolved properties match native builders, independently of decoder seeds.
# These are task initialization choices, not shared material-envelope bounds.
BASE_DENSITY = BODY_DENSITY_KG_M3
MPM_BASE_YOUNG = UNIFORM_MPM_YOUNG_PA
SOLID_BASE_YOUNG = REGULAR_SOLID_YOUNG_PA
MPM_BASE_STRENGTH = UNIFORM_MPM_STRENGTH
SOLID_BASE_STRENGTH = REGULAR_SOLID_STRENGTH


def dim_of(driver: str) -> int:
    """Return the spatial dimension of a registered physics driver."""
    return 2 if driver in ("spring", "mpm2") else 3


def grid(driver: str) -> tuple[int, ...]:
    """Return fixed per-cell design grid extents for this preset."""
    return BASE_GRID[dim_of(driver)]


def size(driver: str) -> tuple[float, ...]:
    """Return the physical envelope dimensions in metres."""
    return tuple(n * CELL for n in grid(driver))


def origin(driver: str) -> tuple[float, ...]:
    """Return the reference envelope origin in metres above the ground."""
    d = dim_of(driver)
    return (0.0, CLEARANCE) + ((-0.5 * size(driver)[2],) if d == 3 else ())


def env_kwargs(driver: str, *, physics_steps_per_action: int | None = None) -> EnvironmentTiming:
    """Action holding at the preset physics dt, with a fixed settling budget.

    The preset episode contains ``HORIZON * SUBSTEPS`` physics steps. Its action
    horizon rounds up when that count is not divisible by the requested hold,
    extending the episode by less than one new control interval. Bundle builders
    still use ``CONTROL_DT / SUBSTEPS``; this option never changes their dt.
    """
    if physics_steps_per_action is not None and (
        type(physics_steps_per_action) is not int or physics_steps_per_action < 1
    ):
        raise ValueError("physics_steps_per_action must be a positive plain int or None")
    hold = SUBSTEPS if physics_steps_per_action is None else physics_steps_per_action
    horizon = (HORIZON * SUBSTEPS + hold - 1) // hold
    return dict(
        substeps=hold,
        settling_steps=round(SETTLE_SECONDS / CONTROL_DT) * SUBSTEPS,
        horizon=horizon,
        length_ref=size(driver)[0],
        work_ref=WORK_REF[dim_of(driver)],
        energy_weight=ENERGY_WEIGHT,
    )


def base_props(driver: str) -> Fields:
    """Warp builder defaults: all cells solid; ``regular_solid`` defaults for solids, ``uniform_body`` for MPM."""
    d, g = dim_of(driver), grid(driver)
    count, channels = int(np.prod(g)), CHANNELS[d]
    direction = np.zeros((count, d))
    if driver == "beam":
        direction[:, int(np.argmax(size(driver)))] = (
            1.0  # regular_solid aligns beam fibres with the longest axis
        )
    else:
        direction[:, 1] = 1.0
    if driver.startswith("mpm"):
        return dict(
            occupancy=np.ones(count),
            young=np.full(count, MPM_BASE_YOUNG),
            density=np.full(count, BASE_DENSITY),
            strength=np.full(count, MPM_BASE_STRENGTH),
            direction=direction,
            kind=np.ones(count),
            channel=np.zeros(count),
            poisson=np.full(count, POISSON_DEFAULT),
        )
    return dict(
        occupancy=np.ones(count),
        young=np.full(count, SOLID_BASE_YOUNG),
        density=np.full(count, BASE_DENSITY),
        strength=np.full(count, SOLID_BASE_STRENGTH),
        direction=direction,
        kind=np.ones(count),
        channel=np.arange(count) % channels,
        poisson=np.full(count, POISSON_DEFAULT),
    )


# --- terrain -------------------------------------------------------------------------------------------------------
TERRAINS = ("flat", "stairs", "gaps")


BOX_ROUNDING = 0.001  # m, matches scenario rounded boxes.


BOX_HALF_DEPTH = 0.25  # m, spans the entire 3D body width.


# Stair geometry mirrors presets.terrain.scenario_courses without importing JAX.
STAIR_START_X = 0.28  # m, first tread centre.
STAIR_SPACING = 0.075  # m, centre-to-centre spacing.
STAIR_HALF_LENGTH = 0.038  # m, slight overlap avoids an unintended crack.
STAIR_RISE = 0.006  # m, height added per step.
STAIR_COUNT = 4
GAP_LEADING_PLATFORM_START = -1.0  # m, inherited finite box extent behind the body.
GAP_TRAILING_PLATFORM_END = 3.0  # m, inherited finite box extent after all gaps.
GAP_FLOOR_OVERLAP = 0.01  # m, platform boxes extend below the lowered floor.

STAIRS = tuple(
    (
        STAIR_START_X + STAIR_SPACING * k,
        0.0,
        0.0,
        STAIR_HALF_LENGTH,
        STAIR_RISE * (k + 1),
        BOX_HALF_DEPTH,
        BOX_ROUNDING,
    )
    for k in range(STAIR_COUNT)
)


# All four values below are in metres; gaps are a chosen stress-test layout.
GAP_START, GAP_WIDTHS, GAP_PLATFORM, GAP_DEPTH = 0.22, (0.04, 0.20, 0.02, 0.08), 0.05, 0.05


def _gap_platforms() -> list[tuple[float, float]]:
    """x-intervals of the walkable platforms: behind/under the body up to the first gap, then between gaps, then onward."""
    edges, x = [(GAP_LEADING_PLATFORM_START, GAP_START)], GAP_START
    for i, width in enumerate(GAP_WIDTHS):
        x += width
        end = x + GAP_PLATFORM if i < len(GAP_WIDTHS) - 1 else GAP_TRAILING_PLATFORM_END
        edges.append((x, end))
        x = end
    return edges


def terrain(driver: str, name: str, dtype: str) -> Fields | None:
    """Resolved Warp terrain dict (plane + rounded boxes) for ``name``; None for the flat ground plane."""
    if name not in TERRAINS:
        raise ValueError(f"unknown terrain {name!r}; choose from {TERRAINS}")
    if name == "flat":
        return None
    d = dim_of(driver)
    if name == "stairs":
        boxes, offset = STAIRS, 0.0
    else:  # platforms reach from below the pit floor up to y = 0
        half_y = (GAP_DEPTH + GAP_FLOOR_OVERLAP) / 2
        boxes, offset = (
            tuple(
                ((a + b) / 2, -half_y, 0.0, (b - a) / 2, half_y, BOX_HALF_DEPTH, BOX_ROUNDING)
                for a, b in _gap_platforms()
            ),
            -GAP_DEPTH,
        )
    b: HostArray = np.asarray(boxes, float)
    return dict(
        normal=np.eye(d, dtype=dtype)[1],
        offset=np.asarray(offset, dtype),
        centers=b[:, :d].astype(dtype),
        half_sizes=b[:, 3 : 3 + d].astype(dtype),
        rounding=b[:, 6].astype(dtype),
        active=np.ones(len(b), np.int32),
    )


# --- builders ------------------------------------------------------------------------------------------------------
# Background-grid padding reproduces the original coarse scenario. Counts are
# nodes/cells, not metres. The fixed eight-cell reference is intentionally kept
# separate from arbitrary body-grid choices; changing it would change capacity.
MPM_MIN_FORWARD_PADDING = 4
MPM_REFERENCE_BODY_CELLS = 8
MPM_FORWARD_GUARD_CELLS = 2
MPM_REAR_PADDING = 4
MPM_LOWER_PADDING = 4
MPM_UPPER_PADDING = 6
MPM_SIDE_PADDING = 4
SOLID_PLACEHOLDER_GRID_NODES = 4  # Historical unused background-grid shape for solid configs.
SPRING_CONTACT_SCALE = 1.0  # Contact modulus is supplied separately to regular_solid.


def _resolved(props: Fields) -> Fields:
    """Hard occupancy and integer fields as the builders expect them."""
    out = {k: np.asarray(v, dtype=float) for k, v in props.items()}
    out["occupancy"] = (out["occupancy"] >= E.OCCUPANCY_THRESHOLD).astype(float)
    out["kind"], out["channel"] = (
        np.rint(out["kind"]).astype(np.int32),
        np.rint(out["channel"]).astype(np.int32),
    )
    return out


def build_bundle(
    driver: str,
    props: Fields,
    dtype: str,
    terrain_name: str = "flat",
) -> Bundle | SolidBundle:
    """Assemble per-cell properties into a resolved bundle on the chosen terrain (backward Euler)."""
    props = _resolved(props)
    bundle = (
        _build_mpm(driver, props, dtype, terrain_name)
        if driver.startswith("mpm")
        else _build_solid(driver, props, dtype)
    )
    return replace(bundle, terrain=terrain(driver, terrain_name, dtype))


def _build_solid(
    driver: str, props: Fields, dtype: str
) -> SolidBundle:
    """Instantiate cell properties as a native spring/beam/FEM body.

    Contact, damping, integration timestep and actuator filtering match the
    registered task preset; material fields retain their per-cell variation."""
    from .engine.warp_native.solid_builders import regular_solid
    from .engine.warp_native.solid_model import SolidConfig

    d = dim_of(driver)
    cfg = SolidConfig(
        physics=driver,
        grid_shape=(SOLID_PLACEHOLDER_GRID_NODES,) * d,
        origin=(0.0,) * d,
        gravity=GRAVITY,
        dt=CONTROL_DT / SUBSTEPS,
        contact_enabled=True,
        contact_height=0.0,
        contact_stiffness=SPRING_CONTACT_SCALE if driver == "spring" else CONTACT_STIFFNESS,
        contact_width=CONTACT_WIDTH,
        friction=FRICTION,
        friction_width=FRICTION_VELOCITY,
        actuator_tau=ACTUATOR_TAU,
        max_log_stretch=MAX_LOG_STRETCH,
        linear_work_reuse=True,
    )
    return regular_solid(
        driver,
        shape=grid(driver),
        size=size(driver),
        origin=origin(driver),
        channels=CHANNELS[d],
        precision=dtype,
        properties=props,
        config=cfg,
        thickness=THICKNESS,
        damping_ratio=DAMPING_RATIO,
        contact_modulus=CONTACT_STIFFNESS * size(driver)[0],
        length_ref=size(driver)[0],
    )


def mpm_grid(driver: str, terrain_name: str = "flat") -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Background grid (node counts, origin) with the scenario's padding: 4 cells behind/below/aside, room ahead to travel.
    On ``gaps`` the grid reaches a body length past the last gap so the body can clear it."""
    d, g = dim_of(driver), grid(driver)
    forward = max(
        MPM_MIN_FORWARD_PADDING,
        math.ceil(TRAVEL_BODY_LENGTHS * MPM_REFERENCE_BODY_CELLS) + MPM_FORWARD_GUARD_CELLS,
    )
    if terrain_name == "gaps":
        clear = (
            GAP_START
            + sum(GAP_WIDTHS)
            + GAP_PLATFORM * (len(GAP_WIDTHS) - 1)
            + size(driver)[0]
        )
        forward = max(
            forward, math.ceil((clear - size(driver)[0]) / CELL) + MPM_FORWARD_GUARD_CELLS
        )
    pads = ((MPM_REAR_PADDING, forward), (MPM_LOWER_PADDING, MPM_UPPER_PADDING)) + (
        ((MPM_SIDE_PADDING, MPM_SIDE_PADDING),) if d == 3 else ()
    )
    shape = tuple(n + lo + hi + 1 for n, (lo, hi) in zip(g, pads))
    return shape, tuple(x - lo * CELL for x, (lo, _) in zip(origin(driver), pads))


def _build_mpm(driver: str, props: Fields, dtype: str, terrain_name: str = "flat") -> Bundle:
    """Instantiate the cell design as particles and a padded native MPM grid.

    Each occupied cell contributes a fixed particle lattice. Hard occupancy
    sets mass/contact participation; capacities remain fixed for graph replay."""
    from .engine.warp_native.config import MPMConfig
    from .engine.warp_native.model import Bundle

    d, g, ppa = dim_of(driver), grid(driver), PARTICLES_PER_AXIS
    sh = tuple(n * ppa for n in g)
    index = np.array(list(product(*(range(n) for n in sh))))
    hp = CELL / ppa
    cell = np.ravel_multi_index(tuple((index // ppa).T), g)  # C order, same as the per-cell vector
    solid = props["occupancy"][cell] > 0
    factor = THICKNESS if d == 2 else 1.0
    offset, area = [], np.zeros((len(index), 2 * d))
    occupied: HostArray = np.zeros(sh, bool)
    occupied[tuple(index.T)] = solid
    for axis in range(d):
        for side in (-1, 1):
            r = np.zeros(d)
            r[axis] = side * hp / 2
            offset.append(r)
            neighbour = index.copy()
            neighbour[:, axis] += side
            inside = (neighbour[:, axis] >= 0) & (neighbour[:, axis] < sh[axis])
            covered: HostArray = np.zeros(len(index), bool)
            covered[inside] = occupied[tuple(neighbour[inside].T)]
            area[solid & ~covered, len(offset) - 1] = (
                hp ** (d - 1) * factor
            )  # exposed faces of solid particles only
    fields = dict(
        reference=(index + 0.5) * hp + np.asarray(origin(driver)),
        volume=np.full(len(index), hp**d * factor),
        alpha=solid.astype(float),
        young=props["young"][cell],
        poisson=props["poisson"][cell],
        density=props["density"][cell],
        strength=props["strength"][cell],
        direction=props["direction"][cell],
        kind=props["kind"][cell],
        channel=props["channel"][cell],
        face_offset=np.array(offset),
        contact_area=area,
    )
    fields = {
        k: np.asarray(v, dtype=np.int32 if k in ("kind", "channel") else dtype)
        for k, v in fields.items()
    }
    shape, grid_origin = mpm_grid(driver, terrain_name)
    cfg = MPMConfig(
        grid_shape=shape,
        origin=grid_origin,
        spacing=CELL,
        dt=CONTROL_DT / SUBSTEPS,
        gravity=GRAVITY,
        actuator_tau=ACTUATOR_TAU,
        max_log_stretch=MAX_LOG_STRETCH,
        mass_weighted=True,
        contact_enabled=True,
        contact_height=0.0,
        contact_stiffness=CONTACT_STIFFNESS,
        contact_width=CONTACT_WIDTH,
        friction=FRICTION,
        friction_width=FRICTION_VELOCITY,
    )
    return Bundle(fields, cfg, CHANNELS[d])


# --- task ----------------------------------------------------------------------------------------------------------
class TimingMetadata(NamedTuple):
    """Counts and physical durations the vector env exposes."""

    physics_steps_per_action: int
    physics_dt: float
    control_dt: float
    horizon: int
    target_episode_duration_s: float
    episode_duration_s: float
    settling_duration_s: float


def validate_options(num_envs: int, driver: str, render_mode: str | None, execution: str) -> None:
    """Validate common constructor options before allocating backend resources."""
    if driver not in DRIVERS:
        raise ValueError(f"unknown soft_walker driver {driver!r}; choose from {DRIVERS}")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive int")
    if render_mode not in (None, "rgb_array", "human"):
        raise ValueError("render_mode must be None, 'rgb_array', or 'human'")
    if execution not in ("auto", "serial", "parallel"):
        raise ValueError("execution must be 'auto', 'serial', or 'parallel'")


def resolve_timing(
    driver: str, physics_steps_per_action: int | None
) -> tuple[EnvironmentTiming, EnvironmentTiming]:
    """Keep the historical default call shape and optionally override action hold."""
    default = env_kwargs(driver)
    timing = (
        default
        if physics_steps_per_action is None
        else env_kwargs(driver, physics_steps_per_action=physics_steps_per_action)
    )
    return default, timing


def task_layout(
    driver: str, dtype: np.dtype, terrain: str
) -> tuple[int, tuple[int, ...], int, HostArray, Bundle | SolidBundle]:
    """Resolve the fixed layout (dimension, grid, channels), the base design and the template bundle."""
    dim, shape = dim_of(driver), grid(driver)
    channels = CHANNELS[dim]
    base = E.from_cells(base_props(driver), dim, dtype)
    props = E.to_cells(base, dim)
    template = build_bundle(driver, props, dtype.name, terrain)
    return dim, shape, channels, base, template


def timing_metadata(
    timing: EnvironmentTiming, default: EnvironmentTiming, physics_dt: float
) -> TimingMetadata:
    """Derive matching episode/settling durations at the unchanged physics step."""
    hold, horizon = timing["substeps"], timing["horizon"]
    control_dt = hold * physics_dt
    return TimingMetadata(
        hold,
        physics_dt,
        control_dt,
        horizon,
        default["horizon"] * default["substeps"] * physics_dt,
        horizon * control_dt,
        timing["settling_steps"] * physics_dt,
    )


def candidate_bundle(
    vector: ArrayLike,
    *,
    driver: str,
    dim: int,
    dtype: np.dtype,
    terrain: str,
    errors: list[str],
) -> tuple[Bundle | SolidBundle | None, list[str]]:
    """Build a structurally checked candidate, preserving common rejection messages.

    Structural errors are supplied by the wrapper so its historical validation
    hooks remain effective. No accepted model or state is changed here.
    """
    if errors:
        return None, errors
    try:
        props = E.to_cells(np.asarray(vector, float), dim)
        bundle = build_bundle(driver, props, dtype.name, terrain)
        bundle.validate()
        return bundle, []
    except (ValueError, TypeError, IndexError) as exc:
        return None, [f"BUILD: {exc}"]
