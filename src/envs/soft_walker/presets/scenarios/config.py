"""Serializable scenario recipes, not claims of optimized walking performance.

A recipe fixes numerical settings independently of embodiment/controller values.
Suite expansion is explicit: there is no implicit enormous Cartesian-product run.
Defaults retain the historical demonstration protocol; no original calibration
record establishes them as optimal locomotion settings. These are task priors,
not general engine defaults. Distances use metres, time seconds and angles radians.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from envs.soft_walker.contracts.fields import (
    DENSITY_BOUNDS,
    POISSON_BOUNDS,
    POISSON_DEFAULT,
    YOUNG_BOUNDS,
)
from envs.soft_walker.presets.embodiments.chamber_profile import (
    BASE_DENSITY,
    BASE_LEG_YOUNG,
    BASE_STRENGTH,
    BASE_TORSO_YOUNG,
    CHAMBER_CHANNELS,
)

if TYPE_CHECKING:
    from envs.soft_walker.terrain.recipes import TerrainRecipe

import math
import re
from dataclasses import asdict, dataclass

DRIVERS = ("spring", "mpm2", "mpm3", "beam", "fem")
REPRESENTATIONS = ("primitives", "direct", "bspline", "cppn")
CONTROLLERS = ("fourier", "pulse", "feedback", "neural")
TERRAINS = (
    "flat",
    "uphill",
    "downhill",
    "cross_slope",
    "low_blocks",
    "rough_blocks",
    "stairs",
    "hurdles",
)
GAITS = ("walk", "crawl", "trot", "bound", "pronk")

# The same body envelope is sampled at 20 mm (coarse) or 10 mm (resolved).
REFERENCE_GRID_SHAPE = (8, 6, 6)
ENVELOPE_SIZE = (0.16, 0.12, 0.12)  # m, x/y/z; slice by spatial dimension.
REFERENCE_CELL_SIZE = 0.02  # m; ENVELOPE_SIZE / REFERENCE_GRID_SHAPE on each axis.
RESOLVED_REFINEMENT = 2  # Double cells per axis without changing physical dimensions.
SMOKE_CONTROL_DT = 0.00025  # s; one physics interval for initialization checks only.
CONTROL_INTERVAL_ATOL = 1e-8  # Dimensionless interval-count allowance for float division.


@dataclass(frozen=True)
class Scenario:
    """A resolved task configuration. +x is travel, +y is up, +z is lateral.

    smoke: one short interval, no settling; initialization test only.
    coarse: h=20 mm. resolved: h=10 mm with identical physical envelope.
    Coarse limbs have two cells across; narrow chamber resolution is flagged.
    """

    driver: str = "mpm2"
    representation: str = "primitives"
    controller: str = "fourier"
    gait: str = "walk"
    terrain: str = "flat"
    tier: str = "coarse"
    dtype: str = "float64"
    seed: int = 0
    cycles: int = 6  # Number of gait periods, excluding settling.
    period: float = 0.4  # s per gait cycle.
    control_dt: float = 0.01  # s per held policy action.
    substeps: int = 40  # Physics intervals per action; default dt is 0.25 ms.
    settle_seconds: float = 0.2  # s with no gait command before evaluation.
    leg_young: float = BASE_LEG_YOUNG
    torso_young: float = BASE_TORSO_YOUNG
    density: float = BASE_DENSITY
    strength: float = BASE_STRENGTH
    poisson: float = POISSON_DEFAULT
    max_log_stretch: float = 0.35  # Dimensionless log-strain actuator scale.
    actuator_tau: float = 0.02  # s; actuator state relaxation time.
    friction: float = 0.8  # Dimensionless Coulomb coefficient.
    friction_velocity: float = 0.002  # m/s; smoothed friction transition speed.
    travel_body_lengths: float = 3.0  # Dimensionless horizontal-domain allowance.
    amplitude: float = -0.35  # Dimensionless controller seed coefficient.
    lift: float = 1.25  # Dimensionless chamber lift multiplier.
    target_height: float = 0.01  # m; target used by airborne assessment/objectives.
    gmres_method: str = "incremental"
    terrain_recipe: TerrainRecipe | None = None

    def __post_init__(self) -> None:
        """Reject unsupported component combinations and inconsistent timing/material settings."""
        for key, options in (
            ("driver", DRIVERS),
            ("representation", REPRESENTATIONS),
            ("controller", CONTROLLERS),
            ("gait", GAITS),
            ("terrain", TERRAINS),
            ("tier", ("smoke", "coarse", "resolved")),
            ("dtype", ("float32", "float64")),
            ("gmres_method", ("incremental", "batched", "fused")),
        ):
            if getattr(self, key) not in options:
                raise ValueError(f"Unknown {key}: {getattr(self, key)!r}")
        for key in ("seed", "cycles", "substeps"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(
                    f"{key} must be a {'nonnegative' if key == 'seed' else 'positive'} integer"
                )
        for key in (
            "period",
            "control_dt",
            "leg_young",
            "torso_young",
            "density",
            "max_log_stretch",
            "actuator_tau",
            "friction_velocity",
            "target_height",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("settle_seconds", "friction", "travel_body_lengths"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not all(
            math.isfinite(v) for v in (self.amplitude, self.lift, self.strength, self.poisson)
        ):
            raise ValueError("Nonfinite controller/material values")
        if not (
            YOUNG_BOUNDS[0] <= self.leg_young <= YOUNG_BOUNDS[1]
            and YOUNG_BOUNDS[0] <= self.torso_young <= YOUNG_BOUNDS[1]
        ):
            raise ValueError("Young's modulus must lie in [5000, 100000] Pa")
        if not (
            DENSITY_BOUNDS[0] <= self.density <= DENSITY_BOUNDS[1]
            and 0 <= self.strength <= 1
            and POISSON_BOUNDS[0] <= self.poisson <= POISSON_BOUNDS[1]
        ):
            raise ValueError("Material/actuation values exceed embodiment bounds")
        for value in (self.period, self.settle_seconds):
            if not math.isclose(
                value / self.control_dt,
                round(value / self.control_dt),
                abs_tol=CONTROL_INTERVAL_ATOL,
            ):
                raise ValueError("Period and settling time must be integer control intervals")
        if self.dim == 2 and self.terrain == "cross_slope":
            raise ValueError("cross_slope requires 3D; use uphill/downhill in 2D")
        if self.terrain_recipe is not None:
            if self.terrain != "flat":
                raise ValueError(
                    "A custom terrain recipe cannot also select a named terrain preset"
                )
            # Keep planning and JSON validation independent of JAX initialization.
            from envs.soft_walker.terrain.recipes import validate_recipe

            object.__setattr__(
                self, "terrain_recipe", validate_recipe(self.terrain_recipe, dim=self.dim)
            )

    @property
    def dim(self) -> int:
        """Spatial dimension selected by the physics driver."""
        return 2 if self.driver in ("spring", "mpm2") else 3

    @property
    def channels(self) -> int:
        """Number of independently controlled physical chambers."""
        return CHAMBER_CHANNELS[self.dim]

    @property
    def refinement(self) -> int:
        """Reference-grid refinement implied by the scenario resolution tier."""
        return RESOLVED_REFINEMENT if self.tier == "resolved" else 1

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Fixed reference grid dimensions for the selected resolution."""
        base = REFERENCE_GRID_SHAPE[: self.dim]
        return tuple(n * self.refinement for n in base)

    @property
    def size(self) -> tuple[float, ...]:
        """Physical design-envelope dimensions in metres."""
        return ENVELOPE_SIZE[: self.dim]

    @property
    def cell_size(self) -> float:
        """Reference cell spacing in metres."""
        return REFERENCE_CELL_SIZE / self.refinement

    @property
    def steps_per_cycle(self) -> int:
        """Control intervals in one exact gait period."""
        return round(self.period / self.control_dt)

    @property
    def steps(self) -> int:
        """Number of intervals evaluated by the resolved scenario."""
        return 1 if self.tier == "smoke" else self.cycles * self.steps_per_cycle

    @property
    def numerical_substeps(self) -> int:
        """Physics substeps per interval, including the special smoke configuration."""
        return 1 if self.tier == "smoke" else self.substeps

    @property
    def effective_control_dt(self) -> float:
        """Actual policy interval in seconds after resolving the smoke profile."""
        return SMOKE_CONTROL_DT if self.tier == "smoke" else self.control_dt

    @property
    def key(self) -> str:
        """Stable artifact identifier for this scenario combination."""
        terrain_key = (
            "custom_" + re.sub(r"[^A-Za-z0-9_-]+", "_", self.terrain_label).strip("_")
            if self.terrain_recipe is not None
            else self.terrain
        )
        return "__".join(
            (
                self.driver,
                self.representation,
                self.controller,
                self.gait,
                terrain_key,
                self.tier,
                f"s{self.seed}",
            )
        )

    @property
    def terrain_label(self) -> str:
        """Human-readable terrain label, including an explicit recipe when present."""
        return self.terrain_recipe["name"] if self.terrain_recipe is not None else self.terrain

    def to_dict(self) -> dict[str, object]:
        """Return a portable scenario record without runtime objects."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> Scenario:
        """Reconstruct and validate a scenario from named settings."""
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown scenario fields: {sorted(unknown)}")
        # Dataclass constructor validates the decoded descriptor.
        return cls(**values)  # type: ignore[arg-type]
