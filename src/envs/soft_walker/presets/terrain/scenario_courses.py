"""Terrain families with metric sizes, fixed spawn area, and replayable samples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import DTypeLike

from envs.soft_walker.engine.physics.terrain import (
    Terrain,
    TerrainConfig,
    generate,
    serialize as serialize,
)
from envs.soft_walker.presets.scenarios.config import Scenario


@dataclass(frozen=True)
class RandomBlockCourse:
    """Inherited NumPy-uniform block course parameters, with distances in metres.

    Draw order is x jitter, width, then height for each block. It is part of the
    seeded preset behavior. These values were not calibrated against walking
    performance and intentionally differ from the editor's Python-random recipe.
    """

    count: int = 4
    start: float = 0.26
    spacing: float = 0.11
    x_jitter: tuple[float, float] = (-0.012, 0.012)
    width_range: tuple[float, float] = (0.045, 0.075)
    low_height_range: tuple[float, float] = (0.004, 0.008)
    rough_height_range: tuple[float, float] = (0.010, 0.020)


@dataclass(frozen=True)
class FixedBlockCourse:
    """Repeated obstacle geometry in metres; stair height is the rise per tread."""

    count: int
    start: float
    spacing: float
    half_width: float
    height: float


RANDOM_BLOCKS = RandomBlockCourse()
# Preserve overlapping stair treads and sparse narrow hurdles from the original
# scenario preset. Their precise calibration history was not recorded.
STAIRS = FixedBlockCourse(count=4, start=0.28, spacing=0.075, half_width=0.038, height=0.006)
HURDLES = FixedBlockCourse(count=3, start=0.29, spacing=0.14, half_width=0.012, height=0.012)
COURSE_SLOPE_DEGREES = 5.0  # Gentle inherited example grade; converted to rise/run.
COURSE_HALF_DEPTH = 0.25  # m; lateral half-size, also the 2D display extrusion.
COURSE_ROUNDING = 0.001  # m; fixed rounded-box radius for every scenario obstacle.


def terrain_config(name: str, *, dim: int = 2, seed: int = 0) -> TerrainConfig:
    """Return a frozen TerrainConfig; steps/hurdles extend below the floor.

    No pits, caves, deformable terrain, or general self-contact are implied.
    Initial body occupies x in [0,.16]; obstacles start beyond x=.22.
    The same explicit sample is retained across reset seeds and backends.
    """
    if name == "flat":
        return TerrainConfig(kind="flat", blocks=0)
    if name in ("uphill", "downhill", "cross_slope"):
        if name == "cross_slope" and dim != 3:
            raise ValueError("cross_slope requires 3D")
        slope = math.tan(math.radians(COURSE_SLOPE_DEGREES))
        slopes = {"uphill": (slope, 0.0), "downhill": (-slope, 0.0), "cross_slope": (0.0, slope)}[
            name
        ]
        return TerrainConfig(kind="slopes", blocks=0, fixed_slopes=slopes)
    rng = np.random.default_rng(seed)
    boxes = []
    if name in ("low_blocks", "rough_blocks"):
        rough = name == "rough_blocks"
        for k in range(RANDOM_BLOCKS.count):
            x = (
                RANDOM_BLOCKS.start
                + RANDOM_BLOCKS.spacing * k
                + rng.uniform(*RANDOM_BLOCKS.x_jitter)
            )
            width = rng.uniform(*RANDOM_BLOCKS.width_range)
            height = rng.uniform(
                *(RANDOM_BLOCKS.rough_height_range if rough else RANDOM_BLOCKS.low_height_range)
            )
            boxes.append((x, 0.0, 0.0, width / 2, height, COURSE_HALF_DEPTH, COURSE_ROUNDING))
    elif name == "stairs":
        for k in range(STAIRS.count):
            # Overlap below ground; each successive tread is 6 mm higher.
            boxes.append(
                (
                    STAIRS.start + STAIRS.spacing * k,
                    0.0,
                    0.0,
                    STAIRS.half_width,
                    STAIRS.height * (k + 1),
                    COURSE_HALF_DEPTH,
                    COURSE_ROUNDING,
                )
            )
    elif name == "hurdles":
        for k in range(HURDLES.count):
            boxes.append(
                (
                    HURDLES.start + HURDLES.spacing * k,
                    0.0,
                    0.0,
                    HURDLES.half_width,
                    HURDLES.height,
                    COURSE_HALF_DEPTH,
                    COURSE_ROUNDING,
                )
            )
    else:
        raise ValueError(f"Unknown terrain family: {name}")
    return TerrainConfig(
        kind="blocks", blocks=len(boxes), rounding=COURSE_ROUNDING, fixed_boxes=tuple(boxes)
    )


def resolved_terrain(
    name: str, *, dim: int = 2, seed: int = 0, dtype: DTypeLike = "float64"
) -> Terrain:
    """Generate one reproducible named terrain for the selected dimension/precision."""
    return generate(terrain_config(name, dim=dim, seed=seed), seed, dim, dtype)


def case_terrain_config(case: Scenario) -> TerrainConfig:
    """Resolve a named preset or the frozen custom recipe embedded in a case."""
    if case.terrain_recipe is None:
        return terrain_config(case.terrain, dim=case.dim, seed=case.seed)
    from envs.soft_walker.terrain.recipes import validate_recipe

    values = dict(validate_recipe(case.terrain_recipe, dim=case.dim)["config"])
    # Dataclass configs are used in compilation/cache keys; JSON lists must not
    # make an otherwise immutable configuration unhashable.
    if values.get("fixed_slopes") is not None:
        values["fixed_slopes"] = tuple(cast(list[float], values["fixed_slopes"]))
    if values.get("fixed_boxes") is not None:
        values["fixed_boxes"] = tuple(
            tuple(box) for box in cast(list[list[float]], values["fixed_boxes"])
        )
    # Dataclass constructor validates the decoded descriptor.
    return TerrainConfig(**values)  # type: ignore[arg-type]


def case_terrain(case: Scenario) -> Terrain:
    """Realized preview/contact geometry; custom recipes never resample on reset."""
    seed = case.terrain_recipe["seed"] if case.terrain_recipe is not None else case.seed
    return generate(case_terrain_config(case), seed, case.dim, case.dtype)
