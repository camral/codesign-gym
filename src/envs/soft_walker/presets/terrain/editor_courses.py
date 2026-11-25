"""Named reproducible editor courses over the portable terrain-recipe schema."""

from __future__ import annotations

import math
import random
from typing import cast

from envs.soft_walker.terrain.recipes import (
    MAX_ABS_SLOPE,
    MAX_BLOCKS,
    SCHEMA,
    TerrainRecipe,
    finite_number,
    validate_recipe,
)

# Named editor courses are inherited demonstration choices, not schema defaults
# or calibrated locomotion difficulty. Distances are m; slope_degrees is degrees.
PRESETS = (
    "flat",
    "uphill",
    "downhill",
    "cross_slope",
    "low_blocks",
    "rough_blocks",
    "stairs",
    "hurdles",
)
PRESET_DEFAULTS = {
    name: dict(
        slope_degrees=5.0,
        count=4,
        start=0.26,
        spacing=0.11,
        width=0.065,
        height=0.012,
        rounding=0.001,
    )
    for name in PRESETS
}
PRESET_DEFAULTS["low_blocks"].update(height=0.007)
PRESET_DEFAULTS["rough_blocks"].update(height=0.016)
PRESET_DEFAULTS["stairs"].update(start=0.28, spacing=0.075, width=0.077, height=0.006)
PRESET_DEFAULTS["hurdles"].update(count=3, start=0.29, spacing=0.14, width=0.024)

# Keep this Python-random policy distinct from NumPy-based scenario courses.
BLOCK_X_JITTER = (-0.008, 0.008)  # m, independent uniform displacement per obstacle.
BLOCK_WIDTH_SCALE = (0.85, 1.15)  # Dimensionless multiplier of requested width.
LOW_BLOCK_HEIGHT_SCALE = (0.7, 1.0)
ROUGH_BLOCK_HEIGHT_SCALE = (0.65, 1.35)
COURSE_HALF_DEPTH = 0.25  # m; lateral half-size, also the 2D display extrusion.
COURSE_ROUNDING_FRACTION = 0.2  # Maximum radius relative to the smaller x/y half-size.
COURSE_CONFIG_ROUNDING_CAP = 0.001  # m; unused procedural fallback in frozen configs.


def make_recipe(
    preset: str,
    *,
    dim: int = 2,
    seed: int = 7,
    slope_degrees: float | None = None,
    count: int | None = None,
    start: float | None = None,
    spacing: float | None = None,
    width: float | None = None,
    height: float | None = None,
    rounding: float | None = None,
) -> TerrainRecipe:
    """Create a reproducible course, with a clear starting area by default.

    For stairs, height is the rise per tread. For low/rough blocks it controls
    their height scale, with fixed seeded variation. In 2D the z dimensions
    only determine the cosmetic extrusion shown by the editor.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown terrain preset: {preset}")
    if type(dim) is not int or dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    values = locals().copy()
    params = {
        key: PRESET_DEFAULTS[preset][key] if values[key] is None else values[key]
        for key in PRESET_DEFAULTS[preset]
    }
    for key in params.keys() - {"count"}:
        params[key] = finite_number(params[key], key)
    count = cast(int, params["count"])
    if type(count) is not int or not 0 <= count <= MAX_BLOCKS:
        raise ValueError("count must be an integer in [0,32]")
    if min(params[key] for key in ("spacing", "width", "height")) <= 0 or params["rounding"] < 0:
        raise ValueError("Dimensions must be positive and rounding nonnegative")
    if abs(params["slope_degrees"]) > math.degrees(math.atan(MAX_ABS_SLOPE)):
        raise ValueError("Slope exceeds the supported 0.5 rise/run limit")
    config = dict(kind="flat", blocks=0)
    if preset in ("uphill", "downhill", "cross_slope"):
        if preset == "cross_slope" and dim != 3:
            raise ValueError("Cross slope requires 3D")
        slope = math.tan(math.radians(abs(params["slope_degrees"])))
        slopes = {"uphill": [slope, 0.0], "downhill": [-slope, 0.0], "cross_slope": [0.0, slope]}[
            preset
        ]
        config.update(kind="slopes", fixed_slopes=slopes)
    elif preset != "flat":
        rng = random.Random(seed)
        boxes = []
        for k in range(count):
            x = params["start"] + k * params["spacing"]
            w, h = params["width"], params["height"]
            if preset in ("low_blocks", "rough_blocks"):
                x += rng.uniform(*BLOCK_X_JITTER)
                w *= rng.uniform(*BLOCK_WIDTH_SCALE)
                h *= rng.uniform(
                    *(
                        ROUGH_BLOCK_HEIGHT_SCALE
                        if preset == "rough_blocks"
                        else LOW_BLOCK_HEIGHT_SCALE
                    )
                )
            elif preset == "stairs":
                h *= k + 1
            radius = min(params["rounding"], COURSE_ROUNDING_FRACTION * min(w / 2, h))
            boxes.append([x, 0.0, 0.0, w / 2, h, COURSE_HALF_DEPTH, radius])
        config.update(
            kind="blocks",
            blocks=count,
            fixed_boxes=boxes,
            rounding=min(params["rounding"], COURSE_CONFIG_ROUNDING_CAP),
        )
    return validate_recipe(
        dict(
            schema=SCHEMA, name=preset.replace("_", " ").title(), dim=dim, seed=seed, config=config
        )
    )
