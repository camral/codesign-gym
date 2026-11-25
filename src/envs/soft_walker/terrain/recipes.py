"""Portable terrain recipes shared by the editor and the physics scenarios.

This module deliberately uses only the standard library so the Newton viewer
can author terrain without importing JAX. Distances are metres; y is up.
Recipes freeze every slope and box rather than drawing new geometry on reset.
"""

from __future__ import annotations

import math
from typing import TypedDict, cast


class RecipeConfig(TypedDict):
    """Canonical, validated terrain fields; geometry is frozen across resets."""

    kind: str
    max_slope: float
    blocks: int
    block_start: float
    block_spacing: float
    block_width: float
    block_height: float
    rounding: float
    height: float
    fixed_slopes: list[float] | None
    fixed_boxes: list[list[float]] | None


class TerrainRecipe(TypedDict):
    """Portable validated terrain recipe, independent of NumPy/JAX imports."""

    schema: str
    name: str
    dim: int
    seed: int
    config: RecipeConfig


class TerrainRecord(TypedDict):
    """Frozen plane/box arrays in JSON form; lengths are metres and normal is unitless."""

    normal: list[float]
    offset: float
    centers: list[list[float]]
    half_sizes: list[list[float]]
    rounding: list[float]
    active: list[bool]


SCHEMA = "codesign_terrain_recipe_v1"
# The portable schema always stores xyz, including a cosmetic z extrusion in 2D.
# Ordering is persisted in JSON and must not follow the requested simulation dim.
BOX_CENTER = slice(0, 3)
BOX_HALF_SIZE = slice(3, 6)
BOX_ROUNDING = 6
BOX_RECORD_WIDTH = 7
MAX_RECIPE_NAME_LENGTH = 120  # Historical portable-label limit, in characters.
MAX_BLOCKS = 32  # Historical bounded recipe capacity; not a physics-engine maximum.
MAX_ABS_SLOPE = 0.5  # Dimensionless rise/run; editor's supported slope envelope.
CONFIG_SIZE_TO_ROUNDING = 4  # Rounding must be below one quarter of width/height.

# Canonical fallbacks preserve the v1 recipe contract when fields are omitted.
# Geometry values are in m; max_slope is rise/run. No calibration record exists.
_DEFAULT_CONFIG = dict(
    kind="flat",
    max_slope=0.15,
    blocks=0,
    block_start=0.25,
    block_spacing=0.1,
    block_width=0.065,
    block_height=0.015,
    rounding=0.002,
    height=0.0,
    fixed_slopes=None,
    fixed_boxes=None,
)
_SLOPE_TOL = 1e-12  # Absorbs degree/tangent round trips at the 0.5 rise/run limit.


def finite_number(value: object, name: str) -> float:
    """Accept finite numeric recipe values while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_recipe(recipe: object, dim: int | None = None) -> TerrainRecipe:
    """Return a detached, canonical JSON recipe; reject ambiguous geometry."""
    if not isinstance(recipe, dict):
        raise ValueError("Terrain recipe must be an object")
    required = {"schema", "name", "dim", "seed", "config"}
    if set(recipe) != required or recipe.get("schema") != SCHEMA:
        raise ValueError(f"Terrain recipe needs schema {SCHEMA} and name/dim/seed/config")
    if type(recipe["dim"]) is not int or recipe["dim"] not in (2, 3):
        raise ValueError("Terrain recipe dim must be 2 or 3")
    if dim is not None and recipe["dim"] != dim:
        raise ValueError(f"Terrain dimension {recipe['dim']} does not match driver dimension {dim}")
    if type(recipe["seed"]) is not int or recipe["seed"] < 0:
        raise ValueError("Terrain recipe seed must be a nonnegative integer")
    if (
        not isinstance(recipe["name"], str)
        or not recipe["name"].strip()
        or len(recipe["name"]) > MAX_RECIPE_NAME_LENGTH
    ):
        raise ValueError("Terrain name must contain 1 to 120 characters")
    given = recipe["config"]
    if not isinstance(given, dict) or set(given) - set(_DEFAULT_CONFIG):
        raise ValueError("Unknown or invalid terrain config fields")
    config = dict(_DEFAULT_CONFIG, **given)
    if config["kind"] not in ("flat", "slopes", "blocks"):
        raise ValueError("Terrain kind must be flat, slopes, or blocks")
    count = config["blocks"]
    if type(count) is not int or not 0 <= count <= MAX_BLOCKS:
        raise ValueError("blocks must be an integer in [0,32]")
    for key in (
        "max_slope",
        "block_start",
        "block_spacing",
        "block_width",
        "block_height",
        "rounding",
        "height",
    ):
        config[key] = finite_number(config[key], key)
    if not 0 <= config["max_slope"] <= MAX_ABS_SLOPE:
        raise ValueError("max_slope must lie in [0,.5]")
    if min(config[k] for k in ("block_spacing", "block_width", "block_height")) <= 0:
        raise ValueError("Block dimensions and spacing must be positive")
    if (
        not 0
        <= config["rounding"]
        < (min(config["block_width"], config["block_height"]) / CONFIG_SIZE_TO_ROUNDING)
    ):
        raise ValueError("Config rounding is too large for its block dimensions")
    slopes = config["fixed_slopes"]
    if config["kind"] == "slopes" and slopes is None:
        raise ValueError("Recipes require fixed_slopes so resetting cannot change the terrain")
    if slopes is not None:
        if config["kind"] != "slopes" or not isinstance(slopes, (list, tuple)) or len(slopes) != 2:
            raise ValueError("fixed_slopes requires kind=slopes and two values")
        slopes = [finite_number(v, "fixed_slopes") for v in slopes]
        if any(abs(v) > MAX_ABS_SLOPE + _SLOPE_TOL for v in slopes):
            raise ValueError("Slopes must lie in [-.5,.5]")
        slopes = [min(max(v, -MAX_ABS_SLOPE), MAX_ABS_SLOPE) for v in slopes]
        if recipe["dim"] == 2 and slopes[1] != 0:
            raise ValueError("A lateral slope requires a 3D terrain")
        config["fixed_slopes"] = slopes
    boxes = config["fixed_boxes"]
    if config["kind"] == "blocks" and boxes is None:
        raise ValueError("Recipes require fixed_boxes so resetting cannot change the terrain")
    if boxes is not None:
        if (
            config["kind"] != "blocks"
            or not isinstance(boxes, (list, tuple))
            or len(boxes) != count
        ):
            raise ValueError("fixed_boxes must match the block capacity")
        canonical = []
        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) != BOX_RECORD_WIDTH:
                raise ValueError("A box needs center xyz, half-size xyz, and rounding")
            values = [finite_number(v, "fixed_boxes") for v in box]
            if min(values[BOX_HALF_SIZE]) <= 0 or not (
                0 <= values[BOX_ROUNDING] < min(values[BOX_HALF_SIZE])
            ):
                raise ValueError("Box half-sizes must be positive; rounding must fit inside")
            canonical.append(values)
        config["fixed_boxes"] = canonical
    return dict(
        schema=SCHEMA,
        name=recipe["name"].strip(),
        dim=recipe["dim"],
        seed=recipe["seed"],
        config=cast(RecipeConfig, config),
    )


def recipe_terrain(recipe: object) -> TerrainRecord:
    """Resolve frozen recipes to the exact public Terrain serialization schema.

    Kept independent of JAX for graphics; tests compare this against generate()
    and the physical contact surface. No visual heightfield replaces contact.
    """
    validated = validate_recipe(recipe)
    dim, config = validated["dim"], validated["config"]
    slopes = config["fixed_slopes"] or [0.0, 0.0]
    normal = [-slopes[0], 1.0] + ([-slopes[1]] if dim == 3 else [])
    scale = math.sqrt(sum(v * v for v in normal))
    normal = [v / scale for v in normal]
    boxes = config["fixed_boxes"]
    count = config["blocks"]
    return dict(
        normal=normal,
        offset=config["height"] / scale,
        centers=[box[BOX_CENTER][:dim] for box in boxes]
        if boxes is not None
        else [[0.0] * dim for _ in range(count)],
        half_sizes=[box[BOX_HALF_SIZE][:dim] for box in boxes]
        if boxes is not None
        else [[1.0] * dim for _ in range(count)],
        rounding=[box[BOX_ROUNDING] for box in boxes] if boxes is not None else [0.0] * count,
        active=[boxes is not None] * count,
    )
