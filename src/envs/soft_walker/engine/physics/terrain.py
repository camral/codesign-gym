"""Seeded analytical terrain, with dynamic arrays shared by every physics model.

The default normal is +y in both dimensions. Slopes are a single bounded plane;
blocks are a union of rounded axis-aligned boxes and the ground half-space.
Union switches / box medial axes are piecewise differentiable, not smooth contact.
All geometry is generated on the host and serialized; no backend RNG parity claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from ..types import HostArray, JsonValue


@dataclass(frozen=True)
class TerrainConfig:
    """Seeded terrain generation settings expressed in SI length units."""

    kind: str = "flat"
    max_slope: float = 0.15
    blocks: int = 4
    block_start: float = 0.25
    block_spacing: float = 0.1
    block_width: float = 0.065
    block_height: float = 0.015
    rounding: float = 0.002
    height: float = 0.0
    # Optional deterministic overrides, leaving the original RNG path unchanged.
    # Slopes are dy/dx and dy/dz. Boxes store center xyz, half-size xyz, rounding.
    fixed_slopes: tuple[float, float] | None = None
    fixed_boxes: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        """Validate procedural capacities and deterministic terrain overrides."""
        if self.kind not in ("flat", "slopes", "blocks"):
            raise ValueError("terrain kind must be flat, slopes, or blocks")
        if type(self.blocks) is not int or not 0 <= self.blocks <= 32:
            raise ValueError("blocks must be an integer in [0,32]")
        vals = (
            self.max_slope,
            self.block_start,
            self.block_spacing,
            self.block_width,
            self.block_height,
            self.rounding,
            self.height,
        )
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("nonfinite terrain parameter")
        if (
            not 0 <= self.max_slope <= 0.5
            or min(self.block_spacing, self.block_width, self.block_height) <= 0
        ):
            raise ValueError("invalid terrain dimensions or slope")
        if not 0 <= self.rounding < min(self.block_width, self.block_height) / 4:
            raise ValueError("rounding too large for smallest sampled block")
        if self.fixed_slopes is not None:
            if len(self.fixed_slopes) != 2 or any(
                not math.isfinite(v) or abs(v) > 0.5 for v in self.fixed_slopes
            ):
                raise ValueError("fixed_slopes needs two finite slopes in [-.5,.5]")
            if self.kind != "slopes":
                raise ValueError("fixed_slopes requires kind=slopes")
        if self.fixed_boxes is not None:
            if self.kind != "blocks" or len(self.fixed_boxes) != self.blocks:
                raise ValueError("fixed_boxes must match the block capacity")
            for box in self.fixed_boxes:
                if (
                    len(box) != 7
                    or not all(math.isfinite(v) for v in box)
                    or min(box[3:6]) <= 0
                    or not 0 <= box[6] < min(box[3:6])
                ):
                    raise ValueError("Invalid explicit rounded box")


class Terrain(NamedTuple):
    """Fixed-capacity analytical support geometry stored as JAX arrays."""

    normal: jax.Array
    offset: jax.Array
    centers: jax.Array
    half_sizes: jax.Array
    rounding: jax.Array
    active: jax.Array


def generate(
    config: TerrainConfig = TerrainConfig(),
    seed: int = 0,
    dim: int = 2,
    dtype: DTypeLike = np.float64,
) -> Terrain:
    """Create fixed-capacity terrain arrays from a host seed in 2D or 3D."""
    if dim not in (2, 3):
        raise ValueError("terrain dimension must be 2 or 3")
    rng = np.random.default_rng(seed)
    slopes = np.zeros(dim)
    slopes[1] = 1.0
    if config.kind == "slopes":
        slopes[0] = -rng.uniform(-config.max_slope, config.max_slope)
        if dim == 3:
            slopes[2] = -rng.uniform(-config.max_slope, config.max_slope)
    if config.fixed_slopes is not None:
        slopes[0] = -config.fixed_slopes[0]
        if dim == 3:
            slopes[2] = -config.fixed_slopes[1]
        elif config.fixed_slopes[1] != 0:
            raise ValueError("Nonzero lateral slope requires 3D")
    norm = np.linalg.norm(slopes)
    normal = slopes / norm
    offset = config.height / norm
    # Fixed capacity even for flat/slopes: changing the seed or terrain kind
    # within a capacity does not change numerical array shapes.
    n = config.blocks
    centers = np.zeros((n, dim))
    half = np.ones((n, dim))
    r = np.zeros(n)
    active: HostArray = np.zeros(n, bool)
    if config.kind == "blocks":
        active[:] = True
        for k in range(n):
            width = config.block_width * rng.uniform(0.8, 1.2)
            height = config.block_height * rng.uniform(0.6, 1.0)
            centers[k, 0] = config.block_start + k * config.block_spacing + rng.uniform(-0.01, 0.01)
            # Box extends below ground so the union has no exposed undersurface.
            centers[k, 1] = config.height
            half[k, 0] = width / 2
            half[k, 1] = height
            if dim == 3:
                half[k, 2] = 0.35
            r[k] = config.rounding
    if config.fixed_boxes is not None:
        boxes: HostArray = np.asarray(config.fixed_boxes, dtype=float).reshape(n, 7)
        centers[:] = boxes[:, :dim]
        half[:] = boxes[:, 3 : 3 + dim]
        r[:] = boxes[:, 6]
        active[:] = True
    return Terrain(
        jnp.asarray(normal, dtype),
        jnp.asarray(offset, dtype),
        jnp.asarray(centers, dtype),
        jnp.asarray(half, dtype),
        jnp.asarray(r, dtype),
        jnp.asarray(active),
    )


def gap_normal(x: jax.Array, terrain: Terrain) -> tuple[jax.Array, jax.Array]:
    """Signed distance and outward normal; positive is outside solid terrain.

    Analytic normals avoid a nested reverse pass in every Newton residual.
    At nonunique union/medial-axis points a deterministic branch is selected.
    Unit normal and energy-gradient consistency are tested away from switches.
    """
    plane = jnp.sum(x * terrain.normal, axis=-1) - terrain.offset
    normal = jnp.broadcast_to(terrain.normal, x.shape)
    if terrain.centers.shape[0] == 0:
        return (plane, normal)
    delta = x[..., None, :] - terrain.centers
    q = jnp.abs(delta) - (terrain.half_sizes - terrain.rounding[:, None])
    outside = jnp.maximum(q, 0.0)
    norm2 = jnp.sum(outside**2, axis=-1)
    norm = jnp.sqrt(jnp.maximum(norm2, jnp.asarray(1e-24, x.dtype)))
    # Select one face for both distance and normal. jnp.max has a different
    # derivative at ties (it averages the tied faces), which would make contact
    # force disagree with the energy gradient and with Warp's first-face rule.
    face_index = jnp.argmax(q, axis=-1)
    interior = jnp.take_along_axis(q, face_index[..., None], axis=-1)[..., 0]
    dist = jnp.where(interior > 0, norm, interior) - terrain.rounding
    # sign(0)=+1 makes deterministic inside-face choices; that point is not smooth.
    sign = jnp.where(delta >= 0, 1.0, -1.0)
    outward = outside / norm[..., None] * sign
    face = jax.nn.one_hot(face_index, x.shape[-1], dtype=x.dtype) * sign
    normals = jnp.where((interior > 0)[..., None], outward, face)
    dist = jnp.where(terrain.active, dist, jnp.inf)
    index = jnp.argmin(dist, axis=-1)
    box = jnp.take_along_axis(dist, index[..., None], axis=-1)[..., 0]
    bn = jnp.take_along_axis(normals, index[..., None, None], axis=-2)[..., 0, :]
    use = box < plane
    return jnp.where(use, box, plane), jnp.where(use[..., None], bn, normal)


def serialize(terrain: Terrain) -> dict[str, JsonValue]:
    """Materialize terrain arrays as non-executable JSON-compatible values."""
    return {k: np.asarray(v).tolist() for k, v in terrain._asdict().items()}


def from_dict(values: dict[str, JsonValue], dtype: DTypeLike = np.float64) -> Terrain:
    """Restore a terrain record after strict shape, finiteness, and geometry checks."""
    if set(values) != set(Terrain._fields):
        raise ValueError("terrain schema mismatch")
    t = Terrain(
        *(jnp.asarray(values[k], bool if k == "active" else dtype) for k in Terrain._fields)
    )
    dim = t.normal.shape[0]
    if t.centers.size == 0 and t.half_sizes.size == 0:
        t = t._replace(centers=t.centers.reshape(0, dim), half_sizes=t.half_sizes.reshape(0, dim))
    if (
        dim not in (2, 3)
        or t.centers.ndim != 2
        or t.centers.shape[1] != dim
        or t.half_sizes.shape != t.centers.shape
    ):
        raise ValueError("terrain shape mismatch")
    if t.active.shape != (t.centers.shape[0],) or t.rounding.shape != t.active.shape:
        raise ValueError("terrain capacity mismatch")
    if not all(np.all(np.isfinite(x)) for x in t):
        raise ValueError("nonfinite terrain")
    if not np.isclose(np.linalg.norm(t.normal), 1.0) or t.normal[1] <= 0:
        raise ValueError("invalid plane normal")
    if (
        np.any(t.half_sizes <= 0)
        or np.any(t.rounding < 0)
        or np.any(t.rounding[:, None] >= t.half_sizes)
    ):
        raise ValueError("invalid box dimensions")
    return t
