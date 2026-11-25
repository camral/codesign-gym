"""Shared bounded field interpretation and host validation for every decoder.

Sample locations use normalized material coordinates. Interpretation returns
JAX arrays and remains traceable; validation intentionally materializes arrays
on the host and must run before differentiating or compiling a rollout.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ...contracts.fields import ACTUATOR_DIRECTIONAL, ACTUATOR_PASSIVE
from ..types import HostArray, Shape
from .constants import (
    BASE_CATEGORY_SCORE,
    BASE_OCCUPANCY_SCORE,
    BASE_YOUNG,
    DIRECTION_NORM2_FLOOR,
    FIELD_BOUND_RTOL,
    FIELD_DIRECTION_ATOL,
    INVERSE_MATERIAL_CLIP,
    RAW_SCORE_LIMIT,
)

if TYPE_CHECKING:
    from .decoders import Fields, FieldSpec


def _poisson_score(value: float | np.generic, spec: FieldSpec) -> float | np.generic:
    """Host inverse of the new bounded score, including both Box endpoints."""
    lo, hi = spec.poisson_bounds
    if float(value) == (lo + hi) / 2:
        return 0.0
    fraction = 2 * (float(value) - lo) / (hi - lo) - 1
    return np.clip(
        np.arctanh(np.clip(fraction, -1.0, 1.0) * np.tanh(RAW_SCORE_LIMIT)),
        -RAW_SCORE_LIMIT,
        RAW_SCORE_LIMIT,
    )


def _log_bounded(score: jax.Array, bounds: tuple[float, float]) -> jax.Array:
    """Map dimensionless scores to strictly positive material coefficients."""
    lo, hi = bounds
    return jnp.exp((jnp.log(lo) + jnp.log(hi)) / 2 + jnp.tanh(score) * jnp.log(hi / lo) / 2)


def interpret(raw: jax.Array, spec: FieldSpec) -> Fields:
    """Convert raw scores (..., width) into bounded physical fields.

    Integer labels select actuator types/channels. Hard geometry uses discrete
    occupancy; relaxed geometry retains its configured occupancy floor.
    """
    from .decoders import Fields

    layout = spec.raw_layout
    raw = jnp.asarray(raw)
    if raw.ndim < 1 or raw.shape[-1] not in (spec.poisson_index, spec.width):
        raise ValueError("Raw fields need the current width or the exact legacy width")
    alpha = (
        (raw[..., layout.occupancy] >= 0).astype(raw.dtype)
        if spec.design_mode == "hard"
        else spec.occupancy_floor
        + (1 - spec.occupancy_floor)
        * jax.nn.sigmoid(raw[..., layout.occupancy] / spec.geometry_temperature)
    )
    e = _log_bounded(raw[..., layout.young], spec.young_bounds)
    d = spec.dim
    if spec.material_mode == "categorical":
        logits = raw[..., layout.material_logits]
        palette = jnp.asarray(spec.palette, dtype=raw.dtype)
        if spec.design_mode == "hard":
            e = palette[jnp.argmax(logits, axis=-1)]
        else:
            # In the spring model energy is linear in E: this is energy mixing.
            e = jnp.sum(jax.nn.softmax(logits / spec.material_temperature) * palette, axis=-1)
    rho = _log_bounded(raw[..., layout.density], spec.density_bounds)
    direction = raw[..., layout.direction]
    norm2 = jnp.sum(direction**2, axis=-1, keepdims=True)
    # Safe inactive directions BEFORE division. No 0*NaN pattern.
    fallback = jnp.broadcast_to(jnp.eye(d, dtype=raw.dtype)[0], direction.shape)
    direction = jnp.where(norm2 > DIRECTION_NORM2_FLOOR, direction, fallback)
    direction = direction / jnp.sqrt(jnp.sum(direction**2, axis=-1, keepdims=True))
    kind = jnp.argmax(raw[..., layout.actuator_logits], axis=-1).astype(jnp.int32)
    channel = jnp.argmax(raw[..., layout.channel_logits], axis=-1).astype(jnp.int32)
    strength = jax.nn.sigmoid(raw[..., layout.strength]) * (kind != ACTUATOR_PASSIVE)
    if raw.shape[-1] == spec.poisson_index:
        poisson = jnp.full_like(alpha, spec.poisson_default)
    else:
        lo, hi = spec.poisson_bounds
        # Normalizing by tanh(6) makes the coefficient Box endpoints represent
        # the physical endpoints exactly. CPPN scores may exceed that Box, so
        # the final bounded field still saturates safely. Interior is smooth.
        score = jnp.clip(
            jnp.tanh(raw[..., spec.poisson_index])
            / jnp.asarray(np.tanh(RAW_SCORE_LIMIT), raw.dtype),
            -1.0,
            1.0,
        )
        poisson = (
            jnp.asarray((lo + hi) / 2, raw.dtype) + jnp.asarray((hi - lo) / 2, raw.dtype) * score
        )
        poisson = jnp.clip(poisson, jnp.asarray(lo, raw.dtype), jnp.asarray(hi, raw.dtype))
        poisson = jnp.where(
            raw[..., spec.poisson_index] >= RAW_SCORE_LIMIT, jnp.asarray(hi, raw.dtype), poisson
        )
        poisson = jnp.where(
            raw[..., spec.poisson_index] <= -RAW_SCORE_LIMIT, jnp.asarray(lo, raw.dtype), poisson
        )
    return Fields(alpha, e, rho, strength, direction, kind, channel, poisson)


def baseline_raw(points: ArrayLike, spec: FieldSpec, dtype: DTypeLike = np.float64) -> HostArray:
    """Create host scores for 20 kPa material and directional actuation."""
    p = np.asarray(points)
    layout = spec.raw_layout
    r = np.zeros(p.shape[:-1] + (spec.width,), dtype=dtype)
    r[..., layout.occupancy] = BASE_OCCUPANCY_SCORE
    lo, hi = spec.young_bounds
    q = 2 * (np.log(BASE_YOUNG) - 0.5 * (np.log(lo) + np.log(hi))) / np.log(hi / lo)
    r[..., layout.young] = np.arctanh(np.clip(q, -INVERSE_MATERIAL_CLIP, INVERSE_MATERIAL_CLIP))
    r[..., layout.direction.start] = 1.0
    r[..., layout.material_logits.start + 1] = BASE_CATEGORY_SCORE  # material palette category 1
    r[..., layout.actuator_logits.start + ACTUATOR_DIRECTIONAL] = (
        BASE_CATEGORY_SCORE  # directional actuator
    )
    centers = np.linspace(-1, 1, spec.channels) if spec.channels > 1 else np.array([0.0])
    r[..., layout.channel_logits] = -((p[..., 0, None] - centers) ** 2)
    r[..., spec.poisson_index] = _poisson_score(spec.poisson_default, spec)
    return r


def grid_points(shape: Shape) -> HostArray:
    """Coefficient-grid points span the closed normalized envelope."""
    return np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij"), axis=-1)


def interpolate(coeff: jax.Array, points: jax.Array, cubic: bool = False) -> jax.Array:
    """Tensor interpolation with clamped coefficient extension at the boundary.

    Direct: multilinear, constant extrapolation. Cubic: cardinal cubic B-spline,
    four coefficients per dimension, repeated boundary coefficient extension.
    Partition of unity is preserved including at envelope endpoints.
    """
    dim = points.shape[-1]
    shape = coeff.shape[:dim]
    u = (jnp.clip(points, -1.0, 1.0) + 1.0) * jnp.asarray(
        [(n - 1) / 2 for n in shape], dtype=points.dtype
    )
    base = jnp.floor(u).astype(jnp.int32)
    t = u - base
    offsets: HostArray = np.asarray(
        list(product(range(4 if cubic else 2), repeat=dim)), dtype=np.int32
    )
    indices = base[:, None, :] + jnp.asarray(offsets)[None, :, :] - (1 if cubic else 0)
    indices = jnp.clip(indices, 0, jnp.asarray(shape) - 1)
    if cubic:
        weights = jnp.stack(
            (
                (1 - t) ** 3 / 6,
                (3 * t**3 - 6 * t**2 + 4) / 6,
                (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6,
                t**3 / 6,
            ),
            axis=-1,
        )
    else:
        weights = jnp.stack((1 - t, t), axis=-1)
    w = jnp.ones(indices.shape[:2], dtype=points.dtype)
    for axis in range(dim):
        w = w * weights[:, axis, offsets[:, axis]]
    values = coeff[tuple(indices[:, :, k] for k in range(dim))]
    return jnp.sum(values * w[..., None], axis=1)


def field_errors(fields: Fields, spec: FieldSpec, ncells: int) -> list[str]:
    """Validate sampled host shapes, finite values, bounds, and actuator directions."""
    shapes = {
        "occupancy": (ncells,),
        "young": (ncells,),
        "density": (ncells,),
        "strength": (ncells,),
        "direction": (ncells, spec.dim),
        "actuator_type": (ncells,),
        "channel": (ncells,),
    }
    if getattr(fields, "poisson", None) is not None:
        shapes["poisson"] = (ncells,)
    errors = []
    for name, shape in shapes.items():
        a = np.asarray(getattr(fields, name))
        if a.shape != shape:
            errors.append(f"FIELD_SHAPE:{name}")
        elif not np.all(np.isfinite(a)):
            errors.append(f"FIELD_NONFINITE:{name}")
    if errors:
        return errors
    for name, bounds in (
        ("occupancy", (0, 1)),
        ("young", spec.young_bounds),
        ("density", spec.density_bounds),
        ("strength", (0, 1)),
        ("actuator_type", (0, 2)),
        ("channel", (0, spec.channels - 1)),
    ):
        a = np.asarray(getattr(fields, name))
        if np.any(a < bounds[0] * (1 - FIELD_BOUND_RTOL)) or np.any(
            a > bounds[1] * (1 + FIELD_BOUND_RTOL)
        ):
            errors.append(f"FIELD_BOUNDS:{name}")
    for name in ("actuator_type", "channel"):
        if np.asarray(getattr(fields, name)).dtype.kind not in "iu":
            errors.append(f"FIELD_INTEGER_REQUIRED:{name}")
    if getattr(fields, "poisson", None) is not None:
        nu = np.asarray(fields.poisson)
        lo, hi = spec.poisson_bounds
        if np.any(nu < lo) or np.any(nu > hi):
            errors.append("FIELD_BOUNDS:poisson")
    norms = np.linalg.norm(np.asarray(fields.direction), axis=-1)
    directional = np.asarray(fields.actuator_type) == ACTUATOR_DIRECTIONAL
    if np.any(directional & (np.abs(norms - 1) > FIELD_DIRECTION_ATOL)):
        errors.append("NONUNIT_ACTUATOR_DIRECTION")
    if spec.design_mode == "hard" and np.any(
        (np.asarray(fields.occupancy) != 0) & (np.asarray(fields.occupancy) != 1)
    ):
        errors.append("HARD_OCCUPANCY_NOT_BINARY")
    return errors
