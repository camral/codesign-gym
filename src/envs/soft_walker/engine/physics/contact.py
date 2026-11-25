"""Smooth unilateral plane contact, with reference-area quadrature.

stiffness has units N/m^3: force = stiffness * reference_area * penetration.
No particle-count dependent constant forces. No hard-contact/self-contact claim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .terrain import Terrain

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PlaneContact:
    """Smooth reference-area contact coefficients in SI units."""

    enabled: bool = False
    height: float = 0.0
    stiffness: float = 1000000.0
    width: float = 0.0005
    friction: float = 0.6
    velocity_width: float = 0.01
    vertical_axis: int = 1

    def __post_init__(self) -> None:
        """Validate the plane orientation and smooth contact coefficients."""
        if not all(
            math.isfinite(x)
            for x in (self.height, self.stiffness, self.width, self.friction, self.velocity_width)
        ):
            raise ValueError("Contact parameters must be finite")
        if self.vertical_axis not in (0, 1, 2):
            raise ValueError("vertical_axis must be 0, 1, or 2")
        if self.stiffness < 0 or self.width <= 0 or self.friction < 0 or self.velocity_width <= 0:
            raise ValueError("Invalid contact regularization or material parameter")


def response(
    x: jax.Array, v: jax.Array, area: jax.Array, cfg: PlaneContact, terrain: Terrain | None = None
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return force, potential energy, and dissipated power at contact samples.

    x/v end in the spatial dimension; reference area broadcasts over samples.
    Terrain normals determine friction directions on nonflat surfaces.
    """
    if cfg.vertical_axis >= x.shape[-1]:
        raise ValueError("Contact vertical axis exceeds the spatial dimension")
    if not cfg.enabled:
        return jnp.zeros_like(x), jnp.zeros(x.shape[:-1], x.dtype), jnp.zeros(x.shape[:-1], x.dtype)
    if terrain is None:
        gap = x[..., cfg.vertical_axis] - cfg.height
        direction = jnp.broadcast_to(
            jnp.eye(x.shape[-1], dtype=x.dtype)[cfg.vertical_axis], x.shape
        )
    else:
        from .terrain import gap_normal

        gap, direction = gap_normal(x, terrain)
    depth = cfg.width * jax.nn.softplus(-gap / cfg.width)
    gate = jax.nn.sigmoid(-gap / cfg.width)
    normal = cfg.stiffness * area * depth * gate
    vt = v - jnp.sum(v * direction, axis=-1, keepdims=True) * direction
    denom = jnp.sqrt(jnp.sum(vt * vt, axis=-1) + cfg.velocity_width**2)
    tang = -cfg.friction * normal[..., None] * vt / denom[..., None]
    force = tang + normal[..., None] * direction
    energy = 0.5 * cfg.stiffness * area * depth**2
    diss = -jnp.sum(tang * vt, axis=-1)
    return (force, energy, diss)
