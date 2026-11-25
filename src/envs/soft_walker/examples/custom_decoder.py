"""Minimal custom continuous parameterization; optimization remains external."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from envs.soft_walker.engine.embodiments.decoders import Fields, FieldSpec
from envs.soft_walker.engine.spaces import Box, Dict

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from envs.soft_walker.engine.embodiments.decoders import Fields
    from envs.soft_walker.engine.spaces import Dict
    from envs.soft_walker.engine.types import (
        Array,
        Embedding,
        HostEmbedding,
    )


class UniformMaterialDecoder:
    """Uniform stiffness/density and a smooth, spatially varying Poisson ratio.

    The two Poisson endpoint coefficients interpolate along reference x; all
    decoded fields are device arrays with one value per sampled reference point.
    """

    def __init__(self, dtype: DTypeLike = "float64") -> None:
        """Select decoder precision and the default two-channel field specification."""
        self.dtype = np.dtype(dtype)
        self.spec = FieldSpec()

    def space(self) -> Dict:
        """Declare parameter bounds in pascals, kg/m3, and dimensionless Poisson ratio."""
        return Dict(
            {
                "young_pa": Box(5000.0, 100000.0, (1,), self.dtype),
                "density_kg_m3": Box(800.0, 1250.0, (1,), self.dtype),
                "poisson_left_right": Box(0.25, 0.45, (2,), self.dtype),
            }
        )

    def base(self) -> HostEmbedding:
        """Return fresh host arrays for the nominal material coefficients."""
        return {
            "young_pa": np.array([20000.0], dtype=self.dtype),
            "density_kg_m3": np.array([1000.0], dtype=self.dtype),
            "poisson_left_right": np.array([0.30, 0.40], dtype=self.dtype),
        }

    def descriptor(self) -> dict[str, object]:
        """Return stable serialization metadata for this custom decoder version."""
        return {"name": "user_uniform_material", "version": 2, "dtype": self.dtype.name}

    def structural_errors(self, embodiment: Embedding) -> list[str]:
        """Add no topology constraints beyond the parameter-space validation."""
        return []

    def decode(self, embodiment: Embedding, points: Array) -> Fields:
        """Sample material/actuation fields at normalized reference points (n, 2)."""
        n = points.shape[0]
        ones = jnp.ones(n, dtype=points.dtype)
        return Fields(
            ones,
            ones * embodiment["young_pa"][0],
            ones * embodiment["density_kg_m3"][0],
            0.5 * ones,
            jnp.tile(jnp.asarray([1.0, 0.0], dtype=points.dtype), (n, 1)),
            jnp.ones(n, jnp.int32),
            (points[:, 0] >= 0).astype(jnp.int32),
            jnp.interp(
                points[:, 0],
                jnp.asarray([-1.0, 1.0], dtype=points.dtype),
                embodiment["poisson_left_right"],
            ),
        )


# Usage after setting JAX_ENABLE_X64=1:
# env, base = envs.soft_walker.engine.make('SoftMPM2D-v0', decoder=UniformMaterialDecoder(), dtype='float64')
# obs, rejected, info = env.reset(embodiment=base)
