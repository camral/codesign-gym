"""Four editable geometry representations with common physical chamber semantics.

Coordinates are normalized material coordinates. Shape scores are not physical
signed distances. Hard thresholding is discrete; only relaxed occupancy has
ordinary topology-surrogate gradients. No physics particles are relocated.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from envs.soft_walker.contracts.fields import (
    DENSITY_BOUNDS,
    POISSON_BOUNDS,
    POISSON_DEFAULT,
    YOUNG_BOUNDS,
)
from envs.soft_walker.engine.embodiments.constants import (
    CPPN_NORM2_FLOOR,
    DEFAULT_OCCUPANCY_FLOOR,
    RAW_SCORE_LIMIT,
)
from envs.soft_walker.engine.embodiments.decoders import Fields, FieldSpec, grid_points, interpolate
from envs.soft_walker.engine.spaces import Box, Dict
from envs.soft_walker.engine.types import Embedding, HostArray, HostEmbedding
from envs.soft_walker.presets.embodiments.chamber_profile import (
    BASE_DENSITY,
    BASE_LEG_YOUNG,
    BASE_MATERIAL_IDS,
    BASE_STRENGTH,
    BASE_TORSO_YOUNG,
    CHAMBER_CHANNELS,
    CPPN_HIDDEN_NODES,
    CPPN_SEED,
    GEOMETRY,
    LEG_MATERIAL_INDEX,
    MATERIAL_REGION_COUNT,
    OCCUPANCY_SCORE_WIDTH,
    POISSON_GRID_SIDE,
    PRIMITIVE_HALF_SIZE_BOUNDS,
    SHAPE_GRID,
    TORSO_MATERIAL_INDEX,
)


def boxes(dim: int) -> tuple[HostArray, HostArray]:
    """Passive torso plus 2/4 attached legs, in normalized coordinates."""
    centers = [[0.0, GEOMETRY.torso_center_y] + ([0.0] if dim == 3 else [])]
    halves = [[1.0, GEOMETRY.torso_half_y] + ([1.0] if dim == 3 else [])]
    for x in (-GEOMETRY.leg_center_x, GEOMETRY.leg_center_x):
        for z in (-GEOMETRY.leg_center_z, GEOMETRY.leg_center_z) if dim == 3 else (None,):
            centers.append([x, GEOMETRY.leg_center_y] + ([cast(float, z)] if dim == 3 else []))
            halves.append(
                [GEOMETRY.leg_half_x, GEOMETRY.leg_half_y]
                + ([GEOMETRY.leg_half_z] if dim == 3 else [])
            )
    return np.asarray(centers), np.asarray(halves)


def box_score(points: jax.Array, centers: jax.Array, half_sizes: jax.Array) -> jax.Array:
    """Positive inside the union of axis-aligned boxes; normalized units."""
    q = jnp.abs(points[:, None, :] - centers[None, :, :]) - half_sizes[None, :, :]
    # L-infinity signed score is exact for occupancy, not Euclidean distance.
    return -jnp.min(jnp.max(q, axis=-1), axis=-1)


class ChamberDecoder:
    """Geometry embedding + two materials + fixed material-coordinate routing.

    Geometry parameters vary by representation; physical parameters remain in SI
    units and have explicit bounds. Primitive centers/sizes, scalar-grid values,
    or CPPN graph weights are genuinely consumed by decode(), not display-only.
    """

    def __init__(
        self,
        dim: int = 2,
        representation: str = "primitives",
        dtype: DTypeLike = "float64",
        design_mode: str = "hard",
        material_mode: str = "continuous",
    ) -> None:
        """Configure editable geometry coefficients and common physical chamber field semantics."""
        if representation not in ("primitives", "direct", "bspline", "cppn"):
            raise ValueError("Unknown geometry representation")
        self.spec = FieldSpec(
            dim=dim,
            channels=CHAMBER_CHANNELS[2 if dim == 2 else 3],
            design_mode=design_mode,
            material_mode=material_mode,
        )
        self.dtype = np.dtype(dtype)
        self.representation = representation
        self.coefficient_shape = SHAPE_GRID[:dim]
        self.centers, self.halves = boxes(dim)
        # CPPN: directional sigmoid gates, two gate-aggregation nodes, output.
        # Inputs x,y[,z],radius/sqrt(dim). One topological loop, shared over sites.
        self.hidden = CPPN_HIDDEN_NODES[dim]
        self.inputs = dim + 1
        self.nodes = self.hidden + 1
        self.allowed = np.arange(self.inputs + self.nodes)[None, :] < (
            self.inputs + np.arange(self.nodes)[:, None]
        )

    def space(self) -> Dict:
        """Expose representation coefficients and spatial material parameters."""
        physical = {
            "density": Box(*DENSITY_BOUNDS, (), self.dtype),
            "strength": Box(0.0, 1.0, (), self.dtype),
            "poisson_values": Box(
                *POISSON_BOUNDS, (POISSON_GRID_SIDE,) * self.spec.dim + (1,), self.dtype
            ),
        }
        if self.spec.material_mode == "continuous":
            physical["young"] = Box(*YOUNG_BOUNDS, (MATERIAL_REGION_COUNT,), self.dtype)
        else:
            physical["material_ids"] = Box(
                0, len(self.spec.palette) - 1, (MATERIAL_REGION_COUNT,), np.int32
            )
        if self.representation == "primitives":
            physical.update(
                centers=Box(-1.0, 1.0, self.centers.shape, self.dtype),
                half_sizes=Box(*PRIMITIVE_HALF_SIZE_BOUNDS, self.halves.shape, self.dtype),
            )
        elif self.representation in ("direct", "bspline"):
            physical["shape_values"] = Box(-1.0, 1.0, self.coefficient_shape + (1,), self.dtype)
        else:
            physical.update(
                weights=Box(-RAW_SCORE_LIMIT, RAW_SCORE_LIMIT, self.allowed.shape, self.dtype),
                bias=Box(-RAW_SCORE_LIMIT, RAW_SCORE_LIMIT, (self.nodes,), self.dtype),
                edge_mask=Box(0, 1, self.allowed.shape, np.int32),
            )
        return Dict(physical)

    def base(self) -> HostEmbedding:
        """Create a fresh baseline chamber embedding on the host."""
        result = {
            "density": np.asarray(BASE_DENSITY, self.dtype),
            "strength": np.asarray(BASE_STRENGTH, self.dtype),
            "poisson_values": np.full(
                (POISSON_GRID_SIDE,) * self.spec.dim + (1,), POISSON_DEFAULT, self.dtype
            ),
        }
        if self.spec.material_mode == "continuous":
            result["young"] = np.asarray([BASE_LEG_YOUNG, BASE_TORSO_YOUNG], self.dtype)
        else:
            result["material_ids"] = np.asarray(BASE_MATERIAL_IDS, np.int32)
        if self.representation == "primitives":
            result.update(
                centers=self.centers.astype(self.dtype), half_sizes=self.halves.astype(self.dtype)
            )
        elif self.representation in ("direct", "bspline"):
            points = grid_points(self.coefficient_shape).reshape(-1, self.spec.dim)
            q = np.abs(points[:, None] - self.centers) - self.halves
            scores = -np.min(np.max(q, axis=-1), axis=-1)
            result["shape_values"] = scores.reshape(self.coefficient_shape + (1,)).astype(
                self.dtype
            )
        else:
            result.update(self._cppn_seed())
        return result

    def upgrade_embedding(self, embodiment: Embedding) -> Embedding:
        """Fill only the missing Poisson parameter in an exact legacy layout."""
        if isinstance(embodiment, dict) and "poisson_values" not in embodiment:
            return dict(
                embodiment,
                poisson_values=np.full(
                    (POISSON_GRID_SIDE,) * self.spec.dim + (1,), POISSON_DEFAULT, self.dtype
                ),
            )
        return embodiment

    def _cppn_seed(self) -> HostEmbedding:
        """Build an acyclic CPPN seed matching the fixed chamber geometry."""
        weights = np.zeros(self.allowed.shape, self.dtype)
        bias = np.zeros(self.nodes, self.dtype)
        d, inp = self.spec.dim, self.inputs
        # The prefix order is part of the saved graph layout, not a tuning knob.
        x_low, x_high, torso_gate, x_union = range(4)
        weights[x_low, 0], weights[x_high, 0] = -CPPN_SEED.gate_gain, CPPN_SEED.gate_gain
        bias[x_low : x_high + 1] = CPPN_SEED.x_edge_bias
        weights[torso_gate, 1], bias[torso_gate] = CPPN_SEED.gate_gain, CPPN_SEED.torso_bias
        weights[x_union, inp + x_low : inp + x_high + 1] = CPPN_SEED.gate_gain
        bias[x_union] = CPPN_SEED.x_union_bias
        if d == 2:
            # h4 sharpens the torso transition; output performs a soft OR.
            torso_sharpener, output = range(4, 6)
            weights[torso_sharpener, inp + torso_gate] = CPPN_SEED.gate_gain
            bias[torso_sharpener] = CPPN_SEED.torso_sharpen_bias
            weights[output, [inp + x_union, inp + torso_sharpener]] = CPPN_SEED.output_gain
            bias[output] = CPPN_SEED.output_bias
        else:
            # h4/h5 z-edge gates; h6 z union; h7 corner-leg intersection.
            z_low, z_high, z_union, corner_intersection, output = range(4, 9)
            weights[z_low, 2], weights[z_high, 2] = -CPPN_SEED.gate_gain, CPPN_SEED.gate_gain
            bias[z_low : z_high + 1] = CPPN_SEED.z_edge_bias
            weights[z_union, inp + z_low : inp + z_high + 1] = CPPN_SEED.gate_gain
            bias[z_union] = CPPN_SEED.z_union_bias
            weights[corner_intersection, [inp + x_union, inp + z_union]] = CPPN_SEED.corner_gain
            bias[corner_intersection] = CPPN_SEED.corner_bias
            weights[output, [inp + torso_gate, inp + corner_intersection]] = CPPN_SEED.output_gain
            bias[output] = CPPN_SEED.output_bias
        return {"weights": weights, "bias": bias, "edge_mask": self.allowed.astype(np.int32)}

    def score(self, embodiment: Embedding, points: jax.Array) -> jax.Array:
        """Evaluate the selected representation occupancy score at material points."""
        if self.representation == "primitives":
            return box_score(points, embodiment["centers"], embodiment["half_sizes"])
        if self.representation in ("direct", "bspline"):
            return interpolate(
                embodiment["shape_values"], points, cubic=self.representation == "bspline"
            )[:, 0]
        radius = jnp.sqrt(jnp.sum(points * points, axis=-1, keepdims=True) + CPPN_NORM2_FLOOR)
        inputs = jnp.concatenate((points, radius / np.sqrt(self.spec.dim)), axis=-1)
        values = jnp.pad(inputs, ((0, 0), (0, self.nodes)))
        weights = jnp.asarray(embodiment["weights"]) * embodiment["edge_mask"] * self.allowed
        bias = jnp.asarray(embodiment["bias"])

        def node(i: jax.Array, current: jax.Array) -> jax.Array:
            """Evaluate one CPPN node in topological order."""
            value = current @ weights[i] + bias[i]
            value = jnp.where(i < self.hidden, jax.nn.sigmoid(value), value)
            return current.at[:, self.inputs + i].set(value)

        values = jax.lax.fori_loop(0, self.nodes, node, values)
        return values[:, -1]

    def regions(self, points: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Stable spatial region identity, independent of current deformation."""
        q = (points + 1.0) / 2.0
        torso = q[:, 1] >= GEOMETRY.torso_start_qy
        lx = (q[:, 0] > GEOMETRY.side_split_q).astype(jnp.int32)
        cx = (q[:, 0] - lx * GEOMETRY.leg_stride_qx >= GEOMETRY.chamber_split_qx).astype(jnp.int32)
        if self.spec.dim == 2:
            return torso, lx, 2 * lx + cx
        lz = (q[:, 2] > GEOMETRY.side_split_q).astype(jnp.int32)
        cz = (q[:, 2] - lz * GEOMETRY.leg_stride_qz >= GEOMETRY.chamber_split_qz).astype(jnp.int32)
        leg = 2 * lx + lz
        return torso, leg, 4 * leg + 2 * cx + cz

    def decode(self, embodiment: Embedding, points: jax.Array) -> Fields:
        """Convert occupancy scores and spatial material coefficients to physical fields."""
        embodiment = self.upgrade_embedding(embodiment)
        score = self.score(embodiment, points)
        if self.spec.design_mode == "hard":
            occupancy = (score >= 0.0).astype(points.dtype)
        else:
            occupancy = DEFAULT_OCCUPANCY_FLOOR + (1.0 - DEFAULT_OCCUPANCY_FLOOR) * jax.nn.sigmoid(
                score / OCCUPANCY_SCORE_WIDTH
            )
        torso, _, channel = self.regions(points)
        active = ~torso
        young = (
            embodiment["young"]
            if self.spec.material_mode == "continuous"
            else jnp.asarray(self.spec.palette, points.dtype)[embodiment["material_ids"]]
        )
        return Fields(
            occupancy,
            jnp.where(torso, young[TORSO_MATERIAL_INDEX], young[LEG_MATERIAL_INDEX]),
            jnp.broadcast_to(embodiment["density"], score.shape),
            jnp.where(active, embodiment["strength"], 0.0),
            jnp.broadcast_to(jnp.eye(self.spec.dim, dtype=points.dtype)[1], points.shape),
            active.astype(jnp.int32),
            channel,
            jnp.clip(interpolate(embodiment["poisson_values"], points)[:, 0], *POISSON_BOUNDS),
        )

    def structural_errors(self, embodiment: Embedding) -> list[str]:
        """Reject forward or cyclic CPPN edges without changing user coefficients."""
        if self.representation == "cppn" and np.any(
            np.asarray(embodiment["edge_mask"])[~self.allowed]
        ):
            return ["CPPN_CYCLE_OR_FORWARD_EDGE"]
        return []

    def descriptor(self) -> dict[str, object]:
        """Return portable chamber representation settings."""
        return {
            "name": "scenario_chamber_shape_v1",
            "representation": self.representation,
            "dtype": self.dtype.name,
            "spec": asdict(self.spec),
            "shape_coordinate_units": "normalized material coordinates [-1,1]",
            "material_routing": "fixed reference regions; not nearest deformed site",
            "cppn_hidden": self.hidden if self.representation == "cppn" else None,
            "coefficient_shape": self.coefficient_shape
            if self.representation in ("direct", "bspline")
            else None,
        }
