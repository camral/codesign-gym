"""Bounded embeddings -> semantic fields, independently of discretization.

All coordinates are normalized material coordinates [-1,1]^dim. Geometry scores
are dimensionless, not signed distances unless explicitly stated. See MODEL.md.
This module contains no optimization algorithm or policy representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ...contracts.fields import (
    ACTUATOR_TYPE_COUNT,
    DENSITY_BOUNDS,
    LABEL_ACTUATOR,
    LABEL_CHANNEL,
    LABEL_MATERIAL,
    LABEL_WIDTH,
    MATERIAL_CATEGORY_COUNT,
    POISSON_BOUNDS,
    POISSON_DEFAULT,
    YOUNG_BOUNDS,
    RawFieldLayout,
)
from ..spaces import Box, Dict
from ..types import Embedding, HostArray, HostEmbedding, Shape
from .constants import (
    CPPN_BASE_COORDINATE_WEIGHT,
    CPPN_BASE_HIDDEN_WEIGHT,
    CPPN_BASE_RADIUS_WEIGHT,
    CPPN_NORM2_FLOOR,
    DEFAULT_GEOMETRY_WIDTH,
    DEFAULT_MATERIAL_PALETTE,
    DEFAULT_OCCUPANCY_FLOOR,
    DEFAULT_UNION_BLEND_WIDTH,
    DISABLED_PRIMITIVE_DISTANCE,
    LABEL_LOGIT,
    MAX_CHANNELS,
    MAX_CPPN_HIDDEN,
    MAX_PRIMITIVES,
    PRIMITIVE_AXIS_MIN_NORM,
    PRIMITIVE_BASE_HALF_LENGTH,
    PRIMITIVE_BASE_HALF_SIZE,
    PRIMITIVE_BASE_RADIUS,
    PRIMITIVE_BASE_ROUNDING,
    PRIMITIVE_HALF_LENGTH_BOUNDS,
    PRIMITIVE_HALF_SIZE_BOUNDS,
    PRIMITIVE_NORM2_FLOOR,
    PRIMITIVE_RADIUS_BOUNDS,
    PRIMITIVE_ROUNDING_BOUNDS,
    RAW_SCORE_LIMIT,
)
from .fields import (
    _log_bounded as _log_bounded,
    _poisson_score as _poisson_score,
    baseline_raw as baseline_raw,
    grid_points as grid_points,
    interpolate as interpolate,
    interpret as interpret,
)


class Fields(NamedTuple):
    """Sampled physical fields; leading dimensions follow the requested material sites.

    Young modulus is in Pa, density in kg/m^3, and Poisson ratio dimensionless.
    Directions end in dim; actuator type and channel arrays contain integer labels."""

    occupancy: jax.Array
    young: jax.Array
    density: jax.Array
    strength: jax.Array
    direction: jax.Array
    actuator_type: jax.Array
    channel: jax.Array
    # Legacy custom decoders may still construct the original seven fields.
    # None means the material default (or an explicit legacy global override).
    poisson: jax.Array | None = None


@dataclass(frozen=True)
class FieldSpec:
    """Immutable interpretation settings shared by every morphology representation."""

    dim: int = 2
    channels: int = 2
    design_mode: str = "hard"
    material_mode: str = "continuous"
    geometry_temperature: float = 1.0
    material_temperature: float = 1.0
    occupancy_floor: float = DEFAULT_OCCUPANCY_FLOOR
    young_bounds: tuple[float, float] = YOUNG_BOUNDS
    density_bounds: tuple[float, float] = DENSITY_BOUNDS
    palette: tuple[float, float, float] = DEFAULT_MATERIAL_PALETTE
    poisson_bounds: tuple[float, float] = POISSON_BOUNDS
    poisson_default: float = POISSON_DEFAULT

    def __post_init__(self) -> None:
        """Reject inconsistent dimensions, material bounds, and decoding modes."""
        if self.dim not in (2, 3) or not (1 <= self.channels <= MAX_CHANNELS):
            raise ValueError("dim must be 2/3; channels must be in [1,16]")
        if self.design_mode not in ("hard", "relaxed") or self.material_mode not in (
            "continuous",
            "categorical",
        ):
            raise ValueError("unsupported design/material mode")
        if not (0 < self.occupancy_floor < 1):
            raise ValueError("occupancy_floor must be in (0,1)")
        if not all(
            np.isfinite(x) and x > 0 for x in (self.geometry_temperature, self.material_temperature)
        ):
            raise ValueError("temperatures must be positive")
        for bounds in (self.young_bounds, self.density_bounds):
            if len(bounds) != 2 or not (0 < bounds[0] < bounds[1] < np.inf):
                raise ValueError("invalid material bounds")
        if len(self.palette) != 3 or any(
            not self.young_bounds[0] <= x <= self.young_bounds[1] for x in self.palette
        ):
            raise ValueError("palette must contain three bounded stiffnesses")
        if (
            len(self.poisson_bounds) != 2
            or not POISSON_BOUNDS[0]
            <= self.poisson_bounds[0]
            < self.poisson_bounds[1]
            <= POISSON_BOUNDS[1]
            or not np.isfinite(self.poisson_default)
            or not self.poisson_bounds[0] <= self.poisson_default <= self.poisson_bounds[1]
        ):
            raise ValueError("Poisson bounds/default must lie in [0.25, 0.45]")

    @property
    def raw_layout(self) -> RawFieldLayout:
        """Computed raw-score offsets; excluded from serialized dataclass fields."""
        return RawFieldLayout(self.dim, self.channels)

    @property
    def width(self) -> int:
        """Number of scalar scores produced per sampled material location."""
        # Append nu AFTER all historical scores; category/channel indices stay.
        return self.raw_layout.width

    @property
    def poisson_index(self) -> int:
        """Index of the appended Poisson score; earlier field indices are stable."""
        # Stable raw layout: four scalar scores, dim direction scores, three
        # material logits, three actuator-kind logits, then channel logits.
        return self.raw_layout.poisson


class Decoder(Protocol):
    """Replaceable embedding-to-fields contract; decode must be JAX traceable."""

    spec: FieldSpec
    dtype: np.dtype

    def space(self) -> Dict:
        """Return host bounds and shapes for each stable embedding key."""
        ...

    def base(self) -> HostEmbedding:
        """Return a deterministic valid coefficient dictionary owned by the caller."""
        ...

    def decode(self, embodiment: Embedding, points: jax.Array) -> Fields:
        """Sample normalized material points (..., dim) into device-resident physical fields."""
        ...

    def descriptor(self) -> dict[str, object]:
        """Describe representation settings for portable design serialization."""
        ...

    def structural_errors(self, embodiment: Embedding) -> list[str]:
        """Return host-side structural rejection codes without mutating coefficients."""
        ...


class GridDecoder:
    """Direct fields; optional integer material/actuator/channel labels."""

    name = "direct"
    cubic = False

    def __init__(
        self,
        spec: FieldSpec = FieldSpec(),
        shape: Shape | None = None,
        dtype: DTypeLike = "float64",
        discrete_labels: bool = False,
    ) -> None:
        """Configure a direct coefficient grid over normalized material coordinates.

        Shape must match the field dimension; dtype controls embedding and field precision.
        """
        self.spec, self.dtype = spec, np.dtype(dtype)
        self.shape = tuple(shape or (3,) * spec.dim)
        if len(self.shape) != spec.dim or any(n < (4 if self.cubic else 2) for n in self.shape):
            raise ValueError("invalid coefficient grid")
        self.discrete_labels = bool(discrete_labels)

    def space(self) -> Dict:
        """Return named coefficient bounds and dtypes for this representation."""
        s = {
            "values": Box(
                -RAW_SCORE_LIMIT, RAW_SCORE_LIMIT, self.shape + (self.spec.width,), self.dtype
            )
        }
        if self.discrete_labels:
            low: HostArray = np.zeros(self.shape + (LABEL_WIDTH,), dtype=np.int32)
            high = np.broadcast_to(
                np.array(
                    [MATERIAL_CATEGORY_COUNT - 1, ACTUATOR_TYPE_COUNT - 1, self.spec.channels - 1],
                    dtype=np.int32,
                ),
                low.shape,
            ).copy()
            s["labels"] = Box(low, high, dtype=np.int32)
        return Dict(s)

    def base(self) -> HostEmbedding:
        """Return a fresh host embedding with a connected baseline geometry."""
        pts = grid_points(self.shape)
        result = {"values": baseline_raw(pts, self.spec, self.dtype)}
        if self.discrete_labels:
            labels: HostArray = np.ones(self.shape + (LABEL_WIDTH,), dtype=np.int32)
            labels[..., LABEL_CHANNEL] = np.minimum(
                ((pts[..., 0] + 1) / 2 * self.spec.channels).astype(np.int32),
                self.spec.channels - 1,
            )
            result["labels"] = labels
        return result

    def raw(self, emb: Embedding, points: jax.Array) -> jax.Array:
        """Evaluate raw semantic scores at normalized material points shaped (N, dim)."""
        raw = interpolate(emb["values"], points, self.cubic)
        if self.discrete_labels:
            idx = jnp.rint(
                (jnp.clip(points, -1, 1) + 1) * jnp.asarray([(n - 1) / 2 for n in self.shape])
            ).astype(jnp.int32)
            labels = emb["labels"][tuple(idx[:, k] for k in range(self.spec.dim))]
            layout = self.spec.raw_layout
            raw = raw.at[:, layout.material_logits].set(
                LABEL_LOGIT
                * jax.nn.one_hot(
                    labels[:, LABEL_MATERIAL], MATERIAL_CATEGORY_COUNT, dtype=raw.dtype
                )
            )
            raw = raw.at[:, layout.actuator_logits].set(
                LABEL_LOGIT
                * jax.nn.one_hot(labels[:, LABEL_ACTUATOR], ACTUATOR_TYPE_COUNT, dtype=raw.dtype)
            )
            raw = raw.at[:, layout.channel_logits].set(
                LABEL_LOGIT
                * jax.nn.one_hot(labels[:, LABEL_CHANNEL], self.spec.channels, dtype=raw.dtype)
            )
        return raw

    def decode(self, emb: Embedding, points: jax.Array) -> Fields:
        """Decode scores into physical arrays with leading sample dimension N."""
        return interpret(self.raw(emb, points), self.spec)

    def structural_errors(self, emb: Embedding) -> list[str]:
        """Report topology or representation constraints beyond coefficient bounds."""
        return []

    def descriptor(self) -> dict[str, object]:
        """Return construction settings for portable non-pickle design serialization."""
        return {
            "name": self.name,
            "spec": asdict(self.spec),
            "shape": self.shape,
            "dtype": self.dtype.name,
            "discrete_labels": self.discrete_labels,
        }


class BSplineDecoder(GridDecoder):
    """Cubic B-spline semantic fields with at least four coefficients per axis.

    Embedding keys and field interpretation match GridDecoder; optional integer
    labels still use nearest-site sampling rather than cubic interpolation.
    """

    name = "bspline"
    cubic = True

    def __init__(
        self,
        spec: FieldSpec = FieldSpec(),
        shape: Shape | None = None,
        dtype: DTypeLike = "float64",
        discrete_labels: bool = False,
    ) -> None:
        """Configure a cubic coefficient grid, rejecting undersized or mismatched axes."""
        super().__init__(spec, shape or (4,) * spec.dim, dtype, discrete_labels)


class CPPNDecoder:
    """Bounded feed-forward graph with explicit topological node order.

    Inputs: x,y[,z],radius/sqrt(dim). Hidden nodes: sigmoid. Output nodes:
    identity-valued semantic scores. A connection may only read an input or an
    EARLIER node. Graph mutations change edge_mask; weights remain differentiable.
    This is a separated-output adaptation, not the exact five-class NEAT output.
    """

    name = "cppn"

    def __init__(
        self, spec: FieldSpec = FieldSpec(), hidden: int = 1, dtype: DTypeLike = "float64"
    ) -> None:
        """Create a fixed acyclic neural field template with typed bounded parameter spaces."""
        self.spec, self.hidden, self.dtype = spec, int(hidden), np.dtype(dtype)
        if not 1 <= self.hidden <= MAX_CPPN_HIDDEN:
            raise ValueError("hidden must be in [1,32]")
        self.inputs = spec.dim + 1
        self.nodes = hidden + spec.width
        self.allowed = np.arange(self.inputs + self.nodes)[None, :] < (
            self.inputs + np.arange(self.nodes)[:, None]
        )

    def space(self) -> Dict:
        """Return named coefficient bounds and dtypes for this representation."""
        return Dict(
            {
                "weights": Box(
                    -RAW_SCORE_LIMIT,
                    RAW_SCORE_LIMIT,
                    (self.nodes, self.inputs + self.nodes),
                    self.dtype,
                ),
                "bias": Box(-RAW_SCORE_LIMIT, RAW_SCORE_LIMIT, (self.nodes,), self.dtype),
                "edge_mask": Box(0, 1, (self.nodes, self.inputs + self.nodes), np.int32),
            }
        )

    def base(self) -> HostEmbedding:
        """Return a fresh host embedding with a connected baseline geometry."""
        mask = self.allowed.astype(np.int32)
        weights = np.zeros(mask.shape, dtype=self.dtype)
        bias = np.zeros(self.nodes, dtype=self.dtype)
        baseline = baseline_raw(np.zeros((1, self.spec.dim)), self.spec, self.dtype)[0]
        bias[self.hidden :] = baseline
        weights[0, : self.inputs] = np.asarray(
            [CPPN_BASE_COORDINATE_WEIGHT] * self.spec.dim + [CPPN_BASE_RADIUS_WEIGHT], self.dtype
        )
        weights[self.hidden, self.inputs] = CPPN_BASE_HIDDEN_WEIGHT
        centers = (
            np.linspace(-1, 1, self.spec.channels) if self.spec.channels > 1 else np.array([0.0])
        )
        for k, c in enumerate(centers):
            row = self.hidden + self.spec.raw_layout.channel_logits.start + k
            # Expand -(x-c)**2; the common -x**2 cancels between channel logits.
            weights[row, 0] = 2 * c
            bias[row] = -c * c
        return {"weights": weights, "bias": bias, "edge_mask": mask}

    def raw(self, emb: Embedding, points: jax.Array) -> jax.Array:
        """Evaluate raw semantic scores at normalized material points shaped (N, dim)."""
        radius = jnp.sqrt(
            jnp.sum(points**2, axis=-1, keepdims=True) + jnp.asarray(CPPN_NORM2_FLOOR, points.dtype)
        ) / np.sqrt(self.spec.dim)
        inputs = jnp.concatenate((points, radius), axis=-1)
        values = jnp.concatenate(
            (inputs, jnp.zeros((points.shape[0], self.nodes), dtype=points.dtype)), axis=-1
        )
        w = emb["weights"] * emb["edge_mask"] * jnp.asarray(self.allowed)

        def one(i: jax.Array, values: jax.Array) -> jax.Array:
            """Evaluate one acyclic network node in the fixed topological order."""
            z = values @ w[i] + emb["bias"][i]
            value = jnp.where(i < self.hidden, jax.nn.sigmoid(z), z)
            return values.at[:, self.inputs + i].set(value)

        values = jax.lax.fori_loop(0, self.nodes, one, values)
        return values[:, self.inputs + self.hidden :]

    def decode(self, emb: Embedding, points: jax.Array) -> Fields:
        """Decode scores into physical arrays with leading sample dimension N."""
        return interpret(self.raw(emb, points), self.spec)

    def structural_errors(self, emb: Embedding) -> list[str]:
        """Report topology or representation constraints beyond coefficient bounds."""
        if np.any(np.asarray(emb["edge_mask"])[~self.allowed] != 0):
            return ["CPPN_CYCLE_OR_FORWARD_EDGE"]
        return []

    def descriptor(self) -> dict[str, object]:
        """Return construction settings for portable non-pickle design serialization."""
        return {
            "name": self.name,
            "spec": asdict(self.spec),
            "hidden": self.hidden,
            "dtype": self.dtype.name,
        }


class PrimitiveDecoder:
    """Ellipsoid implicit distance, capsule SDF, and rounded-box SDF unions.

    kind: 0 ellipsoid, 1 capsule, 2 rounded box. Hard mode uses exact min and
    nearest-primitive property assignment. Relaxed mode uses a soft-min union
    and convex property weights. Ellipsoid distance is NOT an exact SDF.
    """

    name = "primitives"

    def __init__(
        self,
        spec: FieldSpec = FieldSpec(),
        count: int = 3,
        dtype: DTypeLike = "float64",
        blend_width: float = DEFAULT_UNION_BLEND_WIDTH,
        geometry_width: float = DEFAULT_GEOMETRY_WIDTH,
    ) -> None:
        self.spec, self.dtype, self.count = spec, np.dtype(dtype), int(count)
        self.blend_width, self.geometry_width = float(blend_width), float(geometry_width)
        if not 1 <= self.count <= MAX_PRIMITIVES or min(self.blend_width, self.geometry_width) <= 0:
            raise ValueError("invalid primitive capacity or smoothing width")

    def space(self) -> Dict:
        """Return named coefficient bounds and dtypes for this representation."""
        p, d = (self.count, self.spec.dim)
        return Dict(
            {
                "center": Box(-1.0, 1.0, (p, d), self.dtype),
                "half_size": Box(*PRIMITIVE_HALF_SIZE_BOUNDS, (p, d), self.dtype),
                "axis": Box(-1.0, 1.0, (p, d), self.dtype),
                "radius": Box(*PRIMITIVE_RADIUS_BOUNDS, (p,), self.dtype),
                "half_length": Box(*PRIMITIVE_HALF_LENGTH_BOUNDS, (p,), self.dtype),
                "rounding": Box(*PRIMITIVE_ROUNDING_BOUNDS, (p,), self.dtype),
                "kind": Box(0, 2, (p,), np.int32),
                "enabled": Box(0, 1, (p,), np.int32),
                "features": Box(
                    -RAW_SCORE_LIMIT, RAW_SCORE_LIMIT, (p, self.spec.width - 1), self.dtype
                ),
            }
        )

    def base(self) -> HostEmbedding:
        """Return a fresh host embedding with a connected baseline geometry."""
        p, d = (self.count, self.spec.dim)
        axis = np.zeros((p, d), dtype=self.dtype)
        axis[:, 0] = 1
        enabled: HostArray = np.zeros(p, np.int32)
        enabled[0] = 1
        return {
            "center": np.zeros((p, d), self.dtype),
            "half_size": np.full((p, d), PRIMITIVE_BASE_HALF_SIZE, self.dtype),
            "axis": axis,
            "radius": np.full(p, PRIMITIVE_BASE_RADIUS, self.dtype),
            "half_length": np.full(p, PRIMITIVE_BASE_HALF_LENGTH, self.dtype),
            "rounding": np.full(p, PRIMITIVE_BASE_ROUNDING, self.dtype),
            "kind": np.full(p, 2, np.int32),
            "enabled": enabled,
            "features": baseline_raw(np.zeros((p, d)), self.spec, self.dtype)[:, 1:],
        }

    def raw(self, emb: Embedding, points: jax.Array) -> jax.Array:
        """Evaluate raw semantic scores at normalized material points shaped (N, dim)."""
        delta = points[:, None, :] - emb["center"][None, :, :]
        tiny = jnp.asarray(PRIMITIVE_NORM2_FLOOR, points.dtype)
        norm = lambda x: jnp.sqrt(jnp.sum(x * x, axis=-1) + tiny)
        ellipse = (norm(delta / emb["half_size"]) - 1) * jnp.min(emb["half_size"], axis=-1)
        axis = emb["axis"] / jnp.sqrt(jnp.sum(emb["axis"] ** 2, axis=-1, keepdims=True) + tiny)
        proj = jnp.clip(jnp.sum(delta * axis, axis=-1), -emb["half_length"], emb["half_length"])
        capsule = norm(delta - proj[..., None] * axis) - emb["radius"]
        q = jnp.abs(delta) - (emb["half_size"] - emb["rounding"][:, None])
        box = norm(jnp.maximum(q, 0)) + jnp.minimum(jnp.max(q, axis=-1), 0) - emb["rounding"]
        distances = jnp.where(emb["kind"] == 0, ellipse, jnp.where(emb["kind"] == 1, capsule, box))
        distances = jnp.where(
            emb["enabled"] != 0, distances, jnp.asarray(DISABLED_PRIMITIVE_DISTANCE, points.dtype)
        )
        if self.spec.design_mode == "hard":
            nearest = jnp.argmin(distances, axis=-1)
            phi = jnp.min(distances, axis=-1)
            features = emb["features"][nearest]
        else:
            weights = jax.nn.softmax(-distances / self.blend_width, axis=-1)
            phi = -self.blend_width * jax.scipy.special.logsumexp(
                -distances / self.blend_width, axis=-1
            )
            features = weights @ emb["features"]
        return jnp.concatenate((-phi[:, None] / self.geometry_width, features), axis=-1)

    def decode(self, emb: Embedding, points: jax.Array) -> Fields:
        """Decode scores into physical arrays with leading sample dimension N."""
        return interpret(self.raw(emb, points), self.spec)

    def structural_errors(self, emb: Embedding) -> list[str]:
        """Report topology or representation constraints beyond coefficient bounds."""
        errors = []
        if not np.any(emb["enabled"]):
            errors.append("NO_ENABLED_PRIMITIVE")
        if np.any(np.linalg.norm(emb["axis"], axis=-1) < PRIMITIVE_AXIS_MIN_NORM):
            errors.append("DEGENERATE_PRIMITIVE_AXIS")
        if np.any(emb["rounding"] >= np.min(emb["half_size"], axis=-1)):
            errors.append("INVALID_ROUNDING")
        return errors

    def descriptor(self) -> dict[str, object]:
        """Return construction settings for portable non-pickle design serialization."""
        return {
            "name": self.name,
            "spec": asdict(self.spec),
            "count": self.count,
            "dtype": self.dtype.name,
            "blend_width": self.blend_width,
            "geometry_width": self.geometry_width,
        }


def make_decoder(
    name: str = "direct",
    *,
    spec: FieldSpec | None = None,
    dtype: DTypeLike = "float64",
    **kwargs: object,
) -> Decoder:
    """Construct a named representation with the shared physical field contract."""
    constructors = {
        "direct": GridDecoder,
        "cppn": CPPNDecoder,
        "primitives": PrimitiveDecoder,
        "bspline": BSplineDecoder,
    }
    if name not in constructors:
        raise ValueError(f"Unknown decoder {name!r}")
    return constructors[name](spec=spec or FieldSpec(), dtype=dtype, **kwargs)


def decoder_from_descriptor(desc: dict[str, object]) -> Decoder:
    """Rebuild a built-in decoder from its portable constructor descriptor."""
    desc = dict(desc)
    name = cast(str, desc.pop("name"))
    spec = dict(cast(dict[str, object], desc.pop("spec")))
    for k in ("palette", "young_bounds", "density_bounds", "poisson_bounds"):
        if k in spec:
            spec[k] = tuple(cast(list[float], spec[k]))
    # Dataclass constructor validates the decoded descriptor.
    return make_decoder(name, spec=FieldSpec(**spec), **desc)  # type: ignore[arg-type]


def migrate_legacy_embedding(decoder: Decoder, embodiment: Embedding) -> HostEmbedding:
    """Upgrade an exact pre-Poisson built-in layout; never shift old scores.

    Used by the saved-design loader for descriptors predating the appended
    head. This is also an explicit host utility for old in-memory dictionaries.
    A legacy global ``poisson`` override is preserved when present.
    """
    result = {k: np.asarray(v).copy() for k, v in embodiment.items()}
    score = _poisson_score(decoder.spec.poisson_default, decoder.spec)
    if isinstance(decoder, GridDecoder):
        old = result["values"]
        if old.shape != decoder.shape + (decoder.spec.poisson_index,):
            raise ValueError("Legacy coefficient shape mismatch")
        result["values"] = np.concatenate(
            (old, np.full(old.shape[:-1] + (1,), score, old.dtype)), axis=-1
        )
    elif isinstance(decoder, PrimitiveDecoder):
        old = result["features"]
        if old.shape != (decoder.count, decoder.spec.raw_layout.primitive_poisson):
            raise ValueError("Legacy primitive feature shape mismatch")
        result["features"] = np.concatenate(
            (old, np.full((decoder.count, 1), score, old.dtype)), axis=-1
        )
    elif isinstance(decoder, CPPNDecoder):
        n, inputs = decoder.nodes - 1, decoder.inputs
        if (
            result["weights"].shape != (n, inputs + n)
            or result["edge_mask"].shape != (n, inputs + n)
            or result["bias"].shape != (n,)
        ):
            raise ValueError("Legacy CPPN shape mismatch")
        for name in ("weights", "edge_mask"):
            old = result[name]
            enlarged = np.zeros((n + 1, inputs + n + 1), old.dtype)
            enlarged[:n, : inputs + n] = old
            if name == "edge_mask":
                # Match independent semantic-head parameterizations: a new
                # head may read inputs/hidden nodes, never prior output heads.
                enlarged[n, : inputs + decoder.hidden] = 1
            result[name] = enlarged
        result["bias"] = np.concatenate((result["bias"], np.asarray([score], result["bias"].dtype)))
    else:
        raise ValueError("Automatic legacy raw-layout migration supports built-in decoders only")
    return result


def set_poisson(decoder: Decoder, embodiment: Embedding, value: ArrayLike) -> HostEmbedding:
    """Return a copied built-in embedding with a uniform physical nu field.

    Only the appended coefficients (or CPPN output row/bias) change. This host
    convenience supports the closed bounds including 0.25/0.45; spatial fields
    are programmed by varying those coefficients instead. Remove any legacy
    global override before calling, since it would mask the local field.
    """
    if "poisson" in embodiment and "poisson" not in decoder.space().spaces:
        raise ValueError("Remove the legacy global poisson override before setting its local field")
    errors = check_embedding(decoder, embodiment)
    if errors:
        raise ValueError(f"Invalid embedding: {errors}")
    value = np.asarray(value)
    lo, hi = decoder.spec.poisson_bounds
    if (
        value.shape != ()
        or value.dtype.kind not in "iuf"
        or not np.isfinite(value)
        or not lo <= value <= hi
    ):
        raise ValueError("Uniform Poisson ratio must be a finite scalar within the field bounds")
    result = {k: np.asarray(v).copy() for k, v in embodiment.items()}
    score = _poisson_score(value, decoder.spec)
    if isinstance(decoder, GridDecoder):
        result["values"][..., decoder.spec.poisson_index] = score
    elif isinstance(decoder, PrimitiveDecoder):
        result["features"][..., decoder.spec.raw_layout.primitive_poisson] = score
    elif isinstance(decoder, CPPNDecoder):
        result["weights"][-1] = 0
        result["bias"][-1] = score
        result["edge_mask"][-1] = 0
        result["edge_mask"][-1, : decoder.inputs + decoder.hidden] = 1
    else:
        raise ValueError("set_poisson supports built-in decoders only")
    return result


def check_embedding(decoder: Decoder, embodiment: object) -> list[str]:
    """Strict finite/type/shape checks without silent coercion of integer labels."""
    s = decoder.space()
    if isinstance(embodiment, dict) and hasattr(decoder, "upgrade_embedding"):
        try:
            embodiment = decoder.upgrade_embedding(embodiment)
        except (TypeError, ValueError, KeyError):
            return ["EMBEDDING_UPGRADE"]
    if not isinstance(embodiment, dict) or set(embodiment) != set(s.spaces):
        return ["EMBEDDING_KEYS"]
    errors = []
    for k, b in s.spaces.items():
        try:
            a = np.asarray(embodiment[k])
        except (ValueError, TypeError):
            errors.append(f"TYPE:{k}")
            continue
        if a.shape != b.shape:
            errors.append(f"SHAPE:{k}")
            continue
        if a.dtype.kind not in "biuf":
            errors.append(f"TYPE:{k}")
            continue
        if not np.can_cast(a.dtype, b.dtype):
            errors.append(f"DTYPE:{k}")
            continue
        if not np.all(np.isfinite(a)):
            errors.append(f"NONFINITE:{k}")
            continue
        if np.issubdtype(b.dtype, np.integer) and a.dtype.kind not in "biu":
            errors.append(f"INTEGER_REQUIRED:{k}")
            continue
        if np.any(a < b.low) or np.any(a > b.high):
            errors.append(f"BOUNDS:{k}")
    if not errors:
        errors.extend(decoder.structural_errors(embodiment))
    return errors


def cast_embedding(decoder: Decoder, embodiment: Embedding) -> dict[str, jax.Array]:
    """Convert validated coefficients to JAX arrays using each space dtype."""
    if hasattr(decoder, "upgrade_embedding"):
        embodiment = decoder.upgrade_embedding(embodiment)
    return {k: jnp.asarray(embodiment[k], dtype=s.dtype) for k, s in decoder.space().spaces.items()}
