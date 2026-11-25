"""Reproducible random creatures in the four general design representations.

These are deliberately structured random priors, not optimized morphologies or
uniform samples of the design spaces. Only embedding parameters are generated:
the ordinary decoders supply geometry, materials and actuators. Rejection never
repairs a decoded occupancy mask, so saved designs reproduce the accepted body.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from .. import topology_checks
from ..types import Embedding, HostArray, HostEmbedding, Shape
from .decoders import (
    BSplineDecoder,
    CPPNDecoder,
    Decoder,
    Fields,
    FieldSpec,
    GridDecoder,
    PrimitiveDecoder,
    baseline_raw,
    check_embedding,
    grid_points,
)
from .priors import (
    BSPLINE_COEFFICIENTS_PER_AXIS,
    BSPLINE_PRIOR,
    CPPN_HIDDEN_CAPACITY,
    CPPN_PRIOR,
    DEFAULT_SAMPLE_GRIDS,
    DIRECT_PRIOR,
    DIRECTION_NORM_FLOOR,
    MIN_SAMPLE_AXIS,
    MPM_SAMPLING_REFINEMENT,
    OCCUPANCY_FRACTION_BOUNDS,
    PRIMITIVE_CAPACITY,
    PRIMITIVE_PRIOR,
    PROPERTY_PRIOR,
    SCORE_MARGIN_LIMIT,
    SEGMENT_NORM2_FLOOR,
)

REPRESENTATIONS = ("primitives", "direct", "bspline", "cppn")
PRIORS = {
    "primitives": "An ellipsoid torso with random overlapping capsule limbs and rounded appendages; primitive-specific properties.",
    "direct": "Branching random walks painted into direct grid coefficients, with random widths and smooth spatial property fields.",
    "bspline": "Random overlapping anisotropic coefficient lobes and smooth coefficient noise, interpreted by the native cubic B-spline decoder.",
    "cppn": "A radial torso prior plus random sigmoid half-space gates and lobe connections; all geometry and properties are network outputs.",
}


class GenerationError(ValueError):
    """The bounded rejection sampler could not find an admissible embedding."""

    def __init__(self, report: dict[str, object]) -> None:
        self.report = report
        super().__init__(
            f"No valid {report['representation']} morphology in "
            f"{report['attempts']} attempts: {report['rejection_reasons']}"
        )


def cell_center_points(shape: Shape) -> HostArray:
    """Normalized material sample sites, in the drivers' flattening order."""
    shape = tuple(int(n) for n in shape)
    if len(shape) not in (2, 3) or any(n < 1 for n in shape):
        raise ValueError("shape must contain two or three positive dimensions")
    return np.stack(
        np.meshgrid(*[(np.arange(n) + 0.5) * 2.0 / n - 1.0 for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, len(shape))


def sample_fields(
    decoder: Decoder, embodiment: Embedding, shape: Shape
) -> tuple[HostArray, Fields]:
    """Return normalized points and actual decoded fields on a CPU lattice."""
    if len(shape) != decoder.spec.dim:
        raise ValueError("sample shape and decoder dimensions must agree")
    points = cell_center_points(shape).astype(decoder.dtype)
    with jax.default_device(jax.devices("cpu")[0]):
        fields = jax.block_until_ready(
            decoder.decode({k: jnp.asarray(v) for k, v in embodiment.items()}, jnp.asarray(points))
        )
    return points, fields


def _unit(rng: np.random.Generator, dim: int) -> HostArray:
    """Draw one host unit direction with a protected zero-length normalization."""
    direction = rng.normal(size=dim)
    return direction / max(np.linalg.norm(direction), DIRECTION_NORM_FLOOR)


def _properties(
    points: HostArray, spec: FieldSpec, rng: np.random.Generator, dtype: DTypeLike
) -> HostArray:
    """Bounded smooth material scores and a spatial Voronoi actuator map."""
    flat = np.asarray(points).reshape(-1, spec.dim)
    raw = baseline_raw(flat, spec, dtype)
    frequency = rng.normal(
        0.0, PROPERTY_PRIOR.frequency_std, (spec.dim, PROPERTY_PRIOR.fourier_features)
    )
    features = np.sin(
        flat @ frequency + rng.uniform(-np.pi, np.pi, PROPERTY_PRIOR.fourier_features)
    )
    raw[:, spec.raw_layout.young : spec.raw_layout.direction.start] = features @ rng.normal(
        0.0, PROPERTY_PRIOR.score_std, (PROPERTY_PRIOR.fourier_features, 3)
    )
    raw[:, spec.raw_layout.direction] = (
        rng.normal(size=spec.dim) + PROPERTY_PRIOR.direction_slope * flat
    )
    raw[:, spec.raw_layout.actuator_logits] = np.array(PROPERTY_PRIOR.actuator_logits)
    # The sampler uses directional actuators, with varying directions, strengths
    # and routing. Other actuator kinds remain available through decoder edits.
    centers = rng.uniform(*PROPERTY_PRIOR.channel_center_bounds, (spec.channels, spec.dim))
    raw[:, spec.raw_layout.channel_logits] = -PROPERTY_PRIOR.channel_distance_scale * np.sum(
        (flat[:, None] - centers[None]) ** 2, axis=-1
    )
    # Reuse sampled features so adding a material field does not change the
    # random draws that determine the existing creature geometry.
    raw[:, spec.poisson_index] = (
        PROPERTY_PRIOR.poisson_feature_weights[0] * features[:, 0]
        - PROPERTY_PRIOR.poisson_feature_weights[1] * features[:, 1]
    )
    return np.clip(
        raw.reshape(points.shape[:-1] + (spec.width,)), -SCORE_MARGIN_LIMIT, SCORE_MARGIN_LIMIT
    ).astype(dtype)


def _primitives(decoder: PrimitiveDecoder, rng: np.random.Generator) -> HostEmbedding:
    """Sample a connected torso/limb prior using overlapping primitive coefficients."""
    emb, dim = decoder.base(), decoder.spec.dim
    count = int(rng.integers(PRIMITIVE_PRIOR.min_enabled, decoder.count + 1))
    emb["enabled"][:count] = 1
    emb["enabled"][count:] = 0
    center = rng.uniform(-PRIMITIVE_PRIOR.center_jitter, PRIMITIVE_PRIOR.center_jitter, dim)
    emb["center"][0] = center
    emb["kind"][0] = 0
    emb["half_size"][0] = (
        rng.uniform(*PRIMITIVE_PRIOR.torso_radii_2d, dim)
        if dim == 2
        else rng.uniform(*PRIMITIVE_PRIOR.torso_radii_3d, dim)
    )
    emb["half_size"][0, 0] += PRIMITIVE_PRIOR.torso_x_extension
    # Radial limbs need not be bilateral or have any prescribed leg count.
    for i in range(1, count):
        direction = _unit(rng, dim)
        if dim == 2:
            angle = 2 * np.pi * (i - 1) / (count - 1) + rng.uniform(
                -PRIMITIVE_PRIOR.angle_jitter, PRIMITIVE_PRIOR.angle_jitter
            )
            direction = np.array([np.cos(angle), np.sin(angle)])
        length = rng.uniform(*PRIMITIVE_PRIOR.limb_length)
        radius = (
            rng.uniform(*PRIMITIVE_PRIOR.limb_radius_2d)
            if dim == 2
            else rng.uniform(*PRIMITIVE_PRIOR.limb_radius_3d)
        )
        start, end = (
            center + PRIMITIVE_PRIOR.attachment_radius * direction,
            center + length * direction,
        )
        emb["center"][i] = (start + end) / 2
        emb["axis"][i] = direction
        emb["half_length"][i] = np.linalg.norm(end - start) / 2
        emb["radius"][i] = radius
        emb["half_size"][i] = radius + rng.uniform(*PRIMITIVE_PRIOR.half_size_noise, dim)
        emb["kind"][i] = 1 if rng.random() < PRIMITIVE_PRIOR.capsule_probability else 0
    emb["features"] = _properties(emb["center"], decoder.spec, rng, decoder.dtype)[:, 1:]
    return emb


def _segment_distance(points: HostArray, start: HostArray, end: HostArray) -> HostArray:
    """Compute host distances to the finite segment from start to end."""
    axis = end - start
    fraction = np.clip(
        np.sum((points - start) * axis, axis=-1) / max(np.sum(axis**2), SEGMENT_NORM2_FLOOR),
        0.0,
        1.0,
    )
    return np.linalg.norm(points - start - fraction[..., None] * axis, axis=-1)


def _direct(decoder: GridDecoder, rng: np.random.Generator) -> HostEmbedding:
    """Paint a branching prior into direct coefficients without post-decode repairs."""
    dim = decoder.spec.dim
    points = grid_points(decoder.shape)
    raw = _properties(points, decoder.spec, rng, decoder.dtype)
    center = rng.uniform(-DIRECT_PRIOR.center_jitter, DIRECT_PRIOR.center_jitter, dim)
    radii = (
        rng.uniform(*DIRECT_PRIOR.torso_radii_2d, dim)
        if dim == 2
        else rng.uniform(*DIRECT_PRIOR.torso_radii_3d, dim)
    )
    score = (
        1.0 - np.linalg.norm((points - center) / radii, axis=-1)
    ) * DIRECT_PRIOR.torso_score_scale
    branches = int(rng.integers(*DIRECT_PRIOR.branch_count))
    for branch in range(branches):
        direction = _unit(rng, dim)
        if dim == 2:
            angle = 2 * np.pi * branch / branches + rng.uniform(
                -DIRECT_PRIOR.angle_jitter, DIRECT_PRIOR.angle_jitter
            )
            direction = np.array([np.cos(angle), np.sin(angle)])
        start = center.copy()
        width = (
            rng.uniform(*DIRECT_PRIOR.branch_width_2d)
            if dim == 2
            else rng.uniform(*DIRECT_PRIOR.branch_width_3d)
        )
        for _ in range(int(rng.integers(*DIRECT_PRIOR.segment_count))):
            direction = direction + rng.normal(0.0, DIRECT_PRIOR.direction_noise_std, dim)
            direction /= np.linalg.norm(direction)
            end = np.clip(
                start + rng.uniform(*DIRECT_PRIOR.segment_length) * direction,
                -DIRECT_PRIOR.max_extent,
                DIRECT_PRIOR.max_extent,
            )
            score = np.maximum(score, width - _segment_distance(points, start, end))
            start, width = end, width * rng.uniform(*DIRECT_PRIOR.width_multiplier)
    raw[..., decoder.spec.raw_layout.occupancy] = np.clip(
        DIRECT_PRIOR.output_score_scale * score, -SCORE_MARGIN_LIMIT, SCORE_MARGIN_LIMIT
    )
    return {"values": raw}


def _bspline(decoder: BSplineDecoder, rng: np.random.Generator) -> HostEmbedding:
    """Sample smooth coefficient lobes for the native cubic B-spline decoder."""
    dim = decoder.spec.dim
    points = grid_points(decoder.shape)
    raw = _properties(points, decoder.spec, rng, decoder.dtype)
    center = rng.uniform(-BSPLINE_PRIOR.center_jitter, BSPLINE_PRIOR.center_jitter, dim)
    radii = (
        rng.uniform(*BSPLINE_PRIOR.torso_radii_2d, dim)
        if dim == 2
        else rng.uniform(*BSPLINE_PRIOR.torso_radii_3d, dim)
    )
    score = 1.0 - np.sum(((points - center) / radii) ** 2, axis=-1)
    lobes = int(rng.integers(*BSPLINE_PRIOR.lobe_count))
    for i in range(lobes):
        direction = _unit(rng, dim)
        if dim == 2:
            angle = i * 2 * np.pi / lobes + rng.uniform(
                -BSPLINE_PRIOR.angle_jitter, BSPLINE_PRIOR.angle_jitter
            )
            direction = np.array([np.cos(angle), np.sin(angle)])
        location = center + direction * rng.uniform(*BSPLINE_PRIOR.lobe_offset)
        radii = (
            rng.uniform(*BSPLINE_PRIOR.lobe_radii_2d, dim)
            if dim == 2
            else rng.uniform(*BSPLINE_PRIOR.lobe_radii_3d, dim)
        )
        score = np.maximum(score, 1.0 - np.sum(((points - location) / radii) ** 2, axis=-1))
    # Coefficients, not samples of the final surface: cubic interpolation changes
    # these lobes into a genuinely smooth field, including their union corners.
    raw[..., decoder.spec.raw_layout.occupancy] = np.clip(
        BSPLINE_PRIOR.output_score_scale * score
        + rng.normal(0.0, BSPLINE_PRIOR.coefficient_noise_std, decoder.shape),
        -SCORE_MARGIN_LIMIT,
        SCORE_MARGIN_LIMIT,
    )
    return {"values": raw}


def _cppn(decoder: CPPNDecoder, rng: np.random.Generator) -> HostEmbedding:
    """Sample network coefficients for the radial torso and random appendage prior."""
    dim, spec = decoder.spec.dim, decoder.spec
    emb = decoder.base()
    weights, bias = emb["weights"], emb["bias"]
    weights[:] = 0.0
    bias[:] = 0.0
    row = 0
    blobs = CPPN_PRIOR.lobes_2d if dim == 2 else CPPN_PRIOR.lobes_3d
    lobe_nodes = []
    for lobe in range(blobs):
        direction = _unit(rng, dim)
        if dim == 2:
            angle = 2 * np.pi * lobe / blobs + rng.uniform(
                -CPPN_PRIOR.angle_jitter, CPPN_PRIOR.angle_jitter
            )
            direction = np.array([np.cos(angle), np.sin(angle)])
        center = direction * rng.uniform(*CPPN_PRIOR.lobe_offset)
        radius = rng.uniform(*CPPN_PRIOR.lobe_radius, dim)
        first = row
        for axis in range(dim):
            for sign in (-1.0, 1.0):
                weights[row, axis] = sign * CPPN_PRIOR.gate_slope
                bias[row] = np.clip(
                    CPPN_PRIOR.gate_slope * (radius[axis] - sign * center[axis]),
                    -SCORE_MARGIN_LIMIT,
                    SCORE_MARGIN_LIMIT,
                )
                row += 1
        weights[row, decoder.inputs + first : decoder.inputs + row] = (
            CPPN_PRIOR.gate_combination_scale / dim
        )
        bias[row] = CPPN_PRIOR.gate_bias
        lobe_nodes.append(row)
        row += 1
    geometry = decoder.hidden
    # A negative radius edge confines the network-generated shape to the body
    # envelope and joins its randomly gated protrusions through a central core.
    weights[geometry, dim] = CPPN_PRIOR.radial_envelope_weight
    weights[geometry, decoder.inputs + np.asarray(lobe_nodes)] = rng.uniform(
        *CPPN_PRIOR.lobe_output_weight, blobs
    )
    weights[geometry, :dim] = rng.uniform(*CPPN_PRIOR.coordinate_output_weight, dim)
    bias[geometry] = (
        rng.uniform(*CPPN_PRIOR.geometry_bias) - CPPN_PRIOR.per_lobe_bias_penalty * blobs
    )
    # All remaining outputs are independent network property fields.
    base = baseline_raw(np.zeros((1, dim)), spec, decoder.dtype)[0]
    bias[geometry + spec.raw_layout.properties.start :] = base[spec.raw_layout.properties]
    for feature in (spec.raw_layout.young, spec.raw_layout.density, spec.raw_layout.strength):
        target = geometry + feature
        weights[target, :dim] = rng.normal(0.0, CPPN_PRIOR.property_coordinate_std, dim)
        weights[target, decoder.inputs + np.asarray(lobe_nodes)] = rng.normal(
            0.0, CPPN_PRIOR.property_gate_std, blobs
        )
    direction = _unit(rng, dim)
    bias[geometry + spec.raw_layout.direction.start : geometry + spec.raw_layout.direction.stop] = (
        direction
    )
    centers = rng.uniform(*CPPN_PRIOR.channel_center_bounds, (spec.channels, dim))
    for channel, center in enumerate(centers):
        target = geometry + spec.raw_layout.channel_logits.start + channel
        weights[target, :dim] = 2.0 * center
        bias[target] = -np.sum(center**2)
    target = geometry + spec.poisson_index
    weights[target] = (
        CPPN_PRIOR.poisson_young_weight * weights[geometry + spec.raw_layout.young]
        - CPPN_PRIOR.poisson_density_weight * weights[geometry + spec.raw_layout.density]
    )
    emb["edge_mask"] = ((weights != 0) & decoder.allowed).astype(np.int32)
    return emb


def generate_random_design(
    representation: str,
    *,
    dim: int = 2,
    seed: int = 0,
    grid_shape: Shape | None = None,
    channels: int = 4,
    dtype: DTypeLike = "float64",
    max_attempts: int = 100,
) -> tuple[Decoder, HostEmbedding, dict[str, object]]:
    """Generate one portable design, valid at cell and 2x MPM sample sites.

    Occupied fractions must be 0.08--0.8 at both lattices. Validation certifies
    sampled topology and active material, not locomotion or solver convergence.
    Every retry has an independently reproducible SeedSequence child seed.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    if np.dtype(dtype) == np.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError("float64 generation requires JAX_ENABLE_X64=1")
    grid_shape = tuple(grid_shape or DEFAULT_SAMPLE_GRIDS[dim])
    if len(grid_shape) != dim or any(
        not isinstance(n, (int, np.integer)) or n < MIN_SAMPLE_AXIS for n in grid_shape
    ):
        raise ValueError("grid_shape must match dim and have at least three cells per axis")
    decoder: Decoder
    spec = FieldSpec(dim=dim, channels=channels)
    if representation == "primitives":
        decoder = PrimitiveDecoder(spec, count=PRIMITIVE_CAPACITY[dim], dtype=dtype)
    elif representation == "direct":
        decoder = GridDecoder(spec, shape=tuple(n + 1 for n in grid_shape), dtype=dtype)
    elif representation == "bspline":
        decoder = BSplineDecoder(spec, shape=(BSPLINE_COEFFICIENTS_PER_AXIS,) * dim, dtype=dtype)
    else:
        decoder = CPPNDecoder(spec, hidden=CPPN_HIDDEN_CAPACITY, dtype=dtype)
    generate = {"primitives": _primitives, "direct": _direct, "bspline": _bspline, "cppn": _cppn}[
        representation
    ]
    shapes = (grid_shape, tuple(MPM_SAMPLING_REFINEMENT * n for n in grid_shape))
    points = np.concatenate([cell_center_points(shape) for shape in shapes]).astype(dtype)
    counts = [int(np.prod(shape)) for shape in shapes]
    report = {
        "representation": representation,
        "dim": dim,
        "seed": int(seed),
        "grid_shape": list(grid_shape),
        "sampling_prior": PRIORS[representation],
        "validation_scope": "Embedding, finite fields, occupied volume, actuation, face connectivity and local nonmanifold checks at coarse cell and 2x MPM cell centers; no dynamics certification.",
        "post_decode_repairs": False,
        "decoder": decoder.descriptor(),
    }
    rejections: Counter[str] = Counter()
    children = np.random.SeedSequence(int(seed)).spawn(int(max_attempts))
    with jax.default_device(jax.devices("cpu")[0]):
        evaluate = jax.jit(decoder.decode)
        query = jnp.asarray(points)
        for attempt, child in enumerate(children, 1):
            attempt_seed = int(child.generate_state(1, dtype=np.uint64)[0])
            emb = cast(Callable[[Decoder, np.random.Generator], HostEmbedding], generate)(
                decoder, np.random.default_rng(attempt_seed)
            )
            errors = check_embedding(decoder, emb)
            lattice_reports = []
            if not errors:
                fields = jax.block_until_ready(
                    evaluate({k: jnp.asarray(v) for k, v in emb.items()}, query)
                )
                arrays = {name: np.asarray(value) for name, value in zip(fields._fields, fields)}
                if not all(np.all(np.isfinite(value)) for value in arrays.values()):
                    errors.append("NONFINITE_FIELDS")
                offset = 0
                for lattice, shape, count in zip(("cells", "mpm_particles"), shapes, counts):
                    selection = slice(offset, offset + count)
                    occupied = arrays["occupancy"][selection] > 0
                    fraction = float(np.mean(occupied))
                    active = occupied & (arrays["strength"][selection] > 0)
                    topology_errors = topology_checks.errors(occupied, shape)
                    errors += [f"{lattice}:{error}" for error in topology_errors]
                    if not OCCUPANCY_FRACTION_BOUNDS[0] <= fraction <= OCCUPANCY_FRACTION_BOUNDS[1]:
                        errors.append(f"{lattice}:OCCUPANCY_FRACTION")
                    if not np.any(active):
                        errors.append(f"{lattice}:NO_ACTUATION")
                    lattice_reports.append(
                        {
                            "lattice": lattice,
                            "shape": list(shape),
                            "occupied_sites": int(np.count_nonzero(occupied)),
                            "occupied_fraction": fraction,
                            "active_channels": np.unique(arrays["channel"][selection][active])
                            .astype(int)
                            .tolist(),
                            "topology_errors": topology_errors,
                            "shape_sha256": hashlib.sha256(occupied.tobytes()).hexdigest(),
                        }
                    )
                    offset += count
            if errors:
                rejections.update(errors)
                continue
            report.update(
                attempts=attempt,
                attempt_seed=attempt_seed,
                rejection_reasons=dict(sorted(rejections.items())),
                lattices=lattice_reports,
                shape_sha256=lattice_reports[-1]["shape_sha256"],
                occupied_fraction=lattice_reports[-1]["occupied_fraction"],
                embedding_valid=True,
            )
            return decoder, emb, report
    report.update(attempts=int(max_attempts), rejection_reasons=dict(sorted(rejections.items())))
    raise GenerationError(report)
