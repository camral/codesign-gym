"""Inherited random-body sampling choices, separated from decoder mathematics.

These values are hand-chosen priors, not uniform space sampling or calibrated
physics. Their original tuning history is unknown; this audit preserves every
value and RNG draw. Lengths/radii use normalized material coordinates [-1,1],
angles use radians, and neural weights/scores are dimensionless. Wider 3D torso
and limb ranges are inherited sampling heuristics, not guarantees of connectivity.
Integer ranges follow NumPy's lower-inclusive, upper-exclusive convention.
"""

from dataclasses import dataclass

SCORE_MARGIN_LIMIT = 5.9  # Keep sampled scores inside the decoder's +/-6 coefficient Box.
DIRECTION_NORM_FLOOR = 1e-12  # Prevent division by zero for sampled directions.
SEGMENT_NORM2_FLOOR = 1e-12  # Protect a squared segment length; a distinct contract.
DEFAULT_SAMPLE_GRIDS = {2: (12, 10), 3: (8, 6, 6)}
MIN_SAMPLE_AXIS = 3  # Retain enough sites for the local topology tests.
PRIMITIVE_CAPACITY = {2: 8, 3: 10}
BSPLINE_COEFFICIENTS_PER_AXIS = 8
CPPN_HIDDEN_CAPACITY = 32
MPM_SAMPLING_REFINEMENT = 2  # Validate both cell centres and the 2-per-axis particle lattice.
OCCUPANCY_FRACTION_BOUNDS = (0.08, 0.8)  # Reject nearly empty or nearly full sampled bodies.


@dataclass(frozen=True)
class PropertyPrior:
    """Smooth spatial material scores and Voronoi-like actuator routing."""

    fourier_features: int = 4
    frequency_std: float = 1.5
    score_std: float = 0.32
    direction_slope: float = 0.6
    actuator_logits: tuple[float, float, float] = (-1.5, 1.0, -0.5)
    channel_center_bounds: tuple[float, float] = (-0.65, 0.65)
    channel_distance_scale: float = 2.0
    poisson_feature_weights: tuple[float, float] = (0.5, 0.25)  # Second feature is subtracted.


@dataclass(frozen=True)
class PrimitivePrior:
    """An ellipsoid torso with mostly capsule limbs, before topology rejection."""

    min_enabled: int = 4
    center_jitter: float = 0.06
    torso_radii_2d: tuple[float, float] = (0.27, 0.45)
    torso_radii_3d: tuple[float, float] = (0.42, 0.57)
    torso_x_extension: float = 0.06
    angle_jitter: float = 0.32
    limb_length: tuple[float, float] = (0.35, 0.66)
    limb_radius_2d: tuple[float, float] = (0.11, 0.19)
    limb_radius_3d: tuple[float, float] = (0.20, 0.28)
    attachment_radius: float = 0.10
    half_size_noise: tuple[float, float] = (0.0, 0.07)
    capsule_probability: float = 0.85


@dataclass(frozen=True)
class DirectPrior:
    """Branching random walks painted into direct coefficient grids."""

    center_jitter: float = 0.08
    torso_radii_2d: tuple[float, float] = (0.24, 0.37)
    torso_radii_3d: tuple[float, float] = (0.57, 0.68)
    torso_score_scale: float = 0.25
    branch_count: tuple[int, int] = (3, 7)
    angle_jitter: float = 0.4
    branch_width_2d: tuple[float, float] = (0.12, 0.20)
    branch_width_3d: tuple[float, float] = (0.23, 0.30)
    segment_count: tuple[int, int] = (2, 5)
    direction_noise_std: float = 0.30
    segment_length: tuple[float, float] = (0.14, 0.24)
    max_extent: float = 0.72
    width_multiplier: tuple[float, float] = (0.86, 1.02)
    output_score_scale: float = 8.0


@dataclass(frozen=True)
class BSplinePrior:
    """Overlapping coefficient-space lobes smoothed by cubic interpolation."""

    center_jitter: float = 0.06
    torso_radii_2d: tuple[float, float] = (0.29, 0.43)
    torso_radii_3d: tuple[float, float] = (0.47, 0.60)
    lobe_count: tuple[int, int] = (3, 7)
    angle_jitter: float = 0.4
    lobe_offset: tuple[float, float] = (0.25, 0.54)
    lobe_radii_2d: tuple[float, float] = (0.23, 0.36)
    lobe_radii_3d: tuple[float, float] = (0.37, 0.47)
    output_score_scale: float = 2.0
    coefficient_noise_std: float = 0.17


@dataclass(frozen=True)
class CPPNPrior:
    """Sigmoid half-space gates feeding a radial geometry output and property heads."""

    lobes_2d: int = 5
    lobes_3d: int = 4
    angle_jitter: float = 0.5
    lobe_offset: tuple[float, float] = (0.24, 0.55)
    lobe_radius: tuple[float, float] = (0.23, 0.39)
    gate_slope: float = 6.0
    gate_combination_scale: float = 3.0
    gate_bias: float = -4.3
    radial_envelope_weight: float = -5.8
    lobe_output_weight: tuple[float, float] = (1.8, 3.8)
    coordinate_output_weight: tuple[float, float] = (-0.25, 0.25)
    geometry_bias: tuple[float, float] = (0.65, 1.1)
    per_lobe_bias_penalty: float = 0.20
    property_coordinate_std: float = 0.4
    property_gate_std: float = 0.35
    channel_center_bounds: tuple[float, float] = (-0.7, 0.7)
    poisson_young_weight: float = 0.5
    poisson_density_weight: float = 0.3  # Density output is subtracted from the Young output.


PROPERTY_PRIOR = PropertyPrior()
PRIMITIVE_PRIOR = PrimitivePrior()
DIRECT_PRIOR = DirectPrior()
BSPLINE_PRIOR = BSplinePrior()
CPPN_PRIOR = CPPNPrior()
